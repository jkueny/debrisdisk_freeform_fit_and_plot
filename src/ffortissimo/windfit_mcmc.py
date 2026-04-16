# pylint: disable=C0103
"""
MCMC code for fitting a wind 
author: Jay Kueny
adapted from J. Mazoyer's DiskFM MCMC code

# 09/18/2024 
- Currently fitting a disk-inspired model with the IWA tightly butting up against the disk.
This is causing some parameters to go into the non-physical regime for a WDH and I think is causing degeneracies
between some params. Now modifying the prior to the beta param so that it will always be a decaying power law
from the center of the image and not a ramp to the edges of the control reg. Check posteriors after this change?

- Also IWA might need to shrink bc I think it's causing some of the oversubtraction artifacts at the minor axis.
Ref the grid search results to see if there are any correlations b/w snr and KLIP IWA.

- Need to make a noise map to use smaller IWAs, otherwise the model fits to the bright inner speckle features
and not the WDH.
"""

import os
import re
import sys
import argparse


# careful on Python 3.8 mac multiprocessing switched to spawn so the global varialbe do not work

basedir = f'{os.environ["HOME"]}/projects'  # the base directory where is
# your data (using OS environnement variable allow to use same code on
# different computer without changing this).

# default_parameter_file = 'HR4796_g_camsci2_20230312_13.yaml'  # name of the parameter file
# default_parameter_file = 'HR4796_r_camsci1_20230312_13.yaml'  # name of the parameter file
# default_parameter_file = 'HR4796a_z_lco2023a_magao-x_20230309_10.yaml'  # name of the parameter file
# default_parameter_file = 'HR4796_i_camsci1_20230309_10.yaml'  # name of the parameter file
default_parameter_file = "wdh_test.yaml"
# you can also call it with the python function argument -p

# For parallelization stuff...?
MPI = False  ## by default the MCMC is not mpi. you can change it
## in the the python function argument --mpi

import sys
import glob

# import cProfile

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# # because this error was coming up
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from multiprocessing import cpu_count


from datetime import datetime

import math as mt
import numpy as np
import pandas as pd

import astropy.io.fits as fits
# from astropy.convolution import convolve
from scipy.signal import convolve
# from scipy.signal import fftconvolve
from astropy.wcs import FITSFixedWarning

import yaml

from emcee import EnsembleSampler
from emcee import backends
from emcee.moves import StretchMove, DEMove, KDEMove

from numba.core.errors import NumbaWarning

from ffortissimo.dev.pyklip.instruments.Wind import GenericWDH

from ffortissimo.dev.pyklip.fmlib.windfm import WindFM
import ffortissimo.dev.pyklip.fm as fm

from ffortissimo.utils.masks import control_region_mask

import ffortissimo.utils.make_gpi_psf_for_disks as gpidiskpsf
import ffortissimo.utils.astro_unit_conversion as convert

# recommended by emcee https://emcee.readthedocs.io/en/stable/tutorials/parallel/
# and by PyKLIPto avoid that NumPy automatically parallelizes some operations,
# which kill the speed
os.environ["OMP_NUM_THREADS"] = "1"

def sort_parang_monotonic(filename):
    # This regex captures a signed float between "2x2bin_" and "_parang"
    match = re.search(r'bin_([-+]?\d*\.?\d+)_parang', filename)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern.")
    
def prep_image_frames_parangs(filelist, parquet_file, time_bin_sz="min"):
    wind_directions1 = []
    wind_directions2 = []
    derot_angs = []
    frames = []
    df = pd.read_parquet(parquet_file)
    df = df.reset_index()
    for ea,name in enumerate(filelist):
        dat_unit, hdr_unit = fits.getdata(name,header=True)
        # print(name, hdr_unit['PARANG'])
        derot_angs.append(hdr_unit['PARANG'])
        frames.append(dat_unit)
        df["ts6"] = df["timestamp"].str[:20]
        df["ts_dt"] = pd.to_datetime(df["ts6"], format="%Y%m%d%H%M%S%f", utc=True)
        obs = hdr_unit["DATE-OBS"]
        obs_ts = pd.to_datetime(obs, utc=True)
        df = df.set_index("ts_dt").sort_index()
        # TODO generalize this for any temporal bin size (within reason)
        obs_round = obs_ts.round(time_bin_sz) #round to nearest minute, for now
        nearest = df.reindex([obs_round], method="nearest").iloc[0]
        # print(f"Science frame at {obs_ts} → matched wind at {obs_round}")
        # one or both directions may be np.nan, meaning no wind detected
        winddir1 = nearest["direction_1"]
        winddir2 = nearest["direction_2"]
        if ea < 5:
            print(obs, nearest["timestamp"], winddir1, winddir2)
        wind_directions1.append(winddir1)
        wind_directions2.append(winddir2)
    return np.asarray(frames), np.asarray(derot_angs),  \
        np.asarray(wind_directions1), np.asarray(wind_directions2)

def gen_wdh_image(
    x: np.ndarray,
    y: np.ndarray,
    beta: float,
    h0: float,
    sigma: float,
    PA_deg: float,
    x0: float,
    gamma: float,
) -> np.ndarray:
    """
    Generate a single WDH-like intensity map on the same grid as ``x`` and ``y``.

    ``PA_deg`` is the position angle (degrees) used to rotate the halo axis; ``x0`` is an
    offset along the rotated radial coordinate (pixels, same units as ``x``, ``y``).
    """
    # At pixel scale ~0.012"/pixel this is the IWA at g', about 4 lambda/D
    r1 = 10.0
    pa_rad = -np.deg2rad(PA_deg)
    x_rot = np.sin(pa_rad) * x + np.cos(pa_rad) * y
    y_rot = np.cos(pa_rad) * x - np.sin(pa_rad) * y

    dx = x_rot - x0
    r = np.sqrt(dx**2 + dy**2)
    r_safe = np.maximum(r, 1e-6)

    power_law = (1.0 / r_safe) ** beta
    denom = h0 * (dx**2)
    denom = np.where(np.abs(denom) < 1e-12, np.copysign(1e-12, denom + 1e-30), denom)
    radial_term = (r**2 / denom) ** gamma
    exp_term = np.exp(-0.5 * (radial_term + (x_rot / sigma) ** 2))
    i_map = power_law * exp_term
    i_map = np.nan_to_num(i_map, nan=0.0, posinf=0.0, neginf=0.0)
    i_map = np.where(r < r1, 0.0, i_map)


    return i_map.astype(np.float32, copy=False)


def arr_free_params(params_mcmc_yaml):
    free_params = []
    # if bool(params_mcmc_yaml['rscale_state']):
    #     free_params.append('rscale')
    # else:
    #     # free_params.append(False)
    #     pass
    if bool(params_mcmc_yaml['hbeta_state']):
        free_params.append('hbeta')
    else:
        # free_params.append(False)
        pass

    if bool(params_mcmc_yaml['ha_r_state']):
        free_params.append('ha_r')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['hsig_state']):
        free_params.append('hsig')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['hpa_state']):
        free_params.append('hPA')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['hdx_state']):
        free_params.append('hdx')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['hdy_state']):
        free_params.append('hdy')
    else:
        # free_params.append(False)
        pass
    # if bool(params_mcmc_yaml['hoffset_state']):
    #     free_params.append('hoffset')
    # else:
    #     # free_params.append(False)
    #     pass
    if bool(params_mcmc_yaml['hN_state']):
        free_params.append('hNorm')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['dtheta_state']):
        free_params.append('dtheta')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['hbeta2_state']):
        free_params.append('hbeta2')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['ha_r2_state']):
        free_params.append('ha_r2')
    else:
        # free_params.append(False)
        pass
    # if bool(params_mcmc_yaml['hsig2_state']):
    #     free_params.append('hsig2')
    # else:
    #     # free_params.append(False)
    #     pass
    if bool(params_mcmc_yaml['hpa2_state']):
        free_params.append('hPA2')
    else:
        # free_params.append(False)
        pass
    # if bool(params_mcmc_yaml['hdx2_state']):
    #     free_params.append('hdx2')
    # else:
    #     # free_params.append(False)
    #     pass
    if bool(params_mcmc_yaml['hdy2_state']):
        free_params.append('hdy2')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['hN2_state']):
        free_params.append('hNorm2')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['dtheta2_state']):
        free_params.append('dtheta2')
    else:
        # free_params.append(False)
        pass
        # if multiwdh > 1:
        #     if bool(params_mcmc_yaml['ha_r3_state']):
        #         free_params.append('ha_r3')
        #     else:
        #         # free_params.append(False)
        #         pass
        #     if bool(params_mcmc_yaml['hpa3_state']):
        #         free_params.append('hPA3')
        #     else:
        #         # free_params.append(False)
        #         pass
        #     if bool(params_mcmc_yaml['hdy3_state']):
        #         free_params.append('hdy3')
        #     else:
        #         # free_params.append(False)
        #         pass

        #     if bool(params_mcmc_yaml['hN3_state']):
        #         free_params.append('hNorm3')
        #     else:
        #         # free_params.append(False)
        #         pass
        #     if bool(params_mcmc_yaml['dtheta3_state']):
        #         free_params.append('dtheta3')
        #     else:
        #         # free_params.append(False)
        #         pass

    print(f'Fitting params: {free_params}')
    return free_params

def from_theta_to_params(theta):
    '''
    Setup function. This function takes the prior parameters and creates a dictionary and vector of parameters from it.
    '''
    param_disk = {} #disk parameters are put into a dict.
    vector_param = [] #this is for the walker chain plots, free params only

    param_disk['beta_in'] = -10  # we fix the inner power law
    param_disk['hr1'] = 1
    param_disk['hr2'] = 60
    # param_disk['hbeta'] = theta[0]
    # param_disk['ha_r'] = theta[1]  # we fix the aspect ratio
    # # param_disk['beta'] = 1.
    # param_disk['hPA'] = theta[2]
    # param_disk['hdx'] = theta[3]
    # param_disk['hdy'] = theta[4]
    # param_disk['hNorm'] = mt.exp(theta[5])
    # all_params = ['hbeta', 'hbeta2', 'ha_r', 'hsig', 'hPA', 'hdx', 'hdy', 'hNorm']

    all_params = [
                # 'rscale',
                'hbeta',
                'ha_r',
                'hsig',
                'hPA',
                'hdx',
                'hdy',
                # 'hdz',
                'hNorm',
                'dtheta',
                # 'hbeta2',
                'ha_r2',
                # 'hsig2',
                'hPA2',
                # 'hdx2',
                'hdy2',
                # 'hdz2',
                'hNorm2',
                'dtheta2',
                ]
    fixed_params = 0
    delta_params = len(all_params) - len(FREE_PARAMS)
    for ea, p in enumerate(all_params):
        # print(ea, p, fixed_params, (ea - fixed_params))
        if p in FREE_PARAMS:
            # if (p == 'rscale'):
            #     # print(ea, p)
            #     param_disk['rscale'] = mt.exp(theta[ea - fixed_params])
            #     vector_param.append(param_disk['rscale'])
            if (p == 'hNorm'):
                # print(ea, p)
                param_disk['hNorm'] = mt.exp(theta[ea - fixed_params])
                vector_param.append(param_disk['hNorm'])
            elif (p == 'hNorm2'):
                # print(ea, p)
                param_disk['hNorm2'] = mt.exp(theta[ea - fixed_params])
                vector_param.append(param_disk['hNorm2'])
            elif (p == 'hNorm3'):
                # print(ea, p)
                param_disk['hNorm3'] = mt.exp(theta[ea - fixed_params])
                vector_param.append(param_disk['hNorm3'])
            else:
                # print(ea, p)
                param_disk[p] = theta[max(0,ea)]
                vector_param.append(param_disk[p])
        else:
            # print(ea,p)
            fixed_params += 1
            param_disk[p] = THETA_INIT[ea]
            # vector_param.append(param_disk[p])

    # We don't need the DC offset term bc the SPF gets normalized at 90
    # param_spf['dc'] = np.exp(theta[0])
    # param_spf['dc'] = 1. #fix the dc offset
    # vector_param = [
    # param_disk['hbeta'], param_disk['ha_r'],
    # param_disk['hPA'], param_disk['hdx'], param_disk['hdy'], param_disk['hNorm']
    #                 ]
    # print('in theta_to_params',theta)
    # print(theta.shape)
    # print(param_disk)
    # print(param_disk['coeffs'])
    # print(param_disk['dc'])
    # print(vector_param.shape)
    # exit()
    return param_disk, vector_param #return the disk parameter dict. and theta vector of parameters


####################################################### 
# MODEL FUNCTION #
'''
The model function should take as an argument a list representing our theta vector, and return the model evaluated at that theta.
'''

def call_gen_disk(theta):
    """ call the disk model from a set of parameters.
        
        use SPF_MODEL, DIMENSION, PIXSCALE_INS, DISTANCE_STAR
        ALIGNED_CENTER and WHEREMASK2GENERATEDISK 
        as global variables

    Args:
        theta: list of parameters of the MCMC

    Returns:
        a 2d model
    """
    param_disk, _ = from_theta_to_params(theta)
    # print(param_disk)
    R1 = param_disk['hr1']
    R2 = param_disk['hr2']
    beta = param_disk['hbeta']
    a_r = param_disk['ha_r']
    sig = param_disk['hsig']
    pa = param_disk['hPA']
    dx = param_disk['hdx']
    dy = param_disk['hdy']
    # dz = param_disk['hdz']
    # dz = 0
    Norm = param_disk['hNorm']
    # rscale = param_disk["rscale"]
    dtheta = param_disk['dtheta']

    max_fov = DIMENSION / 2. * PIXSCALE_INS  #maximum radial distance in AU from the center to the edge
    n_pts = int(np.floor(DIMENSION / 1))
    xsize = max_fov * DISTANCE_STAR  #maximum radial distance in AU from the center to the edge

    # print(f'max_fov: {max_fov}; xsize: {xsize}')

    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight
    # +ve y is going right from the center
    # +ve z is going up from the center

    # y = np.linspace(0,xsize,num=npts/2)
    y = np.linspace(-xsize, xsize, num=n_pts)
    z = np.linspace(-xsize, xsize, num=n_pts)

    model_left = fastgen_wind(ctrlrad=R2,
                         beta=beta,
                         pa=pa,
                         dx=dx, dy=dy,
                         Norm=Norm,
                         a_r=a_r,sig=sig,
                         y_arr=y,
                         z_arr=z,
                         npts=n_pts,
                         mask=WHEREMASK2GENERATEHALO,
                         left=True)
    model_right = fastgen_wind(ctrlrad=R2,
                         beta=beta,
                         pa=pa,
                         dx=dx, dy=dy,
                         Norm=Norm,
                         a_r=a_r*dtheta,sig=sig,
                         y_arr=y,
                         z_arr=z,
                         npts=n_pts,
                         mask=WHEREMASK2GENERATEHALO)
    model1 = (model_left + model_right)

    a_r2 = param_disk['ha_r2']
    # sig2 = param_disk['hsig2']
    pa2 = param_disk['hPA2']
    # dx2 = param_disk['hdx2']
    dy2 = param_disk['hdy2']
    # dz2 = param_disk['hdz2']
    Norm2 = param_disk['hNorm2']
    dtheta2 = param_disk['dtheta2']

    model2_left = fastgen_wind(ctrlrad=R2,
                        beta=beta,
                        pa=pa2,
                        dx=dx, dy=dy2,
                        Norm=Norm2,
                        a_r=a_r2,sig=sig,
                        y_arr=y,
                        z_arr=z,
                        npts=n_pts,
                        mask=WHEREMASK2GENERATEHALO,
                        left=True)
    model2_right = fastgen_wind(ctrlrad=R2,
                        beta=beta,
                        pa=pa2,
                        dx=dx, dy=dy2,
                        Norm=Norm2,
                        a_r=a_r2*dtheta2,sig=sig,
                        y_arr=y,
                        z_arr=z,
                        npts=n_pts,
                        mask=WHEREMASK2GENERATEHALO)
        
    model2 = (model2_left + model2_right)

        # if ADD_WDH > 1:
        #     a_r3 = param_disk['ha_r3']
        #     pa3 = param_disk['hPA3']
        #     dy3 = param_disk['hdy3']
        #     # dz3 = param_disk['hdz3']
        #     Norm3 = param_disk['hNorm3']
        #     dtheta3 = param_disk['dtheta3']

        #     model3_left = fastgen_wind(ctrlrad=R2,
        #                         beta=beta,
        #                         pa=pa3,
        #                         dx=dx, dy=dy3,
        #                         Norm=Norm3,
        #                         a_r=a_r3,sig=sig,
        #                         y_arr=y,
        #                         z_arr=z,
        #                         npts=n_pts,
        #                         mask=WHEREMASK2GENERATEHALO,
        #                         left=True)
        #     model3_right = fastgen_wind(ctrlrad=R2,
        #                         beta=beta,
        #                         pa=pa3,
        #                         dx=dx, dy=dy3,
        #                         Norm=Norm3,
        #                         a_r=a_r3*dtheta3,sig=sig,
        #                         y_arr=y,
        #                         z_arr=z,
        #                         npts=n_pts,
        #                         mask=WHEREMASK2GENERATEHALO)
        #     model += model3_left + model3_right
    
    # remove the nans to avoid problems when convolving
    model1[model1 != model1] = 0
    model2[model2 != model2] = 0
    # I normalize by value of a_r to avoid degenerascies between a_r and Normalization
    # model = param_spf['dc'] * model / param_disk['a_r']

    # return model, rscale
    return model1, model2


########################################################
# LOG LIKELIHOOD FUNCTION #
'''
Its job is to return a number corresponding to how good a fit your model is to your data for a given set of parameters, weighted by the error in your data points (i.e. it is more important the fit be close to data points with small error bars than points with large error bars).
'''
def logl(theta):
    """ measure the Chisquare (log of the likelyhood) of the parameter set.
        create disk
        convolve by the PSF (psf is global)
        do the forward modeling (diskFM obj is global)
        nan out when it is out of the zone (zone mask is global)
        subctract from data and divide by noise (data and noise are global)

    Args:
        theta: list of parameters of the MCMC

    Returns:
        Chisquare
    """
    # model, scaling = call_gen_disk(theta)
    model1, model2 = call_gen_disk(theta)


    # modelconvolved = convolve(model, PSF, boundary='wrap')
    # modelconvolved = fftconvolve(model, PSF, mode='same')
    model1convolved = convolve(model1, PSF, mode='same')#,method='fft')
    model2convolved = convolve(model2, PSF, mode='same')#,method='fft')
    # if RPROFSUB:
    #     modelconvolved += (MED_PROFILE_EST * scaling)
    model1convolved *= (1 - WHEREMASK2GENERATEHALO)
    model2convolved *= (1 - WHEREMASK2GENERATEHALO)
    DISKOBJ.update_wind(model1convolved, model2convolved)
    # model_fm = DISKOBJ.fm_parallelized_jit()[0]
    model_fm = DISKOBJ.fm_parallelized()[0]

    model_fm[model_fm != model_fm] = 0.

    # reduced data have already been naned outside of the minimization
    # zone, so we don't need to do it also for model_fm
    if USE_NOISE:
        res = (REDUCED_DATA - model_fm) / NOISE
    else:
        res = (REDUCED_DATA - model_fm)


    Chisquare = np.nansum(-0.5 * (res * res))

    return Chisquare


########################################################
# CHECK PRIORS FUNCTION #
'''
check, before running the probability function (last one defined) on any set of parameters, that all variables are within their priors (in fact, this is where we set our priors).

The output of this function is totally arbitrary (it is just encoding True False), but emcee asks that if all priors are satisfied, 0.0 is returned, otherwise return -np.inf. Its input is a theta vector.
'''
def logp(theta):
    """ measure the log of the priors of the parameter set.
     This function still have a lot of parameters hard coded here
     Also you can change the prior shape directly here.

    Args:
        theta: list of parameters of the MCMC

    Returns:
        log of priors
    """
    param_disk, _ = from_theta_to_params(theta)


    prior_rout = 1.
    # define the prior values


    # if param_disk['rscale'] < 1 or param_disk['rscale'] > 500:
    #     print('rscale out of prior.')
    #     return -np.inf
    # else:
    #     prior_rout = prior_rout * 1.

    if param_disk['hbeta'] < -50 or param_disk['hbeta'] > 50:
        print('hbeta out of prior.')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.


    if (param_disk['ha_r'] < -10 or param_disk['ha_r'] > 10):
        print('ha_r out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.
    if (param_disk['hsig'] < 0.01 or param_disk['hsig'] > 224):
        print('hsig out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.
    if (param_disk['hPA'] < -180 or param_disk['hPA'] > 180):
        print('hPA out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.
    if (param_disk['hdx'] < -100) or (param_disk['hdx'] > 100):  #The x offset
        print('hdx out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.
    if (param_disk['hdy'] < -6) or (param_disk['hdy'] > 6):  #The y offset
        print('hdy out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.
    if (param_disk['hNorm'] < 0.001 or param_disk['hNorm'] > 1e5):
        print('hNorm out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (param_disk['dtheta'] < 0.999 or param_disk['dtheta'] > 2.0):
        print('dtheta out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.
    # if param_disk['hbeta2'] < -50 or param_disk['hbeta2'] > 50:
    #     print('hbeta2 out of prior.')
    #     return -np.inf
    # else:
    #     prior_rout = prior_rout * 1.
    if (param_disk['ha_r2'] < -10 or param_disk['ha_r2'] > 10):
        print('ha_r2 out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.
    # if (param_disk['hsig2'] < 0.01 or param_disk['hsig2'] > 224):
    #     print('hsig2 out of prior')
    #     return -np.inf
    # else:
    #     prior_rout = prior_rout * 1.
    # if (param_disk['hPA2'] < (param_disk['hPA'] + (WDH_DPA*0.75)) or param_disk['hPA2'] > (param_disk['hPA'] + (WDH_DPA*1.25))):
    if (param_disk['hPA2'] < -180 or param_disk['hPA2'] > 180):
        print('hPA2 out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.
    # if (param_disk['hdx2'] < -100) or (param_disk['hdx2'] > 100):  #The x offset
    #     print('hdx2 out of prior')
    #     return -np.inf
    # else:
    #     prior_rout = prior_rout * 1.
    if (param_disk['hdy2'] < -6) or (param_disk['hdy2'] > 6):  #The y offset
        print('hdy2 out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.
    if (param_disk['hNorm2'] < 0.001 or param_disk['hNorm2'] > 1e5):
        print('hNorm2 out of prior', param_disk['hNorm2'])
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (param_disk['dtheta2'] < 0.999 or param_disk['dtheta2'] > 2.0):
        print('dtheta2 out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

        # if ADD_WDH > 1:
        #     if (param_disk['ha_r3'] < -10 or param_disk['ha_r3'] > 10):
        #         print('ha_r3 out of prior')
        #         return -np.inf
        #     else:
        #         prior_rout = prior_rout * 1.
        #     # if (param_disk['hPA3'] < (param_disk['hPA2'] - (WDH_DPA2*1.1)) or param_disk['hPA3'] > (param_disk['hPA2'] + (WDH_DPA2*1.1))):
        #     if (param_disk['hPA3'] < -180) or (param_disk['hPA3'] > 180):
        #         print('hPA3 out of prior')
        #         return -np.inf
        #     else:
        #         prior_rout = prior_rout * 1.
        #     if (param_disk['hdy3'] < -4) or (param_disk['hdy3'] > 4):  #The y offset
        #         print('hdy3 out of prior')
        #         return -np.inf
        #     else:
        #         prior_rout = prior_rout * 1.
        #     if (param_disk['hNorm3'] < 0.001 or param_disk['hNorm3'] > 1e5):
        #         print('hNorm3 out of prior')
        #         return -np.inf
        #     else:
        #         prior_rout = prior_rout * 1.
        #     if (param_disk['dtheta3'] < 0.999 or param_disk['dtheta3'] > 2.0):
        #         print('dtheta3 out of prior')
        #         return -np.inf
        #     else:
        #         prior_rout = prior_rout * 1.

    # otherwise ...
    return np.log(prior_rout)


########################################################
# LOG PROBABILITY FUNCTION #
'''
This function combines the steps above by running the lnprior function, and if the function returned -np.inf, passing that through as a return, and if not (if all priors are good), returning the lnlike for that model (by convention we say it’s the lnprior output + lnlike output, since lnprior’s output should be zero if the priors are good).
'''
def lnpb(theta):
    """ sum the logs of the priors (return of the logp funciton)
        and of the likelyhood (return of the logl function)


    Args:
        theta: list of parameters of the MCMC

    Returns:
        log of priors + log of likelyhood
    """
    # from datetime import datetime
    # starttime = datetime.now()
    lp = logp(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = logl(theta)
    # print("Running time model + FM: ", datetime.now() - starttime)

    return lp + ll


########################################################
def make_noise_map_rings(nodisk_data,
                         aligned_center=None,
                         delta_raddii=1):
    """ create a noise map from a image using concentring rings
        and measuring the standard deviation on them

    Args:
        nodisk_data: [dim dim] data array containing speckle without disk
        aligned_center: [pixel,pixel], position of the star in the mask
        delta_raddii: pixel, widht of the small concentric rings

    Returns:
        a [dim,dim] array where each concentric rings is at a constant value
            of the standard deviation of the reduced_data
    """
    if aligned_center == None:
        image_center = nodisk_data.shape[0] // 2, nodisk_data.shape[1] // 2
    if len(nodisk_data.shape) > 2:
        # print('Generating noise cube...')
        insitu_noise_frames = []
        height,width = nodisk_data.shape[1],nodisk_data.shape[2]
        Y,X = np.ogrid[:height,:width]
        center = (int(height/2), int(width/2))
        distance = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
        for j in range(nodisk_data.shape[0]):
            blank_noise = np.zeros_like(nodisk_data[j])
            for i in range(1,int(round(height/2))):
                radius = i
                # annulus = (distance >= radius-0.5) & (distance <= radius+0.5)
                annulus = (distance >= radius-(delta_raddii/2)) & (distance <= radius+(delta_raddii/2))
                region = (nodisk_data[j])[annulus]
                region_std = np.nanstd(region)
                if region_std == 0:
                    pass
                blank_noise[annulus] = region_std
            blank_noise[blank_noise == 0] = np.nan
            insitu_noise_frames.append(blank_noise)
        noise_map = np.asarray(insitu_noise_frames)
    else:
        dim = nodisk_data.shape[1]
        # nodisk_data[nodisk_data != nodisk_data] = 0
        # create rho2D for the rings
        x = np.arange(dim, dtype=float)[None, :] - image_center[0]
        y = np.arange(dim, dtype=float)[:, None] - image_center[1]
        rho2d = np.sqrt(x**2 + y**2)

        noise_map = np.zeros((dim, dim))
        for i_ring in range(0,
                            int(np.floor(image_center[0] / delta_raddii)) - 2):
            wh_rings = (rho2d >= i_ring * delta_raddii) & (rho2d < (i_ring + 1) * delta_raddii)
            noise_map[wh_rings] = np.nanstd(nodisk_data[wh_rings])
        # noise_map[noise_map == 0] = np.nan
    return noise_map

def create_uncertainty_map(params_mcmc_yaml):
    """ measure the uncertainty map using the counter rotation trick
    described in Sec4 of Gerard&Marois SPIE 2016 and probabaly elsewhere

    Args:
        dataset: a pyklip instance of Instrument.Data containing the data
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        delta_raddii: pixel, widht of the small concentric rings
        psflib: a PSF librairy if RDI

    Returns:
        a [dim,dim] array containing only speckles and the disk has been removed

    """
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    
    datadir = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    klipdir = os.path.join(datadir, 'klip_fm_files')

    maskfornoisemap = fits.getdata(
        os.path.join(klipdir,
                     file_prefix + '_mask2minimize.fits'))
    reduced_data = fits.getdata(
        os.path.join(klipdir,
                     file_prefix + '-klipped-KLmodes-all.fits'))
    noise = make_noise_map_rings(reduced_data * maskfornoisemap,
                                 aligned_center=aligned_center,
                                 delta_raddii=1)
    noise[noise == 0] = np.nan  #we are going to divide by this noise

    return noise




########################################################
def initialize_mask_psf_noise(params_mcmc_yaml, quietklip=True):
    """ initialize the MCMC by preparing the useful things to measure the
    likelyhood (measure the data, the psf, the uncertainty map, the masks).

    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        quietklip : if True, pyklip and DiskFM are quiet


    Returns:
        a dataset a pyklip instance of Instrument.Data
    """
    instrument = params_mcmc_yaml['INSTRUMENT']

    # if first_time=True, all the masks, reduced data, noise map, and KL vectors
    # are recalculated. be careful, for some reason the KL vectors are slightly
    # different on different machines. if you see weird stuff in the FM models
    # (for example in plotting the results), just remake them
    first_time = params_mcmc_yaml['FIRST_TIME']

    datadir = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    klipdir = os.path.join(datadir, 'wind_fm_files')

    os.makedirs(klipdir, exist_ok=True)

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    #The PSF centers
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    x_off = params_mcmc_yaml['MASK_DX']
    y_off = params_mcmc_yaml['MASK_DY']
    mask_center = aligned_center[0] + x_off, aligned_center[1] + y_off 
    eff_wl = params_mcmc_yaml['WL']
    noise_multiplication_factor = params_mcmc_yaml["NOISE_MULTIPLICATION_FACTOR"]
    mask_speckles = params_mcmc_yaml['MASK_SPECKLES']
    which_wind_parquet = params_mcmc_yaml["WIND_LOOKUP"]
    path_wind_parquet = f"{basedir}/{which_wind_parquet}"
    print(f"Read wind lookup table -> {path_wind_parquet}")

    ### This is the only part of the code different for GPI IFS anf SPHERE
    # For SPHERE We load and crop the PSF and the parangs
    # For GPI, we load the raw data, emasure hte PSF from sat spots and
    # collaspe the data
    if first_time:
        if instrument == 'MagAO-X':
            filelist = sorted(glob.glob(f'{datadir}/*parang.fits'), key=sort_parang_monotonic)
            if len(filelist) == 0:
                raise ValueError(f"Could not find files in the dir: {datadir}")

            input_data, par_angs, wdh1_angs, wdh2_angs = prep_image_frames_parangs(filelist,
                                                                                   path_wind_parquet,
                                                                                   )
            input_centers = np.array([aligned_center for _ in range(len(filelist))])
            # IWA = 10#use 10 for now, which is ~1.5 lambda/d JKK 01/08/22
            IWA = params_mcmc_yaml['IWA']#use 13 for now, post-optimized bkg sub SNRE says JKK 01/18/23

            dataset = GenericWDH(input_data,
                                 input_centers,
                                 obj_parangs=par_angs,
                                 wdh1_parangs=wdh1_angs,
                                 wdh2_parangs=wdh2_angs,
                                 IWA=IWA,filenames=filelist)
        else:
            print("Unknown instrument. Exiting the script.")
            sys.exit("An error occurred due to not knowing which instrument.")

        #After this, this is for both GPI and SPHERE
        #define the outer working angle
        dataset.OWA = params_mcmc_yaml['OWA']

        if dataset.input.shape[1] != dataset.input.shape[2]:
            raise ValueError(""" Data slices are not square (dimx!=dimy), 
                            please make them square""")

        #create the masks
        #create the mask where the non convoluted disk is going to be generated.
        # To gain time, it is ~tightely adjusted to the expected models BEFORE
        # convolution. Inded, the models are generated pixel by pixels. 0.1 s
        # gained on every model is a day of calculation gain on one million model,
        # so adjust your mask tightly to your model. You can change the harcoded parameter
        # here if you neet to go faster (reduced it) or it the slope beta is very slow (increase it)
        print(
            "\n Create the binary masks to define model zone and chisquare zone"
        )   
        in_scaling = params_mcmc_yaml['MASK_IN_SCALING'] #originally 18
        out_scaling = params_mcmc_yaml['MASK_OUT_SCALING'] #originally 18

        # mask2generatedisk = 1 - mask_disk_zeros
        lyot_sm_rad = 3 #lambda / D
        ctrl_rad = 24 #lambda / D

        apdiam = 6.5 #m
        reselem = eff_wl * 1e-6 / apdiam * 180 / np.pi * 3600 #arcseconds
        reselem_pix = reselem / PIXSCALE_INS #num pixels, int
        coron_reg = lyot_sm_rad * reselem_pix
        # MagAO-X dark hole is square-shaped
        seeing_limited = ctrl_rad * reselem_pix * np.sqrt(2)
        mask2generatehalo = control_region_mask(dataset.input.shape[1:],
                                                coronrad=coron_reg,
                                                seeinglimited=seeing_limited,
                                                )
        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_mask2generatehalo.fits'),
                     mask2generatehalo,
                     overwrite='True')
        # we create a second mask for the minimization a little bit larger
        # (because model expect to grow with the PSF convolution and the FM)
        # and we can also exclude the center region where there are too much speckles
        mask_disk_zeros = gpidiskpsf.make_disk_mask(
            dataset.input.shape[1],
            params_mcmc_yaml['pa_init'],
            params_mcmc_yaml['inc_init'],
            convert.au_to_pix(params_mcmc_yaml['r1_init'],
                              params_mcmc_yaml['PIXSCALE_INS'],
                              params_mcmc_yaml['DISTANCE_STAR']) -
            in_scaling / np.cos(np.radians(params_mcmc_yaml['inc_init'])),
            convert.au_to_pix(params_mcmc_yaml['r2_init'],
                              params_mcmc_yaml['PIXSCALE_INS'],
                              params_mcmc_yaml['DISTANCE_STAR']) +
            out_scaling / np.cos(np.radians(params_mcmc_yaml['inc_init'])),
            aligned_center=mask_center)
        # mask2minimize = (1 - mask_disk_zeros)
        mask2minimize = mask_disk_zeros * mask2generatehalo

        ### a few lines to create a circular central mask to hide center regions with a lot
        ### of speckles. Currently not using it but it's there
        mask_coron_region = np.ones((dataset.input.shape[1], dataset.input.shape[2]))
        mask_speckle_region = np.ones((dataset.input.shape[1], dataset.input.shape[2]))
        x = np.arange(dataset.input.shape[1], dtype=float)[None,:] - aligned_center[0]
        y = np.arange(dataset.input.shape[2], dtype=float)[:,None] - aligned_center[1]
        rho2d = np.sqrt(x**2 + y**2)
        mask_coron_region[np.where(rho2d < coron_reg)] = 0.
        mask_speckle_region[np.where(rho2d < mask_speckles)] = 0.
        mask2minimize = mask2minimize*mask_speckle_region
        mask2minimize[np.where(mask2minimize < 0.5)] = 0
        mask2minimize[np.where(mask2minimize > 0.5)] = 1

        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_mask2minimize.fits'),
                     mask2minimize,
                     overwrite='True')

        # Disable print for pyklip
        if quietklip:
            sys.stdout = open(os.devnull, 'w')

        sys.stdout = sys.__stdout__
        
        noise = create_uncertainty_map(
                                       params_mcmc_yaml
                                       )
        noise *= noise_multiplication_factor
        fits.writeto(os.path.join(klipdir, file_prefix + '_noisemap.fits'),
                     noise,
                     overwrite='True')
        psflib = None
    else:
        psflib = None
        dataset = None

    return dataset, psflib


########################################################
# Removed RDI function. 07/27/2024 JKK


########################################################
def initialize_windfm(dataset, params_mcmc_yaml, psflib=None, quietklip=True):
    """ initialize the MCMC by preparing the diskFM object

    Args:
        dataset: a pyklip instance of Instrument.Data
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        psflib : a librairy of PSF if RDI
        quietklip : if True, pyklip and DiskFM are quiet

    Returns:
        a  diskFM object
    """
    print("\n Initialize WindFM")
    first_time = params_mcmc_yaml['FIRST_TIME']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]
    move_here = params_mcmc_yaml['MOVE_HERE']
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    mode = params_mcmc_yaml['MODE']
    annuli = params_mcmc_yaml['ANNULI']
    hp = params_mcmc_yaml['HP_FILTER']
    datadir = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    klipdir = os.path.join(datadir, 'wind_fm_files')


    if first_time:
        # create a first model to check the begining parameter and initialize the FM.
        # We will clear all useless variables befire starting the MCMC
        # Be careful that this model is close to what you think is the minimum
        # because the FM is not completely linear so you have to measure the FM on
        # something already close to the best one

        # theta_init = from_param_to_theta_init(params_mcmc_yaml)
        theta_init = THETA_INIT


        #generate the model
        # model_here, scaling_here = call_gen_disk(theta_init)
        model1_here, model2_here = call_gen_disk(theta_init)

        fits.writeto(os.path.join(klipdir, file_prefix + '_FirstModel.fits'),
                     model1_here + model2_here,
                     overwrite='True')

        # model_here_convolved = convolve(model_here, PSF, boundary='wrap')
        # model_here_convolved = fftconvolve(model_here, PSF, mode='same')
        model1_here_convolved = convolve(model1_here, PSF, mode='same')#,method='fft')
        model2_here_convolved = convolve(model2_here, PSF, mode='same')#,method='fft')
        # if RPROFSUB:
        #     model_here_convolved += (MED_PROFILE_EST*scaling_here)

        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_FirstModel1_Conv.fits'),
                     model1_here_convolved,
                     overwrite='True')
        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_FirstModel2_Conv.fits'),
                     model2_here_convolved,
                     overwrite='True')

    else:
        model1_here_convolved = fits.getdata(
        os.path.join(klipdir, file_prefix + '_FirstModel1_Conv.fits'))
        model2_here_convolved = fits.getdata(
        os.path.join(klipdir, file_prefix + '_FirstModel2_Conv.fits'))

    if first_time:
        # Disable print for pyklip
        # if quietklip:
        #     sys.stdout = open(os.devnull, 'w')
        # initialize the DiskFM object

        diskobj = WindFM(dataset.input.shape,
                         numbasis,
                         dataset,
                         model_wdh1=model1_here_convolved,
                         model_wdh2=model2_here_convolved,
                         basis_filename=os.path.join(
                             klipdir, file_prefix + '_klbasis.h5'),
                         save_basis=True,
                         aligned_center=aligned_center)
        # measure the KL basis and save it

        maxnumbasis = dataset.input.shape[0]
        if hp > 0:
            fm.klip_dataset(dataset,
                            diskobj,
                            numbasis=numbasis,
                            maxnumbasis=maxnumbasis,
                            annuli=annuli,
                            subsections=1,
                            mode=mode,
                            outputdir=klipdir,
                            fileprefix=file_prefix,
                            aligned_center=aligned_center,
                            mute_progression=True,
                            highpass=hp,
                            minrot=move_here,
                            calibrate_flux=False,
                            numthreads=1,
                            time_collapse='median',
                            psf_library=psflib)
        else:
            fm.klip_dataset(dataset,
                            diskobj,
                            numbasis=numbasis,
                            maxnumbasis=maxnumbasis,
                            annuli=annuli,
                            mode=mode,
                            subsections=1,
                            outputdir=klipdir,
                            fileprefix=file_prefix,
                            aligned_center=aligned_center,
                            mute_progression=True,
                            highpass=False,
                            minrot=move_here,
                            calibrate_flux=False,
                            numthreads=1,
                            time_collapse='median',
                            psf_library=psflib)

        sys.stdout = sys.__stdout__
        reduced_data = fits.getdata(os.path.join(klipdir,
                                                 file_prefix + '-klipped-KLmodes-all.fits'))[0]
        noisemap_rings = make_noise_map_rings(reduced_data)
        noisemap_rings[noisemap_rings == 0] = np.nan

        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_noisemap.fits'),
                                  noisemap_rings, overwrite=True)
    # load the the KL basis and define the diskFM object
    diskobj = WindFM(None,
                     None,
                     None,
                     model_wdh1=model1_here_convolved,
                     model_wdh2=model2_here_convolved,
                     basis_filename=os.path.join(klipdir,
                                                 file_prefix + '_klbasis.h5'),
                     load_from_basis=True)
    

    # test the diskFM object

    diskobj.update_wind(model1_here, model2_here)

    if first_time:
        ### we take only the first KL modemode
        modelfm_here = diskobj.fm_parallelized_jit()[0]
        # if RPROFSUB:
        #     modelfm_here -= MED_PROFILE_EST
        fits.writeto(os.path.join(klipdir,
                                  file_prefix + '_FirstModel_FM.fits'),
                     modelfm_here,
                     overwrite='True')

        return diskobj, reduced_data
    else:
        return diskobj


########################################################
def initialize_walkers_backend(nwalkers,
                               n_dim_mcmc,
                               theta_init,
                               file_prefix='prefix',
                               mcmcresultdir='.',
                               new_backend=False):
    """ initialize the MCMC by preparing the initial position of the
        walkers and the backend file

    Args:
        n_dim_mcmc: int, number of parameter in the MCMC
        nwalkers: int, number of walkers (at least 2 times n_dim_mcmc)
        theta_init: numpy array of dim n_dim_mcmc, set of initial parameters
        file_prefix: prefix name to save the backend
        mcmcresultdir='.': folder where to save the backend
        new_backend: bool, if new_backend=False, reset the backend, 
                           if new_backend=Falserestart the chains.
                           If you change the parameters or walkers numbers,
                            you have to restart with new_backend=True

    Returns:
        if new_backend=True then [intial position of the walkers, a clean BACKEND]
        if new_backend=False then [None, the loaded BACKEND]
    """

    os.makedirs(mcmcresultdir, exist_ok=True)

    # Set up the backend h5
    # Don't forget to clear it in case the file already exists
    filename_backend = os.path.join(mcmcresultdir,
                                    file_prefix + "_backend_file_mcmc.h5")
    backend_ini = backends.HDFBackend(filename_backend)

    #############################################################
    # Initialize the walkers. The best technique seems to be
    # to start in a small ball around the a priori preferred position.
    # I start with a +/-0.1% ball for parameters defined in log and
    # +/-1% ball for the others

    if new_backend:
        p0 = np.zeros((1, nwalkers, n_dim_mcmc))
        for i in range(n_dim_mcmc):
            p0[:, :, i] = np.random.uniform(theta_init[i] * 0.999,
                                            theta_init[i] * 1.001,
                                            size=(nwalkers))

        backend_ini.reset(nwalkers, n_dim_mcmc)
        return p0[0], backend_ini

    return None, backend_ini


########################################################
def from_param_to_theta_init(params_mcmc_yaml):
    """ create a initial set of MCMCparameter from the initial parmeters
        store in the init yaml file
    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        initial set of MCMC parameter
    """
    theta_init = []

    # cosinc_init = np.cos(np.radians(params_mcmc_yaml['inc_init']))
    # theta_init.append(np.log(params_mcmc_yaml['rscale_init']))
    theta_init.append(params_mcmc_yaml['hbeta_init'])
    theta_init.append(params_mcmc_yaml['ha_r_init'])
    theta_init.append(params_mcmc_yaml['hsig_init'])
    theta_init.append(params_mcmc_yaml['hpa_init'])
    theta_init.append(params_mcmc_yaml['hdx_init'])
    theta_init.append(params_mcmc_yaml['hdy_init'])
    # theta_init.append(np.log(params_mcmc_yaml['hoffset_init']))
    theta_init.append(np.log(params_mcmc_yaml['hN_init']))
    theta_init.append(params_mcmc_yaml['dtheta_init'])
    # theta_init.append(params_mcmc_yaml['hbeta2_init'])
    theta_init.append(params_mcmc_yaml['ha_r2_init'])
    # theta_init.append(params_mcmc_yaml['hsig2_init'])
    theta_init.append(params_mcmc_yaml['hpa2_init'])
    # theta_init.append(params_mcmc_yaml['hdx2_init'])
    theta_init.append(params_mcmc_yaml['hdy2_init'])
    theta_init.append(np.log(params_mcmc_yaml['hN2_init']))
    theta_init.append(params_mcmc_yaml['dtheta2_init'])
        # if multiwdh > 1:
        #     theta_init.append(params_mcmc_yaml['ha_r3_init'])
        #     theta_init.append(params_mcmc_yaml['hpa3_init'])
        #     theta_init.append(params_mcmc_yaml['hdy3_init'])
        #     theta_init.append(np.log(params_mcmc_yaml['hN3_init']))
        #     theta_init.append(params_mcmc_yaml['dtheta3_init'])
    


    return np.asarray(theta_init)


if __name__ == '__main__':

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.simplefilter('ignore', FITSFixedWarning)
    warnings.simplefilter('ignore', NumbaWarning)
    # warnings.filterwarnings("ignore", category=UserWarning)
    # warnings.simplefilter('ignore', category=AstropyWarning)
    parser = argparse.ArgumentParser(description='run diskFM MCMC')
    parser.add_argument('-p',
                        '--param_file',
                        required=False,
                        help='parameter file name')
    args = parser.parse_args()

    # Parallel processing stuff
    import multiprocessing as mp
    mp.set_start_method('fork')
    MultiPool = mp.get_context('fork').Pool
    # from multiprocessing import Pool as MultiPool


    if args.param_file is None: #grab param file if no command line input, JKK
        str_yalm = f'initialization_files/{default_parameter_file}'
    else:
        str_yalm = args.param_file

    print("Read " + str_yalm + " parameter file")
    # open the parameter file
    yaml_path_file = os.path.join(os.getcwd(), str_yalm)
    with open(yaml_path_file, 'r') as yaml_file:
        params_mcmc_yaml = yaml.safe_load(yaml_file)
    
    # load in global the Parameters necessary to launch the MCMC
    NWALKERS = params_mcmc_yaml['NWALKERS']  #Number of walkers
    N_ITER_MCMC = params_mcmc_yaml['N_ITER_MCMC']  #Number of interation
    DISK_MODEL = params_mcmc_yaml['DISK_MODEL']
    R_INNER = params_mcmc_yaml['r_inner'] #for modified disk model boundaries
    R_OUTER = params_mcmc_yaml['r_outer'] #for modified disk model boundaries

    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']
    NEW_BACKEND = params_mcmc_yaml['NEW_BACKEND']

    if not os.path.isdir(os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])):
        raise ValueError(
            "Could not find the data directory (BAND_DIR parameter)")

    KLIPDIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'],
                           'wind_fm_files')
    BANDDIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    MCMCRESULTDIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'],
                                 'windfit_MCMC')


    if (DISK_MODEL.lower() == 'modified') or (DISK_MODEL.lower() == 'original'):
        pass
    else:
        raise ValueError(DISK_MODEL + "not a valid disk model. Choose 'original' or 'modified'.")

    # load DISTANCE_STAR & PIXSCALE_INS and make them global
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']
    ALIGNED_CENTER = params_mcmc_yaml['ALIGNED_CENTER']


    # initialize the things necessary to measure the model (PSF, masks,
    # uncertainities). In RDI mode, psflib is also initiliazed here
    dataset, psflib = initialize_mask_psf_noise(params_mcmc_yaml,
                                                quietklip=True)

    ## Load all variables necessary for the MCMC and make them global
    ## to avoid very long transfert time at each iteration

    # load wheremask2generatedisk and make it global
    WHEREMASK2GENERATEHALO = (fits.getdata(
        os.path.join(KLIPDIR, FILE_PREFIX + '_mask2generatehalo.fits')) == 0)

    # load noise and make it global

    # load PSF and make it global
    # PSF = fits.getdata(os.path.join(KLIPDIR, FILE_PREFIX + '_SmallPSF.fits'))
    PSF = fits.getdata(os.path.join(KLIPDIR, FILE_PREFIX + '_instrPSF.fits'))
    PSF /= np.sum(PSF)

    # if ADD_WDH > 0:
    #     WDH_DPA = params_mcmc_yaml["WDH_DPA"]
    #     if ADD_WDH > 1:
    #         WDH_DPA2 = params_mcmc_yaml["WDH_DPA2"]
    USE_NOISE = params_mcmc_yaml["USE_NOISE"]
    RPROFSUB = params_mcmc_yaml["RPROFSUB"]
    # if RPROFSUB:
    #     MED_PROFILE_EST = fits.getdata(f"{KLIPDIR}/estimated_med_profile.fits")
    #     MED_PROFILE_EST /= np.max(MED_PROFILE_EST)
    FREE_PARAMS = arr_free_params(params_mcmc_yaml)

    # load initial parameter value and make them global
    THETA_INIT = from_param_to_theta_init(params_mcmc_yaml)

    # measure the size of images DIMENSION and make it global
    DIMENSION = round(ALIGNED_CENTER[0]) * 2
    N_DIM_MCMC = len(THETA_INIT)
    N_DIM_MOD = round(N_DIM_MCMC / 2)
    
    if params_mcmc_yaml['FIRST_TIME']:
    # initialize_diskfm and make diskobj global
        DISKOBJ, REDUCED_DATA = initialize_windfm(dataset,
                                    params_mcmc_yaml,
                                    psflib=psflib,
                                    quietklip=True)
    else:
        DISKOBJ = initialize_windfm(dataset,
                                    params_mcmc_yaml,
                                    psflib=psflib,
                                    quietklip=True)
        REDUCED_DATA = fits.getdata(
            os.path.join(KLIPDIR, FILE_PREFIX + '-klipped-KLmodes-all.fits'))[
                0]  ### we take only the first KL mode


    NOISE = fits.getdata(os.path.join(KLIPDIR, FILE_PREFIX + '_noisemap.fits'))
    NOISE[NOISE == 0] = np.nan
    # Modification for Justin to save memory, slightly slower
    # del DISKOBJ

    # manager = mp.Manager()
    # KL_BASIS_FILE = manager.dict(_load_dict_from_hdf5(os.path.join(KLIPDIR,
    #                                               FILE_PREFIX + '_klbasis.h5')))

    
    # load reduced_data and make it a global variable
    # we multiply the reduced_data by the nan mask2minimize to avoid having
    # to pass mask2minimize as a global variable
    mask2minimize = fits.getdata(
        os.path.join(KLIPDIR, FILE_PREFIX + '_mask2minimize.fits'))
    mask4disk = mask2minimize.copy()
    mask2minimize[np.where(mask2minimize == 0.)] = np.nan
    mask4data = mask2minimize.copy()
    mask4data[np.where(mask2minimize == 0.)] = np.nan
    # mask4disk[np.where(mask4data == np.nan)] = 0
    MASKED_DATA = REDUCED_DATA * mask4data
    MODEL_MASKED_DATA = REDUCED_DATA * (1 - mask4disk)
    fits.writeto(f'{KLIPDIR}/{FILE_PREFIX}_masked-data.fits',MASKED_DATA,overwrite=True)
    fits.writeto(f'{KLIPDIR}/{FILE_PREFIX}_model-masked_data.fits',MODEL_MASKED_DATA,overwrite=True)
    REDUCED_DATA *= mask4data

    # Before launching th parallel MCMC
    # Make a final test "in c" by printing the likelyhood of the iniatial
    # set of parameter
    startTime = datetime.now()
    lnpb_model = lnpb(THETA_INIT)
    print("""Test: Likelyhood on initial parameter set is {0}. Time 
            from parameter values to Likelyhood (create model+FM+Likelyhood): 
            {1}""".format(lnpb_model,
                          datetime.now() - startTime))
    if params_mcmc_yaml['FIRST_TIME']:
        print('First time initializing, check wind_fm_files directory and modify the yaml file first_time flag.')
        exit()

    #last chance to delete useless big variables to avoid sending them
    # to every CPUs when paralelizing
    # print(globals())
    del mask4data, dataset, psflib, params_mcmc_yaml
    del MASKED_DATA, MODEL_MASKED_DATA
    if not np.isfinite(lnpb_model):
        raise ValueError(
            """Do not launch MCMC, Likelyhood=-inf:your initial guess 
                            is probably out of the prior range for one of the parameter"""
        )

    print("initialize walkers and start the MCMC...")
    startTime = datetime.now()


    with MultiPool() as pool:

        # initialize the walkers if necessary. initialize/load the backend
        # make them global
        init_walkers, BACKEND = initialize_walkers_backend(
            NWALKERS,
            N_DIM_MCMC,
            THETA_INIT,
            file_prefix=FILE_PREFIX,
            mcmcresultdir=MCMCRESULTDIR,
            new_backend=NEW_BACKEND)
        
        # # Create a profiler object
        # profiler = cProfile.Profile()

        # # Start profiling
        # profiler.enable()

        #Let's start the MCMC
        # Set up the Sampler. I purposefully passed the variables (KL modes,
        # reduced data, masks) in global variables to save time as advised in
        # https://emcee.readthedocs.io/en/latest/tutorials/parallel/
        # mode MPI or not


        moves = [(StretchMove(), 0.5), (DEMove(), 0.3), (KDEMove(), 0.2)]

        sampler = EnsembleSampler(NWALKERS,
                                    N_DIM_MCMC,
                                    lnpb,
                                    pool=pool,
                                    backend=BACKEND,
                                    moves=moves)

        sampler.run_mcmc(init_walkers, N_ITER_MCMC, progress=True)
        # # Stop profiling
        # profiler.disable()
        # # Save the stats to a file
        # profiler.dump_stats(f'{basedir}/debrisdisk_mcmc_fit_and_plot/diskfit_mcmc.prof')
    print("time {0} iterations with {1} walkers and {2} cpus: {3}".format(
          N_ITER_MCMC, NWALKERS, cpu_count(),
          datetime.now() - startTime))
