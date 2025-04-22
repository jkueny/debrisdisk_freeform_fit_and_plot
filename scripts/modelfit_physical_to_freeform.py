"""Fit a physics-informed disk model to the output freeform image
from DiskierFM.

Workflow outline:
- Specify the type of disk model to generate, modified or original.
- Generate the model.
- Convolve the model with the appropriate PSF.
- Compare the model image with the freeform image with a Chi**2 metric.
  + Use noise estimate from the KLIP image.
- This procedure should be wrapped with a MCMC routine.
"""

# pylint: disable=C0103
"""
MCMC code for fitting a disk 
author: Johan Mazoyer
"""

import os
import sys
# import copy
import argparse


# careful on Python 3.8 mac multiprocessing switched to spawn so the global varialbe do not work

basedir = f'{os.environ["HOME"]}/projects'  # the base directory where is
# your data (using OS environnement variable allow to use same code on
# different computer without changing this).

# default_parameter_file = 'HR4796_g_camsci2_20230312_13.yaml'  # name of the parameter file
default_parameter_file = 'HR4796a_z_lco2023a_magao-x_20230309_10.yaml'  # name of the parameter file
# default_parameter_file = 'HR4796_i_smlyot_20230309_10.yaml'  # name of the parameter file
# you can also call it with the python function argument -p

import warnings


# # because this error was coming up
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from multiprocessing import cpu_count


from datetime import datetime

import math as mt
import numpy as np
# import matplotlib.pyplot as plt

import astropy.io.fits as fits
# from astropy.convolution import convolve
from scipy.signal import convolve
# from scipy.signal import fftconvolve
from astropy.wcs import FITSFixedWarning

import yaml

from emcee import EnsembleSampler
from emcee import backends

from pyklip.klip import high_pass_filter

from numba.core.errors import NumbaWarning

import cProfile

# These programs were not included in the initial download
# from anadisk_model.anadisk_sum_mask import phase_function_spline, generate_disk #commented 06/04/21 JK

# from disk_models import hg_1g, hg_2g, hg_3g

# from utils.disk_models import gen_disk_dxdy_1g, fastgen_disk_dxdy_2g, fastgen_disk_dxdy_3g
# from utils.disk_models import mod_gen_disk_dxdy_1g, fastmodgen_disk_dxdy_2g, fastmodgen_disk_dxdy_3g

from utils.disk_models import fastgen_disk_dxdy_custom, fastmodgen_disk_custom

from utils.spf_models import calculate_hg_spf, \
                        fit_fourier_to_hg_spf, fit_legendre_to_hg_spf, \
                        legendre_reconstruction, bessel_reconstruction, \
                        fit_bessel_to_hg_spf


# recommended by emcee https://emcee.readthedocs.io/en/stable/tutorials/parallel/
# and by PyKLIPto avoid that NumPy automatically parallelizes some operations,
# which kill the speed
os.environ["OMP_NUM_THREADS"] = "1"


def arr_free_params(params_mcmc_yaml):
    free_params = []
    if DISK_MODEL == 'modified':
        if bool(params_mcmc_yaml['rc_state']):
            free_params.append('rc')
        else:
            # free_params.append(False)
            pass
        if bool(params_mcmc_yaml['alpha_in_state']):
            free_params.append('alpha_in')
        else:
            # free_params.append(False)
            pass
        if bool(params_mcmc_yaml['alpha_out_state']):
            free_params.append('alpha_out')
        else:
            # free_params.append(False)
            pass

    elif DISK_MODEL == 'original':
        if bool(params_mcmc_yaml['r1_state']):
            free_params.append('r1')
        else:
            # free_params.append(False)
            pass
        if bool(params_mcmc_yaml['r2_state']):
            free_params.append('r2')
        else:
            # free_params.append(False)
            pass
        if bool(params_mcmc_yaml['beta_state']):
            free_params.append('beta')
        else:
            # free_params.append(False)
            pass

    if bool(params_mcmc_yaml['a_r_state']):
    # cosinc_init = np.cos(np.radians(params_mcmc_yaml['inc_init']))
        free_params.append('a_r')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['inc_state']):
    # cosinc_init = np.cos(np.radians(params_mcmc_yaml['inc_init']))
        free_params.append('inc')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['pa_state']):
        free_params.append('PA')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['dx_state']):
        free_params.append('dx')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['dy_state']):
        free_params.append('dy')
    else:
        # free_params.append(False)
        pass
    if bool(params_mcmc_yaml['N_state']):
        free_params.append('Norm')
    else:
        # free_params.append(False)
        pass

    print(f'Fitting geometrical params: {free_params}')
    return free_params



def from_theta_to_params(theta):
    '''
    Setup function. This function takes the prior parameters and creates a dictionary and vector of parameters from it.
    '''
    param_disk = {} #disk parameters are put into a dict.
    vector_param = [] #this is for the walker chain plots, free params only
    param_disk['offset'] = 0.  # no vertical offset in KLIP
    fixed_params = 0

    param_disk['beta_in'] = -10  # we fix the inner power law
    if DISK_MODEL.lower() == 'modified':
        all_params = ['rc','alpha_in','alpha_out','a_r','inc','PA','dx','dy','Norm']
        for ea, p in enumerate(all_params):
            if p in FREE_PARAMS:
                if p == 'rc':
                    param_disk['rc'] = mt.exp(theta[ea])
                    vector_param.append(param_disk['rc'])
                elif p == 'Norm':
                    param_disk['Norm'] = mt.exp(theta[ea])
                    vector_param.append(param_disk['Norm'])
                else:
                    param_disk[p] = theta[ea]
                    vector_param.append(param_disk[p])
            else:
                param_disk[p] = THETA_INIT[ea]


    elif DISK_MODEL.lower() == 'original':
        all_params = ['r1','r2','beta','a_r','inc','PA','dx','dy','Norm']
        for ea, p in enumerate(all_params):
            if p in FREE_PARAMS:
                if p == 'r1':
                    param_disk['r1'] = mt.exp(theta[ea - fixed_params])
                    vector_param.append(param_disk['r1'])
                elif p == 'r2':
                    param_disk['r2'] = mt.exp(theta[ea - fixed_params])
                    vector_param.append(param_disk['r2'])
                elif p == 'Norm':
                    param_disk['Norm'] = mt.exp(theta[ea - fixed_params])
                    vector_param.append(param_disk['Norm'])
                elif p == 'a_r':
                    param_disk['a_r'] = theta[ea - fixed_params]
                    vector_param.append(100*param_disk['a_r']) #plot the opening angle in percent
                else:
                    param_disk[p] = theta[ea - fixed_params]
                    vector_param.append(param_disk[p])
            else:
                fixed_params += 1
                if p == 'r1':
                    param_disk['r1'] = mt.exp(THETA_INIT[ea])
                elif p == 'r2':
                    param_disk['r2'] = mt.exp(THETA_INIT[ea])
                elif p == 'Norm':
                    param_disk['Norm'] = mt.exp(THETA_INIT[ea])
                else:
                    param_disk[p] = THETA_INIT[ea]

    # We don't need the DC offset term bc the SPF gets normalized at 90
    # param_disk['dc'] = DC
    if BASIS == "bessel" or BASIS == "legendre":
        # param_disk['dc'] = theta[len(all_params)] #tack onto the end of the geo params
        param_disk['dc'] = DC
        # vector_param.append(param_disk['dc'])
    param_disk['coeffs'] = theta[-int(N_MODES):] #indexing confirmed

    # print(f"from_theta_to_params: {param_disk['coeffs']}")

    for i in range(len(param_disk['coeffs'])):
        vector_param.append(param_disk['coeffs'][i])
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

    a_r = param_disk['a_r']
    inc = param_disk['inc']
    pa = param_disk['PA']
    dx = param_disk['dx']
    dy = param_disk['dy']
    Norm = param_disk['Norm']
    # Norm = dc
    # coeffs_all = np.insert(coeffs,0,dc)
    # coeffs_all = np.append(dc,coeffs)
    dc = param_disk['dc']
    coeffs = param_disk['coeffs']
    if BASIS == "legendre":
        # sc_angs = np.linspace(0,180,1800)
        # sc_angs = np.linspace(SM_ANGLE,LG_ANGLE,(LG_ANGLE-SM_ANGLE)*10)
        # sc_angs_rad = np.deg2rad(sc_angs)
        coeffs_all = np.append(dc,coeffs) #we fit the DC component for Legendres w/ a penalty
        # print(f"call_gen_disk: {coeffs_all}")
        recon_spf = legendre_reconstruction(SC_ANGS_RAD,*coeffs_all)
    elif BASIS == "bessel":
        # coeffs -= CSHIFT
        # sc_angs = np.linspace(0,180,1800)
        # sc_angs = np.linspace(SM_ANGLE,LG_ANGLE,(LG_ANGLE-SM_ANGLE)*10)
        # sc_angs_rad = np.deg2rad(sc_angs)
        coeffs_all = np.append(dc, coeffs) #fitting for the DC offset 01/22/2025
        # print(coeffs_all)
        recon_spf = bessel_reconstruction(X_TOFIT, SELECT_BASIS, *coeffs_all)


    if DISK_MODEL == "original":
        r1 = param_disk['r1']
        r2 = param_disk['r2']
        beta = param_disk['beta']
        model = fastgen_disk_dxdy_custom(r1, r2, beta, inc, pa, dx, dy, Norm, a_r,
                                scatt_angs=SC_ANGS_RAD,
                                spf_model=recon_spf,
                                y_arr=Y_MODEL,
                                z_arr=Z_MODEL,
                                npts=n_pts,
                                mask=MASK2GENERATEDISK)
    elif DISK_MODEL == "modified":
        rc = param_disk['rc']
        m = param_disk['alpha_in']
        n = param_disk['alpha_out']
        model = fastmodgen_disk_custom(R1, R2, rc, m, n, inc, pa, dx, dy, Norm, a_r,
                                scatt_angs=SC_ANGS_RAD,
                                spf_model=recon_spf,
                                y_arr=Y_MODEL,
                                z_arr=Z_MODEL,
                                npts=n_pts,
                                mask=MASK2GENERATEDISK)

    # I normalize by value of a_r to avoid degenerascies between a_r and Normalization
    # model = param_spf['dc'] * model / param_disk['a_r']
    # model[model != model] = 0.

    return model, coeffs_all


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
    model, spf_coeffs = call_gen_disk(theta)

    # modelconvolved = convolve(model, PSF, boundary='wrap')
    # modelconvolved = fftconvolve(model, PSF, mode='same')
    modelconvolved = convolve(model, PSF, mode='same')#,method='fft')
    if isinstance(HP_FILTER, int) and HP_FILTER > 0:
        modelconvolved = high_pass_filter(modelconvolved, filtersize=HP_FILTER)


    res = (FREEFORM_IMAGE - modelconvolved) / NOISE #verified this image looks right

    if BASIS == "bessel" or BASIS == "legendre":
        penalize_coeffs = -LAMBDA_REG * np.sum(spf_coeffs ** 2)
        Chisquare = np.nansum(-0.5 * (res * res)) + penalize_coeffs
    else:
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
    # check_coeffs = np.append(param_spf['dc'],param_spf['coeffs'])
    # # print(check_coeffs)
    # check_spf = legendre_reconstruction(scattering_angles_deg,*check_coeffs)
    # has_negative_values = np.any(check_spf < 0)
    # # print(COEFFS_INIT)
    # # print(param_spf['dc'])

    # if has_negative_values:
    #     print('Unphysical SPF.')
    #     return -np.inf
    # else:
    #     prior_rout = prior_rout * 1.

    # if param_disk['dc'] < 0.1 or param_disk['dc'] > 3:
    #     print('dc term out of prior.')
    #     return -np.inf
    # else:
    #     prior_rout = prior_rout * 1.

    if BASIS == "legendre":
        if all(-len(COEFFS_INIT) < c < len(COEFFS_INIT) for c in param_disk['coeffs']):
            prior_rout = prior_rout * 1.
        else:
            print('One or more Legendre coefficients out of prior.')
            return -np.inf
    elif BASIS == "bessel":
            pass
        # if all(-5e4 < c < 5e4 for c in (param_disk['coeffs'] - CSHIFT)):
        # if all((-len(COEFFS_INIT)*10) < c < (len(COEFFS_INIT)*10) for c in param_disk['coeffs']):
        #     prior_rout = prior_rout * 1.
        # else:
        #     print('One or more Bessel coefficients out of prior.')
            # return -np.inf
    if DISK_MODEL == 'original':
        if (param_disk['beta'] < 5 or param_disk['beta'] > 30):
            print('beta out of prior', param_disk["beta"])
            return -np.inf
        else:
            prior_rout = prior_rout * 1.

        if (param_disk['r1'] < 65 or param_disk['r1'] > 85):
            print('r1 out of prior')
            return -np.inf
        else:
            prior_rout = prior_rout * 1.

        # - rout = Logistic We  cut the prior at r2 = xx
        # because this parameter is very limited by the ADI
        if (param_disk['r2'] < 80 or param_disk['r2'] > 150):
            print('r2 out of prior')
            return -np.inf
        else:
            prior_rout = prior_rout / (1. + np.exp(40. * (param_disk['r2'] - 110)))
            # prior_rout = prior_rout * 1.  # or we can just use a flat prior
    elif DISK_MODEL == 'modified':
        if (param_disk['rc'] < 70 or param_disk['rc'] > 120): #in au
            print('rc out of prior', param_disk['rc'])
            return -np.inf
        else:
            prior_rout = prior_rout * 1.
        
        if (param_disk['alpha_in'] < 1 or param_disk['alpha_in'] > 100): #in au
            print('alpha_in out of prior')
            return -np.inf
        else:
            prior_rout = prior_rout * 1.

        if (param_disk['alpha_out'] < -100 or param_disk['alpha_out'] > -1): #in au
            print('alpha_out out of prior')
            return -np.inf
        else:
            prior_rout = prior_rout * 1.

    if (param_disk['a_r'] < 0.001 or param_disk['a_r'] > 0.1):
        print('a_r out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (param_disk['inc'] < 70 or param_disk['inc'] > 85):
        print('inc out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (param_disk['PA'] < 20 or param_disk['PA'] > 30):
        print('PA out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    # dx is degenerate with the degree of forward or back-scattering
    # use Gaussian prior?
    if (param_disk['dx'] < -15) or (param_disk['dx'] > 15):  #The x offset
        print('dx out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (param_disk['dy'] < -10) or (param_disk['dy'] > 10):  #The y offset
        print('dy out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    if (param_disk['Norm'] < 0.001 or param_disk['Norm'] > 100000):
        print('Norm out of prior')
        return -np.inf
    else:
        prior_rout = prior_rout * 1.

    # prior_uni = np.log(prior_rout)
    # prior_gau = -np.log(dx_sigma*(np.sqrt(2*np.pi)))-0.5*((param_disk['dx']-dx_mu)/dx_sigma)**power
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

    if DISK_MODEL == 'modified':
        theta_init.append(np.log(params_mcmc_yaml['rc_init']))
        theta_init.append(params_mcmc_yaml['alpha_in_init'])
        theta_init.append(params_mcmc_yaml['alpha_out_init'])

    elif DISK_MODEL == 'original':
        theta_init.append(np.log(params_mcmc_yaml['r1_init']))
        theta_init.append(np.log(params_mcmc_yaml['r2_init']))
        theta_init.append(params_mcmc_yaml['beta_init'])

    # cosinc_init = np.cos(np.radians(params_mcmc_yaml['inc_init']))
    theta_init.append(params_mcmc_yaml['a_r_init'])
    theta_init.append(params_mcmc_yaml['inc_init'])
    theta_init.append(params_mcmc_yaml['pa_init'])
    theta_init.append(params_mcmc_yaml['dx_init'])
    theta_init.append(params_mcmc_yaml['dy_init'])
    theta_init.append(np.log(params_mcmc_yaml['N_init']))

    if SPF_MODEL == 'hg_1g':
        theta_init.append(params_mcmc_yaml['g1_init'])

    elif SPF_MODEL == 'hg_2g':
        theta_init.append(params_mcmc_yaml['g1_init'])
        theta_init.append(params_mcmc_yaml['g2_init'])
        theta_init.append(params_mcmc_yaml['alpha1_init'])

    elif SPF_MODEL == 'hg_3g':
        theta_init.append(params_mcmc_yaml['g1_init'])
        theta_init.append(params_mcmc_yaml['g2_init'])
        theta_init.append(params_mcmc_yaml['alpha1_init'])
        theta_init.append(params_mcmc_yaml['g3_init'])
        theta_init.append(params_mcmc_yaml['alpha2_init'])
    


    return np.asarray(theta_init)

########################################################
def from_theta_init_to_PoI(theta_init):
    """ The number of free parameters can be large, so we can specify
    Parameters of Interest (PoI) to fix parameters that are the most
    constrained.
    Args:
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file

    Returns:
        interest set of MCMC parameters
    """
    theta_interest = []
    if DISK_MODEL.lower() == 'modified':
        all_params = ['rc','alpha_in','alpha_out','a_r','inc','PA','dx','dy','Norm']
        for ea, p in enumerate(all_params):
            if p in FREE_PARAMS:
                theta_interest.append(theta_init[ea])
            else:
                continue
    elif DISK_MODEL.lower() == 'original':
        all_params = ['r1','r2','beta','a_r','inc','PA','dx','dy','Norm']
        for ea, p in enumerate(all_params):
            if p in FREE_PARAMS:
                theta_interest.append(theta_init[ea])
            else:
                continue
    print(f"Prepared theta of interest...")
    print(theta_interest)

    return np.asarray(theta_interest)



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
    parser.add_argument("--image",
                        help="Path to freeform fits file to use.",
                        required=True)
    # parser.add_argument("--noise",
    #                     help="Path to noise map fits file to use.",
    #                     required=True)
    args = parser.parse_args()

    # Parallel processing stuff
    import multiprocessing as mp
    mp.set_start_method('fork')
    MultiPool = mp.get_context('fork').Pool
    # from multiprocessing import Pool as MultiPool


    if args.param_file is None: #grab param file if no command line input, JKK
        str_yaml = f'initialization_files/{default_parameter_file}'
    else:
        str_yaml = args.param_file
        str_yaml_prefix = str_yaml.split("/")[-1]
        save_to_dir = str_yaml_prefix.split(".")[0]

    print("Read " + str_yaml + " parameter file")
    # open the parameter file
    yaml_path_file = os.path.join(os.getcwd(), str_yaml)
    with open(yaml_path_file, 'r') as yaml_file:
        params_mcmc_yaml = yaml.safe_load(yaml_file)
    
    # load in global the Parameters necessary to launch the MCMC
    NWALKERS = params_mcmc_yaml['NWALKERS']  #Number of walkers
    N_ITER_MCMC = params_mcmc_yaml['N_ITER_MCMC']  #Number of interation
    SPF_MODEL = params_mcmc_yaml['SPF_MODEL']  #Type of description for the SPF
    DISK_MODEL = params_mcmc_yaml['DISK_MODEL']
    R_INNER = params_mcmc_yaml['r_inner'] #for modified disk model boundaries
    R_OUTER = params_mcmc_yaml['r_outer'] #for modified disk model boundaries
    USE_NOISE = params_mcmc_yaml["USE_NOISE"]
    HP_FILTER = params_mcmc_yaml["HP_FILTER"]

    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']
    NEW_BACKEND = params_mcmc_yaml['NEW_BACKEND']

    if not os.path.isdir(os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])):
        raise ValueError(
            "Could not find the data directory (BAND_DIR parameter)")

    KLIPDIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'],
                           'klip_fm_files')
    os.makedirs(f"{save_to_dir}/results_MCMC", exist_ok=True)
    MCMCRESULTDIR = f"{save_to_dir}/results_MCMC"


    if DISK_MODEL.lower() == 'modified':
        if SPF_MODEL == "hg_1g":  #1g henyey greenstein, SPF described with 1 parameter
            N_DIM_MCMC = 9  #Number of dimension of the parameter space
        elif SPF_MODEL == "hg_2g":  #2g henyey greenstein, SPF described with 3 parameter
            N_DIM_MCMC = 11  #Number of dimension of the parameter space
        elif SPF_MODEL == "hg_3g":  #1g henyey greenstein, SPF described with 5 parameter
            N_DIM_MCMC = 13  #Number of dimension of the parameter space
        else:
            raise ValueError(SPF_MODEL + " not a valid SPF model")
    elif DISK_MODEL.lower() == 'original':
        if SPF_MODEL == "s10pf_fix":  #1g henyey greenstein, SPF described with 1 parameter
            N_DIM_MCMC = 11  #Number of dimension of the parameter space
        elif SPF_MODEL == "hg_1g":  #1g henyey greenstein, SPF described with 1 parameter
            N_DIM_MCMC = 11  #Number of dimension of the parameter space
        elif SPF_MODEL == "hg_2g":  #2g henyey greenstein, SPF described with 3 parameter
            N_DIM_MCMC = 13  #Number of dimension of the parameter space
        elif SPF_MODEL == "hg_3g":  #1g henyey greenstein, SPF described with 5 parameter
            N_DIM_MCMC = 15  #Number of dimension of the parameter space
        else:
            raise ValueError(SPF_MODEL + " not a valid SPF model")
    else:
        raise ValueError(DISK_MODEL + "not a valid disk model. Choose 'original' or 'modified'.")

    # load DISTANCE_STAR & PIXSCALE_INS and make them global
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']
    ALIGNED_CENTER = params_mcmc_yaml['ALIGNED_CENTER']


    ## Load all variables necessary for the MCMC and make them global
    ## to avoid very long transfert time at each iteration

    # load wheremask2generatedisk and make it global
    WHEREMASK2GENERATEDISK = fits.getdata(
        os.path.join(KLIPDIR, FILE_PREFIX + '_mask2generatedisk.fits'))
    
    MASK2GENERATEDISK = WHEREMASK2GENERATEDISK == 0

    # load noise and make it global
    NOISE = fits.getdata(os.path.join(KLIPDIR, FILE_PREFIX + "_noisemap.fits"))

    # load PSF and make it global
    # PSF = fits.getdata(os.path.join(KLIPDIR, FILE_PREFIX + '_SmallPSF.fits'))
    PSF = fits.getdata(os.path.join(KLIPDIR, FILE_PREFIX + '_instrPSF.fits'))
    PSF /= np.sum(PSF)


    # load initial parameter value and make them global
    THETA_INIT = from_param_to_theta_init(params_mcmc_yaml)
    FREE_PARAMS = arr_free_params(params_mcmc_yaml)

    # Separate the HG params from the geo params and call my func to fit the Fourier model to the curve
    # Return the coefficients - these are what we need to use the MCMC to optimize
    # Finally fix the geo params
    SM_ANGLE = params_mcmc_yaml['MIN_SCATT_ANG']
    LG_ANGLE = params_mcmc_yaml['MAX_SCATT_ANG']
    R1 = params_mcmc_yaml["r_inner"]
    R2 = params_mcmc_yaml["r_outer"]
    N_MODES = params_mcmc_yaml['N_BASIS']

    scattering_angles_deg = np.arange(SM_ANGLE,LG_ANGLE,1)
    sc_angs_deg = np.linspace(SM_ANGLE,LG_ANGLE,(LG_ANGLE - SM_ANGLE) * 10)
    XRANGE_MAX = params_mcmc_yaml["XRANGE_MAX"]
    X_TOFIT = np.linspace(0, XRANGE_MAX, len(sc_angs_deg))
    SC_ANGS_RAD = np.deg2rad(sc_angs_deg)
    spf_tofit = calculate_hg_spf(g1=THETA_INIT[-3],
                                 g2=THETA_INIT[-2],
                                 alpha1=THETA_INIT[-1],
                                #  scattangs=scattering_angles_deg)
                                 scattangs=sc_angs_deg)
    #
    DIMENSION = round(ALIGNED_CENTER[0]) * 2
    N_DIM_GEO = len(FREE_PARAMS)
    if SPF_MODEL == 'hg_1g':
        # GEO_PARAMS = THETA_INIT[:-1]
        if params_mcmc_yaml['SPF_BASIS'] == 'fourier':
            BASIS = 'fourier'
            COEFFS_INIT = fit_fourier_to_hg_spf(KLIPDIR,g1=THETA_INIT[-3])
            THETA_INIT = np.append(THETA_INIT[:-3],COEFFS_INIT[1:])
        elif params_mcmc_yaml['SPF_BASIS'] == 'legendre':
            BASIS = 'legendre'
            COEFFS_INIT = fit_legendre_to_hg_spf(KLIPDIR,spf_tofit,scattering_angles_deg,nmodes=N_MODES)
            THETA_INIT = np.append(THETA_INIT[:-3],COEFFS_INIT[1:])
        elif params_mcmc_yaml['SPF_BASIS'] == "bessel":
            BASIS = "bessel"
            COEFFS_INIT = fit_bessel_to_hg_spf(KLIPDIR,spf_tofit,scattering_angles_deg,nmodes=N_MODES)
            THETA_INIT = np.append(THETA_INIT[:-3],COEFFS_INIT)
        N_DIM_MCMC = N_DIM_GEO + len(COEFFS_INIT)
    elif SPF_MODEL == 'hg_2g':
        if params_mcmc_yaml['SPF_BASIS'] == 'fourier':
            BASIS = 'fourier'
            COEFFS_INIT = fit_fourier_to_hg_spf(KLIPDIR,g1=THETA_INIT[-3],g2=THETA_INIT[-2],alpha1=THETA_INIT[-1])
            THETA_INIT = np.append(THETA_INIT[:-3],COEFFS_INIT[1:])
        elif params_mcmc_yaml['SPF_BASIS'] == 'legendre':
            BASIS = 'legendre'
            COEFFS_INIT_ALL = fit_legendre_to_hg_spf(KLIPDIR,
                                                 spf_tofit,
                                                 sc_angs_deg,
                                                 nmodes=N_MODES)
            COEFFS_INIT = COEFFS_INIT_ALL[1:]
            DC = COEFFS_INIT_ALL[0]
            THETA_INIT = np.append(THETA_INIT[:-3],COEFFS_INIT)
            print(f"HG Legendre 0 solution: {DC}")
            print(f"Starting with coefficients: {THETA_INIT[-(N_MODES+1):]}") #include DC offset
        elif params_mcmc_yaml['SPF_BASIS'] == "bessel":
            BASIS = "bessel"
            COEFFS_LEG = fit_legendre_to_hg_spf(KLIPDIR,
                                                spf_tofit,
                                                sc_angs_deg,
                                                nmodes=N_MODES)
            DC = COEFFS_LEG[0]
            # spf_tofit -= np.ones_like(spf_tofit) * DC
            # COEFFS_INIT = fit_bessel_to_hg_spf(KLIPDIR,spf_tofit,scattering_angles_deg,nmodes=N_MODES,cshift=CSHIFT)
            COEFFS_INIT, SELECT_BASIS = fit_bessel_to_hg_spf(KLIPDIR,
                                                             XRANGE_MAX,
                                                             sc_angs_deg,
                                                             nmodes=N_MODES,
                                                             dcoffset=DC)
            COEFFS_INIT_ALL = np.append(DC,COEFFS_INIT)
            THETA_INIT = np.append(THETA_INIT[:-3],COEFFS_INIT_ALL)
        elif params_mcmc_yaml['SPF_BASIS'] == "ortho_bessel":
            XRANGE_MAX = params_mcmc_yaml["XRANGE_MAX"]
            BASIS = "bessel"
            COEFFS_LEG = fit_legendre_to_hg_spf(KLIPDIR,spf_tofit,sc_angs_deg,nmodes=N_MODES)
            DC = COEFFS_LEG[0]
            # spf_tofit -= np.ones_like(spf_tofit) * DC
            # COEFFS_INIT = fit_bessel_to_hg_spf(KLIPDIR,spf_tofit,scattering_angles_deg,nmodes=N_MODES,cshift=CSHIFT)
            COEFFS_INIT, SELECT_BASIS = fit_bessel_to_hg_spf(KLIPDIR,
                                                             XRANGE_MAX,
                                                             spf_tofit,
                                                             sc_angs_deg,
                                                             nmodes=N_MODES,
                                                             dcoffset=DC,
                                                             orthogonalize=True)
            # COEFFS_INIT_ALL = np.append(DC,COEFFS_INIT)
            COEFFS_INIT_ALL = COEFFS_INIT
            THETA_INIT = np.append(THETA_INIT[:-3],COEFFS_INIT_ALL)
        N_DIM_MCMC = len(FREE_PARAMS) + len(COEFFS_INIT)
    else:
        raise ValueError('Not configured for a different SPF model. Choose "hg_1g" or "hg_2g".')
    

    THETA_INTEREST = from_theta_init_to_PoI(THETA_INIT)
    FIRST_THETA = np.append(THETA_INTEREST, COEFFS_INIT_ALL)
    print(f"print(FIRST_THETA) -> {FIRST_THETA}")
    N_DIM_MCMC = len(FIRST_THETA)

    # measure the size of images DIMENSION and make it global
    DIMENSION = round(ALIGNED_CENTER[0]) * 2


    max_fov = DIMENSION / 2. * PIXSCALE_INS  #maximum radial distance in AU from the center to the edge
    n_pts = int(np.floor(DIMENSION / 1))
    xsize = max_fov * DISTANCE_STAR  #maximum radial distance in AU from the center to the edge

    # print(f'max_fov: {max_fov}; xsize: {xsize}')

    #The coordinate system here [x,y,z] is defined :
    # +ve x is the line of sight
    # +ve y is going right from the center
    # +ve z is going up from the center

    # y = np.linspace(0,xsize,num=npts/2)
    Y_MODEL = np.linspace(-xsize, xsize, num=n_pts)
    Z_MODEL = np.linspace(-xsize, xsize, num=n_pts)
    
    FREEFORM_IMAGE = fits.getdata(args.image) * WHEREMASK2GENERATEDISK
    NOISE[NOISE == 0] = np.nan
    LAMBDA_REG = params_mcmc_yaml["LAMBDA_REG"]

    # plt.imshow(FREEFORM_IMAGE, origin="lower")
    # plt.colorbar()
    # plt.show()
    # sys.exit()
    # Before launching th parallel MCMC
    # Make a final test "in c" by printing the likelyhood of the iniatial
    # set of parameter
    startTime = datetime.now()
    lnpb_model = lnpb(FIRST_THETA)
    print("""Test: Likelyhood on initial parameter set is {0}. Time 
            from parameter values to Likelyhood (create model+FM+Likelyhood): 
            {1}""".format(lnpb_model,
                          datetime.now() - startTime))
    if params_mcmc_yaml['FIRST_TIME']:
        print('First time initializing, check klip_fm_files directory and modify the yaml file first_time flag.')
        exit()

    #last chance to delete useless big variables to avoid sending them
    # to every CPUs when paralelizing
    # print(globals())
    if not np.isfinite(lnpb_model):
        raise ValueError(
            """Do not launch MCMC, Likelyhood=-inf:your initial guess 
                            is probably out of the prior range for one of the parameter"""
        )

    print("Initialize walkers and start the MCMC...")
    startTime = datetime.now()

    with MultiPool() as pool:


        # initialize the walkers if necessary. initialize/load the backend
        # make them global
        init_walkers, BACKEND = initialize_walkers_backend(
            NWALKERS,
            N_DIM_MCMC,
            FIRST_THETA,
            file_prefix=FILE_PREFIX,
            mcmcresultdir=MCMCRESULTDIR,
            new_backend=NEW_BACKEND)

        #Let's start the MCMC
        # Set up the Sampler. I purposefully passed the variables (KL modes,
        # reduced data, masks) in global variables to save time as advised in
        # https://emcee.readthedocs.io/en/latest/tutorials/parallel/
        # mode MPI or not

        # # Create a profiler object
        # profiler = cProfile.Profile()

        # # Start profiling
        # profiler.enable()



        sampler = EnsembleSampler(NWALKERS,
                                    N_DIM_MCMC,
                                    lnpb,
                                    pool=pool,
                                    backend=BACKEND)

        sampler.run_mcmc(init_walkers, N_ITER_MCMC, progress=True)

        # # Stop profiling
        # profiler.disable()
        # # Save the stats to a file
        # profiler.dump_stats(f'{basedir}/debrisdisk_mcmc_fit_and_plot/diskfit_mcmc.prof')
        # profiler.dump_stats("/tmp/cProfile/modelfit_physical_to_freeform.prof")

    print("Time {0} iterations with {1} walkers and {2} cpus: {3}".format(
              N_ITER_MCMC, NWALKERS, cpu_count(),
              datetime.now() - startTime))
