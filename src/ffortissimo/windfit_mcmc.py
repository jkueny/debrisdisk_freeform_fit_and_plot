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

# TODO fix basedir to be more general
basedir = f'{os.environ["HOME"]}/data'  # the base directory where is
# your data (using OS environnement variable allow to use same code on
# different computer without changing this).

# default_parameter_file = 'HR4796_g_camsci2_20230312_13.yaml'  # name of the parameter file
# default_parameter_file = 'HR4796_r_camsci1_20230312_13.yaml'  # name of the parameter file
# default_parameter_file = 'HR4796a_z_lco2023a_magao-x_20230309_10.yaml'  # name of the parameter file
# default_parameter_file = 'HR4796_i_camsci1_20230309_10.yaml'  # name of the parameter file
default_parameter_file = "wdh_HR4796_z_20230309_10.yaml"
# you can also call it with the python function argument -p


import sys
import glob

# import cProfile

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from multiprocessing import cpu_count


from datetime import datetime

import math as mt
import numpy as np

import astropy.io.fits as fits
# from astropy.convolution import convolve
from scipy.signal import convolve
from scipy.ndimage import labeled_comprehension
# from scipy.signal import fftconvolve
from astropy.wcs import FITSFixedWarning

import yaml

from emcee import EnsembleSampler
from emcee import backends
# from emcee.moves import StretchMove, DEMove, KDEMove
from emcee.moves import DEMove, DESnookerMove

from numba.core.errors import NumbaWarning

from ffortissimo.dev.pyklip.instruments.Wind import GenericWDH

from ffortissimo.dev.pyklip.fmlib.windfm import WindFM
import ffortissimo.dev.pyklip.fm as fm
import ffortissimo.dev.pyklip.parallelized as parallelized

from ffortissimo.utils.masks import control_region_mask

import ffortissimo.utils.make_gpi_psf_for_disks as gpidiskpsf
import ffortissimo.utils.astro_unit_conversion as convert
from ffortissimo.utils.improc_tools import subtract_radial_profile, get_radial_inds

# recommended by emcee https://emcee.readthedocs.io/en/stable/tutorials/parallel/
# and by PyKLIPto avoid that NumPy automatically parallelizes some operations,
# which kill the speed
os.environ["OMP_NUM_THREADS"] = "1"
# # because this error was coming up
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['ACCELERATE_NUM_THREADS'] = '1'


# Globals consumed by lnpb/logl/call_gen_disk
DIMENSION = None
ALIGNED_CENTER = None
WHEREMASK2GENERATEHALO = None
DISKOBJ = None
REDUCED_DATA = None
NOISE = None
USE_NOISE = None
RPROFSUB = False
RADIAL_INDS = None
PA_PRIOR_SIGMA = 20.0

def sort_parang_monotonic(filename):
    # This regex captures a signed float between "2x2bin_" and "_parang"
    match = re.search(r'bin_([-+]?\d*\.?\d+)_parang', filename)
    if match:
        return float(match.group(1))
    else:
        raise ValueError(f"Filename {filename} does not match expected pattern.")

def subtract_radial_profile_np(image, radial_inds, radii):
    medians = labeled_comprehension(
        image, radial_inds, radii, np.median, float, 0.0
    )
    profile_2d = medians[radial_inds]
    return image - profile_2d


def _wrapped_angle_delta_deg(angle_deg, center_deg):
    """Shortest signed angular distance in degrees."""
    return ((float(angle_deg) - float(center_deg) + 180.0) % 360.0) - 180.0
    
def prep_image_frames_parangs(filelist):
    derot_angs = []
    frames = []
    for name in filelist:
        dat_unit, hdr_unit = fits.getdata(name,header=True)
        derot_angs.append(hdr_unit['PARANG'])
        frames.append(dat_unit)
    return np.asarray(frames), np.asarray(derot_angs)

def gen_wdh_image(
    x: np.ndarray,
    y: np.ndarray,
    wheremask2generatehalo: np.ndarray,
    beta: float,
    h0: float,
    sigma: float,
    PA_deg: float,
    x0: float,
    gamma: float,
) -> np.ndarray:
    """
    Generate a single WDH-like intensity map on the same grid as ``x`` and ``y``.

    Spatial support is set by ``wheremask2generatehalo`` (same convention as the global
    ``WHEREMASK2GENERATEHALO``): pixels where this mask is True are excluded from the
    model (no inner IWA/r1 cutoff — the control-region mask defines extent).

    ``PA_deg`` is the position angle (degrees) used to rotate the halo axis; ``x0`` is an
    offset along the rotated radial coordinate (pixels, same units as ``x``, ``y``).
    """
    pa_rad = -np.deg2rad(PA_deg)
    x_rot = np.sin(pa_rad) * x + np.cos(pa_rad) * y
    y_rot = np.cos(pa_rad) * x - np.sin(pa_rad) * y

    dx = x_rot - x0
    r = np.sqrt(dx**2 + y_rot**2)
    r_safe = np.maximum(r, 1e-6)

    power_law = (1.0 / r_safe) ** beta
    denom = h0 * (dx**2)
    denom = np.where(np.abs(denom) < 1e-12, np.copysign(1e-12, denom + 1e-30), denom)
    radial_term = (r**2 / denom) ** gamma
    sigma_safe = np.maximum(np.abs(sigma), 1e-6)
    exp_term = np.exp(-0.5 * (radial_term + (x_rot / sigma_safe) ** 2))
    i_map = power_law * exp_term
    i_map = np.nan_to_num(i_map, nan=0.0, posinf=0.0, neginf=0.0)
    # Restrict model to the halo generation region (from mask2generatehalo.fits logic).
    exclude = np.asarray(wheremask2generatehalo, dtype=np.float64)
    i_map = i_map * (1.0 - exclude)

    return i_map.astype(np.float32, copy=False)


def _cfg_first_match(cfg, candidates, default=None):
    """Return the first key match from candidates in cfg."""
    for key in candidates:
        if key in cfg:
            return cfg[key]
    return default


def _param_candidates(base_name, comp_idx, suffix):
    """
    Return init/state key candidates for one WDH component parameter.
    Supports both legacy (hbeta/ha_r/...) and newer (beta/a_r/...) conventions.
    """
    init_keys = {
        "beta": [f"hbeta{suffix}_init", f"beta{suffix}_init"],
        "h0": [f"ha_r{suffix}_init", f"a_r{suffix}_init"],
        "sigma": [f"hsig{suffix}_init", f"sig{suffix}_init"],
        "PA": [f"hpa{suffix}_init", f"pa{suffix}_init"],
        "x0": [f"hdx{suffix}_init", f"dx{suffix}_init"],
        "Norm": [f"hN{suffix}_init", f"Norm{suffix}_init"],
    }
    state_keys = {
        "beta": [f"hbeta{suffix}_state", f"beta{suffix}_state"],
        "h0": [f"ha_r{suffix}_state", f"a_r{suffix}_state"],
        "sigma": [f"hsig{suffix}_state", f"sig{suffix}_state"],
        "PA": [f"hpa{suffix}_state", f"pa{suffix}_state"],
        "x0": [f"hdx{suffix}_state", f"dx{suffix}_state"],
        "Norm": [f"hN{suffix}_state", f"Norm{suffix}_state"],
    }
    return init_keys[base_name], state_keys[base_name]


def _component_init_from_yaml(params_mcmc_yaml, comp_idx):
    suffix = "" if comp_idx == 1 else str(comp_idx)
    cfg = params_mcmc_yaml.get("wdh_model", params_mcmc_yaml)
    params = {}
    for p_name in ("beta", "h0", "sigma", "PA", "x0", "Norm"):
        init_candidates, _ = _param_candidates(p_name, comp_idx, suffix)
        default_val = 1.0 if p_name == "Norm" else 0.0
        params[p_name] = float(_cfg_first_match(cfg, init_candidates, default=default_val))
    return params


def _n_wdh_components(params_mcmc_yaml):
    cfg = params_mcmc_yaml.get("wdh_model", params_mcmc_yaml)
    return int(_cfg_first_match(cfg, ["N_WIND_LAYERS", "N_WDH_COMPONENTS"], default=3))


def _shared_component_flags(params_mcmc_yaml):
    cfg = params_mcmc_yaml.get("wdh_model", params_mcmc_yaml)
    return {
        "beta": bool(_cfg_first_match(cfg, ["beta_shared_state"], default=False)),
        "h0": bool(_cfg_first_match(cfg, ["h0_shared_state", "a_r_shared_state"], default=False)),
        "sigma": bool(_cfg_first_match(cfg, ["sigma_shared_state", "sig_shared_state"], default=False)),
        # Keep PA component-specific by design (sharing PA would collapse components).
        "PA": False,
        "x0": bool(_cfg_first_match(cfg, ["x0_shared_state", "dx_shared_state"], default=False)),
        "Norm": bool(_cfg_first_match(cfg, ["Norm_shared_state"], default=False)),
    }


def validate_mcmc_runtime_globals():
    """Ensure globals used by lnpb/logl/call_gen_disk are initialized."""
    required = {
        "DIMENSION": DIMENSION,
        "ALIGNED_CENTER": ALIGNED_CENTER,
        "WHEREMASK2GENERATEHALO": WHEREMASK2GENERATEHALO,
        "DISKOBJ": DISKOBJ,
        "REDUCED_DATA": REDUCED_DATA,
        "USE_NOISE": USE_NOISE,
    }
    if USE_NOISE:
        required["NOISE"] = NOISE
    if RPROFSUB:
        required["RADIAL_INDS"] = RADIAL_INDS

    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            "Missing required MCMC globals before multiprocessing launch: "
            + ", ".join(missing)
        )


def _init_mcmc_worker(
    dimension,
    aligned_center,
    wheremask2generatehalo,
    reduced_data,
    noise,
    use_noise,
    rprofsub,
    radial_inds,
    radii,
    component_init,
    shared_component_flags,
    free_params,
    n_wdh_components,
    gamma_fixed,
    pa_prior_sigma,
    theta_init,
    basis_filename,
    initial_model_list,
):
    """Pool initializer that populates every module-level global consumed by
    ``lnpb`` / ``logl`` / ``call_gen_disk`` in a freshly-spawned worker.

    Needed because on macOS NumPy is linked against Accelerate, and Accelerate
    uses Grand Central Dispatch which is not ``fork()``-safe. We therefore use
    ``forkserver`` (or ``spawn``) as the start method; those methods do not
    inherit parent globals, so each worker has to set its own state here.

    The large ``DISKOBJ`` (WindFM) object is intentionally not passed across
    the pickle boundary: workers rebuild it from the on-disk KL basis, which
    is cheap after the OS file cache has the file warm.
    """
    global DIMENSION, ALIGNED_CENTER, WHEREMASK2GENERATEHALO
    global DISKOBJ, REDUCED_DATA, NOISE, USE_NOISE, RPROFSUB, RADIAL_INDS
    global COMPONENT_INIT, SHARED_COMPONENT_FLAGS, FREE_PARAMS, RADII
    global N_WDH_COMPONENTS, GAMMA_FIXED, PA_PRIOR_SIGMA, THETA_INIT

    DIMENSION = dimension
    ALIGNED_CENTER = aligned_center
    WHEREMASK2GENERATEHALO = wheremask2generatehalo
    REDUCED_DATA = reduced_data
    NOISE = noise
    USE_NOISE = use_noise
    RPROFSUB = rprofsub
    RADIAL_INDS = radial_inds
    RADII = radii
    COMPONENT_INIT = component_init
    SHARED_COMPONENT_FLAGS = shared_component_flags
    FREE_PARAMS = free_params
    N_WDH_COMPONENTS = n_wdh_components
    GAMMA_FIXED = gamma_fixed
    PA_PRIOR_SIGMA = pa_prior_sigma
    THETA_INIT = theta_init

    DISKOBJ = WindFM(
        None,
        None,
        None,
        model_wdh_list=initial_model_list,
        model_pas_mask=None,
        basis_filename=basis_filename,
        load_from_basis=True,
    )
    # Note: we deliberately do NOT call DISKOBJ.update_wind(initial_model_list)
    # here. The very first lnpb/logl invocation in this worker calls update_wind
    # with the proposed walker state, immediately overwriting any model_wdhs we
    # would compute now. validPAs/wdhPAs/aligned_center are all restored by
    # load_from_basis=True, so the object is in a usable state already.


def arr_free_params(params_mcmc_yaml):
    free_params = []
    n_components = _n_wdh_components(params_mcmc_yaml)
    shared_flags = _shared_component_flags(params_mcmc_yaml)
    cfg = params_mcmc_yaml.get("wdh_model", params_mcmc_yaml)

    for p_name in ("beta", "h0", "sigma", "PA", "x0", "Norm"):
        if shared_flags[p_name]:
            init_candidates, state_candidates = _param_candidates(p_name, 1, "")
            is_free = bool(_cfg_first_match(cfg, state_candidates, default=True))
            if is_free:
                free_params.append(f"{p_name}_1")
            continue

        for idx in range(1, n_components + 1):
            suffix = "" if idx == 1 else str(idx)
            _, state_candidates = _param_candidates(p_name, idx, suffix)
            is_free = bool(_cfg_first_match(cfg, state_candidates, default=True))
            if is_free:
                free_params.append(f"{p_name}_{idx}")

    print(f'Fitting params: {free_params}')
    return free_params

def from_theta_to_params(theta):
    '''
    Setup function. This function takes the prior parameters and creates a dictionary and vector of parameters from it.
    '''
    theta = np.asarray(theta, dtype=float)
    comp_params = [dict(item) for item in COMPONENT_INIT]
    vector_param = []

    if theta.size != len(FREE_PARAMS):
        raise ValueError(
            f"Theta size ({theta.size}) does not match free parameter count ({len(FREE_PARAMS)})."
        )

    free_idx = 0
    for p_name in ("beta", "h0", "sigma", "PA", "x0", "Norm"):
        if SHARED_COMPONENT_FLAGS[p_name]:
            token = f"{p_name}_1"
            if token in FREE_PARAMS:
                value = float(theta[free_idx])
                free_idx += 1
            else:
                value = float(comp_params[0][p_name])
            for comp in comp_params:
                comp[p_name] = value
                vector_param.append(value)
            continue

        for idx in range(N_WDH_COMPONENTS):
            token = f"{p_name}_{idx + 1}"
            if token in FREE_PARAMS:
                value = float(theta[free_idx])
                free_idx += 1
                comp_params[idx][p_name] = value
                vector_param.append(value)

    return {"components": comp_params, "gamma": GAMMA_FIXED}, vector_param


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
    x = np.arange(DIMENSION, dtype=np.float64)[None, :] - ALIGNED_CENTER[0]
    y = np.arange(DIMENSION, dtype=np.float64)[:, None] - ALIGNED_CENTER[1]
    model_list = []

    for comp in param_disk["components"]:
        model = gen_wdh_image(
            x=x,
            y=y,
            wheremask2generatehalo=WHEREMASK2GENERATEHALO,
            beta=comp["beta"],
            h0=comp["h0"],
            sigma=comp["sigma"],
            PA_deg=comp["PA"],
            x0=comp["x0"],
            gamma=param_disk["gamma"],
        )
        model = model * comp["Norm"]
        model[np.isnan(model)] = 0
        model_list.append(model.astype(np.float32, copy=False))

    return model_list


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
    model_list = call_gen_disk(theta)
    if RPROFSUB:
        combined_model = np.nansum(np.asarray(model_list), axis=0)
        combined_model_sub = subtract_radial_profile_np(
            combined_model,
            RADIAL_INDS,
            RADII,
        )
        model_list = [np.asarray(combined_model_sub, dtype=np.float32)]
    DISKOBJ.update_wind(model_list)
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
    lp_reg = 0.0
    for idx, comp in enumerate(param_disk["components"], start=1):
        if comp["beta"] < -50 or comp["beta"] > 50:
            print(f'beta_{idx} out of prior.')
            return -np.inf
        if comp["h0"] < 0.01 or comp["h0"] > 10:
            print(f'h0_{idx} out of prior')
            return -np.inf
        if comp["sigma"] < 0.01 or comp["sigma"] > 224:
            print(f'sigma_{idx} out of prior')
            return -np.inf
        if comp["PA"] < -180 or comp["PA"] > 180:
            print(f'PA_{idx} out of prior')
            return -np.inf
        pa0 = float(COMPONENT_INIT[idx - 1]["PA"])
        dpa = _wrapped_angle_delta_deg(comp["PA"], pa0)
        lp_reg += -0.5 * (dpa / float(PA_PRIOR_SIGMA))**2
        if comp["x0"] < -7.5 or comp["x0"] > 7.5:
            print(f'x0_{idx} out of prior')
            return -np.inf
        if comp["Norm"] < 0.001 or comp["Norm"] > 1e10:
            print(f'Norm_{idx} out of prior')
            return -np.inf

    return lp_reg


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
    if aligned_center is None:
        image_center = nodisk_data.shape[0] // 2, nodisk_data.shape[1] // 2
    else:
        image_center = aligned_center
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
        dim = nodisk_data.shape[0]
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

def create_uncertainty_map(dataset, params_mcmc_yaml):
    """ measure the uncertainty map using a KLIP image built from
    negated parallactic angles.
    described in Sec4 of Gerard&Marois SPIE 2016 and probabaly elsewhere

    Args:
        dataset: a pyklip instance of Instrument.Data containing the data
        params_mcmc_yaml: dic, all the parameters of the MCMC and klip
                            read from yaml file
        delta_raddii: pixel, widht of the small concentric rings
    Returns:
        a [dim,dim] array containing only speckles and the disk has been removed

    """
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    delta_raddii = int(params_mcmc_yaml.get("DELTA_RADII", 1))

    datadir = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    klipdir = os.path.join(datadir, 'wind_fm_files')

    maskfornoisemap = fits.getdata(
        os.path.join(klipdir, file_prefix + '_mask2minimize.fits')
    )

    backrot_klip = fits.getdata(
        os.path.join(klipdir, file_prefix + '_backrot-KLmodes-all.fits')
    )[0]

    # we don't need the noise map (which is the disk masked out) because
    # the disk has been medianed out in the backrotated KLIP image
    noise = make_noise_map_rings(
        backrot_klip,
        aligned_center=aligned_center,
        delta_raddii=delta_raddii,
    )
    noise[noise == 0] = np.nan  #we are going to divide by this noise

    return noise


def create_backrotated_klip_image(dataset, params_mcmc_yaml, psflib=None, quietklip=True):
    """Run KLIP once more with negated PAs and save the back-rotated product."""
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]
    move_here = params_mcmc_yaml['MOVE_HERE']
    mode = params_mcmc_yaml['MODE']
    annuli = params_mcmc_yaml['ANNULI']
    hp = params_mcmc_yaml['HP_FILTER']
    datadir = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    klipdir = os.path.join(datadir, 'wind_fm_files')
    backrot_prefix = file_prefix + '_backrot'

    original_pas = np.asarray(dataset._PAs, dtype=float).copy()

    try:
        dataset._PAs = -original_pas

        blank_model = np.zeros(dataset.input.shape[1:], dtype=np.float32)
        blank_model_list = [blank_model.copy() for _ in range(max(int(N_WDH_COMPONENTS), 1))]
        model_pa_mask = np.ones((len(blank_model_list), dataset.input.shape[0]), dtype=bool)
        windobj_backrot = WindFM(
            dataset.input.shape,
            numbasis,
            dataset,
            model_wdh_list=blank_model_list,
            model_pas_mask=model_pa_mask,
            basis_filename=os.path.join(klipdir, backrot_prefix + '_klbasis.h5'),
            save_basis=True,
            aligned_center=aligned_center,
        )
        maxnumbasis = dataset.input.shape[0]
        if quietklip:
            sys.stdout = open(os.devnull, 'w')

        if hp > 0:
            parallelized.klip_dataset(dataset,
                            numbasis=numbasis,
                            maxnumbasis=maxnumbasis,
                            annuli=annuli,
                            subsections=1,
                            mode=mode,
                            outputdir=klipdir,
                            fileprefix=backrot_prefix,
                            aligned_center=aligned_center,
                            highpass=hp,
                            minrot=move_here,
                            calibrate_flux=False,
                            numthreads=1,
                            time_collapse='median',
                            psf_library=psflib)
        else:
            parallelized.klip_dataset(dataset,
                            numbasis=numbasis,
                            maxnumbasis=maxnumbasis,
                            annuli=annuli,
                            mode=mode,
                            subsections=1,
                            outputdir=klipdir,
                            fileprefix=backrot_prefix,
                            aligned_center=aligned_center,
                            highpass=False,
                            minrot=move_here,
                            calibrate_flux=False,
                            numthreads=1,
                            time_collapse='median',
                            psf_library=psflib)
    finally:
        dataset._PAs = original_pas
        if quietklip:
            sys.stdout = sys.__stdout__




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
    ### This is the only part of the code different for GPI IFS anf SPHERE
    # For SPHERE We load and crop the PSF and the parangs
    # For GPI, we load the raw data, emasure hte PSF from sat spots and
    # collaspe the data
    psflib = None

    if first_time:
        if instrument == 'MagAO-X':
            filelist = sorted(glob.glob(f'{datadir}/*parang.fits'), key=sort_parang_monotonic)
            if len(filelist) == 0:
                raise ValueError(f"Could not find files in the dir: {datadir}")

            input_data, par_angs = prep_image_frames_parangs(filelist)
            input_centers = np.array([aligned_center for _ in range(len(filelist))])
            # IWA = 10#use 10 for now, which is ~1.5 lambda/d JKK 01/08/22
            IWA = params_mcmc_yaml['IWA']#use 13 for now, post-optimized bkg sub SNRE says JKK 01/18/23
            wdh_parangs = np.tile(par_angs, (N_WDH_COMPONENTS, 1))
            valid_pa_mask = np.ones_like(wdh_parangs, dtype=bool)

            dataset = GenericWDH(input_data,
                                 input_centers,
                                 obj_parangs=par_angs,
                                 wdh_parangs=wdh_parangs,
                                 IWA=IWA,filenames=filelist)
            dataset._wdhValidMask = valid_pa_mask
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
        
        create_backrotated_klip_image(
            dataset,
            params_mcmc_yaml,
            psflib=psflib,
            quietklip=quietklip,
        )
        noise = create_uncertainty_map(dataset, params_mcmc_yaml)
        noise *= noise_multiplication_factor
        fits.writeto(os.path.join(klipdir, file_prefix + '_noisemap.fits'),
                     noise,
                     overwrite='True')
    else:
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


    model_pa_mask = np.ones((N_WDH_COMPONENTS, dataset.input.shape[0]), dtype=bool) if first_time else None

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
        model_list_here = call_gen_disk(theta_init)

        fits.writeto(os.path.join(klipdir, file_prefix + '_FirstModel.fits'),
                     np.nansum(np.asarray(model_list_here), axis=0),
                     overwrite='True')

    else:
        model_list_here = call_gen_disk(THETA_INIT)

    if first_time:
        # Disable print for pyklip
        # if quietklip:
        #     sys.stdout = open(os.devnull, 'w')
        # initialize the DiskFM object

        diskobj = WindFM(dataset.input.shape,
                         numbasis,
                         dataset,
                         model_wdh_list=model_list_here,
                         model_pas_mask=model_pa_mask,
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
                     model_wdh_list=model_list_here,
                     model_pas_mask=None,
                     basis_filename=os.path.join(klipdir,
                                                 file_prefix + '_klbasis.h5'),
                     load_from_basis=True)
    

    # test the diskFM object

    diskobj.update_wind(model_list_here)

    if first_time:
        ### we take only the first KL modemode
        modelfm_here = diskobj.fm_parallelized()[0]
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
    _ = params_mcmc_yaml  # maintained for compatibility with legacy call sites
    theta_init = []
    component_lookup = {idx + 1: comp for idx, comp in enumerate(COMPONENT_INIT)}

    for token in FREE_PARAMS:
        p_name, idx_str = token.split("_")
        idx = int(idx_str)
        theta_init.append(component_lookup[idx][p_name])

    return np.asarray(theta_init, dtype=float)


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
    parser.add_argument('--nwalkers',
                        type=int,
                        required=False,
                        help='override walker count from parameter file')
    parser.add_argument('--niter',
                        type=int,
                        required=False,
                        help='override MCMC iteration count from parameter file')
    _ft = parser.add_mutually_exclusive_group()
    _ft.add_argument(
        '--first-time',
        action='store_true',
        help='override FIRST_TIME in parameter file (force true: rebuild FM inputs)',
    )
    _ft.add_argument(
        '--do-mcmc',
        action='store_true',
        help='override FIRST_TIME in parameter file (force false: reuse existing wind_fm_files)',
    )
    args = parser.parse_args()

    # Parallel processing stuff.
    #
    # On macOS, NumPy wheels are linked against Apple's Accelerate framework
    # which uses Grand Central Dispatch internally. GCD is NOT fork()-safe, so
    # using mp.get_context('fork').Pool here causes a silent SIGSEGV inside
    # the first np.dot() call in each worker (see pyklip/fm.py::perturb_*).
    # The symptom in that case is that the parent appears to hang because
    # emcee is waiting on pool.map() results that never come back.
    #
    # We use 'forkserver' instead: a clean helper process is spawned once,
    # and workers are forked from it before Accelerate/GCD ever initializes
    # in that server. Each worker loads NumPy post-fork, which is safe.
    import multiprocessing as mp
    mp_ctx = mp.get_context('fork')
    MultiPool = mp_ctx.Pool


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
    NWALKERS = params_mcmc_yaml['NWALKERS']  # Number of walkers
    N_ITER_MCMC = params_mcmc_yaml['N_ITER_MCMC']  # Number of iterations
    if args.nwalkers is not None:
        NWALKERS = int(args.nwalkers)
        print(f"CLI override: NWALKERS={NWALKERS}")
    if args.niter is not None:
        N_ITER_MCMC = int(args.niter)
        print(f"CLI override: N_ITER_MCMC={N_ITER_MCMC}")
    if args.first_time:
        params_mcmc_yaml['FIRST_TIME'] = True
        print("CLI override: FIRST_TIME=True")
    elif args.do_mcmc:
        params_mcmc_yaml['FIRST_TIME'] = False
        print("CLI override: FIRST_TIME=False")
    DISK_MODEL = params_mcmc_yaml['DISK_MODEL']

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


    if DISK_MODEL.lower() in ('wdh'):
        pass
    else:
        raise ValueError(DISK_MODEL + "not a valid disk model. Choose 'wdh'.")

    # load DISTANCE_STAR & PIXSCALE_INS and make them global
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']
    ALIGNED_CENTER = params_mcmc_yaml['ALIGNED_CENTER']


    wdh_cfg = params_mcmc_yaml.get("wdh_model", params_mcmc_yaml)
    N_WDH_COMPONENTS = _n_wdh_components(params_mcmc_yaml)
    SHARED_COMPONENT_FLAGS = _shared_component_flags(params_mcmc_yaml)
    COMPONENT_INIT = [
        _component_init_from_yaml(params_mcmc_yaml, idx)
        for idx in range(1, N_WDH_COMPONENTS + 1)
    ]
    GAMMA_FIXED = float(_cfg_first_match(wdh_cfg, ["gamma_fixed"], default=1.0))
    PA_PRIOR_SIGMA = float(
        _cfg_first_match(
            wdh_cfg,
            ["pa_prior_sigma", "PA_PRIOR_SIGMA", "hpa_prior_sigma"],
            default=20.0,
        )
    )
    FREE_PARAMS = arr_free_params(params_mcmc_yaml)
    THETA_INIT = from_param_to_theta_init(params_mcmc_yaml)

    # initialize the things necessary to measure the model (PSF, masks,
    # uncertainities). In RDI mode, psflib is also initiliazed here
    dataset, psflib = initialize_mask_psf_noise(params_mcmc_yaml, quietklip=True)

    ## Load all variables necessary for the MCMC and make them global
    ## to avoid very long transfert time at each iteration

    # load wheremask2generatedisk and make it global
    WHEREMASK2GENERATEHALO = (fits.getdata(
        os.path.join(KLIPDIR, FILE_PREFIX + '_mask2generatehalo.fits')) == 0)

    # load noise and make it global

    USE_NOISE = params_mcmc_yaml["USE_NOISE"]
    # TODO add a radial profile subtraction for the WDH model
    RPROFSUB = params_mcmc_yaml["RPROFSUB"]
    # if RPROFSUB:
    #     MED_PROFILE_EST = fits.getdata(f"{KLIPDIR}/estimated_med_profile.fits")
    #     MED_PROFILE_EST /= np.max(MED_PROFILE_EST)
    # measure the size of images DIMENSION and make it global
    DIMENSION = round(ALIGNED_CENTER[0]) * 2
    RADIAL_INDS = np.asarray(get_radial_inds((DIMENSION, DIMENSION), ALIGNED_CENTER))
    RADIAL_INDS_NP = np.asarray(RADIAL_INDS, dtype=np.int32)
    MAX_R = int(RADIAL_INDS_NP.max()) + 1
    RADII = np.arange(MAX_R)
    N_DIM_MCMC = len(THETA_INIT)


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

    # Explicitly verify all globals consumed inside lnpb/logl/model calls are ready
    # before handing work to multiprocessing workers.
    validate_mcmc_runtime_globals()

    # forkserver/spawn workers don't inherit the parent's globals, so we pass
    # everything lnpb needs via the pool initializer. DISKOBJ itself is rebuilt
    # inside each worker from the on-disk KL basis to avoid shipping a
    # ~100+ MB pickled WindFM over the IPC queue.
    _worker_basis_filename = os.path.join(KLIPDIR, FILE_PREFIX + '_klbasis.h5')
    _worker_initial_models = call_gen_disk(THETA_INIT)
    _worker_initargs = (
        DIMENSION,
        ALIGNED_CENTER,
        WHEREMASK2GENERATEHALO,
        REDUCED_DATA,
        NOISE,
        USE_NOISE,
        RPROFSUB,
        RADIAL_INDS,
        RADII,
        COMPONENT_INIT,
        SHARED_COMPONENT_FLAGS,
        FREE_PARAMS,
        N_WDH_COMPONENTS,
        GAMMA_FIXED,
        PA_PRIOR_SIGMA,
        THETA_INIT,
        _worker_basis_filename,
        _worker_initial_models,
    )

    print("multiprocessing start method:", mp_ctx.get_start_method())
    with MultiPool(
        initializer=_init_mcmc_worker,
        initargs=_worker_initargs,
    ) as pool:

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


        # moves = [(StretchMove(), 0.5), (DEMove(), 0.3), (KDEMove(), 0.2)]
        moves = [(DEMove(), 0.8), (DESnookerMove(), 0.2)]

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
