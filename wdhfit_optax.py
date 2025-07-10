"""
This is the main script for fitting the WDH to a given KLIP-reduced image
so that it can be removed.

The pipeline now makes use of classes for organization and consolidating
the initialization process. Since JAX was developed for pure functions, we
should keep all JAX-powered computations as functions that are called in the
loss function. This includes:

- WDH image generation
- Forward modeling
- Calculating the MSE

The main func should initialize the WDH model object, then generate and save the
masks, initial model, and initial FM. 


"""

import os
import sys
# import copy
import argparse



basedir = f'{os.environ["HOME"]}/projects'  # the base directory where is
# your data (using OS environnement variable allow to use same code on
# different computer without changing this).

# default_parameter_file = 'HR4796_g_camsci2_20230312_13.yaml'  # name of the parameter file
default_parameter_file = 'HR4796a_z_lco2023a_magao-x_20230309_10.yaml'  # name of the parameter file
# default_parameter_file = 'HR4796_i_smlyot_20230309_10.yaml'  # name of the parameter file
# you can also call it with the python function argument -p


import glob
import re

import time


# # because this error was coming up
# os.environ['OPENBLAS_NUM_THREADS'] = '1'


from functools import partial
import numpy as np

import astropy.io.fits as fits
# from astropy.convolution import convolve
# from scipy.signal import convolve
# from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

import yaml


from dev.pyklip.instruments.Instrument import GenericData

from dev.pyklip.fmlib.jax_diskfm import JDFM
from dev.pyklip.fmlib.funcs_JDFM import update_disk, fm_from_eigen_adi, \
                                        fm_from_eigen_rdi, \
                                        insert_section_into_full_image, \
                                        mass_derotation
from dev.pyklip.fmlib.diskfm import DiskFM
from dev.pyklip.j_klip import rotate_image
import dev.pyklip.fm as fm

from modeling.wdh_parametric import ParametricWDH
from modeling.jax_models.wdh_modeling import gen_multiwdh_image


import utils.make_gpi_psf_for_disks as gpidiskpsf
import utils.astro_unit_conversion as convert
from utils.klip_basis import load_kl_basis, unpack_basis_data
from utils.io import yaml_handling
from utils.model_tools import convolve_model


import jax
from jax import lax
import jax.numpy as jnp
import jax.profiler
from jax.scipy.signal import convolve2d
import optax



def main(config):

    # Init the wdh model object
    wdh_obj = ParametricWDH(config)

    # Make the intial model
    params_init = wdh_obj.params_init

    if params_init["FIRST_TIME"]:


        image_wdh_init = wdh_obj.gen_wdh_image_from_params(params_init)
        

    # Convolve the init model

    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run diskierFM autodiff')
    parser.add_argument('-p',
                        '--param-file',
                        required=False,
                        help='parameter file name')
    parser.add_argument(
                        '--iterations',
                        type=int,
                        required=True,
                        help='Num. iterations')
    args = parser.parse_args()


    main(args.param_file)