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


import utils.make_gpi_psf_for_disks as gpidiskpsf
import utils.astro_unit_conversion as convert
from utils.klip_basis import load_kl_basis, unpack_basis_data


import jax
from jax import lax
import jax.numpy as jnp
import jax.profiler
from jax.scipy.signal import convolve2d
import optax