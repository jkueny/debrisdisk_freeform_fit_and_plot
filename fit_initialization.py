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
from models.wdh_models import gen_wdh_image


import jax
from jax import lax
import jax.numpy as jnp
import jax.profiler
from jax.scipy.signal import convolve2d

def parang_sort(filename):
        """Extracts the float value between '2x2bin_' and '_parang'."""
        match = re.search(r'2x2bin_([-+]?\d*\.\d+|\d+)_parang', filename)
        if match:
            return float(match.group(1))  # Convert extracted string to float
        return float('inf')  # Assign an arbitrary large value if no match is found

def initialize_freeform_model(dimensions):
    # Initialize freeform image parameters (random pixel values)
    rng = jax.random.PRNGKey(42)
    image_params = jax.random.normal(rng, dimensions)  # Trainable parameters
    return image_params


########################################################
def initialize_mask_psf_noise(params_modelfit, quietklip=True):
    """ initialize the MCMC by preparing the useful things to measure the
    likelyhood (measure the data, the psf, the uncertainty map, the masks).

    Args:
        params_modelfit: dic, all the parameters of the MCMC and klip
                            read from yaml file
        quietklip : if True, pyklip and DiskFM are quiet


    Returns:
        a dataset a pyklip instance of Instrument.Data
    """


    datadir = os.path.join(basedir, params_modelfit['BAND_DIR'])
    klipdir = os.path.join(datadir, 'klip_fm_files')

    os.makedirs(klipdir, exist_ok=True)

    file_prefix = params_modelfit['FILE_PREFIX']

    #The PSF centers
    aligned_center = params_modelfit['ALIGNED_CENTER']
    x_off = params_modelfit['MASK_DX']
    y_off = params_modelfit['MASK_DY']
    mask_center = aligned_center[0] + x_off, aligned_center[1] + y_off 
    ### This is the only part of the code different for GPI IFS anf SPHERE
    # For SPHERE We load and crop the PSF and the parangs
    # For GPI, we load the raw data, emasure hte PSF from sat spots and
    # collaspe the data
    filelist = sorted(glob.glob(f'{datadir}/*parang*.fits'), key=parang_sort)


    if len(filelist) == 0:
        raise ValueError(f"Could not find files in the dir: {datadir}")
    frames = []
    # refs = []
    derot_angs = []
    # print(type(first_frame))
    for each,name in enumerate(filelist):
        # with fits.open(name) as input_hdu:
        #     # print(input_hdu[0].header['ROTOFF'])
        #     # exit()
        #     derot_angs.append(input_hdu[0].header['ROTOFF'] - 180. + NORTH_CLIO)
        # #     input_data = np.append(first_frame,input_hdu[0].data,axis=0)
        #     frames.append(input_hdu[0].data)
        dat_unit, hdr_unit = fits.getdata(name,header=True)
        derot_angs.append(hdr_unit['PARANG'])
        frames.append(dat_unit)
    input_data = np.asarray(frames)
    par_angs = np.asarray(derot_angs)
    input_centers = np.array([aligned_center for _ in range(len(filelist))])
    # IWA = 10#use 10 for now, which is ~1.5 lambda/d JKK 01/08/22
    IWA = params_modelfit['IWA']#use 13 for now, post-optimized bkg sub SNRE says JKK 01/18/23
    dataset = GenericData(input_data,input_centers,parangs=par_angs,IWA=IWA,filenames=filelist)

    #After this, this is for both GPI and SPHERE
    #define the outer working angle
    dataset.OWA = params_modelfit['OWA']

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

        ## a few lines to create a circular central mask to hide center regions with a lot
    ## of speckles. Currently not using it but it's there
    mask_owa = np.ones((dataset.input.shape[1], dataset.input.shape[2]))
    x = np.arange(dataset.input.shape[1], dtype=float)[None,:] - aligned_center[0]
    y = np.arange(dataset.input.shape[2], dtype=float)[:,None] - aligned_center[1]
    rho2d = np.sqrt(x**2 + y**2)
    mask_owa[np.where(rho2d > dataset.OWA)] = 0.

    in_scaling = params_modelfit['MASK_IN_SCALING'] #originally 18
    out_scaling = params_modelfit['MASK_OUT_SCALING'] #originally 18

    mask_disk_zeros = gpidiskpsf.make_disk_mask(
        dataset.input.shape[1],
        params_modelfit['pa_init'],
        params_modelfit['inc_init'],
        convert.au_to_pix(params_modelfit['r1_init'],
                            params_modelfit['PIXSCALE_INS'],
                            params_modelfit['DISTANCE_STAR']) -
        in_scaling / np.cos(np.radians(params_modelfit['inc_init'] - 4)),
        convert.au_to_pix(params_modelfit['r2_init'],
                            params_modelfit['PIXSCALE_INS'],
                            params_modelfit['DISTANCE_STAR']) +
        out_scaling / np.cos(np.radians(params_modelfit['inc_init'])),
        aligned_center=mask_center)
    mask2generatedisk = 1 - mask_disk_zeros

    mask2generatedisk *= mask_owa

    print(f"Saving mask to {os.path.join(klipdir,
                                file_prefix + '_mask2generatedisk.fits')}")
    fits.writeto(os.path.join(klipdir,
                                file_prefix + '_mask2generatedisk.fits'),
                    mask2generatedisk,
                    overwrite=True)

    # # we create a second mask for the minimization a little bit larger
    # # (because model expect to grow with the PSF convolution and the FM)
    # # and we can also exclude the center region where there are too much speckles
    # mask_disk_zeros = gpidiskpsf.make_disk_mask(
    #     dataset.input.shape[1],
    #     params_modelfit['pa_init'],
    #     params_modelfit['inc_init'],
    #     convert.au_to_pix(params_modelfit['r1_init'],
    #                         params_modelfit['PIXSCALE_INS'],
    #                         params_modelfit['DISTANCE_STAR']) -
    #     in_scaling / np.cos(np.radians(params_modelfit['inc_init'] - 4)),
    #     convert.au_to_pix(params_modelfit['r2_init'],
    #                         params_modelfit['PIXSCALE_INS'],
    #                         params_modelfit['DISTANCE_STAR']) +
    #     out_scaling / np.cos(np.radians(params_modelfit['inc_init'])),
    #     aligned_center=mask_center)

    # mask2minimize = (1 - mask_disk_zeros)

    # fits.writeto(os.path.join(klipdir,
    #                             file_prefix + '_mask2minimize.fits'),
    #                 mask2minimize,
    #                 overwrite='True')
    psflib = None



    return dataset, psflib


########################################################
def  initialize_diskfm(dataset, params_modelfit, psf, psflib=None, quietklip=True):
    """ initialize the MCMC by preparing the diskFM object

    Args:
        dataset: a pyklip instance of Instrument.Data
        params_modelfit: dic, all the parameters of the MCMC and klip
                            read from yaml file
        psflib : a librairy of PSF if RDI
        quietklip : if True, pyklip and DiskFM are quiet

    Returns:
        a  diskFM object
    """
    print("\n Initialize diskFM")
    aligned_center = params_modelfit['ALIGNED_CENTER']
    numbasis = [params_modelfit['KLMODE_NUMBER']]
    move_here = params_modelfit['MOVE_HERE']
    file_prefix = params_modelfit['FILE_PREFIX']
    mode = params_modelfit['MODE']
    annuli = params_modelfit['ANNULI']
    FIRST_TIME = params_modelfit["FIRST_TIME"]
    MODEL_TYPE = params_modelfit["DISK_MODEL"]
    datadir = os.path.join(basedir, params_modelfit['BAND_DIR'])
    klipdir = os.path.join(datadir, 'klip_fm_files')
    IMAGE_SIZE = round(aligned_center[0]) * 2, round(aligned_center[1]) * 2
   # Initialize freeform image parameters (random pixel values)
    if MODEL_TYPE.lower() == "freeform":
        rng = jax.random.PRNGKey(42)
        freeform_model_here = jax.random.normal(rng, IMAGE_SIZE)  # Trainable parameters
        model_convolved_jax = convolve2d(freeform_model_here, psf, mode="same")
        model_convolved_here = np.asarray(jax.device_get(model_convolved_jax))
    elif MODEL_TYPE.lower() == "wdh":
        # Read in the initial model params
        BETA_INIT = params_modelfit["hbeta_init"]
        HA_R_INIT = params_modelfit["ha_r_init"]
        SIG_INIT = params_modelfit["hsig_init"]
        PA_INIT = params_modelfit["hpa_init"]
        DX_INIT = params_modelfit["hdx_init"]
        NORM_INIT = params_modelfit["hNorm_init"]
        FWHM_INIT = params_modelfit["gfwhm_init"]
        x_vals = np.arange(-round(aligned_center[0]),round(aligned_center[1]), 1)
        y_vals = np.arange(-round(aligned_center[0]),round(aligned_center[1]), 1)
        xx, yy = np.meshgrid(x_vals, y_vals)

        wdh_image = gen_wdh_image(xx, yy, BETA_INIT, HA_R_INIT, SIG_INIT, PA_INIT, DX_INIT,
                                  FWHM_INIT, NORM_INIT)
    # print(type(model_convolved_here))
    # Disable print for pyklip
    if quietklip:
        sys.stdout = open(os.devnull, 'w')

    if FIRST_TIME:
        # initialize the DiskFM object
        diskobj = DiskFM(dataset.input.shape,
                            numbasis,
                            dataset,
                            model_convolved_here,
                            basis_filename=os.path.join(
                                klipdir, file_prefix + '_klbasis.h5'),
                            save_basis=True,
                            aligned_center=aligned_center)
        # measure the KL basis and save it

        maxnumbasis = dataset.input.shape[0]
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
    else:
        # load the the KL basis and define the diskFM object
        diskobj = JDFM(None,
                        None,
                        None,
                        model_convolved_here,
                        basis_filename=os.path.join(klipdir,
                                                    file_prefix + '_klbasis.h5'),
                        load_from_basis=True)

    reduced_data = fits.getdata(os.path.join(klipdir,
                                                file_prefix + '-klipped-KLmodes-all.fits'))[0]


    
    return diskobj, reduced_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run diskierFM autodiff')
    parser.add_argument('-p',
                        '--param-file',
                        required=False,
                        help='parameter file name')
    parser.add_argument(
                        '--initial-model',
                        type=str,
                        required=False,
                        help='Path to starting model fits file')
    args = parser.parse_args()
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
        params_modelfit = yaml.safe_load(yaml_file)
    # Grab the info from the yaml file
    FILE_PREFIX = params_modelfit['FILE_PREFIX']
    KLIPDIR = os.path.join(basedir, params_modelfit['BAND_DIR'],
                           'klip_fm_files')
    RESULTS_DIR = os.path.join(basedir, params_modelfit['BAND_DIR'],
                           'results_freeform')
    # load DISTANCE_STAR & PIXSCALE_INS and make them global
    DISTANCE_STAR = params_modelfit['DISTANCE_STAR']
    PIXSCALE_INS = params_modelfit['PIXSCALE_INS']
    ALIGNED_CENTER = params_modelfit['ALIGNED_CENTER']
    FIRST_TIME = params_modelfit["FIRST_TIME"]
    MODE = params_modelfit["MODE"]
    # if MODE.upper() == "ADI":
    #     mode = 0
    # elif MODE.upper() == "RDI":
    #     mode = 1
    BASIS_FILE = f"{KLIPDIR}/{FILE_PREFIX}_klbasis.h5"

    dataset, psflib = initialize_mask_psf_noise(params_modelfit,
                                                quietklip=True)
    
    # load wheremask2generatedisk
    print(f"Loading mask: {os.path.join(KLIPDIR, FILE_PREFIX + '_mask2generatedisk.fits')}")
    WHEREMASK2GENERATEDISK = fits.getdata(
        os.path.join(KLIPDIR, FILE_PREFIX + '_mask2generatedisk.fits'))


    # load PSF
    psf = fits.getdata(os.path.join(KLIPDIR, FILE_PREFIX + '_instrPSF.fits'))
    JAX_PSF = jnp.array(psf)
    JAX_PSF /= jnp.sum(JAX_PSF)

    # measure the size of images DIMENSION and make it global
    DIMENSION = round(ALIGNED_CENTER[0]) * 2

    if FIRST_TIME:
        # print(FIRST_TIME)
        # initialize_diskfm and make diskobj global
        DISKOBJ, REDUCED_DATA = initialize_diskfm(dataset,
                                    params_modelfit,
                                    psf=JAX_PSF,
                                    psflib=psflib,
                                    quietklip=True)
        print('First time initializing, check klip_fm_files directory and modify the yaml file FIRST_TIME flag.')
        sys.exit(0)
    else:
        print(f"FIRST_TIME set to {FIRST_TIME}; this script is just for making the basis file...")