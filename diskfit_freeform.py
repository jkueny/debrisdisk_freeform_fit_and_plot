'''
Fit a pixel-by-pixel freeform model disk to a KLIP reduced image.

Using JAX.

To prepare, need to normalize the image data bc we're doing this exercise
just to learn the dust morphology of the disk such that we can fit functions
to it later.

Steps to develop:
1. Generate the KLIP image and save the basis for DiskFM.
2. Read in and save the instr. PSF to use for the FM.
3. Generate an image of random numbers as an intial guess and feed it to the
optimizer function.
4. 
'''

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
from scipy.signal import convolve
# from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

import yaml


from dev.pyklip.instruments.Instrument import GenericData

from dev.pyklip.fmlib.jax_diskfm import JDFM
from dev.pyklip.fmlib.funcs_JDFM import update_disk, fm_from_eigen_single, \
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



# jax.config.update("jax_enable_x64", True)

update_disk_jit = jax.jit(update_disk, static_argnames=["min_num_models",])
# reconstruct_full_image_jax = jax.jit(reconstruct_full_image,
#                                      static_argnames=["total_pixels"],
#                                      )
# insert_section_jaxed = jax.jit(insert_section_into_full_image,
#                                static_argnames=["full_shape","section_inds"])
# fm_jaxed_jit = jax.jit(fm_jaxed, static_argnames=[""])

def make_annular_mask(dimensions, inner_radius, outer_radius, center=None):
    """
    Create a binary annular mask for a 2D array.
    
    The pixels whose distance from the center is between inner_radius and outer_radius
    (inclusive) are set to 1; all other pixels are set to 0.
    
    Args:
        dimensions (tuple): (height, width) of the output mask.
        inner_radius (float): The inner radius (in pixels).
        outer_radius (float): The outer radius (in pixels).
        center (tuple, optional): (row, col) coordinates for the center.
                                  If None, defaults to the center of the array.
    
    Returns:
        jnp.ndarray: A binary mask with shape `dimensions` (1's inside the annulus, 0's outside).
    """
    H, W = dimensions
    if center is None:
        center = (H / 2, W / 2)
    
    # Create coordinate grid
    y, x = np.indices((H, W))
    # Compute the radial distance from the center for each pixel.
    # Note: center is given as (row, col) and x corresponds to column indices.
    r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
    # Create the binary mask: 1 inside the annulus, 0 elsewhere.
    mask = np.where((r >= inner_radius) & (r <= outer_radius), 1, 0)
    return mask


def reconstruct_full_image(free_params, total_pixels):
    """
    Given the free parameters (for the unmasked region) and the full image shape,
    create a full image (flattened) where the free parameters are inserted at the positions
    indicated by mask_indices and zeros elsewhere.
    """
    full_shape = (int(np.sqrt(total_pixels)), int(np.sqrt(total_pixels)))
    full_flat = jnp.zeros(total_pixels)
    full_flat = full_flat.at[MASK_INDICES].set(free_params)
    return full_flat.reshape(full_shape)

def initialize_freeform_model_reduced():
    """Initialize freeform model parameters for the unmasked region."""
    rng = jax.random.PRNGKey(42)
    # Instead of full image dimensions, only initialize num_free parameters.
    free_params = jax.random.normal(rng, (NUM_FREE,))

    return free_params


def parang_sort(filename):
        """Extracts the float value between '2x2bin_' and '_parang'."""
        match = re.search(r'2x2bin_([-+]?\d*\.\d+|\d+)_parang', filename)
        if match:
            return float(match.group(1))  # Convert extracted string to float
        return float('inf')  # Assign an arbitrary large value if no match is found

@jax.jit
def convolve_model(input_model, psf):
    # psf = jnp.asarray(psf)
    assert psf.shape[0] == psf.shape[1], "Instr. PSF image is not square. How can this be?!"
    kernel_size = psf.shape[0] #should be square
    # Pad image to maintain size
    # padded_image = jnp.pad(input_model, [(kernel_size//2, kernel_size//2),
    #                                (kernel_size//2, kernel_size//2)], mode='reflect')

    # Apply convoluted convolution
    model_convolved = convolve2d(input_model, psf, mode="same")
    
    return model_convolved

@jax.jit
def convolve_model_lax(input_model, psf):
    """
    Convolve a 2D input image with a 2D PSF using lax.conv_general_dilated,
    mimicking the behavior of jax.scipy.signal.convolve2d(mode="same").

    Args:
        input_model: 2D array of shape (H, W).
        psf: 2D array of shape (kH, kW). Should be square; if not, adjust accordingly.

    Returns:
        2D convolved image of shape (H, W).
    """
    # Ensure PSF is square.
    assert psf.shape[0] == psf.shape[1], "PSF must be square."

    # Expand dimensions: add a batch dimension and a channel dimension.
    # New shape: (1, H, W, 1)
    input_model_exp = input_model[None, ..., None]

    # To perform true convolution (not cross-correlation), flip the kernel along both axes.
    psf_flipped = jnp.flip(psf, axis=(0, 1))
    # Expand kernel dimensions: shape (kH, kW, in_channels, out_channels)
    psf_kernel = psf_flipped[..., None, None]

    # Call the convolution primitive with stride 1 and 'SAME' padding.
    # Using dimension_numbers 'NHWC' for inputs and outputs, and 'HWIO' for the kernel.
    convolved = lax.conv_general_dilated(
        input_model_exp,
        psf_kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=('NHWC', 'HWIO', 'NHWC')
    )

    # Squeeze out the added batch and channel dimensions.
    return jnp.squeeze(convolved, axis=(0, 3))

# @jax.jit(static_argnames=["full_shape", "section_inds"])


@partial(jax.jit, static_argnames=["fixed_refs","aligned_center",
                                   "total_pixels"])
def loss_function(mod_pix_params, disk_image, psf, aligned_images,
                  ref_psfs_stacked, PAs, ref_PAs, fixed_refs, aligned_center,
                  section_inds_arr, klmodes_stacked, evals, evecs_stacked, 
                  total_pixels):
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

    # DM commands scaled to [0,1] fits cubes do like 10 secs of wall clock time
    # Spatil freq. such that speckles end up at 10 lamb/D
    # So the wind is the rate of change of the phase 2pi v k thing maybe over D



    
    # Ensure that the model pixel values range [0,1]
    full_model_image = reconstruct_full_image(mod_pix_params, total_pixels)
    # freeform_model = jax.nn.sigmoid(mod_pix_params)
    # freeform_model = jax.nn.sigmoid(full_model_image)

    # freeform_image = convolve_model(full_model_image, psf)
    freeform_image = convolve_model_lax(full_model_image, psf)

    global_models_prepped, ref_models_stacked = update_disk(model_disk=freeform_image,
                                                            PAs=PAs, ref_PAs=ref_PAs,
                                                            # aligned_center=aligned_center,
                                                            section_inds=section_inds_arr,
                                                            min_num_models=fixed_refs,
                                                            )

    # confirmed shape of model_images_prepped (84, 50176)
    # flat_postklip_psfs = fm_jaxed(aligned_images,
    #                        global_models_prepped,
    #                        ref_models_stacked, ref_psfs_stacked,
    #                        klmodes_stacked, evals, evecs_stacked,
    #                        PAs)
    flat_postklip_psfs = jax.vmap(fm_from_eigen_single
                          )(aligned_images, ref_psfs_stacked,
                            global_models_prepped,ref_models_stacked,
                            klmodes_stacked, evals, evecs_stacked,
                            )
    derotated_postklip_psfs = mass_derotation(flat_postklip_psfs,PAs,
                                              total_pixels,section_inds_arr)

    freeform_fm_full = jnp.mean(derotated_postklip_psfs, axis=0)
    freeform_fm_flat = jnp.reshape(freeform_fm_full, psf.shape[0] * psf.shape[1])
    freeform_fm_interest = freeform_fm_flat[MASK_INDICES]

    # freeform_fm_interest = jnp.where(MASK, freeform_fm_full, jnp.nan)
    # disk_image_interest = jnp.where(MASK, disk_image, jnp.nan)

    # mse = jnp.nanmean((disk_image_interest - freeform_fm_interest) ** 2)
    mse = jnp.mean((disk_image - freeform_fm_interest) ** 2)

    jax.debug.print("print(mse) -> {x}", x=mse)

    return mse

def initialize_freeform_model(dimensions):
    # Initialize freeform image parameters (random pixel values)
    rng = jax.random.PRNGKey(42)
    image_params = jax.random.normal(rng, dimensions)  # Trainable parameters
    return image_params


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


    datadir = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    klipdir = os.path.join(datadir, 'klip_fm_files')

    os.makedirs(klipdir, exist_ok=True)

    file_prefix = params_mcmc_yaml['FILE_PREFIX']

    #The PSF centers
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    x_off = params_mcmc_yaml['MASK_DX']
    y_off = params_mcmc_yaml['MASK_DY']
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
    IWA = params_mcmc_yaml['IWA']#use 13 for now, post-optimized bkg sub SNRE says JKK 01/18/23
    dataset = GenericData(input_data,input_centers,parangs=par_angs,IWA=IWA,filenames=filelist)

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

    mask_disk_zeros = gpidiskpsf.make_disk_mask(
        dataset.input.shape[1],
        params_mcmc_yaml['pa_init'],
        params_mcmc_yaml['inc_init'],
        convert.au_to_pix(params_mcmc_yaml['r1_init'],
                            params_mcmc_yaml['PIXSCALE_INS'],
                            params_mcmc_yaml['DISTANCE_STAR']) -
        in_scaling / np.cos(np.radians(params_mcmc_yaml['inc_init'] - 4)),
        convert.au_to_pix(params_mcmc_yaml['r2_init'],
                            params_mcmc_yaml['PIXSCALE_INS'],
                            params_mcmc_yaml['DISTANCE_STAR']) +
        out_scaling / np.cos(np.radians(params_mcmc_yaml['inc_init'])),
        aligned_center=mask_center)
    mask2generatedisk = 1 - mask_disk_zeros
    fits.writeto(os.path.join(klipdir,
                                file_prefix + '_mask2generatedisk.fits'),
                    mask2generatedisk,
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
        in_scaling / np.cos(np.radians(params_mcmc_yaml['inc_init'] - 4)),
        convert.au_to_pix(params_mcmc_yaml['r2_init'],
                            params_mcmc_yaml['PIXSCALE_INS'],
                            params_mcmc_yaml['DISTANCE_STAR']) +
        out_scaling / np.cos(np.radians(params_mcmc_yaml['inc_init'])),
        aligned_center=mask_center)

    mask2minimize = (1 - mask_disk_zeros)

    ### a few lines to create a circular central mask to hide center regions with a lot
    ### of speckles. Currently not using it but it's there
    # mask_speckle_region = np.ones((dataset.input.shape[1], dataset.input.shape[2]))
    # x = np.arange(dataset.input.shape[1], dtype=np.float)[None,:] - aligned_center[0]
    # y = np.arange(dataset.input.shape[2], dtype=np.float)[:,None] - aligned_center[1]
    # rho2d = np.sqrt(x**2 + y**2)
    # mask_speckle_region[np.where(rho2d < 21)] = 0.
    # mask2minimize = mask2minimize*mask_speckle_region

    fits.writeto(os.path.join(klipdir,
                                file_prefix + '_mask2minimize.fits'),
                    mask2minimize,
                    overwrite='True')
    psflib = None



    return dataset, psflib


########################################################
def initialize_diskfm(dataset, params_mcmc_yaml, psf, psflib=None, quietklip=True):
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
    print("\n Initialize diskFM")
    aligned_center = params_mcmc_yaml['ALIGNED_CENTER']
    numbasis = [params_mcmc_yaml['KLMODE_NUMBER']]
    move_here = params_mcmc_yaml['MOVE_HERE']
    file_prefix = params_mcmc_yaml['FILE_PREFIX']
    mode = params_mcmc_yaml['MODE']
    annuli = params_mcmc_yaml['ANNULI']
    first_time = params_mcmc_yaml["FIRST_TIME"]
    datadir = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'])
    klipdir = os.path.join(datadir, 'klip_fm_files')
    image_size = round(aligned_center[0]) * 2, round(aligned_center[1]) * 2
   # Initialize freeform image parameters (random pixel values)
    rng = jax.random.PRNGKey(42)
    freeform_model_here = jax.random.normal(rng, image_size)  # Trainable parameters
    model_convolved_jax = convolve_model(freeform_model_here, psf)
    model_convolved_here = np.asarray(jax.device_get(model_convolved_jax))
    print(type(model_convolved_here))
    # Disable print for pyklip
    if quietklip:
        sys.stdout = open(os.devnull, 'w')

    if first_time:
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

# --- JIT-Compiled Gradient Computation ---
# loss_and_grad = jax.jit(jax.value_and_grad(loss_function))
loss_and_grad = jax.value_and_grad(loss_function)

def optimize_model(target_image, model_init,
                   psf, basis_data, total_pixels, num_steps=5, lr=0.1):
    
    # dimension = img_dim
    jax_target_image = jnp.array(target_image)

    # Initialize the initial image of random pixels
    image_params = model_init
    # image_params = initialize_freeform_model_reduced()

    # Set up the optimizer
    optimizer = optax.adam(lr)
    opt_state =  optimizer.init(image_params)

    loss_history = []

    basis_data_unpacked = unpack_basis_data(basis_data)

    # num_input_images = int(jax.device_get(basis_data["klparam_dict"]["nfiles"]))
    aligned_image_data = jnp.array(basis_data_unpacked["aligned_images"]) #shape ex. (84, 50176)
    section_inds = basis_data_unpacked["section_inds"][0] #shape ex. (1, 39112)
    klmodes = basis_data_unpacked["klmodes"] #shape (N_images, N_KLmodes, N_pixels) ex. (84, 2, 50176)
    evals = basis_data_unpacked["evals"] # shape (N_images, N_modes)
    # the eigenvectors have been zero-padded at the ends to removed ragged-ness....
    evecs = basis_data_unpacked["evecs"] # shape (N_images, max_N_refs, N_modes)
    # evecs have been unpacked, stacked, and ready to be BATCHED!
    # input_img_nums = basis_data_unpacked["input_img_nums"]
    # These are the images used for the basis for every image in the dataset.
    ref_psfs = basis_data_unpacked["ref_psfs"] # zero-padded at the end to all have the same shape
    ref_PAs = basis_data_unpacked["ref_PAs"]
    fixed_refs = basis_data_unpacked["fixed_refs"]
    # ref_psfs shape (N_images, max_N_refs, N_pixels)
    # ref_psfs have been unpacked, stacked, and ready to be BATCHED!
    # position_angles = tuple(np.asarray(jax.device_get(basis_data["klparam_dict"]["PAs"])))
    position_angles = jnp.array((basis_data["klparam_dict"]["PAs"]))
    aligned_center = tuple(np.asarray(jax.device_get([basis_data["klparam_dict"]["aligned_center_x"],
                                basis_data["klparam_dict"]["aligned_center_y"]])))
    # ref_psfs_indicies = basis_data_unpacked["ref_psfs_indicies"]

    time_now = time.time()
    # jax.profiler.start_trace("/tmp/tensorboard")
    jax.config.update("jax_debug_nans", True)

    @jax.jit
    def step(image_params, opt_state):
        loss, grads = loss_and_grad(image_params, jax_target_image, psf,
                                    aligned_image_data, ref_psfs,
                                    position_angles, ref_PAs, fixed_refs,
                                    aligned_center, section_inds,
                                    klmodes, evals, evecs, total_pixels)
        updates, opt_state = optimizer.update(grads, opt_state)
        image_params = optax.apply_updates(image_params, updates)
        return image_params, opt_state, loss
    
    for step_idx in range(num_steps):
        image_params, opt_state, loss = step(image_params, opt_state)
        loss_history.append(loss.item())

        if step_idx % 1000 == 0:
            print(f"Step {step_idx}/{num_steps} - Loss: {loss:.6f}")

    # jax.profiler.stop_trace()
    print(f"This run took {(time.time() - time_now):.6f} seconds.")
    
    # optimized_model = jax.nn.sigmoid(image_params)
    optimized_model = image_params
    return optimized_model, loss_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run diskFM MCMC')
    parser.add_argument('-p',
                        '--param_file',
                        required=False,
                        help='parameter file name')
    args = parser.parse_args()
    if args.param_file is None: #grab param file if no command line input, JKK
        str_yalm = f'initialization_files/{default_parameter_file}'
    else:
        str_yalm = args.param_file

    print("Read " + str_yalm + " parameter file")
    # open the parameter file
    yaml_path_file = os.path.join(os.getcwd(), str_yalm)
    with open(yaml_path_file, 'r') as yaml_file:
        params_mcmc_yaml = yaml.safe_load(yaml_file)
    # Grab the info from the yaml file
    FILE_PREFIX = params_mcmc_yaml['FILE_PREFIX']
    KLIPDIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'],
                           'klip_fm_files')
    RESULTS_DIR = os.path.join(basedir, params_mcmc_yaml['BAND_DIR'],
                           'results_freeform')
    # load DISTANCE_STAR & PIXSCALE_INS and make them global
    DISTANCE_STAR = params_mcmc_yaml['DISTANCE_STAR']
    PIXSCALE_INS = params_mcmc_yaml['PIXSCALE_INS']
    ALIGNED_CENTER = params_mcmc_yaml['ALIGNED_CENTER']
    FIRST_TIME = params_mcmc_yaml["FIRST_TIME"]
    BASIS_FILE = f"{KLIPDIR}/{FILE_PREFIX}_klbasis.h5"

    dataset, psflib = initialize_mask_psf_noise(params_mcmc_yaml,
                                                quietklip=True)
    
    # load wheremask2generatedisk
    WHEREMASK2GENERATEDISK = fits.getdata(
        os.path.join(KLIPDIR, FILE_PREFIX + '_mask2generatedisk.fits'))


    # load PSF
    psf = fits.getdata(os.path.join(KLIPDIR, FILE_PREFIX + '_instrPSF.fits'))
    JAX_PSF = jnp.array(psf)
    JAX_PSF /= jnp.sum(JAX_PSF)

    # measure the size of images DIMENSION and make it global
    DIMENSION = round(ALIGNED_CENTER[0]) * 2

    if FIRST_TIME:
        print(FIRST_TIME)
        # initialize_diskfm and make diskobj global
        DISKOBJ, REDUCED_DATA = initialize_diskfm(dataset,
                                    params_mcmc_yaml,
                                    psf=JAX_PSF,
                                    psflib=psflib,
                                    quietklip=True)
        print('First time initializing, check klip_fm_files directory and modify the yaml file first_time flag.')
        sys.exit(0)
    else:
        # Read in the basis data
        fm_dict = load_kl_basis(BASIS_FILE)
        # fm_dict contains 
        # dict_keys(['aligned_images_dict', 'evals_dict', 'evecs_dict',
        # 'input_img_num_dict', 'klmodes_dict', 'section_ind_dict'])
        REDUCED_DATA = fits.getdata(os.path.join(KLIPDIR,
                                                FILE_PREFIX + '-klipped-KLmodes-all.fits'))[0]
        # DISKOBJ, REDUCED_DATA = initialize_diskfm(dataset,
        #                             params_mcmc_yaml,
        #                             psf=JAX_PSF,
        #                             psflib=psflib,
        #                             quietklip=True)
    
    # mask the disk image
    TARGET_IMAGE = np.asarray(REDUCED_DATA * WHEREMASK2GENERATEDISK)# * 1e4
    TOTAL_PIXELS = np.prod(TARGET_IMAGE.shape)
    # print(TOTAL_PIXELS)
    TARGET_IMAGE[TARGET_IMAGE != TARGET_IMAGE] = 0.
    # print(INIT_MODEL_FLAT.shape)

    disk_mask = np.array(WHEREMASK2GENERATEDISK)  # convert to JAX array if needed
    annular_mask = make_annular_mask(disk_mask.shape, 10, 112)
    MASK = disk_mask * annular_mask
    MASK_INDICES = jnp.flatnonzero(MASK)  # 1D indices of nonzero (True) entries
    NUM_FREE = MASK_INDICES.shape[0]

    STARTING_DISK = fits.getdata("camsci2_z_20230309_10_FirstModel.fits")
    # STARTING_DISK = fits.getdata("freeform_run_10500iter.fits") #start from the last run
    STARTING_DISK *= MASK
    INIT_MODEL = jnp.array(STARTING_DISK)
    INIT_MODEL_FLAT = INIT_MODEL.reshape(INIT_MODEL.shape[0] * INIT_MODEL.shape[1])
    INIT_MODEL_INTEREST = INIT_MODEL_FLAT[MASK_INDICES]
    TARGET_IMAGE_FLAT = TARGET_IMAGE.reshape(TARGET_IMAGE.shape[0] * TARGET_IMAGE.shape[1])
    TARGET_MODEL_INTEREST = TARGET_IMAGE_FLAT[MASK_INDICES]
    # print(INIT_MODEL_INTEREST.shape)

    # plt.imshow(STARTING_DISK,origin="lower")
    # plt.colorbar()
    # plt.show()
    # sys.exit()
    optimized_model, loss_history = optimize_model(target_image=TARGET_MODEL_INTEREST,
                                                   model_init=INIT_MODEL_INTEREST,
                                                   psf=JAX_PSF, basis_data=fm_dict,
                                                   total_pixels=TOTAL_PIXELS
                                                   )
    optimized_model_image = reconstruct_full_image(optimized_model, TOTAL_PIXELS)
    fits.writeto("freeform_run.fits", np.asarray(optimized_model_image), overwrite=True)
    # --- Visualization ---
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))

    ax[0].imshow(np.asarray(TARGET_IMAGE), cmap='inferno', origin="lower")
    ax[0].set_title("Target Image (Ground Truth)")
    ax[0].axis("off")

    ax[1].imshow(np.array(optimized_model_image), cmap='viridis', origin="lower")
    ax[1].set_title("Optimized Freeform Model")
    ax[1].axis("off")

    ax[2].plot(loss_history)
    ax[2].set_title("Loss Over Time")
    ax[2].set_xlabel("Iteration")
    ax[2].set_ylabel("MSE Loss")
    ax[2].grid()

    plt.show()