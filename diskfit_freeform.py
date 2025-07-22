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
# from scipy.signal import convolve
# from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

import yaml


from modeling.disk_freeform import FreeFormDisk

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



# jax.config.update("jax_enable_x64", True)

# update_disk_jit = jax.jit(update_disk, static_argnames=["min_num_models",])
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


# @jax.jit
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

# @jax.jit
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
                                   "total_pixels", "isRDI"])
def loss_function(mod_pix_params, disk_image, psf, aligned_images,
                  ref_psfs_stacked, PAs, ref_PAs, fixed_refs, mask_indices,
                  section_inds_arr, klmodes_stacked, evals, evecs_stacked,
                  total_pixels, isRDI):
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
    isRDI = bool(isRDI)


    
    # Ensure that the model pixel values range [0,1]
    full_model_image = reconstruct_full_image(mod_pix_params, total_pixels)
    # freeform_model = jax.nn.sigmoid(mod_pix_params)
    # freeform_model = jax.nn.sigmoid(full_model_image)

    freeform_image = convolve_model(full_model_image, psf)
    # freeform_image = convolve_model_lax(full_model_image, psf)
    # freeform_image = full_model_image


    # confirmed shape of model_images_prepped (84, 50176)
    # flat_postklip_psfs = fm_jaxed(aligned_images,
    #                        global_models_prepped,
    #                        ref_models_stacked, ref_psfs_stacked,
    #                        klmodes_stacked, evals, evecs_stacked,
    #                        PAs)
    if not isRDI:
        global_models_prepped, ref_models_stacked = update_disk(model_disk=freeform_image,
                                                                PAs=PAs, ref_PAs=ref_PAs,
                                                                # aligned_center=aligned_center,
                                                                section_inds=section_inds_arr,
                                                                min_num_models=fixed_refs,
                                                                isRDI=isRDI
                                                                )
        
        flat_postklip_psfs = jax.vmap(fm_from_eigen_adi
                            )(aligned_images, ref_psfs_stacked,
                                global_models_prepped,ref_models_stacked,
                                klmodes_stacked, evals, evecs_stacked,
                                )
    elif isRDI:
        global_models_prepped = update_disk(model_disk=freeform_image,
                                            PAs=PAs, ref_PAs=ref_PAs,
                                            # aligned_center=aligned_center,
                                            section_inds=section_inds_arr,
                                            min_num_models=fixed_refs,
                                            isRDI=isRDI
                                            )
        flat_postklip_psfs = jax.vmap(fm_from_eigen_rdi
                            )(aligned_images, 
                                global_models_prepped,
                                klmodes_stacked,
                                )

    derotated_postklip_psfs = mass_derotation(flat_postklip_psfs,PAs,
                                              total_pixels,section_inds_arr)

    freeform_fm_full = jnp.mean(derotated_postklip_psfs, axis=0)
    freeform_fm_flat = jnp.reshape(freeform_fm_full, psf.shape[0] * psf.shape[1])
    freeform_fm_interest = freeform_fm_flat[mask_indices]

    # freeform_fm_interest = jnp.where(MASK, freeform_fm_full, jnp.nan)
    # disk_image_interest = jnp.where(MASK, disk_image, jnp.nan)

    # mse = jnp.nanmean((disk_image_interest - freeform_fm_interest) ** 2)
    mse = jnp.mean((disk_image - freeform_fm_interest) ** 2)

    # jax.debug.print("print(mse) -> {x}", x=mse)

    return mse



# --- JIT-Compiled Gradient Computation ---
# loss_and_grad = jax.jit(jax.value_and_grad(loss_function))
loss_and_grad = jax.value_and_grad(loss_function)

def optimize_model(target_image, model_init, mask_indices,
                   psf, basis_data, total_pixels, num_steps, lr=0.1):
    
    # dimension = img_dim
    jax_target_image = jnp.array(target_image)

    # Initialize the initial image
    image_params = model_init
    # image_params = initialize_freeform_model_reduced()

    # Set up optimizer, use adaptive stochastic grad descent (Adam)
    optimizer = optax.adam(lr)
    # initialize the internal state to track 1st and 2nd moments of the gradients
    opt_state =  optimizer.init(image_params) #this is all zeros initially

    loss_history = []

    basis_data_unpacked = unpack_basis_data(basis_data)

    # num_input_images = int(jax.device_get(basis_data["klparam_dict"]["nfiles"]))
    aligned_image_data = jnp.array(basis_data_unpacked["aligned_images"]) #shape ex. (84, 50176)
    section_inds = basis_data_unpacked["section_inds"][0] #shape ex. (1, 39112)
    aligned_image_sections = jnp.take(aligned_image_data, section_inds[-1], axis=1, fill_value=0.)

    klmodes = basis_data_unpacked["klmodes"] #shape (N_images, N_KLmodes, N_pixels) ex. (84, 2, 50176)
    klmodes_sections = jnp.take(klmodes, section_inds[-1], axis=2, fill_value=0.)

    evals = basis_data_unpacked["evals"] # shape (N_images, N_modes)
    # the eigenvectors have been zero-padded at the ends to removed ragged-ness....
    evecs = basis_data_unpacked["evecs"] # shape (N_images, max_N_refs, N_modes) ex. (84, 78, 2)
    # evecs have been unpacked, stacked, and ready to be BATCHED!
    # input_img_nums = basis_data_unpacked["input_img_nums"]
    # These are the images used for the basis for every image in the dataset.
    ref_psfs = basis_data_unpacked["ref_psfs"] # zero-padded at the end to all have the same shape
    ref_psfs_sections = jnp.take(ref_psfs, section_inds[-1], axis=2, fill_value=0.)

    ref_PAs = basis_data_unpacked["ref_PAs"]

    if bool(basis_data_unpacked["klparams"]["isRDI"]):
        fixed_refs = klmodes.shape[1]
        mode = 1
    elif not bool(basis_data_unpacked["klparams"]["isRDI"]):
        fixed_refs = basis_data_unpacked["fixed_refs"] #this is just a number
        mode = 0
    # ref_psfs shape (N_images, max_N_refs, N_pixels) ex. (84, 78, 50176)
    # ref_psfs have been unpacked, stacked, and ready to be BATCHED!
    # position_angles = tuple(np.asarray(jax.device_get(basis_data["klparam_dict"]["PAs"])))
    position_angles = jnp.array((basis_data["klparam_dict"]["PAs"]))
    aligned_center = tuple(np.asarray(jax.device_get([basis_data["klparam_dict"]["aligned_center_x"],
                                basis_data["klparam_dict"]["aligned_center_y"]])))
    # ref_psfs_indicies = basis_data_unpacked["ref_psfs_indicies"]

    del aligned_image_data
    del klmodes
    del ref_psfs

    time_now = time.time()
    # jax.profiler.start_trace("/tmp/tensorboard")
    # jax.config.update("jax_debug_nans", True)

    @jax.jit
    def step(image_params, opt_state):
        loss, grads = loss_and_grad(image_params, jax_target_image, psf,
                                    aligned_image_sections, ref_psfs_sections,
                                    position_angles, ref_PAs, fixed_refs,
                                    mask_indices, section_inds,
                                    klmodes_sections, evals, evecs, total_pixels, mode)
        updates, opt_state = optimizer.update(grads, opt_state)
        image_params = optax.apply_updates(image_params, updates)
        return image_params, opt_state, loss
    
    for step_idx in range(num_steps):
        image_params, opt_state, loss = step(image_params, opt_state)
        loss_history.append(loss.item())

        if step_idx % round(num_steps / 100) == 0:
            print(f"Step {step_idx}/{num_steps} - Loss: {loss:.6f}")

    # jax.profiler.stop_trace()
    print(f"This run took {(time.time() - time_now):.6f} seconds.")
    
    # optimized_model = jax.nn.sigmoid(image_params)
    optimized_model = image_params
    return optimized_model, loss_history

def main(config, num_iterations, init_model):
    # Grab the info from the yaml file

    # if MODE.upper() == "ADI":
    #     mode = 0
    # elif MODE.upper() == "RDI":
    #     mode = 1
    ffd_obj = FreeFormDisk(config)

    klipdir = ffd_obj.klipdir
    file_prefix = ffd_obj.file_prefix
    aligned_center = ffd_obj.aligned_center
    mode = ffd_obj.mode

    basis_path = os.path.join(klipdir, f"{file_prefix}_klbasis.h5")
    mask2generatedisk = ffd_obj.prep_binary_masks()

    # Get the initial model

    model_firstguess = ffd_obj.get_initial_model(init_model)

    # load PSF
    psf = fits.getdata(os.path.join(klipdir, file_prefix + '_instrPSF.fits'))
    psf = ffd_obj.psf
    jax_psf = jnp.array(psf)
    # jax_psf /= jnp.sum(jax_psf)

    # measure the size of images DIMENSION and make it global



    if ffd_obj.params_file["FIRST_TIME"]:
        # initialize_diskfm and make diskobj global
        dataset = ffd_obj.prep_dataset()
        

        ffd_obj.initialize_diskfm(dataset, model_init=model_firstguess)

        print('First time initializing, check klip_fm_files directory and modify the yaml file first_time flag.')
        sys.exit(0)
    # Read in the basis data
    fm_dict = load_kl_basis(basis_path)
    # fm_dict contains 
    # dict_keys(['aligned_images_dict', 'evals_dict', 'evecs_dict',
    # 'input_img_num_dict', 'klmodes_dict', 'section_ind_dict'])
    reduced_data = fits.getdata(os.path.join(klipdir, f"{file_prefix}-klipped-KLmodes-all.fits"))[0]
    reduced_data[reduced_data != reduced_data] = 0.

    mask2generate_indices = jnp.flatnonzero(jnp.array(mask2generatedisk))[jnp.newaxis, :]
    
    total_pixels = np.prod(reduced_data.shape)

    # model_mask_indices = jnp.vstack((mask_indices, mask_indices, mask_indices))
    # plt.imshow(reduced_data * mask2minimize, origin="lower")
    # plt.show()
    # For 3 models, shape is e.g. (3, 50176)

    reduced_data_flat = reduced_data.flatten()
    reduced_flat_interest = reduced_data_flat[mask2generate_indices]

    disk_mask = np.array(mask2generatedisk)  # convert to JAX array if needed
    annular_mask = make_annular_mask(disk_mask.shape, 10, 112)

    # STARTING_DISK = fits.getdata("freeform_run.fits") #start from the last run
    model_firstguess *= mask2generatedisk
    init_model = jnp.array(model_firstguess)
    init_model_flat = init_model.reshape(init_model.shape[0] * init_model.shape[1])
    init_model_interest = init_model_flat[mask2generate_indices]

    # print(INIT_MODEL_INTEREST.shape)

    # plt.imshow(STARTING_DISK,origin="lower")
    # plt.colorbar()
    # plt.show()
    # sys.exit()
    optimized_model, loss_history = optimize_model(target_image=reduced_flat_interest,
                                                   model_init=init_model_interest,
                                                   psf=jax_psf, basis_data=fm_dict,
                                                   total_pixels=total_pixels,
                                                   num_steps=num_iterations,
                                                   )
    optimized_model_image = reconstruct_full_image(optimized_model, total_pixels)
    os.makedirs(save_to_dir, exist_ok=True)
    fits.writeto(f"{save_to_dir}/freeform_run.fits", np.asarray(optimized_model_image), overwrite=True)
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

    # print("Read " + str_yaml + " parameter file")
    # # open the parameter file
    # yaml_path_file = os.path.join(os.getcwd(), str_yaml)
    # with open(yaml_path_file, 'r') as yaml_file:
    #     yaml_cfg = yaml.safe_load(yaml_file)

    main(config=str_yaml,
         num_iterations=args.iterations,
         init_model=args.initial_model)
    