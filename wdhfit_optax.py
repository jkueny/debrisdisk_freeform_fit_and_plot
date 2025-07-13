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

default_parameter_file = 'HR4796_g_camsci2_20230312_13.yaml'  # name of the parameter file
# default_parameter_file = 'HR4796a_z_lco2023a_magao-x_20230309_10.yaml'  # name of the parameter file
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
from utils.io import yaml_handling, fits_handling
from utils.model_tools import convolve_model


import jax
from jax import lax
import jax.numpy as jnp
# import jax.profiler
from jax.scipy.signal import convolve2d
import optax

@partial(jax.jit, static_argnames=["fixed_refs","aligned_center",
                                   "total_pixels", "isRDI"])
def loss_function(mod_pix_params, disk_image, psf, aligned_images,
                  ref_psfs_stacked, PAs, ref_PAs, fixed_refs, aligned_center,
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
    freeform_fm_interest = freeform_fm_flat[MASK_INDICES]

    # freeform_fm_interest = jnp.where(MASK, freeform_fm_full, jnp.nan)
    # disk_image_interest = jnp.where(MASK, disk_image, jnp.nan)

    # mse = jnp.nanmean((disk_image_interest - freeform_fm_interest) ** 2)
    mse = jnp.mean((disk_image - freeform_fm_interest) ** 2)

    # jax.debug.print("print(mse) -> {x}", x=mse)

    return mse

# --- JIT-Compiled Gradient Computation ---
# loss_and_grad = jax.jit(jax.value_and_grad(loss_function))
loss_and_grad = jax.value_and_grad(loss_function)

def optimize_model(target_image, model_init,
                   psf, basis_data, total_pixels, num_steps, lr=0.1):
    
    # dimension = img_dim
    jax_target_image = jnp.array(target_image)

    # Initialize the initial image
    image_params = model_init
    # image_params = initialize_freeform_model_reduced()

    # Set up optimizer, use adaptive stochastic grad descent
    optimizer = optax.adam(lr)
    opt_state =  optimizer.init(image_params)

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
    wdh_PAs = basis_data_unpacked["wdhPAs"]
    valid_PA_mask = basis_data_unpacked["PAmask"]

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
                                    aligned_center, section_inds,
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



def main(config):
    import matplotlib.pyplot as plt

    # Init the wdh model object
    wdh_obj = ParametricWDH(config)

    # Make the intial model
    params_init = wdh_obj.params_init
    # Ex. {'ps_global': {'fwhm': 3.0}, 'ps_indiv': [{wdh1_params}, {wdh2_params}, {wdh3_params}]}
    # Make the masks and initial model
    mask2generatehalo, mask2minimize = wdh_obj.prep_binary_masks()

    print(mask2generatehalo)

    mask2generatehalo_jax = jnp.array(mask2generatehalo)

    # Calculate a res elem. for the FOV coordinates
    diff_lim = ((wdh_obj.wl * 1e-6) / wdh_obj.params_file["METADATA"]["PRIM_MIRR_SZ"])
    res_el = diff_lim * 180 / np.pi * 3600
    max_fov = (wdh_obj.image_size // 2) * wdh_obj.pixscale
    fov_size = max_fov / res_el
    n_pts = int(np.ceil(wdh_obj.image_size))
    x = jnp.linspace(-fov_size, fov_size, num=n_pts)
    y = jnp.linspace(-fov_size, fov_size, num=n_pts)

    xx, yy = jnp.meshgrid(x, y)
    wdh_models_init = gen_multiwdh_image(xx, yy, params_init, mask2generatehalo_jax)


    if wdh_obj.params_file["FIRST_TIME"]:
        # Prep the dataset
        dataset = wdh_obj.prep_dataset()
        # print(pa_mask)
        
        # wdh_model_test_toplot = np.asarray(np.log10(np.sum(wdh_model_here, axis=0)+1e-6))
        # print(wdh_model_test_toplot.shape)
        # plt.imshow(wdh_model_test_toplot, cmap="jet", origin="lower")
        # plt.show()
        wdh_models_init_np = np.asarray(wdh_models_init)

        wdh_obj.initialize_windfm(dataset, wdh_models_init)
        print('First time initializing... \
              check wind_fm_files directory and modify the yaml conf file first_time flag.')
        sys.exit(0)

    params_file = wdh_obj.params_file
    klipdir = wdh_obj.klipdir
    file_prefix = wdh_obj.file_prefix
    basis_path = os.path.join(klipdir, f"{file_prefix}_klbasis.h5")
    # Read in the masks
    mask2generatehalo = fits.getdata(os.path.join(klipdir, f"{file_prefix}_mask2generatehalo.fits"))
    mask2minimize = fits.getdata(os.path.join(klipdir, f"{file_prefix}_mask2minimize.fits"))
    # Read in the basis
    fm_dict = load_kl_basis(basis_path)
    image_size = fm_dict["klparam_dict"]["input_img_shape"]
    wdh_pas = fm_dict["wdhPAs_dict"]["PAs"]
    valid_pas_mask = fm_dict["wdhPAs_dict"]["PAmask"]
    reduced_data = fits.getdata(os.path.join(klipdir, f"{file_prefix}-klipped-KLmodes-all.fits"))
    reduced_data[reduced_data != reduced_data] = 0.

    mask_indices = jnp.flatnonzero(jnp.array(mask2generatehalo))

    # model_mask_indices = jnp.vstack((mask_indices, mask_indices, mask_indices))

    init_models = jnp.array(wdh_models_init)
    # For 3 models, shape is e.g. (3, 50176)
    init_models_flat = init_models.reshape(init_models.shape[0],
                                           (init_models.shape[1] * init_models.shape[2]))
    init_models_interest = init_models_flat[:, mask_indices]
    reduced_data_flat = reduced_data.reshape(reduced_data.shape[0] * reduced_data.shape[1])
    reduced_flat_interest = reduced_data_flat[mask_indices]





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