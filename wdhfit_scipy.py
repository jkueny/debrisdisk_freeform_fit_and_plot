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

TODO save the klipped images with the masks applied for debug/feedback.

TODO try the L-BFGS optimizer paired with Huber loss

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


import time


# # because this error was coming up
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"


from functools import partial
import numpy as np

import astropy.io.fits as fits


from dev.pyklip.fmlib.funcs_JDFM import update_wind, fm_from_eigen_adi, \
                                        fm_from_eigen_rdi, \
                                        insert_section_into_full_image, \
                                        mass_derotation

from modeling.wdh_parametric import ParametricWDH
from modeling.jax_models.wdh_modeling import gen_multiwdh_image


from utils.klip_basis import load_kl_basis, unpack_basis_data
from utils.io.save_results import save_optimization_outputs


import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jax.scipy import optimize as jsp_opt
# import jax.profiler
import optax
from optax.losses import huber_loss

LAMBDA_REG = 0.1

def huber_loss(residual, delta: float = 1.0):
    """Vectorised Huber loss identical to optax.losses.huber_loss."""
    abs_r = jnp.abs(residual)
    quadratic = 0.5 * residual ** 2
    linear     = delta * (abs_r - 0.5 * delta)
    return jnp.where(abs_r <= delta, quadratic, linear)

def relative_l2(params_all):
    """
    params_all is the parameter dictionary for the WDH models.

    It has a global parameter dict, and individual model paramter
    dicts. 

    Currently (07/16/2025) the only global parameter is the FWHM
    of the Gaussian kernel for model convolution.

    There can be up to 3 WDH models fit at once. If we're only
    fitting for one model, this function won't even get called.

    params_all should have the format:

    {"ps_global":{"fwhm": JAX array}, "ps_indiv":[{WDH1 params},
    {WDH2 params}, {WDH3 params}]}

    The param values are of type jax.lib.xla_client.ArrayImpl.
    
    """
    fwhm = params_all["ps_global"]["fwhm"]
    n_wdh_params = params_all["ps_indiv"]
    

    # Grab the dominant WDH params
    wdh_dom = n_wdh_params[0] #this is a dict
    beta_dom = wdh_dom["beta"]
    a_r_dom = wdh_dom["a_r"]
    sig_dom = wdh_dom["sig"]

    
    penalties = []
    # In this setup we're also computing the differences b/w the dominant
    # model params and themselves, but who cares
    for i in n_wdh_params:
        diff_beta = jnp.square((i["beta"] - beta_dom) / (0.5 * beta_dom))
        diff_a_r = jnp.square((i["a_r"] - a_r_dom) / (0.5 * a_r_dom))
        diff_sig = jnp.square((i["sig"] - sig_dom) / (0.5 * sig_dom))
        penalties.append(diff_beta + diff_a_r + diff_sig)

    # reg_a_r = jnp.square(a_r_dom - 1.0) / jnp.square(2.0) #this params needs more reg...
    # penalties.append(reg_a_r)
    # Now grab the a_r and beta params from the secondary
    # and (if applic) tertiary models
    diff_jax = jnp.array(penalties)


    return jnp.sum(diff_jax) / diff_jax.size


# @partial(jax.jit, static_argnames=["unravel_fn","total_pixels", "isRDI"])
def loss_function(image_params_arr, unravel_fn, x_arr, y_arr, disk_image, aligned_images,
                  ref_psfs_stacked, PAs, ref_inds, wdh_PAs,
                  section_inds_arr, klmodes_stacked, evals, evecs_stacked,
                  total_pixels, isRDI, mask2generatehalo, mask_skip_models,
                  mask2minimize_inds):
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
    mod_params = unravel_fn(image_params_arr)
    # DM commands scaled to [0,1] fits cubes do like 10 secs of wall clock time
    # Spatil freq. such that speckles end up at 10 lamb/D
    # So the wind is the rate of change of the phase 2pi v k thing maybe over D
    isRDI = bool(isRDI)

    # if we're only fitting one model, we out
    reg = jnp.where(len(mod_params["ps_indiv"]) > 1, #if more than one WDH model
                    0., #otherwise, we're good
                    relative_l2(mod_params), #regularize
                    )
    # reg = 0.
    
    # Ensure that the model pixel values range [0,1]
    full_model_images = gen_multiwdh_image(x_arr, y_arr, mod_params, mask2generatehalo)



    # confirmed shape of model_images_prepped (84, 50176)
    # flat_postklip_psfs = fm_jaxed(aligned_images,
    #                        global_models_prepped,
    #                        ref_models_stacked, ref_psfs_stacked,
    #                        klmodes_stacked, evals, evecs_stacked,
    #                        PAs)
    global_models_prepped, ref_models_stacked = update_wind(model_wdhs=full_model_images,
                                                            PAs=wdh_PAs, ref_inds=ref_inds,
                                                            # aligned_center=aligned_center,
                                                            section_inds=section_inds_arr,
                                                            mask_skip_models=mask_skip_models,
                                                            isRDI=isRDI
                                                            )
    if not isRDI:
        
        flat_postklip_psfs = jax.vmap(fm_from_eigen_adi
                            )(aligned_images, ref_psfs_stacked,
                              global_models_prepped,ref_models_stacked,
                              klmodes_stacked, evals, evecs_stacked,
                              )
    elif isRDI:
        flat_postklip_psfs = jax.vmap(fm_from_eigen_rdi
                            )(aligned_images, 
                                global_models_prepped,
                                klmodes_stacked,
                                )

    derotated_postklip_psfs = mass_derotation(flat_postklip_psfs,PAs,
                                              total_pixels,section_inds_arr)

    freeform_fm_full = jnp.mean(derotated_postklip_psfs, axis=0)
    freeform_fm_flat = jnp.reshape(freeform_fm_full,
                                   freeform_fm_full.shape[0]*freeform_fm_full.shape[1])
    freeform_fm_interest = freeform_fm_flat[mask2minimize_inds]

    # disk_image_interest = disk_image[loss_mask]

    # freeform_fm_interest = jnp.where(MASK, freeform_fm_full, jnp.nan)
    # disk_image_interest = jnp.where(MASK, disk_image, jnp.nan)

    # mse = jnp.nanmean((disk_image_interest - freeform_fm_interest) ** 2)
    # mse = jnp.mean((disk_image - freeform_fm_interest) ** 2)
    mse = jnp.mean(huber_loss(freeform_fm_interest, disk_image))
    # mse = jnp.mean(huber_loss(freeform_fm_full, disk_image))

    # jax.debug.print("print(mse) -> {x}", x=jnp.max(disk_image))

    return mse + LAMBDA_REG * reg

# Compute loss and gradients simultaneously
loss_and_grad = jax.value_and_grad(loss_function)

def optimize_model(target_image, params_init, x_arr, y_arr,
                   total_pixels, basis_data, num_steps,
                   mask2generatehalo, mask2minimize_inds,
                   mask_skip_images):
    """
    L-BFGS-B optimisation driven by jax.scipy.optimize.minimize.
    Returns
    -------
    best_params  : pytree    –  parameters in their original structure
    loss_history : list[float]
    result_obj   : OptimizeResult  (full SciPy-style record)
    """
    # ----------- static data & helpers ---------------------------------
    # target_image = jnp.asarray(target_image)
    params0, unravel = ravel_pytree(params_init)

    jax_target_image = jnp.array(target_image)

    bd = unpack_basis_data(basis_data)
    section_inds     = bd["section_inds"][0]
    aligned_image_sections = jnp.take(bd["aligned_images"],
                                      section_inds[-1],
                                      axis=1,
                                      fill_value=0.)
    ref_psfs_sections = jnp.take(bd["ref_psfs"],
                                 section_inds[-1],
                                 axis=2,
                                 fill_value=0.)
    position_angles  = jnp.array(basis_data["klparam_dict"]["PAs"])
    ref_inds         = bd["ref_inds"]
    wdhPAs           = bd["wdhPAs"]
    klmodes_sections = jnp.take(bd["klmodes"],
                                section_inds[-1],
                                axis=2,
                                fill_value=0.)
    evals, evecs     = bd["evals"], bd["evecs"]
    mode             = 1 if bd["klparams"]["isRDI"] else 0

    # ----------- objective closure -------------------------------------
    def objective(p_flat):
        return loss_function(
            p_flat, unravel, x_arr, y_arr,
            jax_target_image, aligned_image_sections, ref_psfs_sections,
            position_angles, ref_inds, wdhPAs,
            section_inds, klmodes_sections, evals, evecs,
            total_pixels, mode, mask2generatehalo,
            mask_skip_images, mask2minimize_inds,
        )

    # ----------- callback for progress & history -----------------------
    loss_history = []


    # ----------- run the optimiser ------------------------------------
    t0 = time.time()
    result = jsp_opt.minimize(
                            objective,          # your scalar loss closure
                            params0,            # 1-D ravelled initial guess
                            method="BFGS",      # <- the only method jax.scipy has today
                            options={"maxiter": num_steps}
                        )
    best_params = unravel(result.x)
    loss_history = None     # jax.scipy doesn't expose per-iter losses
    print(f"Finished in {time.time() - t0:.1f}s - success = {result.success}")

    best_params = unravel(result.x)
    return best_params, loss_history, result



def main(config):
    # Init the wdh model object
    wdh_obj = ParametricWDH(config)

    # Make the intial model
    params_init = wdh_obj.params_init

    # Ex. {'ps_global': {'fwhm': 3.0}, 'ps_indiv': [{wdh1_params}, {wdh2_params}, {wdh3_params}]}
    # Make the masks and initial model
    mask2generatehalo, mask2minimize = wdh_obj.prep_binary_masks()


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
        # wdh_models_init_np = np.asarray(wdh_models_init)

        wdh_obj.initialize_windfm(dataset, wdh_models_init)


        print('First time initializing... \
              check wind_fm_files directory and modify the yaml conf file first_time flag.')
        sys.exit(0)

    params_file = wdh_obj.params_file
    klipdir = wdh_obj.klipdir
    resultsdir = wdh_obj.resultsdir
    file_prefix = wdh_obj.file_prefix
    basis_path = os.path.join(klipdir, f"{file_prefix}_klbasis.h5")
    # Read in the masks
    mask2generatehalo = fits.getdata(os.path.join(klipdir, f"{file_prefix}_mask2generatehalo.fits"))
    mask2minimize = fits.getdata(os.path.join(klipdir, f"{file_prefix}_mask2minimize.fits"))
    # Read in the basis
    fm_dict = load_kl_basis(basis_path)
    image_size = fm_dict["klparam_dict"]["input_img_shape"]
    wdhPAs = fm_dict["wdhPAs_dict"]["PAs"]
    valid_pas_mask = fm_dict["wdhPAs_dict"]["PAmask"]
    reduced_data = fits.getdata(os.path.join(klipdir, f"{file_prefix}-klipped-KLmodes-all.fits"))[0]
    reduced_data[reduced_data != reduced_data] = 0.

    mask2generate_indices = jnp.flatnonzero(jnp.array(mask2generatehalo))[jnp.newaxis, :]
    mask2minimize_indices = jnp.flatnonzero(jnp.array(mask2minimize))#[jnp.newaxis, :]
    total_pixels = np.prod(reduced_data.shape)

    # model_mask_indices = jnp.vstack((mask_indices, mask_indices, mask_indices))
    # plt.imshow(reduced_data * mask2minimize, origin="lower")
    # plt.show()
    init_models = jnp.array(wdh_models_init)
    # For 3 models, shape is e.g. (3, 50176)
    init_models_flat = init_models.reshape(init_models.shape[0],
                                           (init_models.shape[1] * init_models.shape[2]))
    init_models_interest = init_models_flat[:, mask2generate_indices]
    reduced_data_flat = reduced_data.flatten()
    reduced_flat_interest = reduced_data_flat[mask2minimize_indices]

    # print(np.max(reduced_flat_interest))
    # testing_data_masking = insert_section_into_full_image(jnp.array(reduced_flat_interest), reduced_data.shape,
    #                                                       mask2minimize_indices)
    




    opt_models, loss_hist, bestfit_ps = optimize_model(target_image=reduced_flat_interest,
    # opt_models, loss_hist, bestfit_ps = optimize_model(target_image=reduced_data,
                                                   params_init=params_init,
                                                   x_arr=xx, y_arr=yy,
                                                   total_pixels=total_pixels,
                                                   basis_data=fm_dict,
                                                   num_steps=args.iterations,
                                                   mask2generatehalo=mask2generatehalo_jax,
                                                   mask2minimize_inds=mask2minimize_indices,
                                                   mask_skip_images=valid_pas_mask
                                                   )
    
    # print(opt_models)

    optimized_model = jnp.sum(opt_models, axis=0)
    opt_model_np = np.asarray(optimized_model)
    opt_model_fm = wdh_obj.single_fm(np.asarray(opt_models))
    opt_model_fm_np = np.asarray(opt_model_fm)
    residuals_image = reduced_data - opt_model_fm_np
    
    save_optimization_outputs(save_dir=resultsdir,
                              file_prefix=file_prefix,
                              params_dict=bestfit_ps,
                              model_image=opt_model_np,
                              forward_model_image=opt_model_fm_np,
                              residuals_image=residuals_image,
                              loss_value=loss_hist[-1])


    # Convolve the init model

    return True

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='fit a WDH model to KLIP image data.')
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