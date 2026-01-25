import os, sys
import argparse
import time
import multiprocessing
from functools import partial
import numpy as np
import astropy.io.fits as fits
from scipy.signal import fftconvolve

import jax
import jax.numpy as jnp
from jax import lax
import optax
from optax.losses import huber_loss

from ffortissimo.dev.pyklip.fmlib.funcs_JDFM import (
    update_disk, fm_from_eigen_adi, fm_from_eigen_rdi, derotate_and_average)
from ffortissimo.utils.klip_basis import unpack_basis_data
from ffortissimo.utils.improc_tools import reconstruct_full_image, fft_power_spectrum, \
    subtract_radial_profile, high_pass_filter
from ffortissimo.utils.diskfit_tools import convolve_model, \
    penalize_spatial_freq
from ffortissimo.modeling.visualization import plot_training

# Optimization functions (extracted from diskfit_freeform.py)
def fm_scan_func_adi(_, input_pt, full_sample_refs, full_sample_models):
    flat_model_here = input_pt["models"]
    aligned_image = input_pt["images"]
    klmodes = input_pt["modes"]
    evals = input_pt["evals"]
    evecs = input_pt["evecs"]
    reference_images_selector_vec = input_pt["reference_images_selector_vec"]

    flat_postklip_psf_i = fm_from_eigen_adi(
        aligned_image,
        flat_model_here,
        klmodes,
        evals,
        evecs,
        reference_images_selector_vec,
        full_sample_refs,
        full_sample_models,
    )
    
    return _, jnp.array(flat_postklip_psf_i)

def fm_scan_func_rdi(_, input_pt):
    flat_model_here = input_pt["models"]
    aligned_image = input_pt["images"]
    klmodes = input_pt["modes"]

    flat_postklip_psf_i = fm_from_eigen_rdi(
        aligned_image,
        flat_model_here,
        klmodes,
    )
    
    return _, jnp.array(flat_postklip_psf_i)

def asym_weights(res, tau, alpha):
    # r = residuals; weight ~alpha for r<0, ~1 for r>0 (smooth)
    s = jnp.tanh(-res / tau)
    return alpha + (1.0 - alpha) * 0.5*(1.0 + s)

def loss_function(mod_pix_params, disk_image, psf, noise_map, ref_model_psd,
                  aligned_images, PAs, disk_mask_inds, opt_mask_inds, iowa_sec_inds_arr, 
                  klmodes_stacked, radial_inds, aligned_center,
                  isRDI, do_radial_profile_sub, do_clean_final_fm,
                  total_pixels, reg_lambda, 
                  evals=None, evecs_stacked=None,
                  delta=1, hp_filtersize=None,
                  all_reference_images_selectors=None):
    """ measure the huber loss for a given disk freeform disk model."""
    # pos_mod_pix_params = jnp.abs(mod_pix_params)
    pos_mod_pix_params = mod_pix_params
    full_model_image = reconstruct_full_image(pos_mod_pix_params, total_pixels, disk_mask_inds)
    full_noise_image = reconstruct_full_image(noise_map, total_pixels, opt_mask_inds)
    full_model_norm = full_model_image / jnp.linalg.norm(full_model_image)
    full_model_norm_meansub = full_model_norm - jnp.mean(full_model_norm)

    # Compute the model power spectrum and use it to regularize high spatial freq.
    model_psd = fft_power_spectrum(full_model_norm_meansub)
    hsf_penalty = penalize_spatial_freq(model_psd, ref_model_psd, reg_lambda=reg_lambda)

    freeform_image = convolve_model(full_model_image, psf)
    if bool(do_radial_profile_sub):
        freeform_image_profilesub, _ = subtract_radial_profile(freeform_image,
                                                            aligned_center,
                                                            full_noise_image,
                                                            radial_inds,
                                                            )
    else:
        freeform_image_profilesub = freeform_image

    if bool(hp_filtersize):
        freeform_image_profilesub_hp = high_pass_filter(freeform_image_profilesub, filtersize=hp_filtersize)
    else:
        freeform_image_profilesub_hp = freeform_image_profilesub    
    global_models_prepped = update_disk(model_disk=freeform_image_profilesub_hp,
                                        PAs=PAs,
                                        section_inds=iowa_sec_inds_arr,
                                        )
    if bool(isRDI):
        fm_calc_inputs = {
            "models": global_models_prepped,
            "images": aligned_images,
            "modes": klmodes_stacked,
         
        }
        _, flat_postklip_psfs = lax.scan(fm_scan_func_rdi, None, fm_calc_inputs)
    else:
        fm_calc_inputs = {
            "models": global_models_prepped,
            "images": aligned_images,
            "modes": klmodes_stacked,
            "evals": evals,
            "evecs": evecs_stacked,
            "reference_images_selector_vec": all_reference_images_selectors,
        }
        scan_func = partial(
            fm_scan_func_adi,
            full_sample_refs=aligned_images,
            full_sample_models=global_models_prepped
        )
        _, flat_postklip_psfs = lax.scan(scan_func, None, fm_calc_inputs)

    # Reshape the postKLIP PSFs into 2D images and derotate them
    freeform_fm_full = derotate_and_average(flat_postklip_psfs, PAs, total_pixels, iowa_sec_inds_arr)
    if bool(do_clean_final_fm):
        freeform_fm_full_rprofsub, _ = subtract_radial_profile(freeform_fm_full,
                                                           aligned_center,
                                                           full_noise_image,
                                                           radial_inds,
                                                           )
    else:
        freeform_fm_full_rprofsub = freeform_fm_full

    freeform_fm_flat = jnp.reshape(freeform_fm_full_rprofsub, psf.shape[0] * psf.shape[1])

    # Grab just the disk ROI pixels
    freeform_fm_interest = freeform_fm_flat[opt_mask_inds]

    residuals = (freeform_fm_interest - disk_image)
    weights_nominal = 1 / noise_map
    weights_asym = asym_weights(residuals, tau=noise_map, alpha=1.0)
    raw_loss = residuals**2 * weights_nominal
    mean_huber = jnp.mean(huber_loss(raw_loss, delta=delta))

    loss = mean_huber + hsf_penalty
    aux_data = freeform_fm_full, full_model_image, (weights_asym * weights_nominal), weights_nominal
    return loss, aux_data

loss_and_grad = jax.value_and_grad(loss_function, has_aux=True)

def optimize_model(
    target_image, model_init, ref_psd, noise_map, disk_mask_indices, opt_mask_indices,
    psf, basis_data, total_pixels, num_steps, reg_lambda, run_dir, reduced_data, learning_rate,
    radial_inds, delta,
    aligned_center, do_radial_profile_sub, do_clean_final_fm,
    hp_filtersize=None,
):
    target_image = jnp.array(target_image).astype(jnp.float32)
    noise_map = jnp.array(noise_map).astype(jnp.float32)

    basis_data_unpacked = unpack_basis_data(basis_data)
    aligned_image_sections = jnp.array(basis_data_unpacked["aligned_images"])
    n_images = aligned_image_sections.shape[0]
    iowa_sec_inds = basis_data_unpacked["section_inds"][0]
    klmodes_sections = basis_data_unpacked["klmodes"]
    ref_psfs_inds = basis_data_unpacked["ref_inds"]

    # Initialize the initial image
    image_params = model_init.astype(jnp.float32)

    # Set up optimizer, use adaptive stochastic grad descent (Adam)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(image_params)

    loss_history = []

    PAs = jnp.array((basis_data["klparam_dict"]["PAs"]))
    if bool(basis_data_unpacked["klparams"]["isRDI"]):
        all_reference_images_selectors = None
        max_num_refs = klmodes_sections.shape[1]
        mode = 1
        hp_filtersize = hp_filtersize
        evals = None
        evecs = None
    elif not bool(basis_data_unpacked["klparams"]["isRDI"]):
        evals = basis_data_unpacked["evals"]
        n_modes = evals.shape[1]
        subset_evecs = basis_data_unpacked["evecs"]
        evecs = np.zeros((n_images, n_images, n_modes))
        for idx, img_ref_indices in enumerate(ref_psfs_inds):
            evecs[idx, img_ref_indices] = subset_evecs[idx]
        all_reference_images_selectors = np.zeros((aligned_image_sections.shape[0], aligned_image_sections.shape[0]), dtype=bool)
        for i in range(aligned_image_sections.shape[0]):
            for j in ref_psfs_inds[i]:
                all_reference_images_selectors[i, j] = True
        all_reference_images_selectors = jax.device_put(all_reference_images_selectors.astype(float))
        max_num_refs = basis_data_unpacked["fixed_refs"]
        mode = 0

    run_start_ts = time.time()
    total_pixels = int(total_pixels)

    @jax.jit
    def step(image_params, opt_state, reg_lambda_here):
        (loss, aux_data), grads = loss_and_grad(
            image_params, target_image, psf, noise_map, ref_psd,
            aligned_image_sections, PAs, disk_mask_indices, opt_mask_indices, iowa_sec_inds,
            klmodes_sections,
            radial_inds, aligned_center,
            mode, do_radial_profile_sub, do_clean_final_fm,
            total_pixels, reg_lambda_here,
            evals, evecs,
            delta, hp_filtersize,
            all_reference_images_selectors,
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        image_params = optax.apply_updates(image_params, updates)
        return image_params, opt_state, loss, updates, aux_data

    first_step = time.time()
    measure_warmup = True
    plot_idx = 0
    for step_idx in range(num_steps):
        with jax.profiler.StepTraceAnnotation("train", step_num=step_idx):
            image_params, opt_state, loss, updates, aux_data = step(image_params, opt_state, reg_lambda)
        freeform_fm_full, full_model_image, weights_asym, weights_nominal = aux_data
        loss_history.append(loss.item())

        if measure_warmup:
            first_step = time.time() - first_step
            dt = time.time() - run_start_ts
            measure_warmup = False
            print(f"Step {step_idx}/{num_steps} - Loss: {loss:.6f} - {dt:.6f} sec elapsed - ? sec / step")
        else:
            dt = time.time() - run_start_ts - first_step
            print(f"Step {step_idx}/{num_steps} - Loss: {loss:.6f} - {dt:.6f} sec elapsed - {dt / (step_idx+1):.6f} sec / step")
        if not bool(basis_data_unpacked["klparams"]["isRDI"]):
            if step_idx % 10 == 0:
                out_filename = f"{run_dir}/training_{plot_idx:05}.png"
                plot_training(
                    out_filename,
                    reduced_data,
                    freeform_fm_full,
                    full_model_image,
                    reconstruct_full_image(updates, total_pixels, disk_mask_indices),
                    opt_mask_indices
                )
                plot_idx += 1
    
    print(f"This run took {(time.time() - run_start_ts):.6f} seconds.")
    plot_training(f"{run_dir}/training_final.png", reduced_data, freeform_fm_full, full_model_image, np.zeros_like(reduced_data), opt_mask_indices)

    # optimized_model = jnp.abs(image_params)
    optimized_model = image_params
    return optimized_model, loss_history, weights_asym, weights_nominal