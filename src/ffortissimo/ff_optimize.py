'''
Run JAX-based optimization for freeform disk fitting.

This script handles the optimization phase of the freeform disk fitting pipeline:
1. Verifies that ff_klip.py and ff_setup.py were run successfully
2. Optionally performs a single forward-modeling dry run for inspection
3. Executes JAX-based model optimization
4. Saves optimization results to disk

Usage:
    python ff_optimize.py -p initialization_files/config.yaml -i 1000
    python ff_optimize.py -p initialization_files/config.yaml --dry-run
    python ff_optimize.py -p initialization_files/config.yaml -i 1000 --initial-model path/to/model.fits
'''

import os
import sys
import argparse
import multiprocessing
import time
from functools import partial

import numpy as np
import astropy.io.fits as fits
from scipy.signal import fftconvolve
from matplotlib import cm

import jax
import jax.numpy as jnp
from jax import lax
import optax
from optax.losses import huber_loss

from ffortissimo.modeling.disk_freeform import FreeFormDisk
from ffortissimo.utils.io.save_results import save_ffdfit_outputs, get_next_run_dir
from ffortissimo.dev.pyklip.fmlib.funcs_JDFM import (
    update_disk, fm_from_eigen_adi, fm_from_eigen_rdi, derotate_and_average)
from ffortissimo.utils.klip_basis import load_kl_basis, unpack_basis_data
from ffortissimo.utils.masks import make_annular_mask
from ffortissimo.utils.improc_tools import reconstruct_full_image, fft_power_spectrum, \
    subtract_radial_profile, get_radial_inds, high_pass_filter
from ffortissimo.utils.diskfit_tools import convolve_model, record_pyklip_params, \
    penalize_spatial_freq
from ffortissimo.utils.io.fits_handling import save_fits

magma_g = cm.magma.copy()
magma_g.set_bad('0.5')
viridis_g = cm.viridis.copy()
viridis_g.set_bad('0.5')


def verify_prerequisites(ffd_obj):
    """
    Verify that ff_klip.py and ff_setup.py were run successfully.
    
    Args:
        ffd_obj: FreeFormDisk object
        
    Raises:
        FileNotFoundError: If required files are missing
    """
    klipdir = ffd_obj.klipdir
    file_prefix = ffd_obj.file_prefix
    
    # Files from ff_klip.py
    reduced_data_path = os.path.join(klipdir, f"{file_prefix}-klipped-KLmodes-all.fits")
    basis_path = os.path.join(klipdir, f"{file_prefix}_klbasis.h5")
    
    # Files from ff_setup.py
    mask_files = {
        'mask2generatedisk': os.path.join(klipdir, f"{file_prefix}_mask2generatedisk.fits"),
        'mask4noisemap': os.path.join(klipdir, f"{file_prefix}_mask4noisemap.fits"),
        'optimization_mask': os.path.join(klipdir, f"{file_prefix}_optimization_mask.fits"),
        'mask_out_of_bounds': os.path.join(klipdir, f"{file_prefix}_mask_out_of_bounds.fits"),
    }
    noise_map_path = os.path.join(klipdir, f"{file_prefix}_noisemap.fits")
    
    missing_files = []
    
    # Check KLIP files
    if not os.path.exists(reduced_data_path):
        missing_files.append(('KLIP-reduced image', reduced_data_path))
    if not os.path.exists(basis_path):
        missing_files.append(('KL basis file', basis_path))
    
    # Check mask files
    for mask_name, mask_path in mask_files.items():
        if not os.path.exists(mask_path):
            missing_files.append((f'Mask: {mask_name}', mask_path))
    
    # Check noise map
    if not os.path.exists(noise_map_path):
        missing_files.append(('Noise map', noise_map_path))
    
    if missing_files:
        error_msg = (
            f"Error: Prerequisites from ff_klip.py and/or ff_setup.py are missing.\n"
            f"Missing files:\n"
        )
        for file_type, file_path in missing_files:
            error_msg += f"  - {file_type}: {file_path}\n"
        error_msg += "\nPlease run ff_klip.py first, then ff_setup.py before running optimization."
        raise FileNotFoundError(error_msg)
    
    print("✓ Verified prerequisites:")
    print(f"  - KLIP-reduced image: {reduced_data_path}")
    print(f"  - KL basis file: {basis_path}")
    print(f"  - Masks: all present")
    print(f"  - Noise map: {noise_map_path}")


def perform_forward_modeling_dry_run(ffd_obj, init_model_path=None):
    """
    Perform a single forward-modeling dry run for inspection.
    
    Args:
        ffd_obj: FreeFormDisk object
        init_model_path (str, optional): Path to initial model FITS file
    """
    klipdir = ffd_obj.klipdir
    file_prefix = ffd_obj.file_prefix
    
    print("\n[Forward Modeling Dry Run]")
    print("Performing single forward-modeling computation for inspection...")
    
    # Load mask from disk (created by ff_setup.py)
    mask2generatedisk_path = os.path.join(klipdir, f"{file_prefix}_mask2generatedisk.fits")
    if not os.path.exists(mask2generatedisk_path):
        raise FileNotFoundError(
            f"Mask file not found: {mask2generatedisk_path}\n"
            "Please run ff_setup.py first to create the masks."
        )
    ffd_obj.mask2generatedisk = fits.getdata(mask2generatedisk_path)
    
    # Allocate dataset (needed for get_initial_model)
    ffd_obj.allocate_dataset()
    
    # Get initial model
    print("   Creating/loading initial model...")
    model_init = ffd_obj.get_initial_model(init_model_path)
    
    # Convolve initial model with PSF
    print("   Convolving initial model with PSF...")
    model_convolved = fftconvolve(model_init, ffd_obj.psf, mode="same")
    
    # Run forward modeling
    print("   Running forward modeling (this may take a while)...")
    model_fm_init = ffd_obj.single_fm(np.asarray(model_convolved))
    
    # Save forward model
    model_fm_saveto = os.path.join(klipdir, f"{file_prefix}_DryRun_FM.fits")
    save_fits(model_fm_saveto, model_fm_init)
    print(f"   ✓ Saved forward model: {model_fm_saveto}")
    
    print("\n✓ Forward modeling dry run complete!")
    print("You can now inspect the forward model before running the full optimization.")


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


def plot_training(out_filename, reduced_data, freeform_fm_full, full_model_image, updates, mask_indices, min_percent=1.0, max_percent=99.9):
    print('Saving', out_filename, '...', end=' ')
    import matplotlib.pyplot as plt
    from astropy.visualization import simple_norm
    fig, axs = plt.subplots(ncols=4, figsize=(10, 3))
    fig.subplots_adjust(left=0.05, right=0.95)
    mask_tmp = np.zeros(reduced_data.size)
    mask_tmp[mask_indices] = 1.0
    mask_good = (mask_tmp == 1.0).reshape(reduced_data.shape)
    mask_bad = (mask_tmp == 0.0).reshape(reduced_data.shape)
    data_vmax = np.percentile(reduced_data[mask_good], max_percent)
    data_vmin = -data_vmax
    model_vmin, model_vmax = 0, np.percentile(full_model_image[mask_good], max_percent)
    data_space_norm = simple_norm(reduced_data, 'linear', vmin=data_vmin, vmax=data_vmax)
    reduced_data_masked = np.array(reduced_data)
    reduced_data_masked[mask_bad] = np.nan
    plt.colorbar(axs[0].imshow(reduced_data_masked, origin='lower', norm=data_space_norm, cmap=viridis_g))
    axs[0].set(title='Reduced data')
    axs[0].axis('off')
    freeform_fm_full_masked = np.array(freeform_fm_full)
    freeform_fm_full_masked[mask_bad] = np.nan
    plt.colorbar(axs[1].imshow(freeform_fm_full_masked, origin='lower', norm=data_space_norm, cmap=viridis_g))
    axs[1].axis('off')
    full_model_image_masked = np.array(full_model_image)
    full_model_image_masked[mask_bad] = np.nan
    axs[1].set(title='Freeform FM')
    axs[2].axis('off')
    plt.colorbar(axs[2].imshow(full_model_image_masked, origin='lower', norm=simple_norm(full_model_image, 'linear', vmin=model_vmin, vmax=model_vmax), cmap=magma_g))
    axs[2].set(title=r'Model')
    axs[3].axis('off')
    updates_vmax = np.max(np.abs(updates))
    plt.colorbar(axs[3].imshow(updates, origin='lower', vmax=updates_vmax, vmin=-updates_vmax, cmap='RdYlBu_r'))
    axs[3].set(title=r'Updates')
    axs[3].axis('off')
    fig.savefig(out_filename, dpi=128)
    plt.close(fig)
    print('Done.')

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
    pos_mod_pix_params = jnp.abs(mod_pix_params)
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

    optimized_model = jnp.abs(image_params)
    return optimized_model, loss_history, weights_asym, weights_nominal

def harness_optimized_model(optimized_params, ffd_obj, reduced_data, psf, total_pixels, disk_mask_indices, opt_mask_indices,
                            disk_mask, optimization_mask, aligned_center, radial_inds, do_radial_profile_sub, do_clean_final_fm, hp_filtersize,
                            noise_interest, weights_asym, weights_nominal):
    outputs_dict = {}
    print("Reconstructing full image...")
    optimized_model = np.asarray(reconstruct_full_image(optimized_params, total_pixels, disk_mask_indices))

    print("Convolving optimized model image...")
    opt_image_no_rprofsub = fftconvolve(optimized_model, psf, mode="same")

    noise_reconstructed = reconstruct_full_image(jnp.array(noise_interest),
                                                 total_pixels,
                                                 opt_mask_indices,
                                                )
    w_reconstructed = reconstruct_full_image(jnp.array(weights_nominal),
                                             total_pixels,
                                             opt_mask_indices,
                                            )
    asymw_reconstructed = reconstruct_full_image(jnp.array(weights_asym),
                                                 total_pixels,
                                                 opt_mask_indices,
                                                )
    if bool(do_radial_profile_sub):
        opt_model_image, med_prof_image = subtract_radial_profile(opt_image_no_rprofsub,
                                                              aligned_center,
                                                                noise_reconstructed,
                                                                radial_inds,
                                                                )
    else:
        _, med_prof_image = subtract_radial_profile(opt_image_no_rprofsub,
                                                    aligned_center,
                                                    noise_reconstructed,
                                                    radial_inds,
                                                    )
        opt_model_image = opt_image_no_rprofsub
    if bool(hp_filtersize):
        opt_model_image_hp = high_pass_filter(opt_model_image, filtersize=hp_filtersize)
    else:
        opt_model_image_hp = opt_model_image
    optimized_model_image = np.asarray(opt_model_image_hp)
    median_profile_image = np.asarray(med_prof_image)

    print("Generating the optimized forward model image...")
    optimized_fm = ffd_obj.single_fm(np.asarray(optimized_model_image))

    if bool(do_clean_final_fm):
        optimized_fm_rprofsub, _ = subtract_radial_profile(optimized_fm,
                                                          aligned_center,
                                                          noise_reconstructed,
                                                          radial_inds,
                                                          )
    else:
        optimized_fm_rprofsub = optimized_fm

    print("Calculating residuals between reduced data and optimized forward model...")
    residuals = np.asarray(reduced_data - optimized_fm_rprofsub)
    residuals_roi = residuals * optimization_mask
    residuals_roi[residuals_roi == 0.] = np.nan
    residuals_roi = residuals_roi / noise_reconstructed

    outputs_dict["optimized_model"] = np.asarray(optimized_model)
    outputs_dict["optimized_fm"] = np.asarray(optimized_fm_rprofsub)
    outputs_dict["optimized_model_image"] = np.asarray(optimized_model_image)
    outputs_dict["residuals_roi"] = np.asarray(residuals_roi)
    outputs_dict["median_profile_image"] = median_profile_image
    outputs_dict["residuals"] = np.asarray(residuals)
    outputs_dict["weights_asym"] = np.asarray(asymw_reconstructed)
    outputs_dict["weights_nominal"] = np.asarray(w_reconstructed)
    return outputs_dict


def main():
    """
    Main function to run JAX-based optimization for freeform disk fitting.
    """
    multiprocessing.set_start_method('forkserver')
    
    parser = argparse.ArgumentParser(
        description='Run JAX-based optimization for freeform disk fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage (requires iterations)
  python ff_optimize.py -p initialization_files/config.yaml -i 1000
  
  # Forward-modeling dry run only
  python ff_optimize.py -p initialization_files/config.yaml --dry-run
  
  # With custom initial model
  python ff_optimize.py -p initialization_files/config.yaml -i 1000 --initial-model path/to/model.fits
  
  # Custom optimization parameters
  python ff_optimize.py -p initialization_files/config.yaml -i 1000 --reg 0.5 --delta 2.0 --learning-rate 0.01
  
  # Process injected synthetic dataset (PA must be specified)
  python ff_optimize.py -p initialization_files/config.yaml -i 1000 --injected-dir /path/to/injected_data --injected-pa 90.0
        """
    )
    
    parser.add_argument('-p', '--param-file',
                        required=True,
                        help='Path to YAML parameter file')
    parser.add_argument('-i', '--iterations',
                        type=int,
                        required=False,
                        help='Number of optimization iterations (required unless --dry-run)')
    parser.add_argument('--initial-model',
                        type=str,
                        required=False,
                        help='Path to starting model FITS file')
    parser.add_argument('--learning-rate',
                        type=float,
                        default=1e-1,
                        help='Learning rate for optimizer (default: 0.1)')
    parser.add_argument('--reg',
                        type=float,
                        required=False,
                        help='Regularization factor (default: 1.0)')
    parser.add_argument('--delta',
                        type=float,
                        required=False,
                        default=1.0,
                        help='Delta for Huber loss (default: 1.0)')
    parser.add_argument('--dry-run',
                        action='store_true',
                        help='Perform single forward-modeling dry run and exit')
    parser.add_argument('--new-ref',
                        action='store_true',
                        help='Fit new reference model instead of using FirstModel files')
    parser.add_argument('--injected-dir',
                        type=str,
                        required=False,
                        help='Path to directory containing injected synthetic disk data (overrides data directory from config)')
    parser.add_argument('--injected-pa',
                        type=float,
                        required=False,
                        help='Position angle (degrees) of injected disk (required when --injected-dir is provided)')
    
    args = parser.parse_args()
    
    # Validate that --injected-pa is provided when --injected-dir is provided
    if args.injected_dir is not None and args.injected_pa is None:
        print("Error: --injected-pa is required when --injected-dir is provided.")
        print("The injected disk has a different PA than the config file, so the PA prior must be specified.")
        sys.exit(1)
    
    if not os.path.exists(args.param_file):
        print(f"Error: Configuration file not found: {args.param_file}")
        sys.exit(1)
    
    # Check that iterations is provided unless dry-run
    if not args.dry_run and args.iterations is None:
        print("Error: --iterations is required unless --dry-run is specified")
        sys.exit(1)
    
    config = args.param_file
    init_model = args.initial_model if args.initial_model is not None else None
    reg_lambda = args.reg if args.reg is not None else 1.0
    delta = args.delta
    learning_rate = args.learning_rate
    num_iterations = args.iterations if args.iterations is not None else 0
    new_ref = args.new_ref
    dry_run = args.dry_run
    injected_dir = args.injected_dir if args.injected_dir is not None else None
    injected_pa = args.injected_pa if args.injected_pa is not None else None
    
    # Initialize the freeform disk object
    print(f"Initializing FreeFormDisk object with config: {config}")
    ffd_obj = FreeFormDisk(config)
    
    # Override data and output directories if injected directory is provided
    if injected_dir is not None:
        if not os.path.exists(injected_dir):
            print(f"Error: Injected directory not found: {injected_dir}")
            sys.exit(1)
        print(f"Using injected data directory: {injected_dir}")
        print(f"Using injected disk PA: {injected_pa}° (overriding config PA: {ffd_obj.params_file.get('pa_init', 'N/A')}°)")
        # Override datadir to point to injected directory (where FITS files are)
        ffd_obj.datadir = injected_dir
        # Override klipdir to point to klip_fm_files subdirectory in injected directory
        ffd_obj.klipdir = os.path.join(injected_dir, "klip_fm_files")
        os.makedirs(ffd_obj.klipdir, exist_ok=True)
        # Override resultsdir to point to results_freeform subdirectory in injected directory
        ffd_obj.resultsdir = os.path.join(injected_dir, "results_freeform")
        os.makedirs(ffd_obj.resultsdir, exist_ok=True)
        print(f"Output will be saved to: {ffd_obj.klipdir}")
        print(f"Results will be saved to: {ffd_obj.resultsdir}")
        
        # Override PA in params_file for reference model fitting
        # Store original values to restore later if needed
        ffd_obj._original_pa_init = ffd_obj.params_file.get('pa_init')
        ffd_obj._original_pa_best = ffd_obj.params_file.get('pa_best')
        ffd_obj._original_pa_prior = ffd_obj.params_file.get('pa_prior', [None, None])
        
        # Override PA init and best values
        ffd_obj.params_file['pa_init'] = injected_pa
        if 'pa_best' in ffd_obj.params_file:
            ffd_obj.params_file['pa_best'] = injected_pa
        
        # Override PA prior bounds to center around injected PA
        # Calculate a reasonable range around the injected PA (e.g., ±30 degrees)
        pa_prior_range = 30.0  # degrees
        pa_prior_lower = injected_pa - pa_prior_range
        pa_prior_upper = injected_pa + pa_prior_range
        
        # Ensure bounds are in [0, 360) range
        pa_prior_lower = pa_prior_lower % 360
        pa_prior_upper = pa_prior_upper % 360
        
        # If the range wraps around, we might need special handling, but for now
        # just use the modulo values
        ffd_obj.params_file['pa_prior'] = [pa_prior_lower, pa_prior_upper]
        
        # Also update params_init and param_priors in the object (used by fit_simple_disk_model)
        ffd_obj.params_init['pa'] = injected_pa
        ffd_obj.param_priors['pa'] = [pa_prior_lower, pa_prior_upper]
        
        print(f"  PA prior bounds updated to: [{pa_prior_lower:.1f}°, {pa_prior_upper:.1f}°]")
    
    klipdir = ffd_obj.klipdir
    resultsdir = ffd_obj.resultsdir
    file_prefix = ffd_obj.file_prefix
    aligned_center = ffd_obj.aligned_center
    basis_path = os.path.join(klipdir, f"{file_prefix}_klbasis.h5")
    
    print(f"\nRunning optimization for {file_prefix}")
    print(f"Output directory: {klipdir}")
    print(f"Results directory: {resultsdir}")
    print(f"Mode: {ffd_obj.mode}")
    
    # Verify prerequisites
    verify_prerequisites(ffd_obj)
    
    
    # Load PSF
    psf = ffd_obj.psf
    jax_psf = jnp.array(psf)
    
    # Load all required data
    print("\n[1/6] Loading required data...")
    fm_dict = load_kl_basis(basis_path)
    mask4noisemap = fits.getdata(os.path.join(klipdir, f"{file_prefix}_mask4noisemap.fits"))
    mask2generatedisk = fits.getdata(os.path.join(klipdir, f"{file_prefix}_mask2generatedisk.fits"))
    # Set mask on object (needed for get_initial_model later)
    ffd_obj.mask2generatedisk = mask2generatedisk
    reduced_data = fits.getdata(os.path.join(klipdir, f"{file_prefix}-klipped-KLmodes-all.fits"))
    reduced_data = np.squeeze(reduced_data)
    reduced_data[reduced_data != reduced_data] = 0.  # Zero out NaNs
    print("   ✓ Basis, masks, and reduced data loaded")
    
    # Load noise map
    print("\n[2/6] Loading noise map...")
    if ffd_obj.params_file["USE_NOISE"]:
        noise_map = fits.getdata(os.path.join(klipdir, f"{file_prefix}_noisemap.fits"))
        noise_map += 1.  # Get rid of any zeros
        noise_map_flat = noise_map.flatten()
        noise_map_flat[noise_map_flat != noise_map_flat] = 1.
    else:
        noise_map = np.ones_like(reduced_data)
        noise_map_flat = noise_map.flatten()
        noise_map_flat[noise_map_flat != noise_map_flat] = 1.
    print("   ✓ Noise map loaded")
    
    # Set processing flags
    print("\n[3/6] Setting processing flags...")
    hp = ffd_obj.hp
    rprofsub = bool(ffd_obj.params_file["RPROFSUB"])
    clean_final_fm = bool(ffd_obj.clean_final_fm)
    
    if bool(hp):
        hp_filtersize = (psf.shape[0]/hp) / (2*np.sqrt(2*np.log(2)))
    else:
        hp_filtersize = None
    if rprofsub:
        do_radial_profile_sub = 1
    else:
        do_radial_profile_sub = 0
    if clean_final_fm:
        do_clean_final_fm = 1
    else:
        do_clean_final_fm = 0
    print(f"   ✓ High-pass filter: {hp_filtersize}")
    print(f"   ✓ Radial profile subtraction: {do_radial_profile_sub}")
    print(f"   ✓ Clean final FM: {do_clean_final_fm}")
    
    # Get reference model
    print("\n[4/6] Preparing reference model...")
    
    # Load optimization mask from disk (created by ff_setup.py)
    optimization_mask_path = os.path.join(klipdir, f"{file_prefix}_optimization_mask.fits")
    if not os.path.exists(optimization_mask_path):
        raise FileNotFoundError(
            f"Mask file not found: {optimization_mask_path}\n"
            "Please run ff_setup.py first to create the masks."
        )
    optimization_mask_obj = fits.getdata(optimization_mask_path)
    
    ffd_obj.allocate_dataset()
    model_firstguess = ffd_obj.get_initial_model(init_model)
    
    if init_model is None and bool(new_ref):
        reference_model = ffd_obj.fit_reference_model(noise_map, reduced_data)
        reference_model[reference_model != reference_model] = 0.
        # reference_model_psd, window_opt = ffd_obj.get_reference_model_psd(reference_model)
        print("   ✓ New reference model fitted and saved")
        print("   Note: Inspect the reference model and re-run optimization if needed.")
    elif init_model is None:
        # Try to load FirstModel, otherwise use initial guess
        first_model_path = os.path.join(klipdir, f"{file_prefix}_FirstModel.fits")
        if os.path.exists(first_model_path):
            reference_model = fits.getdata(first_model_path)
            reference_model[reference_model != reference_model] = 0.
            print(f"   ✓ Loaded reference model from {first_model_path}")
        else:
            reference_model = model_firstguess
            print("   ✓ Using initial guess model as reference")
    else:
        reference_model = model_firstguess
        print("   ✓ Using provided initial model as reference")
    
    reference_model_psd, window_opt = ffd_obj.get_reference_model_psd(reference_model)
    print("   ✓ Reference model PSD computed")

    # If dry-run, perform forward modeling and exit
    if dry_run:
        perform_forward_modeling_dry_run(ffd_obj, init_model_path=init_model)
        print("\n✓ Dry run complete. Exiting.")
        return
    
    # Prepare data arrays and indices
    print("\n[5/6] Preparing data arrays and indices...")
    total_pixels = np.prod(reduced_data.shape)
    disk_mask = np.array(mask2generatedisk)
    optimization_mask = np.array(optimization_mask_obj)
    annular_mask = make_annular_mask(disk_mask.shape, 10, ffd_obj.owa)
    disk_mask *= annular_mask
    
    disk_mask_indices = jnp.flatnonzero(jnp.array(disk_mask))
    optimization_mask_indices = jnp.flatnonzero(jnp.array(optimization_mask))
    
    reduced_data_flat = reduced_data.flatten()
    reduced_flat_interest = reduced_data_flat[optimization_mask_indices]
    
    model_firstguess *= disk_mask
    init_model_jax = jnp.array(model_firstguess)
    init_model_flat = init_model_jax.reshape(init_model_jax.shape[0] * init_model_jax.shape[1])
    init_model_interest = init_model_flat[disk_mask_indices]
    
    noise_interest = noise_map_flat[optimization_mask_indices]
    run_dir = get_next_run_dir(resultsdir)
    
    aligned_center = float(fm_dict["klparam_dict"]["aligned_center_x"]), float(fm_dict["klparam_dict"]["aligned_center_y"])
    image_shape = (jnp.round(aligned_center[0]) * 2, jnp.round(aligned_center[1]) * 2)
    radial_inds = get_radial_inds(image_shape, aligned_center)
    print("   ✓ Data arrays and indices prepared")
    
    # Run optimization
    print(f"\n[6/6] Running optimization ({num_iterations} iterations)...")
    print(f"   Output directory: {run_dir}")
    
    import jax.profiler
    trace_dest = os.environ.get('profileJaxTraceTo', False)
    if trace_dest:
        jax.profiler.start_trace(trace_dest)
        print(f"   Tracing to {trace_dest}")
    
    optimized_params, loss_history, weights_asym, weights_nominal = optimize_model(
        target_image=reduced_flat_interest,
        model_init=init_model_interest,
        ref_psd=reference_model_psd,
        noise_map=noise_interest,
        disk_mask_indices=disk_mask_indices,
        opt_mask_indices=optimization_mask_indices,
        psf=jax_psf,
        basis_data=fm_dict,
        total_pixels=total_pixels,
        num_steps=num_iterations,
        reg_lambda=reg_lambda,
        run_dir=run_dir,
        reduced_data=reduced_data,
        radial_inds=radial_inds,
        learning_rate=learning_rate,
        hp_filtersize=hp_filtersize,
        delta=delta,
        aligned_center=aligned_center,
        do_radial_profile_sub=do_radial_profile_sub,
        do_clean_final_fm=do_clean_final_fm,
    )
    
    try:
        optimized_params.block_until_ready()
    except Exception as e:
        print(e)
    
    if trace_dest:
        jax.profiler.stop_trace()
        print("   Ended profiler trace")
    
    print("   ✓ Optimization complete")
    
    # Post-process and save results
    print("\n[Saving] Post-processing and saving results...")
    pyklip_params_dict = record_pyklip_params(
        ffd_obj.numbasis,
        ffd_obj.iwa,
        ffd_obj.owa,
        ffd_obj.minrot,
        ffd_obj.aligned_center
    )
    
    outputs_dict = harness_optimized_model(
        optimized_params,
        ffd_obj,
        reduced_data,
        psf,
        total_pixels,
        disk_mask_indices,
        optimization_mask_indices,
        disk_mask,
        optimization_mask,
        aligned_center,
        radial_inds,
        do_radial_profile_sub,
        do_clean_final_fm,
        hp_filtersize,
        noise_interest,
        weights_asym,
        weights_nominal
    )
    
    save_ffdfit_outputs(
        run_dir=run_dir,
        file_prefix=file_prefix,
        model_opt=outputs_dict["optimized_model"],
        model_image_opt=outputs_dict["optimized_model_image"],
        forward_model_opt=outputs_dict["optimized_fm"],
        residuals_image=outputs_dict["residuals"],
        residuals_roi=outputs_dict["residuals_roi"],
        median_profile_image=outputs_dict["median_profile_image"],
        loss_history=loss_history,
        hsf_regularization=reg_lambda,
        huber_delta=delta,
        pyklip_params=pyklip_params_dict,
        weights_asym=outputs_dict["weights_asym"],
        weights_nominal=outputs_dict["weights_nominal"]
    )
    
    print("\n✓ Optimization and saving complete!")
    print(f"\nResults saved to: {run_dir}")
    print(f"  - Optimized model")
    print(f"  - Forward model")
    print(f"  - Residuals")
    print(f"  - Loss history")
    print(f"  - Training plots")


if __name__ == "__main__":
    main()
