'''
Run JAX-based optimization for freeform disk fitting.

This script handles the optimization phase of the 𝒇𝒇 pipeline:
1. Verifies that ff_klip and ff_setup were run successfully
2. Optionally performs a single forward-modeling dry run for inspection (--dry-run, no -i needed)
3. Executes JAX-based model optimization
4. Saves optimization results to disk

Usage:
    ff_optimize -p initialization_files/config.yaml -i 50000
    ff_optimize -p initialization_files/config.yaml --dry-run
    ff_optimize -p initialization_files/config.yaml -i 1000 --loss-tolerance 0.001 --log-to-file

TODO add the option to specify a custom path for the output
klip_fm_files directory
  - This then needs to be reported in the logging/log file to
  remind the user to specify this path in the subsequent steps
  in the pipeline
  - Actually this custom path should be specified in the config file
  as a new parameter KLIP_FM_FILES_PATH
  - This then, *if present* needs to be used to override the default path
  for the klip_fm_files directory. If KLIP_FM_FILES_PATH is null,
  then the default path is used.
  - I've added this new param to HR4796a_z_lco2023a_magao-x_20230309_10.yaml


'''

import os
import sys
import argparse
import logging
import multiprocessing
import numpy as np
import astropy.io.fits as fits
from scipy.signal import fftconvolve

from ffortissimo.modeling.disk_freeform import FreeFormDisk
from ffortissimo.io.save_diskfit_results import save_ffdfit_outputs, \
    get_next_run_dir, harness_optimized_model
from ffortissimo.io.load_diskfit_files import load_diskfit_components
from ffortissimo.io.fits_handling import save_fits, load_pyklip_reduced_data
from ffortissimo.core.diskfit import optimize_model

from ffortissimo.utils.diskfit_tools import record_pyklip_params
from ffortissimo.io.log_handling import configure_logging

logger = logging.getLogger(__name__)


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
    
    logger.info(" Verified prerequisites:")
    logger.info("  - KLIP-reduced image: %s", reduced_data_path)
    logger.info("  - KL basis file: %s", basis_path)
    logger.info("  - Masks: all present")
    logger.info("  - Noise map: %s", noise_map_path)


def perform_forward_modeling_dry_run(ffd_obj, init_model_path=None):
    """
    Perform a single forward-modeling dry run for inspection.
    
    Args:
        ffd_obj: FreeFormDisk object
        init_model_path (str, optional): Path to initial model FITS file
    """
    klipdir = ffd_obj.klipdir
    file_prefix = ffd_obj.file_prefix
    
    logger.info("[Forward Modeling Dry Run]")
    logger.info("Performing single forward-modeling computation for inspection...")
    
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
    logger.info("   Creating/loading initial model...")
    model_init = ffd_obj.get_initial_model(init_model_path)
    
    # Convolve initial model with PSF
    logger.info("   Convolving initial model with PSF...")
    model_convolved = fftconvolve(model_init, ffd_obj.psf, mode="same")
    
    # Run forward modeling
    logger.info("   Running forward modeling (this may take a while)...")
    model_fm_init = ffd_obj.single_fm(np.asarray(model_convolved))
    
    # Save forward model
    model_fm_saveto = os.path.join(klipdir, f"{file_prefix}_DryRun_FM.fits")
    save_fits(model_fm_saveto, model_fm_init)
    logger.info("   ♪ Saved forward model: %s", model_fm_saveto)

    logger.info("♪ Forward modeling dry run complete!")
    logger.info("You can now inspect the forward model before running the full optimization.")




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
  # Basic usage
  ff_optimize -p initialization_files/config.yaml -i 50000

  # Forward-modeling dry run only (no -i needed)
  ff_optimize -p initialization_files/config.yaml --dry-run

  # With custom initial model
  ff_optimize -p initialization_files/config.yaml -i 1000 --initial-model path/to/model.fits

  # Custom optimization parameters
  ff_optimize -p initialization_files/config.yaml -i 1000 --loss-tolerance 0.001 --reg 0.5 --learning-rate 0.005

  # Process injected synthetic dataset (PA from pa_test in config)
  ff_optimize -p initialization_files/config.yaml -i 1000 --injected-dir /path/to/injected_data

  # Write log to file
  ff_optimize -p initialization_files/config.yaml -i 50000 --log-to-file
        """
    )
    
    parser.add_argument('-p', '--param-file',
                        required=True,
                        help='Path to YAML parameter file')
    parser.add_argument('-i', '--iterations',
                        type=int,
                        default=0,
                        help='Maximum optimization iterations (required unless --dry-run)')
    parser.add_argument('--loss-tolerance',
                        type=float,
                        default=1e-6,
                        help='Absolute loss change tolerance for early stopping (default: 5e-3)')
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
    parser.add_argument('--dry-run',
                        action='store_true',
                        help='Perform single forward-modeling dry run and exit')
    parser.add_argument('--injected-dir',
                        type=str,
                        required=False,
                        help='Path to directory containing injected synthetic disk data (overrides data directory from config)')
    parser.add_argument('--log-to-file',
                        action='store_true',
                        help='Write a timestamped log file instead of stdout')

    args = parser.parse_args()
    
    
    if not os.path.exists(args.param_file):
        print(f"Error: Configuration file not found: {args.param_file}")
        sys.exit(1)

    if not args.dry_run and args.iterations <= 0:
        print("Error: -i/--iterations is required for optimization (or use --dry-run)")
        sys.exit(1)

    config = args.param_file
    init_model = args.initial_model if args.initial_model is not None else None
    if args.reg is None:
        reg_lambda = config.get("LAMBDA_REG", 1.0)
    else:
        reg_lambda = args.reg
    delta = 1.0  # Huber loss delta (fixed)
    learning_rate = args.learning_rate
    num_iterations = args.iterations
    loss_tolerance = args.loss_tolerance
    dry_run = args.dry_run
    injected_dir = args.injected_dir if args.injected_dir is not None else None
    
    # Initialize the freeform disk object
    ffd_obj = FreeFormDisk(config)

    # Configure logging (must be before any logger calls)
    save_dir = os.path.join(ffd_obj.datadir, "ff_logs")
    if args.log_to_file:
        os.makedirs(save_dir, exist_ok=True)
    log_path = configure_logging(args.log_to_file, "ff_optimize", save_dir if args.log_to_file else None)
    if log_path:
        print(f"Writing log to {log_path}")

    logger.info("Initializing FreeFormDisk object with config: %s", config)

    # Override data and output directories if injected directory is provided
    if injected_dir is not None:
        if not os.path.exists(injected_dir):
            logger.error("Injected directory not found: %s", injected_dir)
            sys.exit(1)
        
        ffd_obj.inject_recover_mode(injected_dir)
        # assert that the init params have been overridden with the injected test params
        assert ffd_obj.params_init['pa'] == ffd_obj.params_file['pa_test']
        assert ffd_obj.params_init['inc'] == ffd_obj.params_file['inc_test']
        assert ffd_obj.params_init['r1'] == ffd_obj.params_file['r1_test']
        assert ffd_obj.params_init['r2'] == ffd_obj.params_file['r2_test']
        assert ffd_obj.params_init['rc'] == ffd_obj.params_file['rc_test']
        assert ffd_obj.params_init['alpha_in'] == ffd_obj.params_file['alpha_in_test']
        assert ffd_obj.params_init['alpha_out'] == ffd_obj.params_file['alpha_out_test']
        assert ffd_obj.params_init['beta'] == ffd_obj.params_file['beta_test']
        assert ffd_obj.params_init['a_r'] == ffd_obj.params_file['a_r_test']
        assert ffd_obj.params_init['dx'] == ffd_obj.params_file['dx_test']
        assert ffd_obj.params_init['dy'] == ffd_obj.params_file['dy_test']
        assert ffd_obj.params_init['N'] == ffd_obj.params_file['N_test']
        assert ffd_obj.params_init['g1'] == ffd_obj.params_file['g1_test']
        assert ffd_obj.params_init['g2'] == ffd_obj.params_file['g2_test']
        assert ffd_obj.params_init['alpha1'] == ffd_obj.params_file['alpha1_test']
        logger.info("Injected directory: %s", injected_dir)

    klipdir = ffd_obj.klipdir
    resultsdir = ffd_obj.resultsdir
    file_prefix = ffd_obj.file_prefix
    aligned_center = ffd_obj.aligned_center
    logger.info("Running optimization for %s", file_prefix)
    logger.info("Output directory: %s", klipdir)
    logger.info("Results directory: %s", resultsdir)
    logger.info("Mode: %s", ffd_obj.mode)
    
    # Verify prerequisites
    verify_prerequisites(ffd_obj)
    
    
    component_dict = load_diskfit_components(
        ffd_obj=ffd_obj,
        init_model=init_model,
    )
    psf = component_dict["psf"]
    jax_psf = component_dict["jax_psf"]
    fm_dict = component_dict["fm_dict"]
    reduced_data = component_dict["reduced_data"]
    noise_map = component_dict["noise_map"]
    noise_map_flat = component_dict["noise_map_flat"]
    hp_filtersize = component_dict["hp_filtersize"]
    do_radial_profile_sub = component_dict["do_radial_profile_sub"]
    do_clean_final_fm = component_dict["do_clean_final_fm"]
    reference_model_psd = component_dict["reference_model_psd"]
    total_pixels = component_dict["total_pixels"]
    disk_mask = component_dict["disk_mask"]
    disk_mask_apod = component_dict["disk_mask_apod"]
    optimization_mask = component_dict["optimization_mask"]
    disk_mask_indices = component_dict["disk_mask_indices"]#mask2generatedisk * annulus
    optimization_mask_indices = component_dict["optimization_mask_indices"]
    reduced_flat_interest = component_dict["reduced_flat_interest"]
    init_model_interest = component_dict["init_model_interest"]
    noise_interest = component_dict["noise_interest"]
    aligned_center = component_dict["aligned_center"]
    radial_inds = component_dict["radial_inds"]
    counter_rotated_image = component_dict["counter_rotated_image"]

    # TEST: if RDI mode, load and subtract the median background from the reduced data
    if counter_rotated_image is not None:
        reduced_data_no_bkg = reduced_data - counter_rotated_image
        # #debug look at the reduced data no bkg
        # import matplotlib.pyplot as plt
        # plt.imshow(reduced_data_no_bkg, origin='lower')
        # plt.colorbar()
        # plt.show()
        # exit()
        reduced_flat_interest_no_bkg = reduced_data_no_bkg.flatten()[optimization_mask_indices]
    else:
        reduced_flat_interest_no_bkg = reduced_flat_interest

    # If dry-run, perform forward modeling and exit
    if dry_run:
        perform_forward_modeling_dry_run(ffd_obj, init_model_path=init_model)
        logger.info("♪ Dry run complete. Exiting.")
        return
    
    run_dir = get_next_run_dir(resultsdir)
    
    # Run optimization
    logger.info("[6/6] Running optimization (max %s iterations, loss tolerance %s)...", num_iterations, loss_tolerance)
    logger.info("   Output directory: %s", run_dir)
    
    import jax.profiler
    trace_dest = os.environ.get('profileJaxTraceTo', False)
    if trace_dest:
        jax.profiler.start_trace(trace_dest)
        logger.info("   Tracing to %s", trace_dest)
    
    optimized_params, loss_history, weights_nominal = optimize_model(
        target_image=reduced_flat_interest_no_bkg if ffd_obj.mode == "RDI" else reduced_flat_interest,
        model_init=init_model_interest,
        ref_psd=reference_model_psd,
        noise_map=noise_interest,
        disk_mask_indices=disk_mask_indices,
        disk_mask_apod=disk_mask_apod,
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
        loss_tolerance=loss_tolerance,
        aligned_center=aligned_center,
        do_radial_profile_sub=do_radial_profile_sub,
        do_clean_final_fm=do_clean_final_fm,
    )
    
    try:
        optimized_params.block_until_ready()
    except Exception as e:
        logger.exception("%s", e)

    if trace_dest:
        jax.profiler.stop_trace()
        logger.info("   Ended profiler trace")

    logger.info("   ♪ Optimization complete")
    
    # Post-process and save results
    logger.info("[Saving] Post-processing and saving results...")
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
        weights_nominal
    )

    injected_model_path = None
    if injected_dir is not None:
        injected_model_path = os.path.join(klipdir, "disk_model_to_inject.fits")
    
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
        weights_nominal=outputs_dict["weights_nominal"],
        injected_model_path=injected_model_path,
        learning_rate=learning_rate
    )
    
    logger.info("♪ Optimization and saving complete!")
    logger.info("Results saved to: %s", run_dir)
    logger.info("  - Optimized model")
    logger.info("  - Forward model")
    logger.info("  - Residuals")
    logger.info("  - Loss history")
    logger.info("  - Training plots")


if __name__ == "__main__":
    main()
