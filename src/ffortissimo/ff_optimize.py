'''
Run JAX-based optimization for freeform disk fitting.

This script handles the optimization phase of the 𝒇𝒇 pipeline:
1. Verifies that ff_klip.py and ff_setup.py were run successfully
2. Optionally performs a single forward-modeling dry run for inspection
3. Executes JAX-based model optimization
4. Saves optimization results to disk

Usage:
    python ff_optimize.py -p initialization_files/config.yaml -i 1000
    python ff_optimize.py -p initialization_files/config.yaml --dry-run
    python ff_optimize.py -p initialization_files/config.yaml -i 1000 --initial-model path/to/model.fits

TODO offload mask loading to another function in core.diskfit.py
'''

import os, sys
import argparse
import multiprocessing
import numpy as np
import astropy.io.fits as fits
from scipy.signal import fftconvolve

import jax.numpy as jnp

from ffortissimo.modeling.disk_freeform import FreeFormDisk
from ffortissimo.data.save_diskfit_results import save_ffdfit_outputs, \
    get_next_run_dir, harness_optimized_model
from ffortissimo.data.load_diskfit_files import load_diskfit_components
from ffortissimo.io.fits_handling import save_fits
from ffortissimo.core.diskfit import optimize_model

from ffortissimo.utils.diskfit_tools import record_pyklip_params


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
    
    print(" Verified prerequisites:")
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
    print(f"\nRunning optimization for {file_prefix}")
    print(f"Output directory: {klipdir}")
    print(f"Results directory: {resultsdir}")
    print(f"Mode: {ffd_obj.mode}")
    
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

    # If dry-run, perform forward modeling and exit
    if dry_run:
        perform_forward_modeling_dry_run(ffd_obj, init_model_path=init_model)
        print("\n✓ Dry run complete. Exiting.")
        return
    
    run_dir = get_next_run_dir(resultsdir)
    
    # Run optimization
    print(f"\n[6/6] Running optimization ({num_iterations} iterations)...")
    print(f"   Output directory: {run_dir}")
    
    import jax.profiler
    trace_dest = os.environ.get('profileJaxTraceTo', False)
    if trace_dest:
        jax.profiler.start_trace(trace_dest)
        print(f"   Tracing to {trace_dest}")
    
    optimized_params, opt_offset, loss_history, weights_nominal = optimize_model(
        target_image=reduced_flat_interest,
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
        opt_offset,
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
