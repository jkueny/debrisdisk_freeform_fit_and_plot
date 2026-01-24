'''
Create masks and estimate noise map for freeform disk fitting.

This script handles the setup phase after KLIP reduction:
1. Verifies that ff_klip.py was run successfully
2. Creates binary masks for disk generation and noise mapping
3. Estimates spatial noise map from masked reduced data

Usage:
    python ff_setup.py -p initialization_files/params.yaml
    python ff_setup.py -p initialization_files/params.yaml --do-fm
    python ff_setup.py -p initialization_files/params.yaml --injected-dir /path/to/injected_dataset --injected-pa 90.0 --do-fm
'''

import os
import sys
import argparse
import logging
import numpy as np
import astropy.io.fits as fits
from scipy.signal import convolve2d

from ffortissimo.modeling.disk_freeform import FreeFormDisk
from ffortissimo.io.fits_handling import save_fits
from ffortissimo.io.log_handling import configure_logging
logger = logging.getLogger(__name__)


def verify_klip_prerequisites(ffd_obj):
    """
    Verify that ff_klip.py was run successfully.
    
    Args:
        ffd_obj: FreeFormDisk object
        
    Raises:
        FileNotFoundError: If required files are missing
    """
    klipdir = ffd_obj.klipdir
    file_prefix = ffd_obj.file_prefix
    
    reduced_data_path = os.path.join(klipdir, f"{file_prefix}-klipped-KLmodes-all.fits")
    basis_path = os.path.join(klipdir, f"{file_prefix}_klbasis.h5")
    
    missing_files = []
    if not os.path.exists(reduced_data_path):
        missing_files.append(reduced_data_path)
    if not os.path.exists(basis_path):
        missing_files.append(basis_path)
    
    if missing_files:
        error_msg = (
            f"Error: ff_klip.py has not been run successfully.\n"
            f"Missing files:\n"
        )
        for f in missing_files:
            error_msg += f"  - {f}\n"
        error_msg += "\nPlease run ff_klip first to create the reduced image and basis file."
        raise FileNotFoundError(error_msg)
    
    logger.info("Verified that ff_klip.py was run successfully.")
    logger.info("Reduced data: %s", reduced_data_path)
    logger.info("Basis file: %s", basis_path)


def perform_dry_run(ffd_obj, init_model_path=None):
    """
    Perform optimization dry run: create initial model, convolve with PSF, run forward modeling.
    
    Args:
        ffd_obj: FreeFormDisk object
        init_model_path (str, optional): Path to initial model FITS file
    """
    klipdir = ffd_obj.klipdir
    file_prefix = ffd_obj.file_prefix
    
    logger.info("[4/4] Performing optimization dry run...")
    
    # Get initial model
    logger.info("Creating initial model...")
    model_init = ffd_obj.get_initial_model(init_model_path)
    
    # Convolve initial model with PSF
    logger.info("Convolving initial model with PSF...")
    model_convolved = convolve2d(model_init, ffd_obj.psf, mode="same")
    
    # Save initial model files
    model_init_saveto = os.path.join(klipdir, f"{file_prefix}_FirstModel.fits")
    model_convolved_saveto = os.path.join(klipdir, f"{file_prefix}_FirstModel_Conv.fits")
    save_fits(model_init_saveto, model_init)
    save_fits(model_convolved_saveto, model_convolved)
    logger.info("Saved initial model: %s", model_init_saveto)
    logger.info("Saved convolved model: %s", model_convolved_saveto)
    
    # Run forward modeling
    logger.info("Running forward modeling...")
    model_fm_init = ffd_obj.single_fm(np.asarray(model_convolved))
    
    # Save forward model
    model_fm_saveto = os.path.join(klipdir, f"{file_prefix}_FirstModel_FM.fits")
    save_fits(model_fm_saveto, model_fm_init)
    logger.info("Saved forward model: %s", model_fm_saveto)
    
    logger.info("Optimization dry run complete.")


def main():
    """
    Main function to create masks, estimate noise, and perform optimization dry run.
    """
    parser = argparse.ArgumentParser(
        description='Create masks, estimate noise map, and perform optimization dry run for freeform disk fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python ff_setup.py -p initialization_files/params.yaml
  
  # With custom initial model
  python ff_setup.py -p initialization_files/params.yaml --initial-model path/to/model.fits
  
  # Perform initial forward model computation (may be time-consuming)
  python ff_setup.py -p initialization_files/params.yaml --do-fm
  
  # Process injected synthetic dataset (PA must be specified)
  python ff_setup.py -p initialization_files/params.yaml --injected-dir /path/to/injected_data --injected-pa 90.0
        """
    )
    
    parser.add_argument('-p', '--param-file',
                        required=True,
                        help='Path to YAML parameter file')
    parser.add_argument('--initial-model',
                        type=str,
                        required=False,
                        help='Path to initial model FITS file (optional)')
    parser.add_argument('--do-fm',
                        action='store_true',
                        help='Perform initial forward model computation (may be time-consuming)')
    parser.add_argument('--injected-dir',
                        type=str,
                        required=False,
                        help='Path to directory containing injected synthetic disk data (overrides data directory from config)')
    parser.add_argument('--injected-pa',
                        type=float,
                        required=False,
                        help='Position angle (degrees) of injected disk (required when --injected-dir is provided)')
    parser.add_argument('--log-to-file',
                        action='store_true',
                        help='Write a timestamped log file instead of stdout')
    
    args = parser.parse_args()
    

    # Validate that --injected-pa is provided when --injected-dir is provided
    if args.injected_dir is not None and args.injected_pa is None:
        logger.error("--injected-pa is required when --injected-dir is provided.")
        logger.error("The injected disk has a different PA than the config file, so masks must be created with the correct PA.")
        sys.exit(1)

    if not os.path.exists(args.param_file):
        logger.error("Configuration file not found: %s", args.param_file)
        sys.exit(1)
    
    config = args.param_file
    do_fm = args.do_fm
    init_model = args.initial_model if args.initial_model is not None else None
    injected_dir = args.injected_dir if args.injected_dir is not None else None
    injected_pa = args.injected_pa if args.injected_pa is not None else None
    
    # Initialize the freeform disk object
    print(f"Initializing FreeFormDisk object with config: {config}")
    ffd_obj = FreeFormDisk(config)

    save_to_dir = os.path.join(ffd_obj.datadir, 'ff_logs')

    if args.log_to_file:
        print(f"Writing logs to {save_to_dir}")
    
    # Override data and output directories if injected directory is provided
    if injected_dir is not None:
        if not os.path.exists(injected_dir):
            print(f"Injected directory not found: {injected_dir}")
            sys.exit(1)

        # Override datadir to point to injected directory (where FITS files are)
        ffd_obj.datadir = injected_dir
        # Override klipdir to point to klip_fm_files subdirectory in injected directory
        ffd_obj.klipdir = os.path.join(injected_dir, "klip_fm_files")
        os.makedirs(ffd_obj.klipdir, exist_ok=True)
        
        # Override PA in params_file for mask creation
        # Store original values to restore later if needed
        ffd_obj._original_pa_init = ffd_obj.params_file.get('pa_init')
        ffd_obj._original_pa_best = ffd_obj.params_file.get('pa_best')
        ffd_obj.params_file['pa_init'] = injected_pa
        if 'pa_best' in ffd_obj.params_file:
            ffd_obj.params_file['pa_best'] = injected_pa
    
    if save_to_dir:
        save_dir = os.path.join(ffd_obj.datadir, "ff_logs")
        log_path = configure_logging(args.log_to_file,
                                    "ff_setup",
                                    save_dir)
    else:
        log_path = configure_logging(args.log_to_file,
                                    "ff_setup",
                                    None)
    if injected_dir is not None:
        logger.info("Using injected data directory: %s", injected_dir)
        logger.info(
            "Using injected disk PA: %s deg (overriding config PA: %s deg)",
            injected_pa,
            ffd_obj.params_file.get("pa_init", "N/A"),
        )
        logger.info("Output will be saved to: %s", ffd_obj.klipdir)

    klipdir = ffd_obj.klipdir
    file_prefix = ffd_obj.file_prefix
    
    logger.info("Setting up masks and noise map for %s", file_prefix)
    logger.info("Output directory: %s", klipdir)
    logger.info("Mode: %s", ffd_obj.mode)
    
    # Step 1: Verify ff_klip prerequisites
    verify_klip_prerequisites(ffd_obj)
    
    # Step 2: Allocate dataset (needed for mask creation)
    logger.info("[1/4] Allocating dataset...")
    ffd_obj.allocate_dataset()
    logger.info("Dataset allocated")
    
    # Step 3: Generate masks (always regenerate)
    logger.info("[2/4] Creating binary masks...")
    logger.info("Disk generation mask parameters:")
    logger.info("  Inner scaling: %s", ffd_obj.params_file["MASK_IN_SCALING"])
    logger.info("  Outer scaling: %s", ffd_obj.params_file["MASK_OUT_SCALING"])
    logger.info("  Noise inner scaling: %s", ffd_obj.params_file["MASK_NOISE_IN"])
    logger.info("  Noise outer scaling: %s", ffd_obj.params_file["MASK_NOISE_OUT"])
    logger.info("  Mask center offset: (%s, %s)", ffd_obj.params_file["MASK_DX"], ffd_obj.params_file["MASK_DY"])
    
    # Always regenerate masks
    optimization_mask = ffd_obj.prep_binary_masks()
    
    # Read mask_out_of_bounds from disk (it's saved by prep_binary_masks)
    mask_files = {
        'mask_out_of_bounds': os.path.join(klipdir, f"{file_prefix}_mask_out_of_bounds.fits"),
    }
    mask_out_of_bounds = fits.getdata(mask_files['mask_out_of_bounds'])
    
    masks = {
        'mask2generatedisk': ffd_obj.mask2generatedisk,
        'mask4noisemap': ffd_obj.mask4noisemap,
        'engineered_optimization': optimization_mask,
        'mask_out_of_bounds': mask_out_of_bounds,
    }
    
    logger.info("Masks created and saved to %s", klipdir)
    logger.info("  mask2generatedisk: %s pixels", np.sum(masks["mask2generatedisk"]))
    logger.info("  mask4noisemap: %s pixels", np.sum(masks["mask4noisemap"]))
    
    # Step 4: Estimate noise map (always regenerate)
    logger.info("[3/4] Estimating noise map from reduced data...")
    
    noise_map_path = os.path.join(klipdir, f"{file_prefix}_noisemap.fits")
    reduced_data_path = os.path.join(klipdir, f"{file_prefix}-klipped-KLmodes-all.fits")
    
    # Load reduced data
    reduced_data = fits.getdata(reduced_data_path)
    reduced_data = np.squeeze(reduced_data)
    reduced_data[reduced_data != reduced_data] = 0.  # Zero out NaNs
    
    # Create engineered noise mask for ADI mode
    if ffd_obj.mode == "ADI":
        logger.info("ADI mode: Using engineered noise mask")
        bespoke_noise_mask = ffd_obj._engineer_disk_mask(
            masks['mask4noisemap'], 
            angle_sweep_factor=4
        )
        reduced_noise_masked = reduced_data * (1 - bespoke_noise_mask)
        tosave_reduced_noise_masked = reduced_data * bespoke_noise_mask
        
        # Save the engineered noise mask (always regenerate)
        bespoke_noise_path = os.path.join(klipdir, f"{file_prefix}_mask4noisemap.fits")
        save_fits(bespoke_noise_path, bespoke_noise_mask)
    else:
        logger.info("RDI mode: Using standard noise mask")
        reduced_noise_masked = reduced_data * (1 - masks['mask4noisemap'])
        tosave_reduced_noise_masked = reduced_data * masks['mask4noisemap']
    
    # Get noise delta radii parameter
    delta_radii = ffd_obj.params_file.get("NOISE_DELTA_RADII", 1)
    logger.info("Using ring width: %s pixels", delta_radii)
    
    # Estimate noise map (always regenerate)
    noise_map = ffd_obj.make_noise_map_rings(
        reduced_data_no_disk=reduced_noise_masked,
        delta_radii=delta_radii
    )
    
    # Save noise map (always overwrite)
    fits.writeto(noise_map_path, noise_map, overwrite=True)
    
    # Save masked data for inspection (always regenerate)
    masked_data_path = os.path.join(klipdir, f"{file_prefix}_masked_data.fits")
    masked_noise_path = os.path.join(klipdir, f"{file_prefix}_use4noisemap.fits")
    # save_fits(masked_data_path, reduced_data * masks['mask2generatedisk'])
    save_fits(masked_data_path, reduced_data * masks['engineered_optimization'])
    save_fits(masked_noise_path, tosave_reduced_noise_masked)
    
    logger.info("Noise map created and saved to %s", noise_map_path)
    
    # Step 5: Perform optimization dry run (unless skipped)
    if do_fm:
        logger.info("[4/4] Performing initial forward model computation")
        logger.info("Note: Initial model files will not be created.")
        perform_dry_run(ffd_obj, init_model_path=init_model)
    else:
        logger.info("[4/4] Skipping initial forward model computation")
    
    logger.info("Done!")
    logger.info("Check: %s", klipdir)
    logger.info("Next steps:")
    logger.info("  1. Inspect masks and noise map")
    logger.info("  2. Inspect initial model files")
    logger.info("  3. Adjust mask parameters in YAML if needed and re-run")
    logger.info("  4. Run freeform fitting with: ff_optimize")


if __name__ == "__main__":
    main()

