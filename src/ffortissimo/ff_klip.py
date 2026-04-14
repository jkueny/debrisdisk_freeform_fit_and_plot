'''
Run KLIP data reduction and create KLIP basis file.

This script handles the first step of the freeform disk fitting pipeline:
1. Loads and prepares the dataset
2. Runs KLIP reduction
3. Saves the Karhunen-Loeve basis to a HDF5 file.
4. Saves the KLIP-reduced image to a FITS file.

Usage:
    ff_klip -p initialization_files/params.yaml
    ff_klip -p initialization_files/params.yaml --injected-dir /path/to/injected_dataset
    ff_klip -p initialization_files/params.yaml --force --log-to-file
    ff_klip -p initialization_files/params.yaml --make-diskless-image

TODO add the option to specify a custom path for the output
klip_fm_files directory
  - This then needs to be reported in the logging/log file to
  remind the user to specify this path in the subsequent steps
  in the pipeline
  - Actually this custom path should be specified in the config file
  as a new parameter, e.g. KLIP_FM_FILES_PATH
  - This then, *if present* needs to be used to override the default path
  for the klip_fm_files directory. If KLIP_FM_FILES_PATH is null,
  then the default path is used.
  - I've added this new param to HR4796a_z_lco2023a_magao-x_20230309_10.yaml
  - These changes should be propagated to the other config files as well.
  - These changes should be propagated to the ff_setup.py and ff_optimize.py
  scripts as well.
'''

import os
import sys
import argparse
import multiprocessing
import logging

from astropy.io import fits
from ffortissimo.modeling.disk_freeform import FreeFormDisk
from ffortissimo.io.log_handling import configure_logging
logger = logging.getLogger(__name__)




def main():
    """
    Run KLIP data reduction and create basis file.
    
    Accomplishes the first phase of the 𝒇𝒇 pipeline:
    - Initializes the FreeFormDisk object
    - Allocates dataset
    - Runs KLIP reduction
    - Creates and saves the KLIP basis file
    
    Args:
        config (str): Path to YAML parameters/config file
        force (bool): Force regeneration even if basis file exists or
        FIRST_TIME is True in the config file
    """
    multiprocessing.set_start_method('forkserver')
    
    parser = argparse.ArgumentParser(
        description='Run KLIP data reduction and create basis file for freeform disk fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  ff_klip -p initialization_files/params.yaml

  # Force regeneration of existing basis
  ff_klip -p initialization_files/params.yaml --force

  # Process injected synthetic dataset
  ff_klip -p initialization_files/params.yaml --injected-dir /path/to/injected_data

  # Write log to file
  ff_klip -p initialization_files/params.yaml --log-to-file
        """
    )
    
    parser.add_argument('-p', '--param-file',
                        required=True,
                        help='Path to YAML parameter file')
    parser.add_argument('--force',
                        action='store_true',
                        help='Force regeneration even if basis file exists')
    parser.add_argument('--injected-dir',
                        type=str,
                        required=False,
                        help='Path to directory containing injected synthetic disk data (overrides data directory from config)')
    parser.add_argument('--log-to-file',
                        action='store_true',
                        help='Write a timestamped log file instead of stdout')
    parser.add_argument('--make-diskless-image',
                        action='store_true',
                        help='Make a diskless image of the original data')
    
    args = parser.parse_args()

    config = args.param_file
    force = args.force
    make_diskless_image = args.make_diskless_image
    # Initialize the freeform disk object
    ffd_obj = FreeFormDisk(config)
    original_datadir = ffd_obj.datadir
    injected_dir = args.injected_dir if args.injected_dir is not None else None

    # Override data and output directories if injected directory is provided
    if injected_dir is not None:
        if not os.path.exists(injected_dir):
            logger.error("Injected directory not found: %s", injected_dir)
            sys.exit(1)
        logger.info("Using injected data directory: %s", injected_dir)
        # Override datadir to point to injected directory (where FITS files are)
        ffd_obj.datadir = injected_dir
        # Override klipdir to point to klip_fm_files subdirectory in injected directory
        ffd_obj.klipdir = os.path.join(injected_dir, "klip_fm_files")
        os.makedirs(ffd_obj.klipdir, exist_ok=True)
        logger.info("Output will be saved to: %s", ffd_obj.klipdir)
    main_datadir = ffd_obj.datadir
    # data_dir = os.join(os.environ["HOME"], "data", ffd_obj.params_file["BAND_DIR"])
    save_to_dir = args.log_to_file if args.log_to_file is not None else False

    if save_to_dir:
        save_dir = os.path.join(ffd_obj.datadir, "ff_logs")
        print(f"Writing logs to {save_dir}")
        log_path = configure_logging(args.log_to_file,
                                    "ff_klip",
                                    save_dir)
    else:
        log_path = configure_logging(args.log_to_file,
                                    "ff_klip",
                                    None)

    if log_path:
        print(f"Writing log to {log_path}")

    if not os.path.exists(args.param_file):
        print(f"Configuration file not found: {args.param_file}")
        sys.exit(1)


    logger.info("Initializing FreeFormDisk object with config: %s", config)
    
    
    # Define needed variables
    file_prefix = ffd_obj.file_prefix
    aligned_center = ffd_obj.aligned_center
    klipdir = ffd_obj.klipdir
    basis_path = os.path.join(klipdir, f"{file_prefix}_klbasis.h5")
    
    # Check if we should run KLIP reduction
    if ffd_obj.params_file["FIRST_TIME"] or force:
        if make_diskless_image:
            # Counter-rotated KLIP reduction on original dataset (negated parangs)
            ffd_obj.datadir = original_datadir
            ffd_obj.allocate_dataset()
            ffd_obj.par_angs = -ffd_obj.par_angs
            dataset, psflib = ffd_obj.prep_dataset()
            counter_reduced_data = ffd_obj.run_klip_reduction(dataset,
                                                            psflib=psflib)
            counter_path = os.path.join(klipdir, f"{file_prefix}-klipped_counter_rotated.fits")
            fits.writeto(counter_path, counter_reduced_data, overwrite=True)
            logger.info("Counter-rotated KLIP reduction complete.")
            logger.info("Diskless reduced data: %s", counter_path)

        # Prepare dataset and PSF library (for RDI if needed)
        ffd_obj.datadir = main_datadir
        ffd_obj.allocate_dataset()
        dataset, psflib = ffd_obj.prep_dataset()
        
        # Run KLIP reduction and create basis file (without forward modeling)
        
        ffd_obj.run_klip_reduction(dataset,
                                  psflib=psflib)
        
        logger.info("KLIP reduction complete.")
        logger.info("Basis file: %s", basis_path)
        logger.info("Reduced data: %s", os.path.join(klipdir, f"{file_prefix}-klipped-KLmodes-all.fits"))
        logger.info("Next step: run ff_setup to generate masks and noise map.")
    else:
        logger.info("Basis file already exists: %s", basis_path)
        logger.info("Set FIRST_TIME=True in config file or use --force to regenerate.")

    


if __name__ == "__main__":

    
    main()