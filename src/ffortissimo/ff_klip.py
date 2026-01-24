'''
Run KLIP data reduction and create KLIP basis file.

This script handles the first step of the freeform disk fitting pipeline:
1. Loads and prepares the dataset
2. Runs KLIP reduction
3. Saves the Karhunen-Loeve basis to a HDF5 file.
4. Saves the KLIP-reduced image to a FITS file.

Usage:
    python ff_klip.py -p initialization_files/params.yaml
    python ff_klip.py -p initialization_files/params.yaml --injected-dir /path/to/injected_dataset

'''

import os
import sys
import argparse
import multiprocessing

from ffortissimo.modeling.disk_freeform import FreeFormDisk


def main():
    """
    Run KLIP data reduction and create basis file.
    
    This function accomplishes the first phase of the freeform fitting pipeline:
    - Initializes the FreeFormDisk object
    - Allocates dataset
    - Runs KLIP reduction (with optional forward modeling if model provided)
    - Creates and saves the KLIP basis file
    
    Note: This function does NOT require binary masks or initial models for
    pure KLIP reduction. If an initial model is provided, it will be used
    for forward modeling, but it's optional.
    
    Args:
        config (str): Path to YAML configuration file
        init_model (str, optional): Path to initial model FITS file (optional)
        force (bool): Force regeneration even if basis file exists
    
    Returns:
        None
    """
    multiprocessing.set_start_method('forkserver')
    
    parser = argparse.ArgumentParser(
        description='Run KLIP data reduction and create basis file for freeform disk fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python ff_klip.py -p initialization_files/config.yaml
  
  # With custom initial model
  python ff_klip.py -p initialization_files/config.yaml --initial-model path/to/model.fits
  
  # Force regeneration of existing basis
  python ff_klip.py -p initialization_files/config.yaml --force
  
  # Process injected synthetic dataset
  python ff_klip.py -p initialization_files/config.yaml --injected-dir /path/to/injected_data
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
    
    args = parser.parse_args()
    
    if not os.path.exists(args.param_file):
        print(f"Error: Configuration file not found: {args.param_file}")
        sys.exit(1)
    config = args.param_file
    force = args.force
    injected_dir = args.injected_dir if args.injected_dir is not None else None

    # Initialize the freeform disk object
    print(f"Initializing FreeFormDisk object with config: {config}")
    ffd_obj = FreeFormDisk(config)
    
    # Override data and output directories if injected directory is provided
    if injected_dir is not None:
        if not os.path.exists(injected_dir):
            print(f"Error: Injected directory not found: {injected_dir}")
            sys.exit(1)
        print(f"Using injected data directory: {injected_dir}")
        # Override datadir to point to injected directory (where FITS files are)
        ffd_obj.datadir = injected_dir
        # Override klipdir to point to klip_fm_files subdirectory in injected directory
        ffd_obj.klipdir = os.path.join(injected_dir, "klip_fm_files")
        os.makedirs(ffd_obj.klipdir, exist_ok=True)
        print(f"Output will be saved to: {ffd_obj.klipdir}")
    
    # Define needed variables
    file_prefix = ffd_obj.file_prefix
    aligned_center = ffd_obj.aligned_center
    klipdir = ffd_obj.klipdir
    basis_path = os.path.join(klipdir, f"{file_prefix}_klbasis.h5")
    
    # Allocate dataset (load data files)
    ffd_obj.allocate_dataset()
    
    # Check if we should run KLIP reduction
    if ffd_obj.params_file["FIRST_TIME"] or force:
        # Prepare dataset and PSF library (for RDI if needed)
        dataset, psflib = ffd_obj.prep_dataset()
        
        # Run KLIP reduction and create basis file (without forward modeling)
        
        ffd_obj.run_klip_reduction(dataset,
                                  psflib=psflib)
        
        print(f"\n✦ KLIP reduction complete! ✦")
        print(f"  Basis file: {basis_path}")
        print(f"  Reduced data: {os.path.join(klipdir, f'{file_prefix}-klipped-KLmodes-all.fits')}")
        print("\nYou can now run ff_setup to generate the masks and noise map.")
    else:
        print(f"Basis file already exists: {basis_path}")
        print("Set FIRST_TIME=True in config file or use --force to regenerate.")

    


if __name__ == "__main__":

    
    main()