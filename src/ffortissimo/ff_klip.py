'''
Run KLIP data reduction and create KLIP basis file for freeform disk fitting.

This script handles the first step of the freeform disk fitting pipeline:
1. Loads and prepares the dataset
2. Runs KLIP reduction (with optional forward modeling if model provided)
3. Creates and saves the KLIP basis file
4. Saves the reduced image

This script does NOT require:
- Binary masks (mask2generatedisk, mask4noisemap)
- Initial model (unless forward modeling is desired)

Usage:
    python ff_klip.py -p initialization_files/config.yaml
    python ff_klip.py -p initialization_files/config.yaml --initial-model path/to/model.fits
'''

import os
import sys
import argparse
import multiprocessing
from astropy.io import fits

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
        """
    )
    
    parser.add_argument('-p', '--param-file',
                        required=True,
                        help='Path to YAML parameter file')
    parser.add_argument('--initial-model',
                        type=str,
                        required=False,
                        help='Path to initial model FITS file (optional)')
    parser.add_argument('--force',
                        action='store_true',
                        help='Force regeneration even if basis file exists')
    parser.add_argument('--injected-dir',
                        type=str,
                        required=False,
                        help='Path to injected PSF library directory (optional)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.param_file):
        print(f"Error: Configuration file not found: {args.param_file}")
        sys.exit(1)
    config = args.param_file
    force = args.force
    init_model = args.initial_model if args.initial_model is not None else None
    injected_dir = args.injected_dir if args.injected_dir is not None else None

    # Initialize the freeform disk object
    print(f"Initializing FreeFormDisk object with config: {config}")
    ffd_obj = FreeFormDisk(config)
    
    # Define needed variables
    if injected_dir is not None:
        klipdir = os.path.join(injected_dir, "klip_fm_files")
    else:
        klipdir = ffd_obj.klipdir
    file_prefix = ffd_obj.file_prefix
    aligned_center = ffd_obj.aligned_center
    basis_path = os.path.join(klipdir, f"{file_prefix}_klbasis.h5")
    
    # Allocate dataset (load data files)
    ffd_obj.allocate_dataset()
    
    # Check if we should run KLIP reduction
    if ffd_obj.params_file["FIRST_TIME"] or force:
        # Prepare dataset and PSF library (for RDI if needed)
        dataset, psflib = ffd_obj.prep_dataset()
        
        # Run KLIP reduction and create basis file (without forward modeling)
        # If an initial model is provided, load it directly (no mask required for pure KLIP)
        model_init = None
        if init_model is not None:
            if not os.path.exists(init_model):
                print(f"Warning: Initial model file not found: {init_model}")
                print("Proceeding with pure KLIP reduction (no forward modeling)")
            else:
                model_init = fits.getdata(init_model)
        
        ffd_obj.run_klip_reduction(dataset,
                                  model_init=model_init,
                                  psflib=psflib)
        
        print(f"\n✓ KLIP reduction complete!")
        print(f"  Basis file: {basis_path}")
        print(f"  Reduced data: {os.path.join(klipdir, f'{file_prefix}-klipped-KLmodes-all.fits')}")
        print("\nYou can now run the freeform fitting with diskfit_freeform.py")
    else:
        print(f"Basis file already exists: {basis_path}")
        print("Set FIRST_TIME=True in config file or use --force to regenerate.")

    


if __name__ == "__main__":

    
    main()