'''
Run KLIP data reduction and create KLIP basis file for freeform disk fitting.

This script handles the first step of the freeform disk fitting pipeline:
1. Loads and prepares the dataset
2. Creates binary masks for disk generation and noise mapping
3. Runs KLIP reduction with forward modeling
4. Creates and saves the KLIP basis file
5. Generates noise maps and other necessary files

Usage:
    python ff_klip.py -p initialization_files/config.yaml
    python ff_klip.py -p initialization_files/config.yaml --initial-model path/to/model.fits
'''

import os
import sys
import argparse
import multiprocessing

from modeling.disk_freeform import FreeFormDisk

default_parameter_file = 'HR4796a_z_lco2023a_magao-x_20230309_10.yaml'


def run_klip_reduction(config, init_model=None, force=False):
    """
    Run KLIP data reduction and create basis file.
    
    Args:
        config (str): Path to YAML configuration file
        init_model (str, optional): Path to initial model FITS file
        force (bool): Force regeneration even if basis file exists
    
    Returns:
        None
    """
    # Initialize the freeform disk object
    ffd_obj = FreeFormDisk(config)
    
    klipdir = ffd_obj.klipdir
    file_prefix = ffd_obj.file_prefix
    basis_path = os.path.join(klipdir, f"{file_prefix}_klbasis.h5")
    
    # Check if basis file already exists
    if os.path.exists(basis_path) and not force:
        print(f"Basis file already exists: {basis_path}")
        print("Use --force to regenerate it.")
        return
    
    print(f"Starting KLIP reduction for {file_prefix}")
    print(f"Output directory: {klipdir}")
    
    # Step 1: Allocate dataset (load data files)
    print("\n[1/5] Loading dataset...")
    ffd_obj.allocate_dataset()
    print(f"   Loaded {len(ffd_obj.filelist)} files")
    
    # Step 2: Prepare binary masks
    print("\n[2/5] Creating binary masks...")
    optimization_mask = ffd_obj.prep_binary_masks()
    print(f"   Masks saved to {klipdir}")
    
    # Step 3: Get initial model
    print("\n[3/5] Preparing initial model...")
    model_firstguess = ffd_obj.get_initial_model(init_model)
    print(f"   Initial model shape: {model_firstguess.shape}")
    
    # Step 4: Prepare dataset and PSF library (for RDI if needed)
    print("\n[4/5] Preparing dataset and PSF library...")
    dataset, psflib = ffd_obj.prep_dataset()
    if psflib is not None:
        print("   RDI mode: PSF library prepared")
    else:
        print("   ADI mode: No PSF library needed")
    
    # Step 5: Run KLIP reduction and create basis
    print("\n[5/5] Running KLIP reduction and creating basis file...")
    print("   This may take a while...")
    ffd_obj.initialize_diskfm(dataset,
                              model_init=model_firstguess,
                              psflib=psflib)
    
    print(f"\n✓ KLIP reduction complete!")
    print(f"  Basis file: {basis_path}")
    print(f"  Reduced data: {os.path.join(klipdir, f'{file_prefix}-klipped-KLmodes-all.fits')}")
    print(f"  Noise map: {os.path.join(klipdir, f'{file_prefix}_noisemap.fits')}")
    print("\nYou can now run the freeform fitting with diskfit_freeform.py")


if __name__ == "__main__":
    multiprocessing.set_start_method('forkserver')
    
    parser = argparse.ArgumentParser(
        description='Run KLIP data reduction and create basis file for freeform disk fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default config
  python ff_klip.py -p initialization_files/config.yaml
  
  # With custom initial model
  python ff_klip.py -p initialization_files/config.yaml --initial-model path/to/model.fits
  
  # Force regeneration of existing basis
  python ff_klip.py -p initialization_files/config.yaml --force
        """
    )
    
    parser.add_argument('-p', '--param-file',
                        required=False,
                        help='Path to YAML parameter file')
    parser.add_argument('--initial-model',
                        type=str,
                        required=False,
                        help='Path to initial model FITS file (optional)')
    parser.add_argument('--force',
                        action='store_true',
                        help='Force regeneration even if basis file exists')
    
    args = parser.parse_args()
    
    if args.param_file is None:
        str_yaml = f'initialization_files/{default_parameter_file}'
    else:
        str_yaml = args.param_file
    
    if not os.path.exists(str_yaml):
        print(f"Error: Configuration file not found: {str_yaml}")
        sys.exit(1)
    
    run_klip_reduction(
        config=str_yaml,
        init_model=args.initial_model,
        force=args.force
    )