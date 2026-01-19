'''
Create masks and estimate noise map for freeform disk fitting.

This script handles mask creation and noise estimation, which are critical steps
that may need to be iterated to get the right mask shapes. The script:
1. Creates binary masks for disk generation and noise mapping
2. Loads the KLIP-reduced data
3. Estimates spatial noise map from masked reduced data

Usage:
    python ff_mask.py -p initialization_files/config.yaml
    python ff_mask.py -p initialization_files/config.yaml --force
'''

import os
import sys
import argparse
import numpy as np
import astropy.io.fits as fits

from modeling.disk_freeform import FreeFormDisk
from utils.io.fits_handling import save_fits

default_parameter_file = 'HR4796a_z_lco2023a_magao-x_20230309_10.yaml'


def create_masks(ffd_obj, force=False):
    """
    Create all binary masks needed for disk fitting.
    
    Args:
        ffd_obj: FreeFormDisk object
        force (bool): Force regeneration even if masks exist
    
    Returns:
        dict: Dictionary containing all created masks
    """
    klipdir = ffd_obj.klipdir
    file_prefix = ffd_obj.file_prefix
    
    # Check if masks already exist
    mask_files = {
        'mask2generatedisk': os.path.join(klipdir, f"{file_prefix}_mask2generatedisk.fits"),
        'mask4noisemap': os.path.join(klipdir, f"{file_prefix}_mask4noisemap.fits"),
        'engineered_optimization': os.path.join(klipdir, f"{file_prefix}_engineered_optimization_map.fits"),
        'mask_out_of_bounds': os.path.join(klipdir, f"{file_prefix}_mask_out_of_bounds.fits"),
    }
    
    all_exist = all(os.path.exists(f) for f in mask_files.values())
    if all_exist and not force:
        print("Masks already exist. Use --force to regenerate.")
        masks = {}
        for key, path in mask_files.items():
            masks[key] = fits.getdata(path)
        return masks
    
    print("\n[1/2] Creating binary masks...")
    print(f"   Disk generation mask parameters:")
    print(f"     Inner scaling: {ffd_obj.params_file['MASK_IN_SCALING']}")
    print(f"     Outer scaling: {ffd_obj.params_file['MASK_OUT_SCALING']}")
    print(f"     Noise inner scaling: {ffd_obj.params_file['MASK_NOISE_IN']}")
    print(f"     Noise outer scaling: {ffd_obj.params_file['MASK_NOISE_OUT']}")
    print(f"     Mask center offset: ({ffd_obj.params_file['MASK_DX']}, {ffd_obj.params_file['MASK_DY']})")
    
    optimization_mask = ffd_obj.prep_binary_masks()
    
    # Read mask_out_of_bounds from disk (it's saved by prep_binary_masks)
    mask_out_of_bounds = fits.getdata(mask_files['mask_out_of_bounds'])
    
    masks = {
        'mask2generatedisk': ffd_obj.mask2generatedisk,
        'mask4noisemap': ffd_obj.mask4noisemap,
        'engineered_optimization': optimization_mask,
        'mask_out_of_bounds': mask_out_of_bounds,
    }
    
    print(f"   ✓ Masks created and saved to {klipdir}")
    print(f"     - mask2generatedisk: {np.sum(masks['mask2generatedisk'])} pixels")
    print(f"     - mask4noisemap: {np.sum(masks['mask4noisemap'])} pixels")
    
    return masks


def estimate_noise_map(ffd_obj, masks, force=False):
    """
    Estimate spatial noise map from KLIP-reduced data.
    
    Args:
        ffd_obj: FreeFormDisk object
        masks (dict): Dictionary of masks
        force (bool): Force regeneration even if noise map exists
    
    Returns:
        np.ndarray: Noise map
    """
    klipdir = ffd_obj.klipdir
    file_prefix = ffd_obj.file_prefix
    noise_map_path = os.path.join(klipdir, f"{file_prefix}_noisemap.fits")
    reduced_data_path = os.path.join(klipdir, f"{file_prefix}-klipped-KLmodes-all.fits")
    
    # Check if noise map exists
    if os.path.exists(noise_map_path) and not force:
        print("\n[2/2] Noise map already exists. Use --force to regenerate.")
        return fits.getdata(noise_map_path)
    
    # Check if reduced data exists
    if not os.path.exists(reduced_data_path):
        raise FileNotFoundError(
            f"Reduced data not found: {reduced_data_path}\n"
            "Please run ff_klip.py first to create the KLIP-reduced data."
        )
    
    print("\n[2/2] Estimating noise map from reduced data...")
    
    # Load reduced data
    reduced_data = fits.getdata(reduced_data_path)
    reduced_data = np.squeeze(reduced_data)
    reduced_data[reduced_data != reduced_data] = 0.  # Zero out NaNs
    
    # Create engineered noise mask for ADI mode
    if ffd_obj.mode == "ADI":
        print("   ADI mode: Using engineered noise mask")
        bespoke_noise_mask = ffd_obj._engineer_disk_mask(
            masks['mask4noisemap'], 
            angle_sweep_factor=4
        )
        reduced_noise_masked = reduced_data * (1 - bespoke_noise_mask)
        tosave_reduced_noise_masked = reduced_data * bespoke_noise_mask
        
        # Save the engineered noise mask
        bespoke_noise_path = os.path.join(klipdir, f"{file_prefix}_mask4noisemap.fits")
        save_fits(bespoke_noise_path, bespoke_noise_mask)
    else:
        print("   RDI mode: Using standard noise mask")
        reduced_noise_masked = reduced_data * (1 - masks['mask4noisemap'])
        tosave_reduced_noise_masked = reduced_data * masks['mask4noisemap']
    
    # Get noise delta radii parameter
    delta_radii = ffd_obj.params_file.get("NOISE_DELTA_RADII", 1)
    print(f"   Using ring width: {delta_radii} pixels")
    
    # Estimate noise map
    noise_map = ffd_obj.make_noise_map_rings(
        reduced_data_no_disk=reduced_noise_masked,
        delta_radii=delta_radii
    )
    
    # Save noise map
    fits.writeto(noise_map_path, noise_map, overwrite=True)
    
    # Save masked data for inspection
    masked_data_path = os.path.join(klipdir, f"{file_prefix}_masked_data.fits")
    masked_noise_path = os.path.join(klipdir, f"{file_prefix}_use4noisemap.fits")
    save_fits(masked_data_path, reduced_data * masks['mask2generatedisk'])
    save_fits(masked_noise_path, tosave_reduced_noise_masked)
    
    print(f"   ✓ Noise map created and saved to {noise_map_path}")
    print(f"     Noise map statistics:")
    print(f"       Mean: {np.nanmean(noise_map):.6f}")
    print(f"       Median: {np.nanmedian(noise_map):.6f}")
    print(f"       Std: {np.nanstd(noise_map):.6f}")
    print(f"       Min: {np.nanmin(noise_map):.6f}")
    print(f"       Max: {np.nanmax(noise_map):.6f}")
    
    return noise_map


def main(config, force=False):
    """
    Main function to create masks and estimate noise.
    
    Args:
        config (str): Path to YAML configuration file
        force (bool): Force regeneration of existing files
    """
    # Initialize the freeform disk object
    ffd_obj = FreeFormDisk(config)
    
    klipdir = ffd_obj.klipdir
    file_prefix = ffd_obj.file_prefix
    
    print(f"Creating masks and noise map for {file_prefix}")
    print(f"Output directory: {klipdir}")
    print(f"Mode: {ffd_obj.mode}")
    
    # Step 1: Create masks
    masks = create_masks(ffd_obj, force=force)
    
    # Step 2: Estimate noise map
    noise_map = estimate_noise_map(ffd_obj, masks, force=force)
    
    print("\n✓ Mask creation and noise estimation complete!")
    print(f"\nNext steps:")
    print(f"  1. Inspect masks in: {klipdir}")
    print(f"  2. Adjust mask parameters in YAML if needed and re-run")
    print(f"  3. Run freeform fitting with: diskfit_freeform.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create masks and estimate noise map for freeform disk fitting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python ff_mask.py -p initialization_files/config.yaml
  
  # Force regeneration
  python ff_mask.py -p initialization_files/config.yaml --force
        """
    )
    
    parser.add_argument('-p', '--param-file',
                        required=False,
                        help='Path to YAML parameter file')
    parser.add_argument('--force',
                        action='store_true',
                        help='Force regeneration even if files exist')
    
    args = parser.parse_args()
    
    if args.param_file is None:
        str_yaml = f'initialization_files/{default_parameter_file}'
    else:
        str_yaml = args.param_file
    
    if not os.path.exists(str_yaml):
        print(f"Error: Configuration file not found: {str_yaml}")
        sys.exit(1)
    
    main(
        config=str_yaml,
        force=args.force
    )

