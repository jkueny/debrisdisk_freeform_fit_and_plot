'''
Inject a synthetic disk model into a library of FITS images.

This script creates a synthetic dataset by:
1. Reading disk model parameters from YAML configuration file
2. Creating a base disk model with user-specified PA offset
3. Rotating the disk model for each image based on conjugate PARANG values
4. Adding the rotated disk to each image
5. Saving modified images to a new directory

The original PARANG values are multiplied by -1 to counter-rotate the images,
which allows the real disk signal to be "medianed out" during ADI reduction.

Usage:
    python build_synthetic_dataset.py -p initialization_files/config.yaml -o output_dir
    python build_synthetic_dataset.py -p initialization_files/config.yaml -o output_dir --pa-offset 5.0
'''

import os
import sys
import glob
import argparse
import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt
from ffortissimo.utils.io.yaml_handling import read_config
from ffortissimo.modeling.numba_models.hg_disk import fastmodgen_disk_dxdy_2g
from ffortissimo.dev.pyklip.klip import rotate


def get_basedir():
    """Get the base directory for data files."""
    basedir = os.environ.get("DISKFIT_BASEDIR", f'{os.environ["HOME"]}/data')
    return basedir


def get_disk_params(params, use_best=True):
    """
    Extract disk parameters from YAML config, preferring _best over _init.
    
    Args:
        params: Dictionary from YAML config
        use_best: If True, prefer _best parameters, otherwise use _init
    
    Returns:
        dict: Dictionary of disk parameters
    """
    disk_params = {}
    
    # Parameter mapping: (best_key, init_key, default)
    param_map = [
        ('pa', 'pa_best', 'pa_init', None),
        ('r1', 'r1_best', 'r1_init', None),
        ('r2', 'r2_best', 'r2_init', None),
        ('rc', 'rc_best', 'rc_init', None),
        ('alpha_in', 'alpha_in_best', 'alpha_in_init', None),
        ('alpha_out', 'alpha_out_best', 'alpha_out_init', None),
        ('beta', 'beta_best', 'beta_init', None),
        ('a_r', 'a_r_best', 'a_r_init', None),
        ('inc', 'inc_best', 'inc_init', None),
        ('dx', 'dx_best', 'dx_init', None),
        ('dy', 'dy_best', 'dy_init', None),
        ('Norm', 'N_best', 'N_init', None),
        ('g1', 'g1_best', 'g1_init', None),
        ('g2', 'g2_best', 'g2_init', None),
        ('alpha1', 'alpha1_best', 'alpha1_init', None),
    ]
    
    for param_name, best_key, init_key, default in param_map:
        if use_best and best_key in params:
            disk_params[param_name] = params[best_key]
        elif init_key in params:
            disk_params[param_name] = params[init_key]
        elif default is not None:
            disk_params[param_name] = default
        else:
            raise ValueError(f"Missing required parameter: {best_key} or {init_key}")
    
    return disk_params


def main():
    """
    Main function to inject synthetic disk into FITS images.
    """
    parser = argparse.ArgumentParser(
        description='Inject synthetic disk model into FITS images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python build_synthetic_dataset.py -p initialization_files/config.yaml -o synthetic_data
  
  # With PA offset
  python build_synthetic_dataset.py -p initialization_files/config.yaml -o synthetic_data --pa-offset 5.0
  
  # Override data directory
  python build_synthetic_dataset.py -p initialization_files/config.yaml -o synthetic_data --data-dir /path/to/data
        """
    )
    
    parser.add_argument('-p', '--param-file',
                        type=str,
                        required=True,
                        help='Path to YAML parameter file')
    parser.add_argument('-o', '--output-dir',
                        type=str,
                        required=True,
                        help='Output directory for synthetic dataset')
    parser.add_argument('--pa-offset',
                        type=float,
                        default=90.0,
                        help='Position angle offset in degrees (default: 90.0)')
    parser.add_argument('--data-dir',
                        type=str,
                        required=False,
                        help='Override data directory from YAML (uses BAND_DIR if not provided)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.param_file):
        print(f"Error: Parameter file not found: {args.param_file}")
        sys.exit(1)
    
    # Read YAML configuration
    print(f"Reading configuration from: {args.param_file}")
    params = read_config(args.param_file)
    
    # Get data directory
    if args.data_dir:
        data_dir = args.data_dir
    elif "BAND_DIR" in params:
        basedir = get_basedir()
        data_dir = os.path.join(basedir, params["BAND_DIR"])
    else:
        print("Error: Data directory not specified. Provide --data-dir or BAND_DIR in YAML.")
        sys.exit(1)
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Get metadata
    distance = params.get("DISTANCE_STAR")
    pixscale = params.get("PIXSCALE_INS")
    aligned_center = params.get("ALIGNED_CENTER")
    file_prefix = params.get("FILE_PREFIX")
    
    if distance is None or pixscale is None or aligned_center is None or file_prefix is None:
        print("Error: Missing required metadata (DISTANCE_STAR, PIXSCALE_INS, ALIGNED_CENTER, or FILE_PREFIX)")
        sys.exit(1)
    
    # Load instrument PSF
    print("\nLoading instrument PSF...")
    basedir = get_basedir()
    if "BAND_DIR" in params:
        klipdir = os.path.join(basedir, params["BAND_DIR"], "klip_fm_files")
    else:
        # Try to construct klipdir from data_dir
        klipdir = os.path.join(os.path.dirname(data_dir), "klip_fm_files")
    
    psf_path = os.path.join(klipdir, f"{file_prefix}_instrPSF.fits")
    
    if not os.path.exists(psf_path):
        print(f"Error: PSF file not found: {psf_path}")
        print("Please ensure the instrument PSF file exists in the klip_fm_files directory.")
        sys.exit(1)
    
    psf = fits.getdata(psf_path)
    psf = psf / np.sum(psf)  # Normalize PSF
    print(f"  ✓ PSF loaded: {psf.shape}, normalized")
    
    # Get disk parameters
    print("Extracting disk parameters from YAML...")
    disk_params = get_disk_params(params, use_best=True)
    
    # Apply PA offset
    base_pa = disk_params['pa'] + args.pa_offset
    print(f"Base disk PA: {disk_params['pa']}° + offset {args.pa_offset}° = {base_pa}°")
    
    # Find FITS files
    print(f"\nSearching for FITS files in: {data_dir}")
    file_pattern = os.path.join(data_dir, "camsci*.fits")
    fits_files = sorted(glob.glob(file_pattern))
    
    if len(fits_files) == 0:
        print(f"Error: No FITS files found matching pattern: {file_pattern}")
        sys.exit(1)
    
    print(f"Found {len(fits_files)} FITS files")
    
    # Load first image to get dimensions
    print("\nLoading first image to determine dimensions...")
    with fits.open(fits_files[0]) as hdul:
        first_image = hdul[0].data
        first_header = hdul[0].header
    
    if len(first_image.shape) > 2:
        first_image = np.squeeze(first_image)
    
    image_shape = first_image.shape
    print(f"Image dimensions: {image_shape}")
    
    # Use image dimensions for coordinate arrays (assume square or use max dimension)
    # The disk model will be created at the larger dimension, then we'll crop/resize if needed
    image_size = max(image_shape)
    
    # Set up coordinate arrays for disk model
    max_fov = image_size / 2. * pixscale  # Maximum radial distance in arcsec
    n_pts = image_size
    xsize = max_fov * distance  # Maximum radial distance in AU
    
    y_arr = np.linspace(-xsize, xsize, num=n_pts)
    z_arr = np.linspace(-xsize, xsize, num=n_pts)
    
    # Create mask (all zeros for full disk - no masking)
    mask = np.zeros((n_pts, n_pts), dtype=bool)
    
    # Create base disk model
    print(f"\nCreating base disk model...")
    print(f"  R1: {disk_params['r1']} AU")
    print(f"  R2: {disk_params['r2']} AU")
    print(f"  Inclination: {disk_params['inc']}°")
    print(f"  PA: {base_pa}°")
    
    # Note: beta is set to 1.0 in the render function, but we'll use beta from params
    # Actually, looking at the code, beta is set to 1.0 and then rc, m, n are used
    beta = 1.0  # As per the render_initial_disk_model function
    
    base_disk = fastmodgen_disk_dxdy_2g(
        R1=disk_params['r1'],
        R2=disk_params['r2'],
        beta=beta,
        inc=disk_params['inc'],
        pa=base_pa,
        dx=disk_params['dx'],
        dy=disk_params['dy'],
        Norm=disk_params['Norm'],
        g1=disk_params['g1'],
        g2=disk_params['g2'],
        alpha1=disk_params['alpha1'],
        a_r=disk_params['a_r'],
        Rc=disk_params['rc'],
        m=disk_params['alpha_in'],
        n=disk_params['alpha_out'],
        y_arr=y_arr,
        z_arr=z_arr,
        npts=n_pts,
        mask=mask
    )
    
    # Zero out any NaNs in the base disk model
    nan_count = np.sum(np.isnan(base_disk))
    if nan_count > 0:
        print(f"  Warning: Found {nan_count} NaNs in base disk model, zeroing them out")
        base_disk = np.nan_to_num(base_disk, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Crop or pad disk model to match image dimensions exactly
    if base_disk.shape != image_shape:
        from scipy.ndimage import zoom
        if base_disk.shape[0] == base_disk.shape[1] and image_shape[0] == image_shape[1]:
            # Both square, simple zoom
            zoom_factor = image_shape[0] / base_disk.shape[0]
            base_disk = zoom(base_disk, zoom_factor, order=1)
        else:
            # Different aspect ratios, zoom each dimension separately
            zoom_factors = (image_shape[0] / base_disk.shape[0], 
                           image_shape[1] / base_disk.shape[1])
            base_disk = zoom(base_disk, zoom_factors, order=1)
        print(f"  Resized disk model from {base_disk.shape} to {image_shape}")
    
    # Ensure exact match
    if base_disk.shape != image_shape:
        # If still not matching, crop or pad
        if base_disk.shape[0] >= image_shape[0] and base_disk.shape[1] >= image_shape[1]:
            # Crop
            y_start = (base_disk.shape[0] - image_shape[0]) // 2
            x_start = (base_disk.shape[1] - image_shape[1]) // 2
            base_disk = base_disk[y_start:y_start+image_shape[0], x_start:x_start+image_shape[1]]
        else:
            # Pad with zeros
            pad_y = (image_shape[0] - base_disk.shape[0]) // 2
            pad_x = (image_shape[1] - base_disk.shape[1]) // 2
            base_disk = np.pad(base_disk, ((pad_y, image_shape[0] - base_disk.shape[0] - pad_y),
                                          (pad_x, image_shape[1] - base_disk.shape[1] - pad_x)),
                              mode='constant', constant_values=0)
        print(f"  Adjusted disk model to match image: {base_disk.shape}")
    
    # Final NaN check after resizing/cropping
    nan_count = np.sum(np.isnan(base_disk))
    if nan_count > 0:
        print(f"  Warning: Found {nan_count} NaNs after resizing, zeroing them out")
        base_disk = np.nan_to_num(base_disk, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  ✓ Base disk model created: {base_disk.shape}")
    
    # Convolve base disk model with instrument PSF
    print(f"\nConvolving disk model with instrument PSF...")
    base_disk_convolved = fftconvolve(base_disk, psf, mode="same")
    
    # Zero out any NaNs after convolution
    nan_count_conv = np.sum(np.isnan(base_disk_convolved))
    if nan_count_conv > 0:
        print(f"  Warning: Found {nan_count_conv} NaNs after convolution, zeroing them out")
        base_disk_convolved = np.nan_to_num(base_disk_convolved, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"  ✓ Disk model convolved with PSF: {base_disk_convolved.shape}")
    
    # Use convolved model for rotation and injection
    base_disk = base_disk_convolved
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory: {output_dir}")
    klipdir = os.path.join(output_dir, "klip_fm_files")
    os.makedirs(klipdir, exist_ok=True)
    # Save the disk model to be injected
    disk_model_path = os.path.join(klipdir, "injected_disk_model.fits")
    fits.writeto(disk_model_path, base_disk, overwrite=True)
    print(f"  Saved disk model to: {disk_model_path}")
    
    # Process each image
    print(f"\nProcessing {len(fits_files)} images...")
    for idx, fits_file in enumerate(fits_files):
        filename = os.path.basename(fits_file)
        # print(f"  [{idx+1}/{len(fits_files)}] Processing {filename}...", end=' ')
        
        # Read image and header
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data.copy()
            header = hdul[0].header.copy()
        
        # Squeeze if needed
        if len(image_data.shape) > 2:
            image_data = np.squeeze(image_data)
        
        # Zero out any NaNs in the original image data
        nan_count_img = np.sum(np.isnan(image_data))
        if nan_count_img > 0:
            print(f"\n    Warning: Found {nan_count_img} NaNs in original image, zeroing them out")
            image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get PARANG value
        if 'PARANG' not in header:
            print(f"\n    Warning: PARANG not found in header, skipping {filename}")
            continue
        
        parang = float(header['PARANG'])
        
        # Calculate rotation angle: base_pa - PARANG (conjugate PARANG)
        rotation_angle = base_pa - parang - 90.0
        # print(f"PARANG={parang:.2f}°, rotation={rotation_angle:.2f}°", end=' ')
        
        # Rotate disk model
        rotated_disk = rotate(
            img=base_disk,
            angle=-rotation_angle,
            center=aligned_center,
            new_center=None,
            flipx=False
        )

        # #debug print and display the rotated disk
        # print(f"Rotated angle: {rotation_angle}")
        # print(f"base_pa: {base_pa}")
        # print(f"parang: {parang}")
        # plt.imshow(rotated_disk, origin='lower')
        # plt.colorbar()
        # plt.show()
        # sys.exit()

        
        # Zero out any NaNs in the rotated disk model before adding to image
        nan_count_rot = np.sum(np.isnan(rotated_disk))
        if nan_count_rot > 0:
            # print(f"\n    Warning: Found {nan_count_rot} NaNs in rotated disk, zeroing them out")
            rotated_disk = np.nan_to_num(rotated_disk, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Add rotated disk to image
        image_data = image_data + rotated_disk
        
        # Final check: ensure no NaNs in the final result
        nan_count_final = np.sum(np.isnan(image_data))
        if nan_count_final > 0:
            print(f"\n    Warning: Found {nan_count_final} NaNs in final image, zeroing them out")
            image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Update PARANG header to -PARANG (counter-rotate for median subtraction)
        header['PARANG'] = -parang
        
        # Save to output directory
        output_path = os.path.join(output_dir, filename)
        fits.writeto(output_path, image_data, header, overwrite=True)
        
    
    print(f"\n✓ Synthetic dataset created successfully!")
    print(f"  Output directory: {output_dir}")
    print(f"  Number of images processed: {len(fits_files)}")
    print(f"  Base disk PA: {base_pa}°")
    print(f"  PARANG values have been negated in headers for counter-rotation")


if __name__ == "__main__":
    main()
