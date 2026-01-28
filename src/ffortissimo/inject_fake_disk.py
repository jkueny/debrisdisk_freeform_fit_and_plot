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
from ffortissimo.io.yaml_handling import read_config
from ffortissimo.modeling.numba_models.hg_disk import fastmodgen_disk_dxdy_2g
from ffortissimo.dev.pyklip.klip import rotate



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
        ('r_inner', 'r_inner', None, None),
        ('r_outer', 'r_outer', None, None),
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

def add_hi_res_features(base_disk:np.ndarray, feature_PA:float, feature_radius:int) -> np.ndarray:
    """
    Add a small high-res feature to the base disk model.
    Add it at the specified position angle and radius from the center of the image.

    TODO new strategy: add a single bright pixel by the ansae.

    Args:
        base_disk: The base disk model
        feature_PA: The position angle of the feature
        feature_radius: The radius of the feature in pixels
    Returns:
        The base disk model with the feature added
    """

    def get_feature_coordinates(feature_PA, feature_radius):
        PA_rad = np.radians(feature_PA + 90)
        Y,X = np.indices(base_disk.shape)
        im_center = [int(base_disk.shape[0]/2), int(base_disk.shape[1]/2)]
        distance_from_center = np.sqrt((X - im_center[0])**2 + (Y - im_center[1])**2)
        feature_x1 = round(im_center[0] + feature_radius * np.sin(PA_rad))
        feature_y1 = round(im_center[1] + feature_radius * np.cos(PA_rad))
        feature_x2 = round(im_center[0] - feature_radius * np.sin(PA_rad))
        feature_y2 = round(im_center[1] - feature_radius * np.cos(PA_rad))
        return feature_x1, feature_y1, feature_x2, feature_y2
    PA_rad = np.radians(feature_PA + 90)
    Y,X = np.indices(base_disk.shape)
    print(f"Base disk shape: {base_disk.shape}")
    im_center = [int(base_disk.shape[0]/2), int(base_disk.shape[1]/2)]
    distance_from_center = np.sqrt((X - im_center[0])**2 + (Y - im_center[1])**2)
    blank_feature = np.zeros_like(base_disk)
    pixel_value = np.max(base_disk)
    # # Given the feature_PA and feature_radius, calculate the coordinates of the feature
    distance_offsets = [0, 2] #let's try 2 pixels side-by-side
    for d in distance_offsets:
        coords = get_feature_coordinates(feature_PA, feature_radius + d)
        feature_x1 = coords[0]
        feature_y1 = coords[1]
        feature_x2 = coords[2]
        feature_y2 = coords[3]
        blank_feature[feature_x1, feature_y1] = pixel_value
        blank_feature[feature_x2, feature_y2] = pixel_value
    base_disk += blank_feature
    return base_disk
    

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
    parser.add_argument('--pa',
                        type=float,
                        default=115.0,
                        help='Injected disk position angle in degrees (default: 115.0)')
    parser.add_argument('--data-dir',
                        type=str,
                        required=False,
                        help='Override data directory from YAML (uses BAND_DIR if not provided)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.param_file):
        raise FileNotFoundError(f"Parameter file not found: {args.param_file}")
    
    # Read YAML configuration
    print(f"Reading configuration from: {args.param_file}")
    params = read_config(args.param_file)
    
    # Get data directory
    if args.data_dir:
        data_dir = args.data_dir
    elif "BAND_DIR" in params:
        basedir = os.environ.get("DISKFIT_BASEDIR", f'{os.environ["HOME"]}/data')
        data_dir = os.path.join(basedir, params["BAND_DIR"])
    else:
        print("Error: Data directory not specified. Provide --data-dir or BAND_DIR in YAML.")
        sys.exit(1)
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
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
    basedir = os.environ.get("DISKFIT_BASEDIR", f'{os.environ["HOME"]}/data')
    if "BAND_DIR" in params:
        klipdir = os.path.join(basedir, params["BAND_DIR"], "klip_fm_files")
    else:
        # Try to construct klipdir from data_dir
        klipdir = os.path.join(os.path.dirname(data_dir), "klip_fm_files")
        print(f"INFO: BAND_DIR not found in params, using klip_fm_files from data_dir: {klipdir}")
    
    psf_path = os.path.join(klipdir, f"{file_prefix}_instrPSF.fits")
    
    if not os.path.exists(psf_path):
        print("Please add the instrument PSF file to the klip_fm_files directory.")
        raise FileNotFoundError(f"PSF file not found: {psf_path}")
    
    psf = fits.getdata(psf_path)
    psf = psf / np.sum(psf)  # Normalize PSF
    print(f"PSF loaded: {psf.shape}, L1-normalized")
    
    # Get disk parameters
    print("Extracting disk parameters from YAML...")
    disk_params = get_disk_params(params, use_best=True)
    
    # Apply PA offset
    # base_pa = disk_params['pa'] + args.pa_offset
    base_pa = args.pa
    print(f"Base disk PA: {disk_params['pa']} + offset {args.pa} = {base_pa} deg")
    
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
    print(f"  Inclination: {disk_params['inc']} deg")
    print(f"  PA: {base_pa} deg")
    print(f"  alpha_in: {disk_params['alpha_in']}")
    print(f"  alpha_out: {disk_params['alpha_out']}")
    
    
    
    
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput dir: {output_dir}")
    klipdir = os.path.join(output_dir, "klip_fm_files")
    os.makedirs(klipdir, exist_ok=True)
    
    # Process each image
    # We need to generate the disk model + disk image per image to avoid interpolation artifacts
    print(f"\nProcessing {len(fits_files)} images...")
    for ea, fits_file in enumerate(fits_files):
        filename = os.path.basename(fits_file)
        
        # Read in image and header
        with fits.open(fits_file) as hdul:
            image_data = hdul[0].data.copy()
            header = hdul[0].header.copy()
        
        # Check image dimensions
        if len(image_data.shape) > 2:
            raise ValueError(f"Image {filename} has more than 2 dimensions!")
        
        # Zero out any NaNs in the original image data
        nan_count_img = np.sum(np.isnan(image_data))
        if nan_count_img > 0:
            print(f"\n    Warning: Found {nan_count_img} NaNs in original image, zeroing them out")
            image_data = np.nan_to_num(image_data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get PARANG value
        if 'PARANG' not in header:
            raise ValueError(f"PARANG not found in header for {filename}")


        parang = float(header['PARANG'])
        
        # Calculate rotation angle: base_pa - PARANG (conjugate PARANG) - 90.0 for N up E left
        rotation_angle = base_pa + parang #- 90.0
        print(f"base_pa: {base_pa}, parang: {parang}, rotation_angle: {rotation_angle}")
        
        # # Rotate disk model
        # rotated_disk = rotate(
        #     img=base_disk_convolved,
        #     angle=-rotation_angle,
        #     center=aligned_center,
        #     new_center=None,
        #     flipx=False
        # )

        base_disk = fastmodgen_disk_dxdy_2g(
            R1=disk_params['r_inner'],
            R2=disk_params['r_outer'],
            # beta=disk_params['beta'],
            beta=1.0,
            inc=disk_params['inc'],
            # pa=base_pa,
            pa=rotation_angle,
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

        # # Add hi-res feature to the base disk model
        # base_disk = add_hi_res_features(base_disk, rotation_angle, int(85))
        # print(f"Rotation angle: {rotation_angle}")

        # # debug view the base disk model
        # plt.imshow(base_disk, origin='lower')
        # plt.colorbar()
        # plt.show()
        # sys.exit()
        
        # Zero out any NaNs in the base disk model
        nan_count = np.sum(np.isnan(base_disk))
        if nan_count > 0:
            print(f"  Warning: Found {nan_count} NaNs in base disk model, zeroing them out")
            base_disk = np.nan_to_num(base_disk, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Crop or pad disk model to match image dimensions exactly
        if base_disk.shape != image_shape:
            raise ValueError(f"Base disk model shape {base_disk.shape} does not match image shape {image_shape}")
        
        
        # Final NaN check after resizing/cropping
        nan_count = np.sum(np.isnan(base_disk))
        if nan_count > 0:
            print(f"  Warning: Found {nan_count} NaNs after resizing, zeroing them out")
            base_disk = np.nan_to_num(base_disk, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        # Convolve base disk model with instrument PSF
        base_disk_convolved = fftconvolve(base_disk, psf, mode="same")
        
        # check for NaNs after convolution
        nan_count_conv = np.sum(np.isnan(base_disk_convolved))
        if nan_count_conv > 0:
            raise ValueError(f"Found {nan_count_conv} NaNs after convolution, aborting...")
        
        if ea == 0:
            print(f"Base disk model created: {base_disk.shape}")
            # Save the base disk model to a FITS file
            base_disk_path = os.path.join(klipdir, "disk_model_to_inject.fits")
            fits.writeto(base_disk_path, base_disk, overwrite=True)
            # Save the disk model to be injected
            disk_model_path = os.path.join(klipdir, "injected_disk_image.fits")
            fits.writeto(disk_model_path, base_disk_convolved, overwrite=True)
            print(f"  Saved disk model to: {disk_model_path}")
        # #debug print and display the rotated disk
        # print(f"Rotated angle: {rotation_angle}")
        # print(f"base_pa: {base_pa}")
        # print(f"parang: {parang}")
        # plt.imshow(rotated_disk, origin='lower')
        # plt.colorbar()
        # plt.show()
        # sys.exit()

        
        # Add conv disk model to image
        image_data = image_data + base_disk_convolved
        
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
        
    
    print(f"\nDisk injection successful...!")
    print(f"  Output dir: {output_dir}")
    print(f"  Num. images processed: {len(fits_files)}")
    print(f"  Injected disk PA: {base_pa}°")
    print(f"  PARANG values have been set to -PARANG in headers for counter-rotation....")


if __name__ == "__main__":
    main()
