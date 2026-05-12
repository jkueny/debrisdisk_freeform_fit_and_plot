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
from scipy.ndimage import shift
import matplotlib.pyplot as plt
from ffortissimo.io.yaml_handling import read_config
from ffortissimo.modeling.numba_models.hg_disk import fastmodgen_disk_dxdy_2g
from ffortissimo.dev.pyklip.klip import rotate_image



def get_disk_params(params):
    """
    Extract disk parameters from YAML config.

    Args:
        params: Dictionary from YAML config

    Returns:
        dict: Dictionary of disk parameters
    """
    disk_params = {}

    # Parameter mapping: (best_key, init_key, default)
    param_map = [
        ('pa', 'pa_test'),
        ('r1', 'r1_test'),
        ('r2', 'r2_test'),
        ('rc', 'rc_test'),
        ('alpha_in', 'alpha_in_test'),
        ('alpha_out', 'alpha_out_test'),
        ('beta', 'beta_test'),
        ('a_r', 'a_r_test'),
        ('inc', 'inc_test'),
        ('dx', 'dx_test'),
        ('dy', 'dy_test'),
        ('Norm', 'N_test'),
        ('g1', 'g1_test'),
        ('g2', 'g2_test'),
        ('alpha1', 'alpha1_test'),
        ('r_inner', 'r_inner'),
        ('r_outer', 'r_outer'),
    ]

    for param_name, test_key in param_map:
        if test_key in params:
            disk_params[param_name] = params[test_key]
        else:
            raise ValueError(f"Missing required parameter: {test_key}")

    return disk_params

def generate_hi_res_features(
    base_disk:np.ndarray, instr_psf: np.ndarray,
    feature_PA:float, feature_radius:int) -> np.ndarray:
    """
    Add a small high-res feature to the base disk model.
    Add it at the specified position angle and radius from the center of the image.

    TODO new strategy: add 2 bright pixels by the ansae.
    - Place the delta functions at the specified radius
    - Convolve them with the PSF
    - Then do the rotation and add it to the base disk image

    Args:
        base_disk: The base disk model
        feature_PA: The position angle of the feature
        feature_radius: The radius of the feature in pixels
    Returns:
        The base disk model with the feature added
    """
    # Make the empty array that will hold the features
    blank_feature = np.zeros_like(base_disk)
    pixel_value = np.max(base_disk) * 3
    feature_PA += 90. #manual offset for the rotation function

    # Place the delta functions at the specified radius
    Y,X = np.indices(base_disk.shape)
    im_center = [int(base_disk.shape[0]/2), int(base_disk.shape[1]/2)]
    blank_feature[im_center[0], im_center[1] - 1] = pixel_value
    blank_feature[im_center[0], im_center[1] + 1] = pixel_value
    feature_convolved = fftconvolve(blank_feature, instr_psf, mode="same")
    feature1_conv_shift = shift(feature_convolved, (0,feature_radius))
    feature2_conv_shift = shift(feature_convolved, (0,-feature_radius))
    feature_conv_shift = feature1_conv_shift + feature2_conv_shift
    feature_conv_shift_rot = rotate_image(
        img=feature_conv_shift, angle=feature_PA, center=im_center)

    # plt.imshow(feature_conv_shift_rot, origin="lower")
    # plt.show()
    # exit()

    return feature_conv_shift_rot


def generate_hi_res_ellipses(
    pixscale: float,
    distance: float,
    image_shape: tuple,
    center: tuple,
    semi_major_au: float,
    inc_deg: float,
    pa_deg: float,
    value: float
    spacing_px: float = 5.0,
) -> np.ndarray:
    """
    Generate three nested 1-pixel-width ellipses with specified spacing.

    Args:
        image_shape: Shape of the output image (ny, nx)
        center: Center of the ellipses in (y, x) pixel coordinates
        semi_major_au: Semi-major axis of the middle ellipse in au
        inc_deg: Inclination in degrees (sets axis ratio via cos(i))
        pa_deg: Position angle in degrees (rotation of ellipse major axis)
        value: Constant pixel value for the ellipse ring

    Returns:
        Image with three nested ellipses
    """
    semi_major_px = semi_major_au / pixscale / distance
    ellipse_image = np.zeros(image_shape, dtype=float)
    base_ellpse = np.zeros(image_shape, dtype=float)
    y_idx, x_idx = np.indices(image_shape)
    y = y_idx - center[0]
    x = x_idx - center[1]

    theta = np.deg2rad(pa_deg + 90.)
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    cosi = np.cos(np.deg2rad(inc_deg))
    for ea, semi_major in enumerate((semi_major_px - spacing_px, semi_major_px, semi_major_px + spacing_px)):
        if semi_major <= 0:
            continue
        empty_image = np.zeros(image_shape, dtype=float)
        semi_minor = max(1.0, semi_major * cosi)
        r_scaled = np.sqrt((x_rot / semi_major) ** 2 + (y_rot / semi_minor) ** 2)

        ring_mask = np.abs(r_scaled - 1.) <= (0.75 / semi_major)
        empty_image[ring_mask] = value
        if ea == 1:
            base_ellpse = empty_image
        ellipse_image += empty_image

    return ellipse_image, base_ellpse


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
  python inject_fake_disk.py -p initialization_files/config.yaml -o synthetic_data

  # With PA offset
  python inject_fake_disk.py -p initialization_files/config.yaml -o synthetic_data --pa 115.0

  # Override data directory
  python inject_fake_disk.py -p initialization_files/config.yaml -o synthetic_data --data-dir /path/to/data
        """
    )

    parser.add_argument('-p', '--param-file',
                        type=str,
                        required=True,
                        help='Path to YAML parameter file')
    parser.add_argument('-o', '--injected-dir',
                        type=str,
                        required=True,
                        help='Output directory for synthetic dataset')
    parser.add_argument('--pa',
                        type=float,
                        default=115.0,
                        help='Injected disk position angle \
                             in degrees (default: 115.0)')
    parser.add_argument('--rad',
                        type=float,
                        default=75,
                        help='Radial dist. of injected hi-res \
                             features in pixels (default: 75). In \
                             --hires-ellipses mode this is the \
                             semi-major axis of the middle ellipse.')
    parser.add_argument('--hires',
                        action='store_true',
                        help='Use three nested 1-pixel ellipses for the \
                             high-res feature test (opt-in)')
    parser.add_argument('--data-dir',
                        type=str,
                        required=False,
                        help='Override data directory from YAML \
                             (uses BAND_DIR if not provided)')

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
    disk_params = get_disk_params(params)

    # Apply PA offset
    # base_pa = disk_params['pa'] + args.pa_offset
    if args.pa is None:
        base_pa = disk_params['pa']
    else:
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

    print(f"  Semi-major axis in pixels: {disk_params['rc']/distance/pixscale}")

    # Create output directory
    output_dir = args.injected_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput dir: {output_dir}")
    klipdir = os.path.join(output_dir, "klip_fm_files")
    os.makedirs(klipdir, exist_ok=True)
    print("Hi-res ellipse mode enabled (no disk model injection)")
    if args.hires:
        initial_disk, initial_base = generate_hi_res_ellipses(
            pixscale=pixscale,
            distance=distance,
            image_shape=image_shape,
            center=(aligned_center[0], aligned_center[1]),
            semi_major_au=disk_params["rc"],
            inc_deg=disk_params['inc'],
            pa_deg=base_pa,
            value=disk_params['Norm'] / 4
        )
    else:
        base_ellipse = None
        initial_disk = fastmodgen_disk_dxdy_2g(
            R1=disk_params['r_inner'],
            R2=disk_params['r_outer'],
            # beta=disk_params['beta'],
            beta=1.0,
            inc=disk_params['inc'],
            # pa=base_pa,
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
    initial_disk_convolved = fftconvolve(initial_disk, psf, mode="same")
    # Save the base disk model and image to a FITS file
    initial_disk_convolved_path = os.path.join(klipdir, "injected_disk_image.fits")
    fits.writeto(initial_disk_convolved_path, initial_disk_convolved, overwrite=True)
    initial_disk_path = os.path.join(klipdir, "disk_model_to_inject.fits")
    fits.writeto(initial_disk_path, initial_disk, overwrite=True)
    print(f"  Saved initial disk model to: {initial_disk_path}")
    if args.hires:
        # Save the base ellipse to a FITS file
        if initial_base is not None:
            base_ellipse_path = os.path.join(klipdir, "base_ellipse_for_reference.fits")
            fits.writeto(base_ellipse_path, initial_base, overwrite=True)
            print(f"  Saved base ellipse to: {base_ellipse_path}")


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
        # print(f"base_pa: {base_pa}, parang: {parang}, rotation_angle: {rotation_angle}")

        # # Rotate disk model
        # rotated_disk = rotate(
        #     img=base_disk_convolved,
        #     angle=-rotation_angle,
        #     center=aligned_center,
        #     new_center=None,
        #     flipx=False
        # )

        if args.hires:
            base_disk, base_ellipse = generate_hi_res_ellipses(
                image_shape=image_shape,
                center=(aligned_center[0], aligned_center[1]),
                semi_major_au=disk_params["rc"],
                pixscale=pixscale,
                distance=distance,
                inc_deg=disk_params['inc'],
                pa_deg=rotation_angle,
                value=disk_params['Norm'] / 4
            )
        else:
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


        # Convolve base model with instrument PSF
        base_disk_convolved = fftconvolve(base_disk, psf, mode="same")
        base_disk_image_add_feats = base_disk_convolved
        # # debug view the base disk model
        # if ea < 2:
        #     plt.imshow(base_disk_image_add_feats, origin='lower')
        #     plt.colorbar()
        #     plt.show()
        # if ea > len(fits_files) - 2:
        #     plt.imshow(base_disk_image_add_feats, origin='lower')
        #     plt.colorbar()
        #     plt.show()
        # # sys.exit()

        # check for NaNs after convolution
        nan_count_conv = np.sum(np.isnan(base_disk_image_add_feats))
        if nan_count_conv > 0:
            raise ValueError(f"Found {nan_count_conv} NaNs after convolution, aborting...")

        # #debug print and display the rotated disk
        # print(f"Rotated angle: {rotation_angle}")
        # print(f"base_pa: {base_pa}")
        # print(f"parang: {parang}")
        # plt.imshow(rotated_disk, origin='lower')
        # plt.colorbar()
        # plt.show()
        # sys.exit()


        # Add conv disk model to image
        image_data = image_data + base_disk_image_add_feats

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
