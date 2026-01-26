import os

import numpy as np
import astropy.io.fits as fits
import jax.numpy as jnp

from ffortissimo.utils.klip_basis import load_kl_basis
from ffortissimo.utils.masks import make_annular_mask
from ffortissimo.utils.improc_tools import get_radial_inds


def load_diskfit_components(ffd_obj, init_model=None):
    """
    Load data products and build arrays needed for optimization.

    Args:
        ffd_obj: FreeFormDisk object
        init_model (str, optional): Path to starting model FITS file

    Returns:
        dict: Loaded data and derived arrays for optimization
    """
    klipdir = ffd_obj.klipdir
    file_prefix = ffd_obj.file_prefix
    basis_path = os.path.join(klipdir, f"{file_prefix}_klbasis.h5")

    # Load PSF
    psf = ffd_obj.psf
    jax_psf = jnp.array(psf)

    # Load all required data
    print("\n[1/6] Loading required data...")
    fm_dict = load_kl_basis(basis_path)
    mask4noisemap = fits.getdata(os.path.join(klipdir, f"{file_prefix}_mask4noisemap.fits"))
    mask2generatedisk = fits.getdata(os.path.join(klipdir, f"{file_prefix}_mask2generatedisk.fits"))
    # Set mask on object (needed for get_initial_model later)
    ffd_obj.mask2generatedisk = mask2generatedisk
    reduced_data = fits.getdata(os.path.join(klipdir, f"{file_prefix}-klipped-KLmodes-all.fits"))
    reduced_data = np.squeeze(reduced_data)
    reduced_data[reduced_data != reduced_data] = 0.  # Zero out NaNs
    print("   ✓ Basis, masks, and reduced data loaded")

    # Load noise map
    print("\n[2/6] Loading noise map...")
    if ffd_obj.params_file["USE_NOISE"]:
        noise_map = fits.getdata(os.path.join(klipdir, f"{file_prefix}_noisemap.fits"))
        noise_map += 1.  # Get rid of any zeros
        noise_map_flat = noise_map.flatten()
        noise_map_flat[noise_map_flat != noise_map_flat] = 1.
    else:
        noise_map = np.ones_like(reduced_data)
        noise_map_flat = noise_map.flatten()
        noise_map_flat[noise_map_flat != noise_map_flat] = 1.
    print("   ✓ Noise map loaded")

    # Set processing flags
    print("\n[3/6] Setting processing flags...")
    hp = ffd_obj.hp
    rprofsub = bool(ffd_obj.params_file["RPROFSUB"])
    clean_final_fm = bool(ffd_obj.clean_final_fm)

    if bool(hp):
        hp_filtersize = (psf.shape[0]/hp) / (2*np.sqrt(2*np.log(2)))
    else:
        hp_filtersize = None
    if rprofsub:
        do_radial_profile_sub = 1
    else:
        do_radial_profile_sub = 0
    if clean_final_fm:
        do_clean_final_fm = 1
    else:
        do_clean_final_fm = 0
    print(f"   ✓ High-pass filter: {hp_filtersize}")
    print(f"   ✓ Radial profile subtraction: {do_radial_profile_sub}")
    print(f"   ✓ Clean final FM: {do_clean_final_fm}")

    # Get reference model
    print("\n[4/6] Preparing reference model...")

    # Load optimization mask from disk (created by ff_setup.py)
    optimization_mask_path = os.path.join(klipdir, f"{file_prefix}_optimization_mask.fits")
    if not os.path.exists(optimization_mask_path):
        raise FileNotFoundError(
            f"Mask file not found: {optimization_mask_path}\n"
            "Please run ff_setup.py first to create the masks."
        )
    optimization_mask = fits.getdata(optimization_mask_path)

    ffd_obj.allocate_dataset()
    model_firstguess = ffd_obj.get_initial_model(init_model)

    if init_model is None:
        reference_model = ffd_obj.fit_reference_model(noise_map, reduced_data)
        reference_model[reference_model != reference_model] = 0.
        # reference_model_psd, window_opt = ffd_obj.get_reference_model_psd(reference_model)
        print("   ✓ New reference model fitted and saved")
        print("   Note: Inspect the reference model and re-run optimization if needed.")
    # elif init_model is None:
    #     # Try to load FirstModel, otherwise use initial guess
    #     first_model_path = os.path.join(klipdir, f"{file_prefix}_FirstModel.fits")
    #     if os.path.exists(first_model_path):
    #         reference_model = fits.getdata(first_model_path)
    #         reference_model[reference_model != reference_model] = 0.
    #         print(f"   ✓ Loaded reference model from {first_model_path}")
    #     else:
    #         reference_model = model_firstguess
    #         print("   ✓ Using initial guess model as reference")
    else:
        reference_model = model_firstguess
        print("   ✓ Using provided initial model as reference")

    reference_model_psd, window_opt = ffd_obj.get_reference_model_psd(reference_model)
    print("   ✓ Reference model PSD computed")

    # Prepare data arrays and indices
    print("\n[5/6] Preparing data arrays and indices...")
    total_pixels = np.prod(reduced_data.shape)
    disk_mask = np.array(mask2generatedisk)
    optimization_mask = np.array(optimization_mask)
    annular_mask = make_annular_mask(disk_mask.shape, 10, ffd_obj.owa)
    disk_mask *= annular_mask

    disk_mask_indices = jnp.flatnonzero(jnp.array(disk_mask))
    optimization_mask_indices = jnp.flatnonzero(jnp.array(optimization_mask))

    reduced_data_flat = reduced_data.flatten()
    reduced_flat_interest = reduced_data_flat[optimization_mask_indices]

    model_firstguess *= disk_mask
    init_model_jax = jnp.array(model_firstguess)
    init_model_flat = init_model_jax.reshape(init_model_jax.shape[0] * init_model_jax.shape[1])
    init_model_interest = init_model_flat[disk_mask_indices]

    noise_interest = noise_map_flat[optimization_mask_indices]

    aligned_center = (
        float(fm_dict["klparam_dict"]["aligned_center_x"]),
        float(fm_dict["klparam_dict"]["aligned_center_y"]),
    )
    image_shape = (jnp.round(aligned_center[0]) * 2, jnp.round(aligned_center[1]) * 2)
    radial_inds = get_radial_inds(image_shape, aligned_center)
    print("   ✓ Data arrays and indices prepared")

    return {
        "psf": psf,
        "jax_psf": jax_psf,
        "fm_dict": fm_dict,
        "mask4noisemap": mask4noisemap,
        "mask2generatedisk": mask2generatedisk,
        "reduced_data": reduced_data,
        "noise_map": noise_map,
        "noise_map_flat": noise_map_flat,
        "hp_filtersize": hp_filtersize,
        "do_radial_profile_sub": do_radial_profile_sub,
        "do_clean_final_fm": do_clean_final_fm,
        "reference_model_psd": reference_model_psd,
        "window_opt": window_opt,
        "total_pixels": total_pixels,
        "disk_mask": disk_mask, #mask2generatedisk * annulus
        "optimization_mask": optimization_mask,
        "disk_mask_indices": disk_mask_indices,
        "optimization_mask_indices": optimization_mask_indices,
        "reduced_flat_interest": reduced_flat_interest,
        "init_model_interest": init_model_interest,
        "noise_interest": noise_interest,
        "aligned_center": aligned_center,
        "radial_inds": radial_inds,
    }
