import json
from pathlib import Path
from datetime import datetime
from astropy.io import fits
import numpy as np
import types

from ffortissimo.utils.improc_tools import reconstruct_full_image, \
    subtract_radial_profile, get_radial_inds, high_pass_filter
import jax.numpy as jnp
from jax.scipy.signal import fftconvolve


def save_wdhfit_outputs(save_dir: str,
                        file_prefix: str,
                        params_dict: dict,
                        model_image: np.ndarray,
                        forward_model_image: np.ndarray,
                        residuals_image: np.ndarray,
                        loss_value: float,
                        optimizer_name: str = "Optax Adam",
                        model_version: str = None):
    '''
    Dumps relevant outputs from WDH optimization run into a JSON file.

    Parameters
    ----------
    run_dir_root : str
        Base directory where result folders are created (e.g. "results").
    params_dict : dict
        Nested dictionary of best-fit parameters.
    model_image : ndarray
        Optimized WDH model image.
    forward_model_image : ndarray
        Final forward-modeled image.
    loss_value : float
        Final loss achieved during optimization.
    optimizer_name : str
        Name of the optimizer used (default: "Optax AdamW").
    model_version : str, optional
        Optional version tag or hash for the model code used.
    '''
    

    run_root = Path(save_dir)
    run_root.mkdir(exist_ok=True)

    # auto-increment run number
    existing = [p for p in run_root.glob("opt_run_*") if p.is_dir()]
    run_id = 0
    if existing:
        nums = [int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()]
        if nums:
            run_id = max(nums) + 1
    run_dir = run_root / f"opt_run_{run_id:04d}"
    run_dir.mkdir()

    # Output params into JSON, need to convert JAX types first

    _JAX_ARRAY_TYPES = []                 # will become tuple later
    try:
        import jax
        # jax>=0.4 unified type
        if hasattr(jax, "Array"):
            _JAX_ARRAY_TYPES.append(jax.Array)
        # jax 0.3 / internal alias
        from jax.core import Tracer
    except ModuleNotFoundError:
        jax = None
        Tracer = types.SimpleNamespace()  # dummy placeholder

        # legacy aliases (<0.4 or still emitted by some C++ paths)
    for mod_path in (
            "jaxlib.xla_extension",
            "jax.lib.xla_client",
    ):
        try:
            mod = __import__(mod_path, fromlist=["ArrayImpl"])
            _JAX_ARRAY_TYPES.append(mod.ArrayImpl)
        except (ModuleNotFoundError, AttributeError):
            pass

    ARRAY_LIKE_TYPES = tuple([np.ndarray] + _JAX_ARRAY_TYPES)

    # ---------------------------------------------------------------------
    # 2.  Recursive cleaner
    # ---------------------------------------------------------------------
    def clean(obj):
        """
        Recursively convert NumPy/JAX arrays (and scalars) into
        JSON-serialisable Python types.
        """

        # ---- (a) arrays or things that can become arrays ----------------
        if isinstance(obj, ARRAY_LIKE_TYPES) or hasattr(obj, "__array__"):
            arr = np.asarray(obj)                 # pulls to host if needed
            return arr.item() if arr.ndim == 0 else arr.tolist()

        # ---- (b) NumPy/JAX scalar objects and tracers -------------------
        if isinstance(obj, (np.generic, Tracer)):
            return np.asarray(obj).item()

        # ---- (c) containers ---------------------------------------------
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple, set)):
            return [clean(v) for v in obj]

        # ---- (d) everything else is assumed JSON ready ------------------
        return obj
        
    clean_params_dict = clean(params_dict)

    with open(run_dir / "bestfit_params.json", "w") as f:
        json.dump(clean_params_dict, f, indent=2)


    fits.writeto(run_dir / f"{file_prefix}_BestModel.fits", model_image, overwrite=True)
    fits.writeto(run_dir / f"{file_prefix}_FM.fits", forward_model_image, overwrite=True)
    fits.writeto(run_dir / f"{file_prefix}_Res.fits", residuals_image, overwrite=True)

    metadata = {
        "timestamp_utc": str(datetime.now()),
        "loss": float(loss_value),
        "optimizer": optimizer_name,
        "model_version": model_version or "unknown",
        "param_file": "bestfit_params.json",
        "model_file": "wdh_model.fits",
        "forward_model_file": "forward_model.fits"
    }

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done; optimization results saved to: {run_dir}")
    return run_dir

def get_next_run_dir(save_dir: str):
    run_root = Path(save_dir)
    run_root.mkdir(exist_ok=True)

    # auto-increment run number
    existing = [p for p in run_root.glob("opt_run_*") if p.is_dir()]
    run_id = 0
    if existing:
        nums = [int(p.name.split("_")[-1]) for p in existing if p.name.split("_")[-1].isdigit()]
        if nums:
            run_id = max(nums) + 1
    run_dir = run_root / f"opt_run_{run_id:04d}"
    run_dir.mkdir()
    return run_dir

def save_ffdfit_outputs(run_dir: Path,
                        file_prefix: str,
                        model_opt: np.ndarray,
                        model_image_opt: np.ndarray,
                        forward_model_opt: np.ndarray,
                        residuals_image: np.ndarray,
                        residuals_roi: np.ndarray,
                        median_profile_image: np.ndarray,
                        loss_history: list,
                        hsf_regularization: float,
                        huber_delta: float,
                        opt_offset: float,
                        optimizer_name: str = "Optax Adam",
                        pyklip_params: dict = None,
                        weights_nominal: np.ndarray = None):
    """Saves output from the freeform diskfit optimization code.

    Args:
        save_dir (str): Location to save outputs.
        file_prefix (str): Prefix for saved FITS.
        model_opt (np.ndarray): Optimized disk model.
        model_image_opt (np.ndarray): Convolved optimized model.
        forward_model_opt (np.ndarray): Optimized FM.
        residuals_image (np.ndarray): KLIP data minus optimized FM.
        loss_history (list): loss vs. n_iterations.
        hsf_regularization (float): Chosen reg_lambda value for run.
        huber_delta (float): Chosen huber delta value for run.
        optimizer_name (str, optional): Chosen optimizer. Defaults to "Optax Adam".
        pyklip_params (dict, optional): Chosen pyklip parameters for target image.

    Returns:
        str: path to the save directory.
    """
    import matplotlib.pyplot as plt
    




    fits.writeto(run_dir / f"{file_prefix}_BestModel.fits", model_opt, overwrite=True)
    fits.writeto(run_dir / f"{file_prefix}_BestModel_Conv.fits", model_image_opt, overwrite=True)
    fits.writeto(run_dir / f"{file_prefix}_BestModel_FM.fits", forward_model_opt, overwrite=True)
    fits.writeto(run_dir / f"{file_prefix}_BestModel_Res.fits", residuals_image, overwrite=True)
    fits.writeto(run_dir / f"{file_prefix}_BestModel_Res_ROI.fits", residuals_roi, overwrite=True)
    fits.writeto(run_dir / f"{file_prefix}_MedProf.fits", median_profile_image, overwrite=True)
    fits.writeto(run_dir / f"{file_prefix}_Weights_Nominal.fits", weights_nominal, overwrite=True)
    metadata = {
        "timestamp_utc": str(datetime.now()),
        "loss": float(loss_history[-1]),
        "lambda_reg": float(hsf_regularization),
        "huber_delta": float(huber_delta),
        "opt_offset": float(opt_offset),
        "optimizer": optimizer_name,
        "pyklip_params": pyklip_params or "unknown",
        # "param_file": "bestfit_params.json",
        # "model_file": "wdh_model.fits",
        # "forward_model_file": "forward_model.fits"
    }
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(loss_history, linewidth=3, alpha=0.8)
    ax.set(xlabel="Iterations")
    ax.set(ylabel="Loss")
    ax.set(yscale='log')
    ax.grid()
    fig.savefig(f"{run_dir}/loss_history.png")
    fig.savefig(f"{run_dir}/loss_history.pdf")
    plt.close(fig)

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done; optimization results saved to: {run_dir}")
    return run_dir


def harness_optimized_model(optimized_params, opt_offset, ffd_obj, reduced_data, psf, total_pixels, disk_mask_indices, opt_mask_indices,
                            disk_mask, optimization_mask, aligned_center, radial_inds, do_radial_profile_sub, do_clean_final_fm, hp_filtersize,
                            noise_interest, weights_nominal):
    outputs_dict = {}
    print("Reconstructing full image...")
    optimized_model = np.asarray(reconstruct_full_image(optimized_params, total_pixels, disk_mask_indices))

    print("Convolving optimized model image...")
    opt_image_no_rprofsub = fftconvolve(optimized_model, psf, mode="same")

    noise_reconstructed = reconstruct_full_image(jnp.array(noise_interest),
                                                 total_pixels,
                                                 opt_mask_indices,
                                                )
    w_reconstructed = reconstruct_full_image(jnp.array(weights_nominal),
                                             total_pixels,
                                             opt_mask_indices,
                                            )
    if bool(do_radial_profile_sub):
        opt_model_image, med_prof_image = subtract_radial_profile(opt_image_no_rprofsub,
                                                              aligned_center,
                                                                noise_reconstructed,
                                                                radial_inds,
                                                                )
    else:
        _, med_prof_image = subtract_radial_profile(opt_image_no_rprofsub,
                                                    aligned_center,
                                                    noise_reconstructed,
                                                    radial_inds,
                                                    )
        opt_model_image = opt_image_no_rprofsub
    if bool(hp_filtersize):
        opt_model_image_hp = high_pass_filter(opt_model_image, filtersize=hp_filtersize)
    else:
        opt_model_image_hp = opt_model_image
    optimized_model_image = np.asarray(opt_model_image_hp)
    median_profile_image = np.asarray(med_prof_image)

    print("Generating the optimized forward model image...")
    optimized_fm = ffd_obj.single_fm(np.asarray(optimized_model_image))

    if bool(do_clean_final_fm):
        optimized_fm_rprofsub, _ = subtract_radial_profile(optimized_fm,
                                                          aligned_center,
                                                          noise_reconstructed,
                                                          radial_inds,
                                                          )
    else:
        optimized_fm_rprofsub = optimized_fm
    opt_offset = np.asarray(opt_offset)
    optimized_fm_rprofsub += opt_offset

    print("Calculating residuals between reduced data and optimized forward model...")
    residuals = np.asarray(reduced_data - optimized_fm_rprofsub)
    residuals_roi = residuals.copy() * optimization_mask
    residuals_roi[residuals_roi == 0.] = np.nan
    residuals_roi = residuals_roi / noise_reconstructed

    outputs_dict["optimized_model"] = np.asarray(optimized_model)
    outputs_dict["optimized_fm"] = np.asarray(optimized_fm_rprofsub)
    outputs_dict["optimized_model_image"] = np.asarray(optimized_model_image)
    outputs_dict["residuals_roi"] = np.asarray(residuals_roi)
    outputs_dict["median_profile_image"] = median_profile_image
    outputs_dict["residuals"] = np.asarray(residuals)
    outputs_dict["weights_nominal"] = np.asarray(w_reconstructed)
    return outputs_dict