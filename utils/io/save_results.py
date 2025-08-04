import json
from pathlib import Path
from datetime import datetime
from astropy.io import fits
import numpy as np
import types

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
                        loss_history: list,
                        hsf_regularization: float,
                        optimizer_name: str = "Optax Adam",
                        pyklip_params: dict = None):
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
        optimizer_name (str, optional): Chosen optimizer. Defaults to "Optax Adam".
        pyklip_params (dict, optional): Chosen pyklip parameters for target image.

    Returns:
        str: path to the save directory.
    """
    import matplotlib.pyplot as plt
    




    fits.writeto(run_dir / f"{file_prefix}_BestModel.fits", model_opt, overwrite=True)
    fits.writeto(run_dir / f"{file_prefix}_BestModel_Conv.fits", model_image_opt, overwrite=True)
    fits.writeto(run_dir / f"{file_prefix}_FM.fits", forward_model_opt, overwrite=True)
    fits.writeto(run_dir / f"{file_prefix}_Res.fits", residuals_image, overwrite=True)

    metadata = {
        "timestamp_utc": str(datetime.now()),
        "loss": float(loss_history[-1]),
        "lambda_reg": float(hsf_regularization),
        "optimizer": optimizer_name,
        "pyklip_params": pyklip_params or "unknown",
        # "param_file": "bestfit_params.json",
        # "model_file": "wdh_model.fits",
        # "forward_model_file": "forward_model.fits"
    }

    plt.plot(loss_history, linewidth=3, alpha=0.8)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.grid()
    plt.savefig(f"{run_dir}/loss_history.png")
    plt.savefig(f"{run_dir}/loss_history.pdf")

    with open(run_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Done; optimization results saved to: {run_dir}")
    return run_dir
