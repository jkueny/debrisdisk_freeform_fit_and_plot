import json
from pathlib import Path
from datetime import datetime
from astropy.io import fits
import numpy as np
import jax
import jaxlib

def save_optimization_outputs(save_dir: str,
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

    def clean(obj):
        """
        Recursively convert NumPy or JAX arrays (or scalars) into
        JSON-friendly Python types.
        """
        if isinstance(obj, (np.ndarray, jax.Array,        # jax>=0.4
                            jax.lib.xla_client.ArrayImpl, # older alias
                            jaxlib.xla_extension.ArrayImpl)):
            arr = np.asarray(obj)           # pulls to host if it’s on GPU/TPU
            return arr.item() if arr.ndim == 0 else arr.tolist()

        if isinstance(obj, (np.generic, jax.core.Tracer)):
            return np.asarray(obj).item()

        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]

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
