from astropy.io import fits
import jax.numpy as jnp
import numpy as np

def save_fits(save_to, image_arr):
    if not isinstance(image_arr, np.ndarray):
        if isinstance(image_arr, jnp.ndarray):
            image_arr_saveable = np.asarray(image_arr)
        else:
            raise TypeError("In save_fits() -> input array is not an array!")
    elif isinstance(image_arr, np.ndarray):
        image_arr_saveable = image_arr
    if not isinstance(save_to, str):
        raise TypeError("In save_fits() -> save to path is not a path string!")

    if save_to.endswith(".fits"):
        fits.writeto(save_to, image_arr_saveable, overwrite=True)
    else:
        fits.writeto(f"{save_to}.fits", image_arr_saveable, overwrite=True)
        
    return True    