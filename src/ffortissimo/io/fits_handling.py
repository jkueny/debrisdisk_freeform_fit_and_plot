from astropy.io import fits
import jax.numpy as jnp
import numpy as np
import os

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

def load_pyklip_reduced_data(reduced_data_path):
    reduced_data = fits.getdata(reduced_data_path)
    reduced_data = np.squeeze(reduced_data)
    reduced_data[reduced_data != reduced_data] = 0.  # Zero out NaNs
    return reduced_data