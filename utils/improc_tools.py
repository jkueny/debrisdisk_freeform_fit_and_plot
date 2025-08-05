import numpy as np
import jax.numpy as jnp

def fft_power_spectrum(image):
    """
    Compute the 2D power spectrum of an image (shifted so DC is at center).
    """
    fft = jnp.fft.fftshift(jnp.fft.rfft2(image))
    power = jnp.abs(fft) ** 2
    return power #TODO the sum of this should be the variance of the mean-subbed image (Parseval's theorem)

def reconstruct_full_image(free_params, total_pixels, mask_indices):
    """
    Given the free parameters (for the unmasked region) and the full image shape,
    create a full image (flattened) where the free parameters are inserted at the positions
    indicated by mask_indices and zeros elsewhere.
    """
    full_shape = (int(np.sqrt(total_pixels)), int(np.sqrt(total_pixels)))
    full_flat = jnp.zeros(total_pixels)
    full_flat = full_flat.at[mask_indices].set(jnp.abs(free_params))
    return full_flat.reshape(full_shape)