import numpy as np
import jax
import jax.numpy as jnp

def fft_power_spectrum(image):
    """
    Compute the 2D power spectrum of an image (shifted so DC is at center).
    """
    fft = jnp.fft.fftshift(jnp.fft.fft2(image))
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

def get_radial_inds(image_shape, center):
    y, x = image_shape
    yy = jnp.arange(y)
    xx = jnp.arange(x)
    c_y, c_x = center

    ygrid, xgrid = jnp.meshgrid(yy, xx, indexing="ij")    

    rad = jnp.sqrt((xgrid - c_x) ** 2 + (ygrid - c_y)**2 )

    return jnp.array(rad).astype(jnp.int32)

def median_radial_profile(image, center, radial_inds):
    """Measure the median radial profile with respect to the image center.

    Args:
        image (jnp.Array): 2D square image.
    Returns:
        median_radial_profile (jnp.Array): 2D radial profile image.
    """

    max_distance = np.hypot(center[1], center[0])

    radii = jnp.arange(max_distance + 1)
    
    def median_at_radius(r):
        mask = radial_inds == r
        # For JAX compatibility, we need to avoid boolean indexing
        # Instead, we'll use a weighted approach where we set non-ring pixels to NaN
        ring_vals = jnp.where(mask, image, jnp.nan)
        
        # Use nanmedian to compute median ignoring NaN values
        return jnp.nanmedian(ring_vals)

    median_profile_1d = jax.vmap(median_at_radius)(radii)

    # Cast the profile to a 2D array
    median_profile_image = median_profile_1d[radial_inds]

    return median_profile_image

def subtract_radial_profile(image, center, radial_inds):

    median_profile_image = median_radial_profile(image, center, radial_inds)

    subtracted = image - median_profile_image

    return subtracted, median_profile_image


    