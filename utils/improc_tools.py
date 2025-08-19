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

def reconstruct_full_image(free_params, total_pixels, size, mask_indices):
    """
    Given the free parameters (for the unmasked region) and the full image shape,
    create a full image (flattened) where the free parameters are inserted at the positions
    indicated by mask_indices and zeros elsewhere.
    """
    full_shape = (size, size)
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

def median_radial_profile(image, center, noise, radial_inds):
    """Measure the median radial profile with respect to the image center.

    Args:
        image (jnp.Array): 2D square image.
    Returns:
        median_radial_profile (jnp.Array): 2D radial profile image.
    """

    max_distance = np.hypot(center[1], center[0])

    radii = jnp.arange(max_distance + 1)

    thresholded_image = jnp.where(image > (noise * 3), image, 0.) #tested, works 20250814 JKK
    
    def median_at_radius(r):
        mask = radial_inds == r
        ring_vals = jnp.where(mask, thresholded_image, jnp.nan)
        # jax.debug.print("nanmedian ring_vals -> {x}", x=jnp.nanmedian(ring_vals))
        return jnp.nanmedian(ring_vals)

    median_profile_1d = jax.vmap(median_at_radius)(radii)
    median_profile_1d_no_nan = jnp.where(jnp.isnan(median_profile_1d), 0., median_profile_1d)
    # jax.debug.print("med prof 1D -> {x}", x=median_profile_1d_no_nan)

    # Cast the profile to a 2D array
    median_profile_image = median_profile_1d_no_nan[radial_inds]

    # jax.debug.print("median_profile_image sum -> {x}", x=jnp.sum(median_profile_image))

    return median_profile_image

def subtract_radial_profile(image, center, noise, radial_inds):

    median_profile_image = median_radial_profile(image, center, noise, radial_inds)
    # jax.debug.print("before subtracting, sum -> {x}", x=jnp.sum(median_profile_image))
    subtracted = image - median_profile_image

    return subtracted, median_profile_image


    