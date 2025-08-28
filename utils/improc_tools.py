import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy import fft

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
    full_flat = full_flat.at[mask_indices].set(free_params)
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

def high_pass_filter(img, filtersize):
    """
    A FFT implmentation of high pass filter.

    Args:
        img: a 2D image
        filtersize: size in Fourier space of the size of the space. In image space, size=img_size/filtersize

    Returns:
        filtered: the filtered image
    """

    transform = jnp.fft.fft2(img)

    # coordinate system in FFT image
    u,v = jnp.meshgrid(jnp.fft.fftfreq(transform.shape[1]), jnp.fft.fftfreq(transform.shape[0]))
    # scale u,v so it has units of pixels in FFT space
    rho = jnp.sqrt((u*transform.shape[1])**2 + (v*transform.shape[0])**2)
    # scale rho up so that it has units of pixels in FFT space
    # rho *= transform.shape[0]
    # create the filter
    filt = 1. - jnp.exp(-(rho**2/filtersize**2))

    filtered = jnp.real(jnp.fft.ifft2(transform*filt))


    return filtered

def calculate_radial_distances_np(image_shape, center=None):
    """
    This makes a 2D array with each value being the radial distance from the center.

    image_shape should be a (x,y) or [x,y]

    The center coord is optional.
    """
    if center is None:
        #default true center of the image if no center is provided
        center = ((image_shape[0] // 2) - 0.5, (image_shape[1] // 2) - 0.5)
    y, x = np.indices(image_shape)
    center_y, center_x = center
    return np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)

def median_radial_profile_np(image, center=None):
    """
    Calculates the median radial profile of the input 2D image.

    image should be a 2D numpy array
    """
    distances = calculate_radial_distances_np(image.shape, center)
    radial_distances = np.round(distances).astype(int)

    #Grab the maximum radial distance
    max_distance = np.max(radial_distances)

    #Calculate med for each distance
    median_profile = np.array([np.median(image[radial_distances == r]) for r in range(max_distance + 1)])

    return median_profile

def subtract_median_profile_np(image, center=None):
    """
    Subtract the median radial profile from an image.
    """
    if len(image.shape) > 2:
        image = np.squeeze(image)
    distances = calculate_radial_distances_np(image.shape, center)
    radial_distances = np.round(distances).astype(int)
    # print(radial_distances)

    # Compute the median radial profile
    median_profile = median_radial_profile_np(image, center)

    # Create a 2D array from the median profile based on radial distances
    median_image = median_profile[radial_distances]

    # Subtract the median profile from the original image
    subtracted_image = image - median_image

    return subtracted_image, median_image
    