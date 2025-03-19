import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from functools import partial

def bilinear_interpolate(image, i_coords, j_coords):
    """
    Bilinear interpolation on a 2D image.
    """
    H, W = image.shape
    i0 = jnp.floor(i_coords).astype(jnp.int32)
    j0 = jnp.floor(j_coords).astype(jnp.int32)
    i1 = i0 + 1
    j1 = jnp.clip(j0 + 1, 0, W - 1)
    i0 = jnp.clip(i0, 0, H - 1)
    i1 = jnp.clip(i1, 0, H - 1)
    j0 = jnp.clip(j0, 0, W - 1)
    j1 = jnp.clip(j1, 0, W - 1)
    Ia = image[i0, j0]
    Ib = image[i0, j1]
    Ic = image[i1, j0]
    Id = image[i1, j1]
    wa = (i1 - i_coords) * (j1 - j_coords)
    wb = (i1 - i_coords) * (j_coords - j0)
    wc = (i_coords - i0) * (j1 - j_coords)
    wd = (i_coords - i0) * (j_coords - j0)
    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def rotate_image(image: jnp.ndarray, angle_deg: float) -> jnp.ndarray:
    """
    Rotate a 2D image by a given angle (in degrees, CCW positive) about a specified center,
    using a fully differentiable procedure based on map_coordinates for bilinear interpolation.

    Args:
        image: 2D JAX array representing the image.
        angle_deg: Rotation angle in degrees (counter-clockwise positive).
        center: Tuple (cy, cx) representing the center of rotation (row, col).

    Returns:
        A rotated 2D JAX array.
    """
    H, W = image.shape
    cy, cx = (image.shape[0] - 1) / 2, (image.shape[1] - 1) / 2,

    # Create coordinate grid for the output image.
    i, j = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    i = i.astype(jnp.float32)
    j = j.astype(jnp.float32)

    # Shift coordinates so that the rotation center is at the origin.
    i_centered = i - cy
    j_centered = j - cx

    # Convert the rotation angle to radians and compute the inverse rotation.
    theta = -jnp.deg2rad(angle_deg)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    # Compute the input coordinates corresponding to each output pixel via inverse rotation.
    j_in = j_centered * cos_theta - i_centered * sin_theta + cx
    i_in = j_centered * sin_theta + i_centered * cos_theta + cy

    # Use map_coordinates for bilinear interpolation (order=1), which is differentiable.
    rotated = map_coordinates(image, [i_in, j_in], order=1, mode='constant', cval=0.0)

    return rotated