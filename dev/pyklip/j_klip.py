import jax.numpy as jnp
import jax

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

def rotate_image(image: jnp.ndarray, angle_deg: float, center: tuple, flip_x: bool = True) -> jnp.ndarray:
    """
    Rotate a 2D image (JAX array) by a given angle about a specified center.
    
    Args:
        image: 2D jnp.array representing the image.
        angle_deg: Angle in degrees (CCW positive) by which to rotate.
        center: Tuple (cy, cx) representing the center of rotation.
        flip_x: If True, apply a horizontal flip after rotation.
        
    Returns:
        A 2D jnp.array of the rotated (and optionally flipped) image.
    """
    H, W = image.shape
    cy, cx = center
    i, j = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    i = i.astype(jnp.float32)
    j = j.astype(jnp.float32)
    i_centered = i - cy
    j_centered = j - cx
    theta = -jnp.deg2rad(angle_deg)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)
    j_in = j_centered * cos_theta - i_centered * sin_theta + cx
    i_in = j_centered * sin_theta + i_centered * cos_theta + cy
    rotated = bilinear_interpolate(image, i_in, j_in)
    if flip_x:
        rotated = jnp.flip(rotated, axis=1)
    return rotated
