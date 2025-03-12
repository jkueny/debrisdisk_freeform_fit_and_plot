import jax.numpy as jnp
import jax

from functools import partial

def bilinear_interpolate(image, i_coords, j_coords):
    """
    Perform bilinear interpolation on a 2D image.
    
    image: 2D jnp.array
    i_coords, j_coords: arrays of the same shape with float coordinates
                       in the image to sample from.
    
    Returns:
        Interpolated image sampled at (i_coords, j_coords).
    """
    H, W = image.shape
    # Get integer parts of the coordinates
    i0 = jnp.floor(i_coords).astype(jnp.int32)
    j0 = jnp.floor(j_coords).astype(jnp.int32)
    # The next indices
    i1 = i0 + 1
    j1 = j0 + 1

    # Clip indices to be within image bounds
    i0 = jnp.clip(i0, 0, H - 1)
    i1 = jnp.clip(i1, 0, H - 1)
    j0 = jnp.clip(j0, 0, W - 1)
    j1 = jnp.clip(j1, 0, W - 1)

    # Gather pixel values at the four surrounding coordinates
    Ia = image[i0, j0]
    Ib = image[i0, j1]
    Ic = image[i1, j0]
    Id = image[i1, j1]

    # Calculate the weights for each neighboring pixel
    wa = (i1 - i_coords) * (j1 - j_coords)
    wb = (i1 - i_coords) * (j_coords - j0)
    wc = (i_coords - i0) * (j1 - j_coords)
    wd = (i_coords - i0) * (j_coords - j0)

    # Compute the interpolated value
    return wa * Ia + wb * Ib + wc * Ic + wd * Id

def rotate_image(image: jnp.ndarray, angle_deg: float, center: tuple) -> jnp.ndarray:
    """
    Rotate a 2D image (JAX array) by a given angle (in degrees, CCW positive)
    about a specified center. Optionally, flip the rotated image along the x-axis.
    
    Args:
        image: 2D jnp.array representing the image.
        angle_deg: Angle in degrees by which to rotate the image (CCW positive).
        center: Tuple (cy, cx) representing the center of rotation in pixel coordinates.
                Here, cy is the row coordinate and cx is the column coordinate.
        flip_x: If True, the rotated image is flipped along the x-axis (columns reversed).
        
    Returns:
        A 2D jnp.array of the rotated (and optionally flipped) image.
    """
    H, W = image.shape
    cy, cx = center  # center coordinates (row, col)

    # Create a meshgrid of coordinates (rows, cols) for the output image.
    i, j = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing="ij")
    # Convert to float for precise computations.
    i = i.astype(jnp.float32)
    j = j.astype(jnp.float32)

    # Shift the grid so that the center becomes (0, 0).
    i_centered = i - cy
    j_centered = j - cx

    # Convert angle from degrees to radians. For inverse mapping,
    # rotate by -theta (i.e., apply a clockwise rotation).
    theta = -jnp.deg2rad(angle_deg)
    cos_theta = jnp.cos(theta)
    sin_theta = jnp.sin(theta)

    # When treating (j, i) as (x, y), the inverse rotation is:
    # x_in = j_centered * cos(theta) - i_centered * sin(theta)
    # y_in = j_centered * sin(theta) + i_centered * cos(theta)
    # Then, shift back by the center.
    j_in = j_centered * cos_theta - i_centered * sin_theta + cx
    i_in = j_centered * sin_theta + i_centered * cos_theta + cy

    # Use bilinear interpolation to sample the input image at the computed coordinates.
    rotated = bilinear_interpolate(image, i_in, j_in)

    # Optionally flip the image along the x-axis (i.e. reverse columns).
    rotated = jnp.flip(rotated, axis=1)

    return rotated
