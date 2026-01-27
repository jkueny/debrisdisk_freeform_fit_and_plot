import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

from functools import partial

def cubic_weight(x):
    """
    Computes the weights for bicubic interpolation using the Catmull-Rom spline.
    """
    a = -0.5
    abs_x = jnp.abs(x)
    
    # Condition 1: |x| <= 1
    cond1 = abs_x <= 1.0
    val1 = (a + 2.0) * abs_x**3 - (a + 3.0) * abs_x**2 + 1.0
    
    # Condition 2: 1 < |x| < 2
    cond2 = (abs_x > 1.0) & (abs_x < 2.0)
    val2 = a * abs_x**3 - 5.0 * a * abs_x**2 + 8.0 * a * abs_x - 4.0 * a
    
    return jnp.where(cond1, val1, jnp.where(cond2, val2, 0.0))

def get_pixel_value(img, x, y):
    """
    Safe pixel retrieval with zero padding (cval=0.0).
    """
    h, w = img.shape
    # Check bounds
    in_bounds = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    # Clip coordinates to be safe for array indexing (though we mask the result later)
    x_clamped = jnp.clip(jnp.int32(x), 0, w - 1)
    y_clamped = jnp.clip(jnp.int32(y), 0, h - 1)
    
    val = img[y_clamped, x_clamped]
    return jnp.where(in_bounds, val, 0.0)

def bicubic_interp_2d(image, coordinates):
    """
    Bicubic interpolation (order=3) equivalent to map_coordinates.
    
    Args:
        image: 2D array (H, W)
        coordinates: List of two arrays [y_coords, x_coords] matching scipy.map_coordinates
        
    Returns:
        Interpolated image of shape matching coordinates
    """
    y_coords, x_coords = coordinates
    
    # Floor to get the top-left integer coordinate of the 4x4 kernel
    x_f = jnp.floor(x_coords)
    y_f = jnp.floor(y_coords)
    
    # The convolution kernel iterates from -1 to 2 around the floor
    # Sum over the 4x4 neighborhood
    output = jnp.zeros_like(x_coords)
    
    for dy in range(-1, 3):
        for dx in range(-1, 3):
            # Coordinates of the neighbor pixel
            ix = x_f + dx
            iy = y_f + dy
            
            # Weight for this neighbor
            wx = cubic_weight(x_coords - ix)
            wy = cubic_weight(y_coords - iy)
            
            # Pixel value
            val = get_pixel_value(image, ix, iy)
            
            output += val * wx * wy
            
    return output

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
    # rotated = map_coordinates(image, [i_in, j_in], order=1, mode='constant', cval=0.0)
    rotated = bicubic_interp_2d(image, [i_in, j_in])

    return rotated