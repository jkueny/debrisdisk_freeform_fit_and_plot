import numpy as np
import jax
from jax.scipy.ndimage import map_coordinates
import jax.numpy as jnp
import matplotlib.pyplot as plt


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

# --- Vectorized Rotation/Derotation Procedure ---
def vectorized_rotate_and_derotate(image, angles):
    """
    Copy the model image N times, rotate each by a given angle, then derotate each 
    (using the inverse rotation) and mean-stack the derotated images.
    
    Args:
        image: 2D jnp.array, the input model image.
        angles: 1D jnp.array of rotation angles in degrees.
        center: Tuple (cy, cx) for the center of rotation.
        flip_x: Boolean to decide if horizontal flip is applied.
        
    Returns:
        A tuple (rotated_images, derotated_images, mean_image).
    """
    rotated_images = jax.vmap(lambda ang: rotate_image(image, ang))(angles)
    jax.debug.print("print(rotate_images.shape) -> {x}", x=rotated_images.shape)
    rotated_images_flip1 = jnp.flip(rotated_images, axis=2)
    rotated_images_flip2 = jnp.flip(rotated_images_flip1, axis=2)
    derotated_images = jax.vmap(lambda ang, im: rotate_image(im, -ang))(angles, rotated_images_flip2)
    mean_image = jnp.mean(derotated_images, axis=0)
    return rotated_images, derotated_images, mean_image

# --- JIT-Compile the Vectorized Procedure ---
jit_vectorized_rotate_and_derotate = jax.jit(vectorized_rotate_and_derotate)

# --- Create a Composite Test Image with NaNs outside the ROI ---
img_shape = (128, 128)
center = (64, 64)
IWA, OWA = 10, 100

start_angle = -90
end_angle = 90

# Create coordinate grid.
y, x = np.indices(img_shape)
r = np.sqrt((x - center[0])**2 + (y - center[1])**2)

# Build an annulus mask.
mask_roi = (r >= IWA) & (r < OWA)

# Create an annulus and an off-center Gaussian.
annulus = np.where(mask_roi, 1.0, np.nan)
gaussian_center = (90, 40)
sigma = 5.0
gaussian = np.exp(-(((x - gaussian_center[0])**2 + (y - gaussian_center[1])**2) / (2 * sigma**2)))

# Combine them into a composite image.
composite = annulus + gaussian
# Set values outside the ROI to NaN.
test_image = np.where(mask_roi, composite, np.nan)
test_image_jax = jnp.array(test_image)

# --- Define an Array of Test Rotation Angles ---
angles = jnp.linspace(start_angle, end_angle, num=120)

# --- Run the JIT-Compiled Vectorized Rotation/Derotation Procedure ---
# We set flip_x=False here to obtain a pure rotation/derotation inverse.
rotated_images, derotated_images, mean_image = jit_vectorized_rotate_and_derotate(
    test_image_jax, angles)

# --- Visualization ---
plt.figure(figsize=(8, 5))

plt.subplot(2, 3, 1)
plt.imshow(np.array(test_image_jax), cmap='inferno')
plt.title("Original Composite Image\n(NaNs outside ROI)")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(np.array(rotated_images[0]), cmap='inferno')
plt.title(f"Rotated Image ({start_angle}°)")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(np.array(rotated_images[-1]), cmap='inferno')
plt.title(f"Rotated Image ({end_angle}°)")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(np.array(derotated_images[0]), cmap='inferno')
plt.title(f"Derotated Image ({start_angle}°)")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(np.array(derotated_images[-1]), cmap='inferno')
plt.title(f"Derotated Image ({end_angle}°)")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(np.array(mean_image), cmap='inferno')
plt.title("Mean-Stacked Derotated")
plt.axis("off")

plt.tight_layout()
plt.show()
