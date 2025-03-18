import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# --- Amended rotate_image() Function with Optional Flip ---
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

# --- Vectorized Rotation/Derotation Procedure ---
def vectorized_rotate_and_derotate(image, angles, center, flip_x: bool = True):
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
    rotated_images = jax.vmap(lambda ang: rotate_image(image, ang, center, flip_x=flip_x))(angles)
    rotated_images_corrected = jnp.flip(rotated_images, axis=2)
    derotated_images = jax.vmap(lambda ang, im: rotate_image(im, -ang, center, flip_x=False))(angles, rotated_images_corrected)
    mean_image = jnp.mean(derotated_images, axis=0)
    return rotated_images, derotated_images, mean_image

# --- JIT-Compile the Vectorized Procedure ---
jit_vectorized_rotate_and_derotate = jax.jit(vectorized_rotate_and_derotate, static_argnames=('flip_x',))

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
    test_image_jax, angles, center, flip_x=True)

# --- Visualization ---
plt.figure(figsize=(14, 10))

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
