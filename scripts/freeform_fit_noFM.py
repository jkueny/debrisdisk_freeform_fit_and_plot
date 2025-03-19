'''
Fit a pixel-by-pixel freeform model disk to a KLIP reduced image.

Using JAX.

To prepare, need to normalize the image data bc we're doing this exercise
just to learn the dust morphology of the disk such that we can fit functions
to it later.

Steps to develop:
1. Generate the KLIP image and save the basis for DiskFM.
2. Read in and save the instr. PSF to use for the FM.
3. Generate an image of random numbers as an intial guess and feed it to the
optimizer function.
4. 
'''

import os
import sys
# import copy
import argparse


basedir = f'{os.environ["HOME"]}/projects'  # the base directory where is
# your data (using OS environnement variable allow to use same code on
# different computer without changing this).

# default_parameter_file = 'HR4796_g_camsci2_20230312_13.yaml'  # name of the parameter file
default_parameter_file = 'HR4796a_z_lco2023a_magao-x_20230309_10.yaml'  # name of the parameter file
# default_parameter_file = 'HR4796_i_smlyot_20230309_10.yaml'  # name of the parameter file
# you can also call it with the python function argument -p

# For parallelization stuff...?
MPI = False  ## by default the MCMC is not mpi. you can change it
## in the the python function argument --mpi




from datetime import datetime

import math as mt
import jax
import jax.numpy as jnp
from jax.scipy.signal import convolve2d
from jax.scipy.ndimage import map_coordinates
import optax
import matplotlib.pyplot as plt
import numpy as np

import astropy.io.fits as fits
# from astropy.convolution import convolve
from scipy.signal import convolve
# from scipy.signal import fftconvolve
from astropy.wcs import FITSFixedWarning

# import yaml

def initialize_freeform_model_reduced():
    """Initialize freeform model parameters for the unmasked region."""
    rng = jax.random.PRNGKey(42)
    # Instead of full image dimensions, only initialize num_free parameters.
    free_params = jax.random.normal(rng, (NUM_FREE,))

    return free_params

def reconstruct_full_image(free_params, full_shape):
    """
    Given the free parameters (for the unmasked region) and the full image shape,
    create a full image (flattened) where the free parameters are inserted at the positions
    indicated by mask_indices and zeros elsewhere.
    """
    total_pixels = np.prod(full_shape)
    full_flat = jnp.zeros(total_pixels)
    full_flat = full_flat.at[MASK_INDICES].set(free_params)
    return full_flat.reshape(full_shape)

def insert_section_into_full_image(flat_section, full_shape, section_inds):
    """
    Given:
      - flat_section: a 1D JAX array of length N_pixels_section (the forward-modeled section)
      - full_shape: tuple (height, width) for the full image (e.g. (224, 224))
      - section_inds: a JAX array or tuple of indices that select the section in the full image.
                     In your case, section_inds has shape (1, N_pixels_section), so we'll flatten it.
    
    This function flattens a blank full image, updates it at the given 1D indices with the values from flat_section,
    and then reshapes it back to full_shape.
    
    Returns:
      A full image (JAX array) of shape full_shape with the section inserted.
    """
    # Ensure section_inds is a 1D index array.
    section_inds = jnp.ravel(jnp.array(section_inds))
    # Create a blank full image flattened.
    total_pixels = np.prod(full_shape) # Ex. 50176
    full_image_flat = jnp.zeros(total_pixels)
    # Use .at to update the flattened array.
    full_image_flat = full_image_flat.at[section_inds].set(flat_section)
    # Reshape back to full_shape.
    full_image = full_image_flat.reshape(full_shape)
    return full_image

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

def convolve_model(input_model, psf):
    # psf = jnp.asarray(psf)
    assert psf.shape[0] == psf.shape[1], "Instr. PSF image is not square. How can this be?!"
    kernel_size = psf.shape[0] #should be square
    # Pad image to maintain size
    padded_image = jnp.pad(input_model, [(kernel_size//2, kernel_size//2),
                                   (kernel_size//2, kernel_size//2)], mode='reflect')

    # Apply convoluted convolution
    model_convolved = convolve2d(input_model, psf, mode="same")
    
    return model_convolved

# --- Loss Function ---
def loss_function(image_params, psf, target_image, parang):
    """Computes MSE loss between forward-modeled image and target."""
    # freeform_image = jax.nn.sigmoid(image_params)  # Ensure values stay within [0,1]
    freeform_image = insert_section_into_full_image(image_params, psf.shape, MASK_INDICES)
    convolved_image = convolve_model(freeform_image, psf)
    mod_img_rot = rotate_image(convolved_image, parang)
    rot_flipx = jnp.flip(mod_img_rot, axis=1)
    rot_unflip = jnp.flip(rot_flipx, axis=1)
    modeled_image = rotate_image(rot_unflip, -parang)

    return jnp.mean((modeled_image - target_image) ** 2)

# --- JIT-Compiled Gradient Computation ---
loss_and_grad = jax.jit(jax.value_and_grad(loss_function))

# --- Optimization Routine ---
def optimize_image(target_image, psf, parang=90, num_steps=100, lr=0.1):
    """Optimizes a pixel-wise freeform model to match the target image."""
    size = target_image.shape[0]
    
    # # Initialize freeform image parameters (random pixel values)
    # rng = jax.random.PRNGKey(42)
    # image_params = jax.random.normal(rng, (size, size))  # Trainable parameters
    image_params = initialize_freeform_model_reduced()

    # Optimizer setup
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(image_params)

    loss_history = []

    @jax.jit
    def step(image_params, opt_state):
        loss, grads = loss_and_grad(image_params, psf, target_image, parang)
        updates, opt_state = optimizer.update(grads, opt_state)
        image_params = optax.apply_updates(image_params, updates)
        return image_params, opt_state, loss

    for step_idx in range(num_steps):
        image_params, opt_state, loss = step(image_params, opt_state)
        loss_history.append(loss.item())

        if step_idx % 50 == 0:
            print(f"Step {step_idx}/{num_steps} - Loss: {loss:.6f}")

    # Convert optimized parameters to final freeform image
    # optimized_image = jax.nn.sigmoid(image_params)  # Normalize to [0,1]
    optimized_image = insert_section_into_full_image(image_params, psf.shape, MASK_INDICES)  # Normalize to [0,1]
    return optimized_image, loss_history

# --- Run the Simulation ---
MASK2GENERATEDISK = fits.getdata("/Users/jkueny/projects/HR4796a_lco2023a_magao-x_20230309_10/raws_20230310T054736_s_lyot_stop/camsci2/lite_psflib/klip_fm_files/camsci2_z_20230309_10_mask2generatedisk.fits")
MASK = jnp.array(MASK2GENERATEDISK)
MASK_INDICES = jnp.flatnonzero(MASK)  # 1D indices of nonzero (True) entries
NUM_FREE = MASK_INDICES.shape[0]
# Read in the target image
target_image = fits.getdata("/Users/jkueny/projects/HR4796a_lco2023a_magao-x_20230309_10/raws_20230310T054736_s_lyot_stop/camsci2/norm_lite_psflib/klip_fm_files/camsci2_z_20230309_10_FirstModel_Conv.fits")
target_image = target_image #/ (np.max(target_image) * 2) #normalized
target_image *= MASK2GENERATEDISK
size = target_image.shape[0]
jax_target_image = jnp.array(target_image)
# Read in the instr PSF
psf = fits.getdata("/Users/jkueny/projects/HR4796a_lco2023a_magao-x_20230309_10/raws_20230310T054736_s_lyot_stop/camsci2/norm_lite_psflib/klip_fm_files/camsci2_z_20230309_10_instrPSF.fits")
# psf = psf / np.sum(psf) #normalize
jax_psf = jnp.array(psf)

optimized_image, loss_history = optimize_image(jax_target_image,
                                               jax_psf,
                                               num_steps=400)

# --- Visualization ---

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax)
    plt.sca(last_axes)
    return cbar

fig, ax = plt.subplots(2, 2, figsize=(8, 8))

rng = jax.random.PRNGKey(42)
init_image = jax.random.normal(rng, (size, size))

best_convolved = convolve2d(optimized_image, jax_psf, mode="same")

cax1 = ax[0,0].imshow(np.array(init_image), cmap='inferno',origin="lower")
ax[0,0].set_title("Initial model (random pixels)")
colorbar(cax1)
ax[0,0].axis("off")

cax2 = ax[0,1].imshow(np.array(optimized_image), cmap='inferno',origin="lower")
ax[0,1].set_title(f"Best-fit model after {len(loss_history)} steps")
colorbar(cax2)
ax[0,1].axis("off")

cax3 = ax[1,0].imshow(np.array(jax_target_image), cmap='inferno', origin="lower")
ax[1,0].set_title("Convolved disk model (target image)")
colorbar(cax3)
ax[1,0].axis("off")

cax4 = ax[1,1].imshow(np.array(jax_target_image - best_convolved), cmap="inferno", origin="lower",
                      vmin=np.min(np.array(jax_target_image)),vmax=np.max(np.array(jax_target_image)))
ax[1,1].set_title("Target image minus convolved best model")
# ax[1,1].set_xlabel("Iteration")
# ax[1,1].set_ylabel("MSE Loss")
colorbar(cax4)
ax[1,1].axis("off")
plt.tight_layout()

plt.show()