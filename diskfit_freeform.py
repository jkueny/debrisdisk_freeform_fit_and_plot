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
import optax
import matplotlib.pyplot as plt
import numpy as np

import astropy.io.fits as fits
# from astropy.convolution import convolve
from scipy.signal import convolve
# from scipy.signal import fftconvolve
from astropy.wcs import FITSFixedWarning

import yaml

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
def loss_function(image_params, psf, target_image):
    """Computes MSE loss between forward-modeled image and target."""
    freeform_image = jax.nn.sigmoid(image_params)  # Ensure values stay within [0,1]
    modeled_image = convolve_model(freeform_image, psf)
    return jnp.mean((modeled_image - target_image) ** 2)

# --- JIT-Compiled Gradient Computation ---
loss_and_grad = jax.jit(jax.value_and_grad(loss_function))

# --- Optimization Routine ---
def optimize_image(target_image, psf, num_steps=100, lr=0.1):
    """Optimizes a pixel-wise freeform model to match the target image."""
    size = target_image.shape[0]
    
    # Initialize freeform image parameters (random pixel values)
    rng = jax.random.PRNGKey(42)
    image_params = jax.random.normal(rng, (size, size))  # Trainable parameters

    # Optimizer setup
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(image_params)

    loss_history = []

    @jax.jit
    def step(image_params, opt_state):
        loss, grads = loss_and_grad(image_params, psf, target_image)
        updates, opt_state = optimizer.update(grads, opt_state)
        image_params = optax.apply_updates(image_params, updates)
        return image_params, opt_state, loss

    for step_idx in range(num_steps):
        image_params, opt_state, loss = step(image_params, opt_state)
        loss_history.append(loss.item())

        if step_idx % 50 == 0:
            print(f"Step {step_idx}/{num_steps} - Loss: {loss:.6f}")

    # Convert optimized parameters to final freeform image
    optimized_image = jax.nn.sigmoid(image_params)  # Normalize to [0,1]
    return optimized_image, loss_history

# --- Run the Simulation ---
# Read in the target image
target_image = fits.getdata("/Users/jkueny/projects/HR4796a_lco2023a_magao-x_20230309_10/raws_20230310T054736_s_lyot_stop/camsci2/norm_lite_psflib/klip_fm_files/camsci2_z_20230309_10_FirstModel_Conv.fits")
target_image = target_image / (np.max(target_image) * 2) #normalized
size = target_image.shape[0]
jax_target_image = jnp.array(target_image)
# Read in the instr PSF
psf = fits.getdata("/Users/jkueny/projects/HR4796a_lco2023a_magao-x_20230309_10/raws_20230310T054736_s_lyot_stop/camsci2/norm_lite_psflib/klip_fm_files/camsci2_z_20230309_10_instrPSF.fits")
psf = psf / np.sum(psf) #normalize
jax_psf = jnp.array(psf)

optimized_image, loss_history = optimize_image(jax_target_image, jax_psf, num_steps=500)

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