import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import sys
import time

# Add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the update_disk function from funcs_JDFM.py.
# Adjust the import path as needed.
from dev.pyklip.fmlib.funcs_JDFM import update_disk

# For testing purposes, we need a rotate_image function.
# The update_disk() function calls rotate_image from dev.pyklip.j_klip,
# so we override it here with a simple implementation.
# from jax.scipy.ndimage import rotate

# Monkey-patch the rotate_image in funcs_JDFM so update_disk uses our test version.
from dev.pyklip.fmlib.funcs_JDFM import rotate_image

def main():
    # Define image dimensions.
    img_shape = (224, 224)
    
    # Create an asymmetric test disk model.
    # For example, an off-center Gaussian so rotation effects are visible.
    x = jnp.linspace(-1, 1, img_shape[1])
    y = jnp.linspace(-1, 1, img_shape[0])
    X, Y = jnp.meshgrid(x, y)
    model_disk = jnp.exp(-5 * ((X - 0.3)**2 + (Y + 0.2)**2))
    
    # Set the number of input images.
    num_input_images = 5
    # Define different position angles for each image.
    PAs = jnp.array([0, 45, 90, 135, 180], dtype=jnp.float32)
    # Define the aligned center as the center of the image.
    aligned_center = (img_shape[1] // 2, img_shape[0] // 2)  # e.g., (112, 112)
    
    # Create a jitted version of update_disk.
    # Mark 'aligned_center' and 'num_input_images' as static since they do not change.
    update_disk_jit = jax.jit(update_disk, static_argnames=["aligned_center", "num_input_images"])
    
    # Warm-up call for the jitted function (to compile it).
    _ = update_disk_jit(model_disk, PAs, aligned_center, num_input_images).block_until_ready()

    # Time the jitted version.
    jit_times = []
    n_iter = 10
    for i in range(n_iter):
        start = time.time()
        result_jit = update_disk_jit(model_disk, PAs, aligned_center, num_input_images)
        result_jit.block_until_ready()
        jit_times.append(time.time() - start)
    avg_jit = sum(jit_times) / len(jit_times)
    print(f"Average JIT execution time over {n_iter} iterations: {avg_jit:.6f} seconds")
    
    # Time the non-jitted version.
    nonjit_times = []
    for i in range(n_iter):
        start = time.time()
        result_nonjit = update_disk(model_disk, PAs, aligned_center, num_input_images)
        # Use block_until_ready to ensure computation is finished.
        result_nonjit.block_until_ready()
        nonjit_times.append(time.time() - start)
    avg_nonjit = sum(nonjit_times) / len(nonjit_times)
    print(f"Average non-JIT execution time over {n_iter} iterations: {avg_nonjit:.6f} seconds")
    
    # For visualization, use the jitted version's output.
    updated_model_disks = result_jit
    updated_model_disks_reshaped = updated_model_disks.reshape((num_input_images, img_shape[0], img_shape[1]))
    
    # Plot the output disks.
    fig, axes = plt.subplots(1, num_input_images, figsize=(15, 3))
    for i in range(num_input_images):
        axes[i].imshow(np.array(updated_model_disks_reshaped[i]), cmap='inferno')
        axes[i].set_title(f'PA {PAs[i]:.0f}°')
        axes[i].axis('off')
    plt.suptitle("Output of update_disk() with jax.jit")
    plt.show()

if __name__ == "__main__":
    main()
