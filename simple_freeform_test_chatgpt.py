import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import numpy as np

# --- Generate Target Image ---
def generate_target_image(size=32, sigma=5):
    """Creates a synthetic image: a 2D Gaussian centered in the middle."""
    x = jnp.linspace(-size//2, size//2, size)
    y = jnp.linspace(-size//2, size//2, size)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    target = jnp.exp(-(X**2 + Y**2) / (2 * sigma**2))
    return target

# --- Forward Model: Apply Gaussian Blur ---
def gaussian_blur(image, blur_sigma=1.5):
    """Applies a Gaussian blur using a convolutional kernel."""
    kernel_size = 7  # Choose an odd number
    x = jnp.arange(kernel_size) - kernel_size // 2
    gauss_kernel_1d = jnp.exp(-0.5 * (x / blur_sigma) ** 2)
    gauss_kernel_1d /= jnp.sum(gauss_kernel_1d)

    # Create 2D kernel from outer product
    gauss_kernel_2d = jnp.outer(gauss_kernel_1d, gauss_kernel_1d)

    # Pad image to maintain size
    padded_image = jnp.pad(image, [(kernel_size//2, kernel_size//2),
                                   (kernel_size//2, kernel_size//2)], mode='reflect')

    # Apply 2D convolution using `jax.scipy.signal.convolve2d`
    from jax.scipy.signal import convolve2d
    blurred_image = convolve2d(padded_image, gauss_kernel_2d, mode="valid")
    
    return blurred_image

# --- Loss Function ---
def loss_function(image_params, target_image):
    """Computes MSE loss between forward-modeled image and target."""
    freeform_image = jax.nn.sigmoid(image_params)  # Ensure values stay within [0,1]
    modeled_image = gaussian_blur(freeform_image)
    return jnp.mean((modeled_image - target_image) ** 2)

# --- JIT-Compiled Gradient Computation ---
loss_and_grad = jax.jit(jax.value_and_grad(loss_function))

# --- Optimization Routine ---
def optimize_image(target_image, num_steps=10000, lr=0.1):
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
        loss, grads = loss_and_grad(image_params, target_image)
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
size = 224
target_image = generate_target_image(size)
optimized_image, loss_history = optimize_image(target_image)

# --- Visualization ---
fig, ax = plt.subplots(1, 3, figsize=(12, 4))

ax[0].imshow(np.array(target_image), cmap='inferno')
ax[0].set_title("Target Image (Ground Truth)")
ax[0].axis("off")

ax[1].imshow(np.array(optimized_image), cmap='inferno')
ax[1].set_title("Optimized Freeform Model")
ax[1].axis("off")

ax[2].plot(loss_history)
ax[2].set_title("Loss Over Time")
ax[2].set_xlabel("Iteration")
ax[2].set_ylabel("MSE Loss")

plt.show()
