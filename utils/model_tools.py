from jax.scipy.signal import convolve2d


def convolve_model(input_model, psf):
    # psf = jnp.asarray(psf)
    assert psf.shape[0] == psf.shape[1], "Instr. PSF image is not square. How can this be?!"
    kernel_size = psf.shape[0] #should be square
    # Pad image to maintain size
    # padded_image = jnp.pad(input_model, [(kernel_size//2, kernel_size//2),
    #                                (kernel_size//2, kernel_size//2)], mode='reflect')

    # Apply convoluted convolution
    model_convolved = convolve2d(input_model, psf, mode="same")
    
    return model_convolved