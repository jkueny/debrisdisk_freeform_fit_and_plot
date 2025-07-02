import jax.numpy as jnp
import jax.scipy as jsp


def gen_wdh_image(x, y, beta, h0, sigma,
                  PA_deg, x0, gaussian_fwhm, scaling):
    """

    """
    R1 = 5 #At pixel scale 0.012"/pixel this is the IWA at g', about 4 lamb/D
    # Rotate to disk PA
    PA_rad = jnp.deg2rad(PA_deg)
    x_rot =  jnp.cos(PA_rad) * x + jnp.sin(PA_rad) * y
    y_rot = -jnp.sin(PA_rad) * x + jnp.cos(PA_rad) * y

    r = jnp.sqrt((x_rot - x0)**2 + (y_rot)**2)

    # Base intensity
    power_law = (1. / r)**beta
    exp_term = jnp.exp(-0.5 * (((r)**2 / (h0 * (x_rot - x0)**2)**gamma) + (x_rot / sigma)**2))
    I = power_law * exp_term
    I[jnp.isnan(I)] = 0.0
    I[jnp.isinf(I)] = 0.0

    I_coron = I.at[r < R1].set(0.0)


    # Gaussian convolution
    if gaussian_fwhm is not None and gaussian_fwhm > 0:
        sigma_pix = gaussian_fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
        sig2 = 2 * sigma_pix * sigma_pix
        window = jnp.exp(-(x**2 + y**2) / sig2)
        I_image = jsp.signal.convolve2d(I_coron,window)
    else:
        I_image = I_coron

    return I_image * scaling