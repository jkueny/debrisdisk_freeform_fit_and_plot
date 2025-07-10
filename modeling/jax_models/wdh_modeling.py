import jax.numpy as jnp
import jax.scipy as jsp


def render_wdh_model(x, y, params):
    """

    """
    # Unpack
    beta = params["beta"]
    h0 = params["a_r"]
    sigma = params["sigma"]
    PA_deg = params["PA"]
    x0 = params["dx"]
    scaling = params["Norm"]
    fwhm = fwhm
    R1 = 5 #At pixel scale 0.012"/pixel this is the IWA at g', about 4 lamb/D
    # Rotate to disk PA
    PA_rad = jnp.deg2rad(PA_deg)
    x_rot =  jnp.cos(PA_rad) * x + jnp.sin(PA_rad) * y
    y_rot = -jnp.sin(PA_rad) * x + jnp.cos(PA_rad) * y

    r = jnp.sqrt((x_rot - x0)**2 + (y_rot)**2)

    # Base intensity
    power_law = (1. / r)**beta
    exp_term = jnp.exp(-0.5 * (((r)**2 / (h0 * (x_rot - x0)**2)) + (x_rot / sigma)**2))
    I = power_law * exp_term
    I_nanless = I.at[jnp.isnan(I)].set(0.0)
    I_valid = I_nanless.at[jnp.isinf(I_nanless)].set(0.0)
    # I[jnp.isnan(I)] = 0.0
    # I[jnp.isinf(I)] = 0.0

    I_coron = I_valid.at[r < R1].set(0.0)

    return I_coron * scaling

def gen_multiwdh_image(x, y, all_params):
    # The convolutional kernel is the same for all WDHs
    fwhm = all_params["ps_global"]["fwhm"]

    n_wdh_params = all_params["ps_individual"]

    model_image = jnp.zeros_like((x,y))

    for _, params in n_wdh_params.items():
        model_image += render_wdh_model(x, y, params, fwhm)

    # Gaussian convolution
    if fwhm is not None and fwhm > 0:
        sigma_pix = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
        sig2 = 2 * sigma_pix * sigma_pix
        window = jnp.exp(-(x**2 + y**2) / sig2)
        I_image = jsp.signal.convolve2d(model_image,window)
    else:
        I_image = model_image

    return I_image