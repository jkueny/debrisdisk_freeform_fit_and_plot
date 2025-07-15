import jax.numpy as jnp
import jax.scipy as jsp

import jax

def gaussian_1d(fwhm):
    sigma = fwhm / (2.0 * jnp.sqrt(2.0 * jnp.log(2.0)))
    half_width = 5 * sigma # truncate at 5 sigma
    x = jnp.arange(-half_width, half_width + 1)
    g = jnp.exp(-0.5 * (x / sigma)**2)
    return g / g.sum()                          # L1-normalise

def convolve_fft_xy(model, kernel_1d):
    # Convolve along x
    tmp = jsp.signal.fftconvolve(model, kernel_1d[None, :], mode="same")
    # Convolve along y
    image = jsp.signal.fftconvolve(tmp,   kernel_1d[:,  None], mode="same")

    return image


@jax.jit
def render_wdh_model(x, y, params, mask, fwhm):
    """

    """
    # Unpack
    beta = params["beta"]
    h0 = params["a_r"]
    sigma = params["sig"]
    PA_deg = params["PA"]
    x0 = params["dx"]
    scaling = params["Norm"]
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
    I_nanless = jnp.where(jnp.isnan(I), 0., I)
    I_valid = jnp.where(jnp.isinf(I_nanless), 0., I_nanless)


    I_coron = jnp.where(r < R1, 0, I_valid)

    # Gaussian convolution
    # if fwhm is not None and fwhm > 0:
    sigma_pix = fwhm / (2 * jnp.sqrt(2 * jnp.log(2)))
    # kernel_1d = gaussian_1d(fwhm)
    # I_image = convolve_fft_xy(I_coron, kernel_1d)
    sig2 = 2 * sigma_pix * sigma_pix
    window = jnp.exp(-(x**2 + y**2) / sig2)
    I_image = jsp.signal.convolve2d(I_coron,window, mode="same")
    # else:
    #     I_image = I_coron

    return I_image * scaling * mask

def gen_multiwdh_image(x, y, all_params, mask):
    """
    Generate a list of unrotated WDH model images, one per component in `ps_individual`.

    Parameters
    ----------
    x, y : 2D jnp.ndarray
        Coordinate grid.
    params : dict
        Dictionary with keys 'ps_global' and 'ps_individual'.
        'ps_individual' should be a dict of WDH component parameter dicts.

    Returns
    -------
    image_list : list of 2D jnp.ndarray
        List of WDH model images, one per component.
    """
    assert "ps_indiv" in all_params, "params must contain 'ps_indiv' key"
    assert "ps_global" in all_params, "params must contain 'ps_global' key"

    fwhm = all_params["ps_global"]["fwhm"]
    n_wdh_params = all_params["ps_indiv"]

    
    model_list = []

    for wdh_params in n_wdh_params:


        model = render_wdh_model(x, y, wdh_params, mask, fwhm)
        model_list.append(model)


    return jnp.asarray(model_list)    # The convolutional kernel is the same for all WDHs