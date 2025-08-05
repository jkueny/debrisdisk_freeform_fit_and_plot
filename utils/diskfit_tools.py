import jax.numpy as jnp
import numpy as np
from jax.scipy.signal import fftconvolve

def penalize_spatial_freq(model_ps, ref_model_ps, reg_lambda):
    '''
    We have an ideal scattered light disk model from a prior MCMC analysis.

    We can use this as a power spectrum reference to penalize high spatial frequencies.
    '''
    # Compute where freeform power exceeds ref model power
    excess_mask = model_ps > (ref_model_ps)
    excess_power = jnp.where(excess_mask, 
                             model_ps - ref_model_ps, #grab power at high freqs
                             0.0) #zero everything else

    # Return total excess as a scalar penalty
    # jax.debug.print("print(reg_lambda) -> {x}", x=reg_lambda)
    penalty = jnp.mean(excess_power)
    return reg_lambda * penalty



def convolve_model(input_model, psf):
    # psf = jnp.asarray(psf)
    assert psf.shape[0] == psf.shape[1], "Instr. PSF image is not square. How can this be?!"

    # Apply convoluted convolution
    model_convolved = fftconvolve(input_model, psf, mode="same")
    
    return model_convolved


def record_pyklip_params(n_klmodes, iwa, owa, minrot, centering):
    params = {}

    params["n_KLmodes"] = n_klmodes
    params["IWA"] = iwa
    params["OWA"] = owa
    params["minrot"] = minrot
    params["aligned_center"] = centering

    return params

