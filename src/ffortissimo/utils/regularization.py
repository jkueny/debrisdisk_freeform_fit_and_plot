import numpy as np
from jax.scipy.signal import fftconvolve
import jax.numpy as jnp
from scipy.optimize import minimize
from scipy.ndimage import binary_dilation


def preprocess(img: np.ndarray) -> np.ndarray:
    img = img - img.mean()
    return img / (np.linalg.norm(img) + 1e-12)


def power_spectrum(img: np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(img))
    return np.abs(f) ** 2



def asymmetric_tukey(shape, wx, wy, angle_deg, alpha=0.5):
    H, W = shape
    y = np.linspace(-(H - 1) / 2, (H - 1) / 2, H)
    x = np.linspace(-(W - 1) / 2, (W - 1) / 2, W)
    yy, xx = np.meshgrid(y, x, indexing="ij")

    theta = np.deg2rad(angle_deg)
    xr = xx * np.cos(theta) + yy * np.sin(theta)
    yr = -xx * np.sin(theta) + yy * np.cos(theta)

    r = np.sqrt((xr / wx) ** 2 + (yr / wy) ** 2)

    r0 = 1.0 - alpha / 2.0
    r1 = 1.0 + alpha / 2.0

    w = np.where(
        r <= r0,
        1.0,
        np.where(
            r <= r1,
            0.5 * (1.0 + np.cos(np.pi * (r - r0) / alpha)),
            0.0,
        ),
    )
    return w

def elliptical_gaussian(shape, sig_x, sig_y, theta_deg):
    H, W = shape
    y = np.linspace(-(H-1)/2, (H-1)/2, H)
    x = np.linspace(-(W-1)/2, (W-1)/2, W)
    yy, xx = np.meshgrid(y, x, indexing="ij")
    t = np.deg2rad(theta_deg)
    xr =  xx*np.cos(t) + yy*np.sin(t)
    yr = -xx*np.sin(t) + yy*np.cos(t)
    return np.exp(-(xr/sig_x)**2 - (yr/sig_y)**2)


def make_objective(psd_norm, lam_area=0.001):
    # psd_norm -= np.min(psd_norm)

    def _loss(params):          # params = [sig_x, sig_y, theta_deg, scale]
        sx, sy, theta, scale = params
        window = elliptical_gaussian(psd_norm.shape, sx, sy, theta) * scale
        area = np.sum(window) / psd_norm.size
        return np.mean((window - psd_norm)**2) + lam_area * area

    return _loss

# def make_objective(psd_norm, alpha=0.5):
#     # psd_norm = np.sqrt(psd) #/ psd.max()          # [0,1] scale for comparison
#     def _loss(p):
#         wx, wy, theta_deg, scale = p
#         # wx, wy, theta_deg = p
#         window = asymmetric_tukey(psd_norm.shape, wx, wy, theta_deg, alpha)
#         window *= scale
#         return np.mean((window - psd_norm) ** 2)
#     return _loss

def penalize_residual_disk(disk_spine, residuals_image, reg_lambda):
    residual_disk = fftconvolve(residuals_image, disk_spine, mode="same")
    residual_disk_norm = residual_disk / jnp.linalg.norm(residual_disk)
    return jnp.max(residual_disk_norm) * (10 * reg_lambda)


def fit_elgauss_window(ref_psd, generosity=1., verbose=True):
    psd_norm = ref_psd

    H, W = ref_psd.shape
    guess = min(H, W) / 8.0
    x0 = np.array([guess,
                   guess,
                   0.0,
                   1.0,
                   ])

    bounds = [(1e-2, None),
              (1e-2, None),
              (None, None),
              (0.1, None),
              ] 

    result = minimize(make_objective(psd_norm),
                   x0,
                   method="L-BFGS-B",
                   bounds=bounds,
                   options=dict(maxiter=1000, ftol=1e-10, gtol=1e-8))

    if verbose:
        print(f"Converged in {result.nit} iterations | final loss = {result.fun:.4e}")

    wx, wy, theta_deg, scale = result.x
    # wx, wy, theta_deg = res.x
    scale_by = generosity * psd_norm.max()
    # window_opt = asymmetric_tukey(psd.shape, wx, wy, theta_deg, alpha) * scale
    window_opt = elliptical_gaussian(ref_psd.shape,
                                     wx*generosity,
                                     wy*generosity,
                                     theta_deg) * scale_by
    params = dict(width_x=wx, width_y=wy, angle_deg=theta_deg, scale=scale)
    # params = dict(width_x=wx, width_y=wy, angle_deg=theta_deg)
    return params, window_opt

def estimate_median_background(
    klipped_data: np.ndarray,
    fitting_region_mask:np.ndarray,
    dilation_size: int = 10) -> np.ndarray:
    """
    Estimate the median background of the klipped data using the dilated
    fitting region mask (mask2generatedisk).

    Args:
        klipped_data: The klipped data.
        fitting_region_mask: The fitting region mask.
        dilation_size: The size of the dilation in pixels.

    Returns:
        The fitting region mask inpainted with the estimated
        median background.
    """
    # Ensure the fitting region mask is a boolean array
    median_background = fitting_region_mask.copy().astype(float)
    fitting_region_mask = fitting_region_mask.copy().astype(bool)
    if len(klipped_data.shape) > 2:
        klipped_data = np.squeeze(klipped_data)
    dilated_mask = binary_dilation(fitting_region_mask, structure=np.ones((dilation_size, dilation_size)))
    use_for_median = (dilated_mask - median_background) * klipped_data
    median_for_inpaint = np.median(use_for_median[use_for_median > 0])
    median_background[fitting_region_mask] = median_for_inpaint
    # # debug look at the dilated mask
    # import matplotlib.pyplot as plt
    # plt.imshow(klipped_data)
    # plt.show()
    # exit()
    return median_background