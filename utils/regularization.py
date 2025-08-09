import numpy as np


from scipy.optimize import minimize


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


def make_objective(psd_norm, lam_area=1e-3):
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


def fit_elgauss_window(ref_psd, verbose=True):
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

    res = minimize(make_objective(psd_norm),
                   x0,
                   method="L-BFGS-B",
                   bounds=bounds,
                   options=dict(maxiter=1000, ftol=1e-10, gtol=1e-8))

    if verbose:
        print(f"Converged in {res.nit} iterations | final loss = {res.fun:.4e}")

    wx, wy, theta_deg, scale = res.x
    # wx, wy, theta_deg = res.x
    # scale = scale * psd.max()
    # window_opt = asymmetric_tukey(psd.shape, wx, wy, theta_deg, alpha) * scale
    window_opt = elliptical_gaussian(ref_psd.shape, wx, wy, theta_deg) * (scale + psd_norm.max())
    params = dict(width_x=wx, width_y=wy, angle_deg=theta_deg, scale=scale)
    # params = dict(width_x=wx, width_y=wy, angle_deg=theta_deg)
    return params, window_opt