"""
MCMC fit: parametric WDH (`gen_wdh_image`) vs a reference WDH model image (FITS).

This script is scoped to recover a reference WDH map from another modeling
architecture, not to fit raw science data. See `main()` for paths.
"""
from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import yaml
from astropy.io import fits

import ffortissimo.utils.astro_unit_conversion as convert
import ffortissimo.utils.make_gpi_psf_for_disks as gpidiskpsf

# Repo root: .../src/ffortissimo/wdhfit.py -> parents[2] == project root
_REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_WDH_FITS = _REPO_ROOT / "starting_models" / "wdh_camsci2_z_20230309_10.fits"


def gen_wdh_image(
    x, y, beta, h0, sigma, norm, pa_deg, x0, y0, gamma, inner_radius, fwhm
):
    """Generate a parametric WDH model image; optional Gaussian blur (fwhm in pixels)."""
    pa_rad = -np.deg2rad(pa_deg)
    x_rot = np.sin(pa_rad) * x + np.cos(pa_rad) * y
    y_rot = np.cos(pa_rad) * x - np.sin(pa_rad) * y

    x_shift = x_rot - x0
    y_shift = y_rot - y0
    r_shift = np.sqrt(x_shift**2 + y_shift**2)
    r = np.sqrt(x_rot**2 + y_rot**2)
    r_safe = np.maximum(r_shift, 1e-6)
    x_safe = np.sign(x_shift) * np.maximum(np.abs(x_shift), 1e-6)
    h0_safe = max(h0, 1e-6)
    sigma_safe = max(sigma, 1e-6)

    power_law = r_safe ** (-beta)
    exp_term = np.exp(
        -0.5
        * (
            (r_safe**2 / (h0_safe * np.abs(x_safe) ** 2) ** gamma)
            + (x_safe / sigma_safe) ** 2
        )
    )
    image = np.nan_to_num(power_law * exp_term, nan=0.0, posinf=0.0, neginf=0.0)
    image *= norm
    image[r < inner_radius] = 0.0
    if fwhm > 0:
        sigma_pix = fwhm / (2 * np.sqrt(2 * np.log(2)))
        image = ndi.gaussian_filter(image, sigma=sigma_pix, mode="nearest")
    return image


def coronagraph_only_mask(shape, center, coronrad):
    """
    Binary mask: 1 outside the inner coronagraph radius only (no outer cutoff).
    Pixels with rho < coronrad are masked to 0.
    """
    yy, xx = np.indices(shape, dtype=float)
    rho = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mask = np.ones(shape, dtype=float)
    mask[rho < coronrad] = 0.0
    return (mask > 0.5).astype(float)


def build_mask2generatehalo(params, image_shape):
    """Coronagraph inner mask only; λ/D from YAML."""
    center = np.asarray(params["ALIGNED_CENTER"], dtype=float)
    eff_wl = float(params.get("WL", 0.762))
    pixscale_ins = float(params["PIXSCALE_INS"])
    apdiam = float(params.get("APDIAM_M", 6.5))
    lyot_sm_rad = float(params.get("LYOT_LAMBDA_OVER_D", 3.0))

    reselem_arcsec = eff_wl * 1e-6 / apdiam * 180 / np.pi * 3600
    reselem_pix = reselem_arcsec / pixscale_ins
    coron_reg = lyot_sm_rad * reselem_pix
    return coronagraph_only_mask(image_shape, center=center, coronrad=coron_reg)


# --- Helpers kept for a future mask2minimize pipeline (not used in this script) ---
def control_region_mask(shape, center, coronrad, seeinglimited):
    """Annular control region (inner + outer boundary)."""
    yy, xx = np.indices(shape, dtype=float)
    rho = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mask = np.ones(shape, dtype=float)
    mask[rho < coronrad] = 0.0
    mask[rho > seeinglimited] = 0.0
    return (mask > 0.5).astype(float)


def sparkle_mask(shape, sparkle_coords):
    yy, xx = np.indices(shape, dtype=float)
    mask = np.ones(shape, dtype=float)
    for row in np.atleast_2d(sparkle_coords):
        x0, y0, semimajor, semiminor, angle_deg = row[:5]
        angle_rad = np.deg2rad(angle_deg)
        x = xx - x0
        y = yy - y0
        x_rot = x * np.cos(angle_rad) + y * np.sin(angle_rad)
        y_rot = -x * np.sin(angle_rad) + y * np.cos(angle_rad)
        ellipse = (x_rot / max(semimajor, 1e-6)) ** 2 + (y_rot / max(semiminor, 1e-6)) ** 2
        mask[ellipse <= 1.0] = 0.0
    return mask


def resolve_sparkle_path(params, params_yaml_path):
    sparkle_rel = Path(params["SPARKLE_COORDS_FILE"])
    candidates = []
    if sparkle_rel.is_absolute():
        candidates.append(sparkle_rel)
    base_dir = Path(params_yaml_path).resolve().parent
    candidates.append(base_dir / sparkle_rel)
    if "BAND_DIR" in params:
        band_dir = Path(params["BAND_DIR"])
        if not band_dir.is_absolute():
            band_dir = Path("/Users/jkueny/data") / band_dir
        candidates.append(band_dir / sparkle_rel)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve sparkle file from {sparkle_rel}. Tried: {candidates}")


def make_disk_exclusion_mask(shape, params, parang_deg):
    center = np.asarray(params["ALIGNED_CENTER"], dtype=float)
    inner_pad = float(params.get("MASK_IN_SCALING", 3.0))
    outer_pad = float(params.get("MASK_OUT_SCALING", 3.0))
    r1_pix = convert.au_to_pix(params["r1_init"], params["PIXSCALE_INS"], params["DISTANCE_STAR"])
    r2_pix = convert.au_to_pix(params["r2_init"], params["PIXSCALE_INS"], params["DISTANCE_STAR"])
    cos_inc = max(np.cos(np.radians(params["inc_init"])), 1e-6)
    disk_zeros = gpidiskpsf.make_disk_mask(
        shape[0],
        params["pa_init"],
        params["inc_init"],
        r1_pix - inner_pad / cos_inc,
        r2_pix + outer_pad / cos_inc,
        aligned_center=center,
    )
    disk_zeros = np.asarray(disk_zeros, dtype=float)
    disk_rot = ndi.rotate(disk_zeros, angle=parang_deg, reshape=False, order=0, mode="nearest")
    return (disk_rot > 0.5).astype(float)


def hourglass_directional_mask(shape, center, pa_deg, opening_angle_deg):
    opening = float(np.clip(opening_angle_deg, 1e-3, 180.0))
    half_open = np.deg2rad(opening / 2.0)
    yy, xx = np.indices(shape, dtype=float)
    x = xx - center[0]
    y = yy - center[1]
    pa_rad = -np.deg2rad(pa_deg)
    x_rot = np.sin(pa_rad) * x + np.cos(pa_rad) * y
    y_rot = np.cos(pa_rad) * x - np.sin(pa_rad) * y
    ang_from_axis = np.arctan2(np.abs(y_rot), np.abs(x_rot) + 1e-12)
    mask = np.zeros(shape, dtype=float)
    mask[ang_from_axis <= half_open] = 1.0
    return mask


# def build_masks_legacy_with_minimize(params, image_shape, parang_deg, params_yaml_path):
#     """Former pipeline: annulus + sparkles + disk + hourglass -> mask2minimize."""
#     center = np.asarray(params["ALIGNED_CENTER"], dtype=float)
#     eff_wl = float(params.get("WL", 0.762))
#     pixscale_ins = float(params["PIXSCALE_INS"])
#     apdiam = float(params.get("APDIAM_M", 6.5))
#     lyot_sm_rad = float(params.get("LYOT_LAMBDA_OVER_D", 3.0))
#     ctrl_rad = float(params.get("CTRL_LAMBDA_OVER_D", 24.0))
#     reselem_arcsec = eff_wl * 1e-6 / apdiam * 180 / np.pi * 3600
#     reselem_pix = reselem_arcsec / pixscale_ins
#     coron_reg = lyot_sm_rad * reselem_pix
#     seeing_limited = ctrl_rad * reselem_pix
#     mask2generatehalo = control_region_mask(
#         image_shape, center=center, coronrad=coron_reg, seeinglimited=seeing_limited
#     )
#     sparkle_path = resolve_sparkle_path(params, params_yaml_path)
#     sparkle_coords = np.loadtxt(sparkle_path)
#     sparkle_exclusion = sparkle_mask(image_shape, sparkle_coords)
#     disk_exclusion = make_disk_exclusion_mask(image_shape, params, parang_deg=parang_deg)
#     opening_angle = float(params.get("MASK_OPENING_ANGLE", 180.0))
#     directional_exclusion = hourglass_directional_mask(
#         image_shape,
#         center=center,
#         pa_deg=float(params.get("hpa_init", params.get("pa_init", 0.0))),
#         opening_angle_deg=opening_angle,
#     )
#     mask2minimize = (
#         mask2generatehalo * sparkle_exclusion * disk_exclusion * directional_exclusion
#     )
#     mask2minimize = (mask2minimize > 0.5).astype(float)
#     return mask2generatehalo, mask2minimize


def build_parameterization(params):
    """Free parameters; sigma and gamma stay fixed from YAML."""
    fixed_gamma = float(params["hgamma_init"])
    fixed_sigma = float(params["hsig_init"])
    mapping = [
        ("hbeta_init", "beta"),
        ("ha_r_init", "h0"),
        ("hN_init", "norm"),
        ("hfwhm_init", "fwhm"),
        ("hdx_init", "x0"),
        ("hpa_init", "PA"),
        ("hdy_init", "y0"),
    ]
    theta_names = []
    theta0 = []
    for yaml_key, label in mapping:
        if yaml_key not in params:
            raise KeyError(f"Missing required key in YAML: {yaml_key}")
        theta_names.append(label)
        theta0.append(float(params[yaml_key]))
    return (
        theta_names,
        np.array(theta0, dtype=float),
        fixed_gamma,
        fixed_sigma,
    )


def theta_to_model_params(theta, fixed_gamma, fixed_sigma):
    return {
        "beta": theta[0],
        "h0": theta[1],
        "norm": theta[2],
        "fwhm": theta[3],
        "x0": theta[4],
        "pa_deg": theta[5],
        "y0": theta[6],
        "sigma": fixed_sigma,
        "gamma": fixed_gamma,
    }


def log_prior(theta):
    beta, h0, norm, fwhm, x0, pa_deg, y0 = theta
    if not (-3.0 < beta < 3.0):
        return -np.inf
    if not (0.1 < h0 < 10.0):
        return -np.inf
    if not (1e-8 < norm < 1e8):
        return -np.inf
    if not (0.0 <= fwhm < 30.0):
        return -np.inf
    if not (-5.0 < x0 < 5.0):
        return -np.inf
    if not (-180.0 < pa_deg < 180.0):
        return -np.inf
    if not (-5.0 < y0 < 5.0):
        return -np.inf
    return 0.0


def log_likelihood(theta, data):
    params = theta_to_model_params(
        theta,
        fixed_gamma=data["fixed_gamma"],
        fixed_sigma=data["fixed_sigma"],
    )
    model = gen_wdh_image(
        data["xgrid"],
        data["ygrid"],
        beta=params["beta"],
        h0=params["h0"],
        sigma=params["sigma"],
        norm=params["norm"],
        pa_deg=params["pa_deg"],
        x0=params["x0"],
        y0=params["y0"],
        gamma=params["gamma"],
        inner_radius=data["inner_radius"],
        fwhm=params["fwhm"],
    )
    residual = data["target_wdh"] - model
    fit_mask = data["mask2generatehalo"] > 0.5
    # masked_residual = residual[fit_mask]
    masked_residual = residual * fit_mask
    if masked_residual.size == 0:
        return -np.inf
    # var = np.nanvar(masked_residual)
    # if not np.isfinite(var) or var <= 0:
    var = 1.0
    # #debug see the model and data
    # plt.imshow(residual, origin="lower", cmap="viridis")
    # plt.colorbar()
    # plt.show()
    # plt.imshow(residual * data["mask2generatehalo"], origin="lower", cmap="viridis")
    # plt.colorbar()
    # plt.show()
    # exit()
    return -0.5 * np.nansum(masked_residual**2 / var)


def log_probability(theta, data):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, data)
    if not np.isfinite(ll):
        return -np.inf
    return lp + ll


def run_mcmc(theta0, data, nwalkers, nsteps, rng_seed=0):
    ndim = theta0.size
    if nwalkers < 2 * ndim:
        raise ValueError(f"NWALKERS={nwalkers} is too small for ndim={ndim}. Use at least {2 * ndim}.")
    rng = np.random.default_rng(rng_seed)
    scales = np.maximum(np.abs(theta0), 1.0) * 1e-2
    p0 = theta0 + rng.normal(scale=scales, size=(nwalkers, ndim))
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(data,))
    sampler.run_mcmc(p0, nsteps, progress=True)
    return sampler


def save_chain_plot(chains, theta_names, output_path):
    """Walker traces vs iteration (one subplot per free parameter)."""
    nsteps, _, ndim = chains.shape
    fig, axes = plt.subplots(ndim, 1, figsize=(12, 2.0 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    steps = np.arange(nsteps)
    for idx, ax in enumerate(axes):
        ax.plot(steps, chains[:, :, idx], alpha=0.2, lw=0.7)
        ax.set_ylabel(theta_names[idx])
        ax.set_yscale("symlog", linthresh=1e-3)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_corner_plot(flat_chain, theta_names, output_path):
    """Posterior corner plot from flattened post-burn chain."""
    import corner

    fig = corner.corner(
        flat_chain,
        labels=theta_names,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".4g",
    )
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def to_log_display(image):
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros_like(image)
    abs_vals = np.abs(image[finite])
    scale = np.nanpercentile(abs_vals, 50)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.sign(image) * np.log10(1.0 + np.abs(image) / scale)


def save_final_pngs(model_display, residual_display, output_dir):
    """Only the two diagnostic figures requested: best model and residuals."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(model_display, origin="lower", cmap="viridis")
    ax.set_title("best fit WDH model (symlog)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "best_fit_wdh_model.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(residual_display, origin="lower", cmap="viridis")
    ax.set_title("residuals: target WDH - model (symlog)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "best_fit_residual.png", dpi=180)
    plt.close(fig)


def main():
    print("\nMCMC: parametric WDH fit to reference WDH model image")
    params_yaml_path = "initialization_files/wdh_HR4796_i_20230309_10.yaml"
    with open(params_yaml_path, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    ref_path = REFERENCE_WDH_FITS
    if not ref_path.is_file():
        raise FileNotFoundError(f"Reference WDH model not found: {ref_path}")
    target_wdh = fits.getdata(str(ref_path)).astype(float)
    if target_wdh.ndim != 2:
        target_wdh = np.squeeze(target_wdh)

    mask2generatehalo = build_mask2generatehalo(params, target_wdh.shape)
    valid_pix = int(np.sum(mask2generatehalo > 0.5))
    if valid_pix == 0:
        raise ValueError("mask2generatehalo has no valid pixels; check coronagraph settings.")

    theta_names, theta0, fixed_gamma, fixed_sigma = build_parameterization(params)
    center = np.asarray(params["ALIGNED_CENTER"], dtype=float)
    yy, xx = np.indices(target_wdh.shape, dtype=float)
    data = {
        "xgrid": xx - center[0],
        "ygrid": yy - center[1],
        "target_wdh": target_wdh,
        "mask2generatehalo": mask2generatehalo,
        "fixed_gamma": fixed_gamma,
        "fixed_sigma": fixed_sigma,
        "inner_radius": float(params.get("IWA", 10.0)),
    }

    nwalkers = int(params.get("NWALKERS", 40))
    nsteps = int(params.get("N_ITER_MCMC", 3000))
    burnin = int(params.get("BURNIN", 300))
    thin = int(params.get("THIN", 1))

    sampler = run_mcmc(theta0, data, nwalkers=nwalkers, nsteps=nsteps, rng_seed=0)
    save_dir = Path("results/wdhfit_prototype")
    save_dir.mkdir(parents=True, exist_ok=True)
    chains = sampler.get_chain()
    save_chain_plot(chains, theta_names, save_dir / "walker_chains.png")

    discard = min(burnin, max(0, nsteps - 1))
    thin_step = max(thin, 1)
    flat_chain = sampler.get_chain(discard=discard, thin=thin_step, flat=True)
    flat_logprob = sampler.get_log_prob(discard=discard, thin=thin_step, flat=True)
    best_idx = int(np.argmax(flat_logprob))
    best_theta = flat_chain[best_idx]
    best_model_params = theta_to_model_params(
        best_theta,
        fixed_gamma=fixed_gamma,
        fixed_sigma=fixed_sigma,
    )

    best_model = gen_wdh_image(
        data["xgrid"],
        data["ygrid"],
        beta=best_model_params["beta"],
        h0=best_model_params["h0"],
        sigma=best_model_params["sigma"],
        norm=best_model_params["norm"],
        pa_deg=best_model_params["pa_deg"],
        x0=best_model_params["x0"],
        y0=best_model_params["y0"],
        gamma=best_model_params["gamma"],
        inner_radius=data["inner_radius"],
        fwhm=best_model_params["fwhm"],
    )
    best_model_display = best_model * mask2generatehalo
    best_residual = target_wdh - best_model

    save_corner_plot(flat_chain, theta_names, save_dir / "corner.png")
    save_final_pngs(best_model_display, best_residual, save_dir)

    fits.writeto(
        save_dir / "best_fit_wdh_model.fits",
        best_model_display.astype(np.float32),
        overwrite=True,
    )
    fits.writeto(
        save_dir / "best_fit_residual.fits",
        best_residual.astype(np.float32),
        overwrite=True,
    )

    summary_path = save_dir / "fit_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("WDH parametric fit to reference WDH model\n")
        handle.write(f"Reference FITS: {ref_path}\n")
        handle.write(f"Valid pixels in mask2generatehalo: {valid_pix}\n")
        handle.write(f"Best log prob: {flat_logprob[best_idx]:.6e}\n")
        for name, value in zip(theta_names, best_theta):
            handle.write(f"{name}: {value:.8f}\n")

    print(
        f"Saved: {save_dir / 'walker_chains.png'}, {save_dir / 'corner.png'}, "
        f"{save_dir / 'best_fit_wdh_model.png'}, {save_dir / 'best_fit_residual.png'}, "
        f"{save_dir / 'best_fit_wdh_model.fits'}, {save_dir / 'best_fit_residual.fits'}"
    )
    print(f"Summary: {summary_path}")
    print("Best-fit theta:", best_theta)


if __name__ == "__main__":
    main()
