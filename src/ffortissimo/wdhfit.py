from pathlib import Path

import emcee
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi
import yaml
from astropy.io import fits

import ffortissimo.utils.astro_unit_conversion as convert
import ffortissimo.utils.make_gpi_psf_for_disks as gpidiskpsf


def gen_wdh_image(
    x, y, beta, h0, sigma, norm, pa_deg, x0, y0, gamma, inner_radius
):  # fwhm disabled for this prototype
    """Generate a parametric WDH model image."""
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

    power_law = r_safe**(-beta)
    exp_term = np.exp(
        -0.5 * ((r_safe**2 / (h0_safe * np.abs(x_safe) ** 2) ** gamma) + (x_safe / sigma_safe) ** 2)
    )
    image = np.nan_to_num(power_law * exp_term, nan=0.0, posinf=0.0, neginf=0.0)
    image *= norm
    image[r < inner_radius] = 0.0

    # Gaussian smoothing disabled for current dataset prototype.
    # if fwhm > 0:
    #     sigma_pix = fwhm / (2 * np.sqrt(2 * np.log(2)))
    #     image = ndi.gaussian_filter(image, sigma=sigma_pix, mode="nearest")
    return image


def control_region_mask(shape, center, coronrad, seeinglimited):
    """Build a binary annular mask for the WDH control region."""
    yy, xx = np.indices(shape, dtype=float)
    rho = np.sqrt((xx - center[0]) ** 2 + (yy - center[1]) ** 2)
    mask = np.ones(shape, dtype=float)
    mask[rho < coronrad] = 0.0
    mask[rho > seeinglimited] = 0.0
    return (mask > 0.5).astype(float)


def sparkle_mask(shape, sparkle_coords):
    """Return binary mask with sparkle ellipses set to 0."""
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
    """Resolve sparkle coordinates file path robustly."""
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
    """Create disk mask from initial geometry and rotate by PARANG."""
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


def build_masks(params, image_shape, parang_deg, params_yaml_path):
    """Build mask2generatehalo and mask2minimize."""
    center = np.asarray(params["ALIGNED_CENTER"], dtype=float)
    eff_wl = float(params.get("WL", 0.762))
    pixscale_ins = float(params["PIXSCALE_INS"])
    apdiam = float(params.get("APDIAM_M", 6.5))
    lyot_sm_rad = float(params.get("LYOT_LAMBDA_OVER_D", 3.0))
    ctrl_rad = float(params.get("CTRL_LAMBDA_OVER_D", 24.0))

    reselem_arcsec = eff_wl * 1e-6 / apdiam * 180 / np.pi * 3600
    reselem_pix = reselem_arcsec / pixscale_ins
    coron_reg = lyot_sm_rad * reselem_pix
    seeing_limited = ctrl_rad * reselem_pix# * np.sqrt(2)

    mask2generatehalo = control_region_mask(
        image_shape,
        center=center,
        coronrad=coron_reg,
        seeinglimited=seeing_limited,
    )

    sparkle_path = resolve_sparkle_path(params, params_yaml_path)
    sparkle_coords = np.loadtxt(sparkle_path)
    sparkle_exclusion = sparkle_mask(image_shape, sparkle_coords)
    disk_exclusion = make_disk_exclusion_mask(image_shape, params, parang_deg=parang_deg)

    mask2minimize = mask2generatehalo * sparkle_exclusion * disk_exclusion
    mask2minimize = (mask2minimize > 0.5).astype(float)
    return mask2generatehalo, mask2minimize


def build_parameterization(params):
    """Build free-parameter names and fixed prototype values."""
    fixed_pa = float(params["hpa_init"])
    fixed_gamma = float(params["hgamma_init"])
    fixed_sigma = float(params["hsig_init"])
    fixed_y0 = float(params["hdy_init"])
    mapping = [
        ("hbeta_init", "beta"),
        ("ha_r_init", "h0"),
        # ("hsig_init", "sigma"),
        ("hN_init", "norm"),
        ("hdx_init", "x0"),
        # ("hdy_init", "y0"),
        # ("hfwhm_init", "fwhm"),
        # ("hgamma_init", "gamma"),
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
        fixed_pa,
        fixed_gamma,
        fixed_sigma,
        fixed_y0,
    )


def theta_to_model_params(theta, fixed_pa, fixed_gamma, fixed_sigma, fixed_y0):
    return {
        "beta": theta[0],
        "h0": theta[1],
        "sigma": fixed_sigma,
        "norm": theta[2],
        "pa_deg": fixed_pa,
        "x0": theta[3],
        "y0": fixed_y0,
        # "fwhm": theta[6],
        "gamma": fixed_gamma,
    }


def log_prior(theta):
    beta, h0, norm, x0 = theta
    if not (-0.3 < beta < 1.0):
        return -np.inf
    if not (0.1 < h0 < 10.0):
        return -np.inf
    # if not (10.0 < sigma < 40):
    #     return -np.inf
    if not (1e-8 < norm < 1e8):
        return -np.inf
    if not (-5.0 < x0 < 0.1):
        return -np.inf
    # if not (0.0 <= fwhm < 20.0):
    #     return -np.inf
    # if not (1.0 < gamma < 2.0):
    #     return -np.inf
    return 0.0


def log_likelihood(theta, data):
    params = theta_to_model_params(
        theta,
        fixed_pa=data["fixed_pa"],
        fixed_gamma=data["fixed_gamma"],
        fixed_sigma=data["fixed_sigma"],
        fixed_y0=data["fixed_y0"],
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
        # fwhm=params["fwhm"],
        gamma=params["gamma"],
        inner_radius=data["inner_radius"],
    )
    data_minus_model = data["science_image"] - model
    residual = data_minus_model - data["psf_estimate"]
    masked_residual = residual[data["mask2minimize"] > 0.5]
    if masked_residual.size == 0:
        return -np.inf
    var = np.nanvar(masked_residual)
    if not np.isfinite(var) or var <= 0:
        var = 1.0
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
    """Run ensemble MCMC and return sampler with chains."""
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
    nsteps, _, ndim = chains.shape
    fig, axes = plt.subplots(ndim, 1, figsize=(12, 2.0 * ndim), sharex=True)
    if ndim == 1:
        axes = [axes]
    x = np.arange(nsteps)
    for idx, ax in enumerate(axes):
        ax.plot(x, chains[:, :, idx], alpha=0.2, lw=0.7)
        ax.set_ylabel(theta_names[idx])
        ax.set_yscale("symlog", linthresh=1e-3)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def to_log_display(image):
    """Symmetric log-style display transform for signed images."""
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros_like(image)
    abs_vals = np.abs(image[finite])
    scale = np.nanpercentile(abs_vals, 50)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    return np.sign(image) * np.log10(1.0 + np.abs(image) / scale)


def save_result_images(
    model,
    # model_minus_psf,
    residual,
    mask2minimize,
    output_dir,
    ):
    for name, image in [
        ("best_fit_wdh_model.png", model),
        # ("best_fit_model_minus_psf_estimate.png", model_minus_psf),
        ("best_fit_residual.png", residual),
        ("best_fit_residual_masked.png", residual * mask2minimize),
    ]:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        if name == "best_fit_wdh_model.png":
            # Keep best-fit WDH model in linear scale for direct intensity interpretation.
            display_image = to_log_display(image)
            title_suffix = " (symlog)"
        else:
            display_image = to_log_display(image)
            title_suffix = " (symlog)"
        im = ax.imshow(display_image, origin="lower", cmap="viridis")
        ax.set_title(name.replace(".png", "").replace("_", " ") + title_suffix)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output_dir / name, dpi=180)
        plt.close(fig)


def save_single_image_png(image, output_path, title):
    """Save one image panel in symlog display for diagnostics."""
    log_image = to_log_display(image)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(log_image, origin="lower", cmap="magma")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main():
    print("\nCreate masks and run prototype WDH MCMC fit")
    params_yaml_path = "initialization_files/wdh_HR4796_i_20230309_10.yaml"
    with open(params_yaml_path, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    image_path = (
        "/Users/jkueny/data/HR4796a_lco2023a_magao-x_20230309_10/"
        "raws_20230310T054736_s_lyot_stop/camsci1/lite_psflib/psf_subtracted/"
        "camsci1_2x2bin_47.5_parang_subtracted.fits"
    )
    psf_estimate_path = (
        "/Users/jkueny/data/HR4796a_lco2023a_magao-x_20230309_10/"
        "raws_20230310T054736_s_lyot_stop/camsci1/lite_psflib/psf_subtracted/"
        "psf_estimate_med.fits"
    )
    science_image_path = (
        "/Users/jkueny/data/HR4796a_lco2023a_magao-x_20230309_10/"
        "raws_20230310T054736_s_lyot_stop/camsci1/lite_psflib/camsci1_2x2bin_47.5_parang.fits"
    )
    science_image = fits.getdata(science_image_path).astype(float)
    psf_subtracted = fits.getdata(image_path).astype(float)
    psf_sub_header = fits.getheader(image_path)
    psf_estimate = fits.getdata(psf_estimate_path).astype(float)
    if psf_subtracted.shape != psf_estimate.shape or science_image.shape != psf_estimate.shape:
        raise ValueError("Science image, PSF-subtracted image, and PSF estimate must have same shape.")

    parang_deg = float(psf_sub_header.get("PARANG", 0.0))
    mask2generatehalo, mask2minimize = build_masks(params, psf_subtracted.shape, parang_deg, params_yaml_path)
    # #debug view the mask2minimize image
    # plt.imshow(mask2minimize, origin="lower", cmap="magma")
    # plt.show()
    # exit()
    valid_pix = int(np.sum(mask2minimize > 0.5))
    if valid_pix == 0:
        raise ValueError("mask2minimize has zero valid pixels; adjust mask settings.")

    theta_names, theta0, fixed_pa, fixed_gamma, fixed_sigma, fixed_y0 = build_parameterization(params)
    center = np.asarray(params["ALIGNED_CENTER"], dtype=float)
    yy, xx = np.indices(psf_subtracted.shape, dtype=float)
    data = {
        "xgrid": xx - center[0],
        "ygrid": yy - center[1],
        "science_image": science_image,
        "psf_subtracted": psf_subtracted,
        "psf_estimate": psf_estimate,
        "mask2minimize": mask2minimize,
        "fixed_pa": fixed_pa,
        "fixed_gamma": fixed_gamma,
        "fixed_sigma": fixed_sigma,
        "fixed_y0": fixed_y0,
        "inner_radius": float(params.get("IWA", 10.0)),
    }

    nwalkers = int(params.get("NWALKERS", 40))
    nsteps = int(params.get("N_ITER_MCMC", 3000))
    burnin = int(params.get("BURNIN", 300))
    thin = int(params.get("THIN", 1))

    sampler = run_mcmc(theta0, data, nwalkers=nwalkers, nsteps=nsteps, rng_seed=0)
    chains = sampler.get_chain()
    log_probs = sampler.get_log_prob()
    save_dir = Path("results/wdhfit_prototype")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_chain_plot(chains, theta_names, save_dir / "walker_chains.png")

    discard = min(burnin, max(0, nsteps - 1))
    thin_step = max(thin, 1)
    flat_chain = sampler.get_chain(discard=discard, thin=thin_step, flat=True)
    flat_logprob = sampler.get_log_prob(discard=discard, thin=thin_step, flat=True)
    best_idx = int(np.argmax(flat_logprob))
    best_theta = flat_chain[best_idx]
    best_model_params = theta_to_model_params(
        best_theta,
        fixed_pa=fixed_pa,
        fixed_gamma=fixed_gamma,
        fixed_sigma=fixed_sigma,
        fixed_y0=fixed_y0,
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
        # fwhm=best_model_params["fwhm"],
        gamma=best_model_params["gamma"],
        inner_radius=data["inner_radius"],
    )
    best_model_masked = best_model * mask2generatehalo
    # Residual for this strategy: (science - best_model) - psf_estimate
    best_residual = (science_image - best_model) - psf_estimate
    best_residual_masked = best_residual * mask2minimize
    save_result_images(best_model_masked, best_residual, mask2minimize, save_dir)
    psf_subtracted_masked = psf_subtracted * mask2minimize
    save_single_image_png(
        psf_subtracted_masked,
        save_dir / "psf_subtracted_data_image.png",
        "psf subtracted data image masked (symlog)",
    )

    summary_path = save_dir / "fit_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("Prototype WDH MCMC fit summary\n")
        handle.write(f"Fixed PA (deg): {fixed_pa:.6f}\n")
        handle.write(f"PARANG from data (deg): {parang_deg:.6f}\n")
        handle.write(f"Valid pixels in mask2minimize: {valid_pix}\n")
        handle.write(f"Best log prob: {flat_logprob[best_idx]:.6e}\n")
        for name, value in zip(theta_names, best_theta):
            handle.write(f"{name}: {value:.8f}\n")

    np.save(save_dir / "best_theta.npy", best_theta)
    np.save(save_dir / "flat_chain.npy", flat_chain)
    np.save(save_dir / "flat_logprob.npy", flat_logprob)
    fits.writeto(save_dir / "mask2generatehalo.fits", mask2generatehalo.astype(np.float32), overwrite=True)
    fits.writeto(save_dir / "mask2minimize.fits", mask2minimize.astype(np.float32), overwrite=True)
    fits.writeto(
        save_dir / "psf_subtracted_data_image.fits",
        psf_subtracted_masked.astype(np.float32),
        overwrite=True,
    )
    fits.writeto(save_dir / "best_fit_wdh_model.fits", best_model_masked.astype(np.float32), overwrite=True)
    # fits.writeto(
    #     save_dir / "best_fit_model_minus_psf_estimate.fits",
    #     best_model_minus_psf_masked.astype(np.float32),
    #     overwrite=True,
    # )
    fits.writeto(save_dir / "best_fit_residual.fits", best_residual_masked.astype(np.float32), overwrite=True)

    print(f"Saved prototype outputs to: {save_dir}")
    print(f"Fixed PA from hpa_init: {fixed_pa:.3f} deg")
    print("Free parameters:", theta_names)
    print("Best-fit theta:", best_theta)
    print("MCMC chain shape (steps, walkers, ndim):", chains.shape)
    print("Log-prob shape (steps, walkers):", log_probs.shape)


if __name__ == "__main__":
    main()