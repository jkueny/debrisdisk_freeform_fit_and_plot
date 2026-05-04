"""Subtract best-fit WDH components from individual science frames."""
import argparse
import glob
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.optimize import nnls
import scipy.ndimage as ndi
import yaml
from astropy.io import fits

_SRC_ROOT = Path(__file__).resolve().parents[1]
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

import ffortissimo.utils.astro_unit_conversion as convert
from ffortissimo.utils.masks import control_region_mask


_REPO_ROOT = Path(__file__).resolve().parents[2]


def get_basedir():
    """Return the base data directory used by the WDH MCMC scripts."""
    return Path(os.environ.get("DATA_DIR", f'{os.environ["HOME"]}/data')).expanduser()


def resolve_parameter_file(param_file):
    """Resolve a YAML parameter path from a direct path or initialization_files."""
    candidate = Path(param_file).expanduser()
    search_paths = []

    if candidate.is_absolute():
        search_paths.append(candidate)
    else:
        search_paths.extend(
            [
                Path.cwd() / candidate,
                _REPO_ROOT / candidate,
                _REPO_ROOT / "initialization_files" / candidate.name,
            ]
        )

    for path in search_paths:
        if path.is_file():
            return path

    tried = ", ".join(str(path) for path in search_paths)
    raise FileNotFoundError(f"Could not resolve parameter file {param_file!r}. Tried: {tried}")


def load_params(param_file):
    """Load the YAML parameter file."""
    yaml_path = resolve_parameter_file(param_file)
    with yaml_path.open("r", encoding="utf-8") as handle:
        params = yaml.safe_load(handle)
    if params is None:
        raise ValueError(f"Parameter file is empty: {yaml_path}")
    return params, yaml_path

def resolve_band_dir(params):
    """Resolve BAND_DIR relative to DATA_DIR / ~/data, matching existing scripts."""
    if "BAND_DIR" not in params:
        raise KeyError("Parameter file is missing required key: BAND_DIR")

    band_dir = Path(params["BAND_DIR"]).expanduser()
    if not band_dir.is_absolute():
        band_dir = get_basedir() / band_dir
    if not band_dir.is_dir():
        raise FileNotFoundError(f"Could not find BAND_DIR: {band_dir}")
    return band_dir


def sort_parang_monotonic(filename):
    """Sort MagAO-X science frames by the PARANG value embedded in the filename."""
    match = re.search(r"bin_([-+]?\d*\.?\d+)_parang", Path(filename).name)
    if match:
        return float(match.group(1))
    raise ValueError(
        f"Filename {filename} does not match expected '*bin_<parang>_parang.fits' pattern."
    )


def find_science_frames(band_dir):
    """Return coronagraphic science frame FITS paths from BAND_DIR."""
    filelist = glob.glob(str(band_dir / "camsci*.fits"))
    if not filelist:
        raise FileNotFoundError(f"Could not find science frames matching 'camsci*.fits' in {band_dir}")
    return [Path(path) for path in sorted(filelist, key=sort_parang_monotonic)]


def component_number(path):
    """Extract the one-based WDH component number from a best-model component filename."""
    match = re.search(r"_BestModel_comp(\d+)\.fits$", path.name)
    if not match:
        raise ValueError(f"Could not parse WDH component number from {path}")
    return int(match.group(1))


def find_component_files(band_dir):
    """Return 1-4 best-fit WDH model component FITS paths from windfit_MCMC."""
    mcmc_dir = band_dir / "windfit_MCMC"
    if not mcmc_dir.is_dir():
        raise FileNotFoundError(f"Could not find windfit_MCMC directory: {mcmc_dir}")

    component_files = sorted(
        mcmc_dir.glob("*_BestModel_comp*.fits"),
        key=component_number,
    )
    if not component_files:
        raise FileNotFoundError(f"Could not find '*_BestModel_comp*.fits' files in {mcmc_dir}")
    if len(component_files) > 4:
        raise ValueError(
            f"Expected at most 4 WDH component files in {mcmc_dir}, found {len(component_files)}"
        )
    return component_files


def read_science_frame(path):
    """Read one science frame and its FITS header."""
    data, header = fits.getdata(path, header=True)
    data = np.asarray(np.squeeze(data), dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(f"Science frame must be 2D after squeezing: {path} has shape {data.shape}")
    if "PARANG" not in header:
        raise KeyError(f"Science frame is missing PARANG header keyword: {path}")
    return data, header


def read_components(component_files):
    """Read WDH component images."""
    components = []
    for path in component_files:
        data = np.asarray(np.squeeze(fits.getdata(path)), dtype=np.float64)
        if data.ndim != 2:
            raise ValueError(f"WDH component must be 2D after squeezing: {path} has shape {data.shape}")
        components.append(data)
    return components


def build_median_psf_estimate(science_frames, image_shape):
    """Build a detector-frame PSF estimate from the median science frame."""
    frames = []
    for science_path in science_frames:
        science_frame, _ = read_science_frame(science_path)
        if science_frame.shape != image_shape:
            raise ValueError(
                f"Science frame shape {science_frame.shape} does not match expected "
                f"image shape {image_shape}: {science_path}"
            )
        frames.append(science_frame)
    return np.nanmedian(np.asarray(frames, dtype=np.float64), axis=0)


def build_median_psf_estimate_from_frames(science_frames, image_shape):
    """Build a detector-frame PSF estimate from science frame arrays."""
    frames = []
    for idx, science_frame in enumerate(science_frames):
        science_frame = np.asarray(science_frame, dtype=np.float64)
        if science_frame.shape != image_shape:
            raise ValueError(
                f"Science frame index {idx} has shape {science_frame.shape}; "
                f"expected {image_shape}"
            )
        frames.append(science_frame)
    return np.nanmedian(np.asarray(frames, dtype=np.float64), axis=0)


def load_noise_map(params, band_dir, image_shape):
    """Load the spatial noise map used to weight the component fit."""
    if "FILE_PREFIX" not in params:
        raise KeyError("Parameter file is missing required key: FILE_PREFIX")

    noise_path = band_dir / "wind_fm_files" / f"{params['FILE_PREFIX']}_noisemap.fits"
    if not noise_path.is_file():
        raise FileNotFoundError(f"Could not find noise map: {noise_path}")

    noise_map = np.asarray(np.squeeze(fits.getdata(noise_path)), dtype=np.float64)
    if noise_map.shape != image_shape:
        raise ValueError(
            f"Noise map shape {noise_map.shape} does not match image shape {image_shape}: {noise_path}"
        )
    return noise_map, noise_path


def calculate_radial_distances(image_shape, center=None):
    """Return pixel radii from the image center."""
    if center is None:
        center = ((image_shape[0] // 2) - 0.5, (image_shape[1] // 2) - 0.5)
    y, x = np.indices(image_shape)
    center_y, center_x = center
    return np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2)


def subtract_median_profile_np(image, center=None):
    """Subtract a median radial profile without importing the JAX utility module."""
    image = np.asarray(np.squeeze(image), dtype=np.float64)
    radial_distances = np.round(calculate_radial_distances(image.shape, center)).astype(int)
    max_distance = int(np.max(radial_distances))
    median_profile = np.array(
        [np.nanmedian(image[radial_distances == radius]) for radius in range(max_distance + 1)]
    )
    median_image = median_profile[radial_distances]
    return image - median_image, median_image


def rotate_components(components, parang):
    """Rotate WDH components into one science frame's parallactic angle."""
    return [
        ndi.rotate(
            component,
            angle=float(parang),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
        )
        for component in components
    ]


def make_disk_exclusion_mask(dim, pa_deg, inc_deg, inner_radius, outer_radius, aligned_center):
    """Return 1 outside the disk annulus and 0 inside it."""
    inner_radius = max(float(inner_radius), 0.0)
    pa_rad = np.radians(90.0 + float(pa_deg))
    x = np.arange(dim, dtype=float)[None, :] - aligned_center[0]
    y = np.arange(dim, dtype=float)[:, None] - aligned_center[1]
    x_rot = x * np.cos(pa_rad) + y * np.sin(pa_rad)
    y_rot = -x * np.sin(pa_rad) + y * np.cos(pa_rad)
    y_deprojected = y_rot / np.cos(np.radians(float(inc_deg)))
    rho_elliptical = np.sqrt(x_rot**2 + y_deprojected**2)
    mask = np.ones((dim, dim), dtype=float)
    mask[(rho_elliptical > inner_radius) & (rho_elliptical < outer_radius)] = 0.0
    return mask


def load_speckle_ellipses(speckle_mask_file):
    """Load x, y, major, minor, theta rows for optional speckle masking."""
    if speckle_mask_file is None:
        return None

    speckle_path = Path(speckle_mask_file).expanduser()
    if not speckle_path.is_file():
        raise FileNotFoundError(f"Could not find speckle mask file: {speckle_path}")

    ellipses = np.loadtxt(speckle_path, dtype=float)
    ellipses = np.atleast_2d(ellipses)
    if ellipses.shape[1] != 5:
        raise ValueError(
            f"Speckle mask file must have 5 columns: x y major minor theta. "
            f"Found shape {ellipses.shape} in {speckle_path}"
        )
    return ellipses


def make_speckle_exclusion_mask(image_shape, speckle_ellipses):
    """Return a binary mask that excludes configured elliptical DM speckles."""
    mask = np.ones(image_shape, dtype=float)
    if speckle_ellipses is None:
        return mask

    y, x = np.indices(image_shape, dtype=float)
    for x0, y0, major_axis, minor_axis, theta_deg in speckle_ellipses:
        if major_axis <= 0 or minor_axis <= 0:
            raise ValueError("Speckle ellipse major/minor axes must be positive.")
        theta = np.radians(theta_deg)
        x_shift = x - x0
        y_shift = y - y0
        x_rot = x_shift * np.cos(theta) + y_shift * np.sin(theta)
        y_rot = -x_shift * np.sin(theta) + y_shift * np.cos(theta)
        ellipse = (x_rot / major_axis) ** 2 + (y_rot / minor_axis) ** 2
        mask[ellipse <= 1.0] = 0.0
    return mask


def binarize_mask(mask):
    """Force a mask to strict 0/1 values."""
    mask = np.asarray(mask, dtype=np.float64)
    mask[mask < 0.5] = 0.0
    mask[mask >= 0.5] = 1.0
    return mask


def build_fitting_masks(params, image_shape, speckle_ellipses=None):
    """Build static detector and rotatable disk masks for WDH coefficient fitting."""
    if image_shape[0] != image_shape[1]:
        raise ValueError(f"WDH fitting mask requires square frames, got {image_shape}")

    aligned_center = np.asarray(params["ALIGNED_CENTER"], dtype=float)
    mask_center = (
        aligned_center[0] + float(params.get("MASK_DX", 0.0)),
        aligned_center[1] + float(params.get("MASK_DY", 0.0)),
    )
    eff_wl = float(params["WL"])
    pixscale_ins = float(params["PIXSCALE_INS"])
    apdiam = float(params.get("APDIAM_M", 6.5))
    lyot_sm_rad = float(params.get("LYOT_LAMBDA_OVER_D", 3.0))
    ctrl_rad = float(params.get("CTRL_LAMBDA_OVER_D", 24.0))
    in_scaling = float(params.get("MASK_IN_SCALING", 1.0))
    out_scaling = float(params.get("MASK_OUT_SCALING", 1.0))
    inc = float(params["inc_init"])

    reselem = eff_wl * 1e-6 / apdiam * 180 / np.pi * 3600
    reselem_pix = reselem / pixscale_ins
    coron_reg = lyot_sm_rad * reselem_pix
    seeing_limited = ctrl_rad * reselem_pix * np.sqrt(2)

    mask2generatehalo = control_region_mask(
        image_shape,
        coronrad=coron_reg,
        seeinglimited=seeing_limited,
    )
    disk_exclusion = make_disk_exclusion_mask(
        image_shape[0],
        float(params["pa_init"]),
        inc,
        convert.au_to_pix(float(params["r1_init"]), pixscale_ins, float(params["DISTANCE_STAR"]))
        - in_scaling / np.cos(np.radians(inc)),
        convert.au_to_pix(float(params["r2_init"]), pixscale_ins, float(params["DISTANCE_STAR"]))
        + out_scaling / np.cos(np.radians(inc)),
        aligned_center=mask_center,
    )

    mask_speckles = float(params.get("MASK_SPECKLES", 0.0))
    y, x = np.indices(image_shape, dtype=float)
    rho2d = np.sqrt((x - aligned_center[0]) ** 2 + (y - aligned_center[1]) ** 2)
    speckle_exclusion = np.ones(image_shape, dtype=float)
    speckle_exclusion[rho2d < mask_speckles] = 0.0
    dm_speckle_exclusion = make_speckle_exclusion_mask(image_shape, speckle_ellipses)

    static_fitting_mask = binarize_mask(mask2generatehalo * speckle_exclusion * dm_speckle_exclusion)
    disk_exclusion = binarize_mask(disk_exclusion)
    return static_fitting_mask, disk_exclusion


def rotate_binary_mask(mask, parang):
    """Rotate a binary mask into the science frame and re-binarize it."""
    rotated_mask = ndi.rotate(
        mask,
        # I checked the disk position and it's positive
        angle=float(parang),
        reshape=False,
        order=0,
        mode="constant",
        cval=0.0,
    )
    return binarize_mask(rotated_mask)


def mask2minimize_for_parang(static_fitting_mask, disk_exclusion_mask, parang):
    """Combine static detector exclusions with the rotated disk exclusion mask."""
    rotated_disk_exclusion = rotate_binary_mask(disk_exclusion_mask, parang)

    return binarize_mask(static_fitting_mask * rotated_disk_exclusion)
    # return binarize_mask(rotated_disk_exclusion)


def fit_wdh_components(science_frame, rotated_components, psf_estimate, fitting_mask, noise_map):
    """Fit rotated WDH components plus a constant background term."""
    # #debug view the masked rotated components
    # comp_vmin = np.percentile(rotated_components[0], 1.0)
    # comp_vmax = np.percentile(rotated_components[0], 99.0)
    # plt.imshow(rotated_components[-1], origin="lower", vmin=comp_vmin, vmax=comp_vmax)
    # plt.colorbar()
    # plt.show()
    # exit()
    # vmin = np.percentile(science_frame, 1.0)
    # vmax = np.percentile(science_frame, 99.0)
    # plt.imshow(fitting_mask*science_frame, origin="lower", vmin=vmin, vmax=vmax)
    # plt.colorbar()
    # plt.show()
    # exit()
    science_vector = science_frame.ravel()
    noise_vector = noise_map.ravel()
    component_vectors = [component.ravel() for component in rotated_components]
    psf_est_vec = psf_estimate.ravel()
    finite_mask = (
        np.isfinite(science_vector)
        & np.isfinite(noise_vector)
        & (noise_vector > 0.0)
        & (fitting_mask.ravel() > 0.5)
    )
    for vector in component_vectors:
        finite_mask &= np.isfinite(vector)

    if not np.any(finite_mask):
        raise ValueError("No finite pixels are available for the WDH component fit.")

    component_columns = [vector[finite_mask] for vector in component_vectors]
    component_l1_norms = np.asarray(
        [np.nansum(np.abs(column)) for column in component_columns],
        dtype=np.float64,
    )
    safe_component_l1_norms = np.where(component_l1_norms > 0.0, component_l1_norms, 1.0)
    normalized_component_columns = [
        column / norm
        for column, norm in zip(component_columns, safe_component_l1_norms)
    ]
    design_matrix = np.column_stack(
        normalized_component_columns
        # + [np.ones(np.count_nonzero(finite_mask), dtype=np.float64)]
        + [psf_est_vec[finite_mask]]
    )
    target_vector = science_vector[finite_mask]
    weights = 1.0 / noise_vector[finite_mask]
    weighted_design_matrix = design_matrix * weights[:, None]
    weighted_target_vector = target_vector * weights
    solution, residual = nnls(weighted_design_matrix, weighted_target_vector)
    # solution, residuals, rank, singular_values = np.linalg.lstsq(weighted_design_matrix, weighted_target_vector, rcond=None)
    rank = np.linalg.matrix_rank(weighted_design_matrix)
    singular_values = np.linalg.svd(weighted_design_matrix, compute_uv=False)
    residuals = np.asarray([residual**2], dtype=np.float64)

    return (
        solution[:-1],
        safe_component_l1_norms,
        float(solution[-1]),
        residuals,
        rank,
        singular_values,
    )# , delta_chi2
    # return solution, float(0.0), residuals, rank, singular_values, delta_chi2


def zero_negative_coefficients(coefficients):
    """Treat negative component fits as non-detections for subtraction and reporting."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    return np.where(coefficients < 0.0, 0.0, coefficients)


def subtract_wdh_components(science_frame, rotated_components, coefficients, psf_scaling, psf_estimate):
    """Subtract the fitted WDH component model from one science frame."""
    model = np.zeros_like(science_frame, dtype=np.float64)
    for coefficient, component in zip(coefficients, rotated_components):
        model += coefficient * component
    model += psf_scaling * psf_estimate
    cleaned = science_frame - model
    # #debug view the model and the science frame side by side
    # print(f"Coefficients: {coefficients}")
    # vmin = np.percentile(science_frame, 5.0)
    # vmax = np.percentile(science_frame, 95.0)
    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(model, origin="lower", vmin=vmin, vmax=vmax)
    # cax = axs[1].imshow(cleaned, origin="lower", vmin=vmin, vmax=vmax)
    # plt.colorbar(cax)
    # plt.show()
    # exit()
    cleaned[~np.isfinite(science_frame)] = science_frame[~np.isfinite(science_frame)]
    return cleaned


def output_filename(science_path, output_dir):
    """Return the destination filename for one WDH-subtracted science frame."""
    return output_dir / f"{science_path.stem}_wdhsub.fits"


def add_history(header, science_path, component_files, coefficients, background):
    """Annotate an output FITS header with WDH subtraction provenance."""
    header = header.copy()
    header["WDHSUB"] = (True, "WDH components subtracted")
    header["WDHNCOMP"] = (len(component_files), "Number of fitted WDH components")
    header["WDHSRC"] = (science_path.name[:68], "Input science frame")
    header["WDHBG"] = (float(background), "Fitted background/intercept term")
    header["WDHPSF"] = (True, "Median PSF subtracted during WDH coefficient fit")
    header["WDHL1"] = (True, "WDH coefficients fit with L1-normalized components")
    for idx, coefficient in enumerate(coefficients, start=1):
        header[f"WDHCO{idx}"] = (
            float(coefficient),
            f"L1-normalized coefficient for WDH component {idx}",
        )
    for idx, component_file in enumerate(component_files, start=1):
        header.add_history(f"WDH component {idx}: {component_file.name}")
    return header

def parse_date_obs(date_obs):
    """Parse DATE-OBS values with optional fractional seconds and UTC suffix."""
    date_obs_str = str(date_obs).strip()
    if date_obs_str.endswith("Z"):
        date_obs_str = f"{date_obs_str[:-1]}+00:00"
    return datetime.fromisoformat(date_obs_str)


def date_obs_to_minutes_since_start(date_obs_values):
    """Convert DATE-OBS values to integer minutes since the first frame timestamp."""
    if not date_obs_values:
        return []

    timestamps = [parse_date_obs(date_obs) for date_obs in date_obs_values]
    start_time = timestamps[0]
    return [int((timestamp - start_time).total_seconds() / 60.0) for timestamp in timestamps]


def image_for_thumbnail(image):
    """Scale a model component image for a compact plot thumbnail."""
    image = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros_like(image)

    scale = np.nanpercentile(np.abs(image[finite]), 99)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0
    return np.clip(image / scale, -1.0, 1.0)


def plot_component_key(fig, grid_slot, components, component_files, colors):
    """Draw a legend-style key with one thumbnail per WDH model component."""
    n_components = len(components)
    subgrid = grid_slot.subgridspec(1, n_components, wspace=0.10, hspace=-0.30)
    for idx, (component, component_file) in enumerate(zip(components, component_files)):
        ax_key = fig.add_subplot(subgrid[0, idx])
        ax_key.imshow(
            # image_for_thumbnail(component),
            component+0.1,
            origin="lower", cmap="bone",
            norm=LogNorm())
        ax_key.set_xticks([])
        ax_key.set_yticks([])
        ax_key.set_xlabel(
            f"Component {idx + 1}",
            color=colors[idx],
            fontweight="bold",
            backgroundcolor="black",
            fontsize=15,
            labelpad=-22,
        )


def plot_coefficients(
    coefficients_by_frame,
    date_obs_values,
    output_dir,
    components,
    component_files,
    band_name,
    file_prefix,
):
    """Plot fitted WDH component coefficients versus minutes from observation start."""
    if coefficients_by_frame is None or len(coefficients_by_frame) == 0:
        return None
    fig_min_fontsize = 20
    fig_maj_fontsize = 36
    minutes_since_obs_start = date_obs_to_minutes_since_start(date_obs_values)
    coefficient_array = np.asarray(coefficients_by_frame, dtype=np.float64)
    output_path = output_dir / f"{file_prefix}_wdh_component_coefficients"
    n_frames, n_components = coefficient_array.shape
    frame_spacing = 1.0
    frame_positions = np.arange(n_frames, dtype=float) * frame_spacing
    group_width = 0.82 * frame_spacing
    bar_width = group_width / max(n_components, 1)
    # bar_width = 0.5
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    if n_components > len(colors):
        colors = [plt.cm.tab10(idx % 10) for idx in range(n_components)]

    fig = plt.figure(figsize=(18, 7.2))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[2.0, 4.6],
        width_ratios=[2.0, 1.8],
        hspace=0.08,
        wspace=0.04,
        top=0.94,
        bottom=0.09,
        left=0.055,
        right=0.99,
    )
    ax = fig.add_subplot(grid[1, :])
    for idx in range(n_components):
        offset = (idx - (n_components - 1) / 2.0) * bar_width
        ax.bar(
            frame_positions + offset,
            coefficient_array[:, idx],
            width=bar_width,
            color=colors[idx],
            label=f"Component {idx + 1}",
            alpha=0.6,
        )
        # ax.plot(
        #     frame_positions + offset,
        #     coefficient_array[:, idx],
        #     color=colors[idx],
        #     label=f"Component {idx + 1}",
        #     linewidth=1.0,
        #     marker="o",
        #     alpha=0.5,
        #     markersize=4,
        # )
    tick_step = max(1, int(np.ceil(n_frames / 12)))
    tick_indices = np.arange(0, n_frames, tick_step)
    ax.set_xticks(frame_positions[tick_indices])
    ax.set_xticklabels(
        [str(minutes_since_obs_start[idx]) for idx in tick_indices],
        rotation=0,
        fontsize=fig_min_fontsize,
    )
    ax.tick_params(axis="y", labelsize=fig_min_fontsize)
    ax.set_xlabel("Minutes since observation start", fontsize=fig_min_fontsize)
    ax.set_ylabel("Normalized Coefficient", fontsize=fig_min_fontsize)
    ax.set_title(
        f"{band_name}",
        fontsize=fig_maj_fontsize,
        pad=8,
        fontweight="bold",
        loc="left",
    )
    # ax.legend(loc="upper right", fontsize=16)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax_title_space = fig.add_subplot(grid[0, 0])
    ax_title_space.axis("off")
    plot_component_key(fig, grid[0, 1], components, component_files, colors)
    fig.savefig(f"{output_path}.png", dpi=150)
    fig.savefig(f"{output_path}.pdf", dpi=150)
    plt.close(fig)
    return f"{output_path}.png", f"{output_path}.pdf"

def determine_relative_weights(components):
    """Determine the relative weights of the WDH components."""
    components_arr = np.asarray(components)
    weights = []
    for component in components:
        weights.append(np.sum(component) / np.sum(components_arr))
    return weights

def make_noise_map(median_psf_estimate):
    """Make a noise map by estimating the Poisson variance of the median PSF estimate."""
    denom = np.sqrt(median_psf_estimate)
    denom_safe = np.where(denom > 0.0, denom, 1.0)
    return 1.0 / denom_safe

def process_science_frame(
    science_path,
    components,
    component_files,
    output_dir,
    static_fitting_mask,
    disk_exclusion_mask,
    noise_map,
    median_psf_estimate,
    save_frame=False,
    subtract_median_profile=False,
    science_frame_override=None,
    iteration_idx=1,
    n_iterations=1,
):
    """Fit and subtract WDH components from one science frame."""
    science_frame, header = read_science_frame(science_path)
    if science_frame_override is not None:
        science_frame = np.asarray(science_frame_override, dtype=np.float64)
    parang = float(header["PARANG"])
    if "DATE-OBS" not in header:
        raise KeyError(f"Science frame is missing DATE-OBS header keyword: {science_path}")
    date_obs = header["DATE-OBS"]

    for component_file, component in zip(component_files, components):
        if component.shape != science_frame.shape:
            raise ValueError(
                f"Shape mismatch for {science_path.name}: science frame has {science_frame.shape}, "
                f"but {component_file.name} has {component.shape}"
            )

    science_for_fit = science_frame# - median_psf_estimate
    rotated_components = rotate_components(components, parang)
    mask2minimize = mask2minimize_for_parang(static_fitting_mask, disk_exclusion_mask, parang)
    subtracted_components = []
    for component in rotated_components:
        if subtract_median_profile:
            subtracted_component, _ = subtract_median_profile_np(component)
        else:
            subtracted_component = component
        subtracted_components.append(subtracted_component)
    fit_components = [
        subtracted_component #- median_psf_estimate
        for subtracted_component in subtracted_components
    ]
    # rel_weights = determine_relative_weights(rotated_components)
    coefficients, component_l1_norms, psf_scaling, residuals, rank, singular_values = fit_wdh_components(
        science_for_fit,
        fit_components,
        median_psf_estimate,
        mask2minimize,
        noise_map,
    )
    # print(f"Parang: {parang}")
    # if parang > 1.0 and parang < 2.0:
    #     print(f"Parang: {parang}")
    #     print(f"Coefficients: {coefficients}")
    #     plt.imshow(mask2minimize*rotated_components[-1], origin="lower")
    #     plt.colorbar()
    #     plt.show()
    #     exit()
    # coefficients = zero_negative_coefficients(coefficients)
    weighted_coefficients = np.asarray(coefficients)# * np.asarray(rel_weights)
    # print(f"Relative weights: {rel_weights}")
    # print(f"Weighted coefficients: {weighted_coefficients}")
    # print(f"Sum of model component 1: {np.sum(rotated_components[0])}")
    # print(f"Sum of model component 2: {np.sum(rotated_components[1])}")
    # print(f"Sum of model component 3: {np.sum(rotated_components[2])}")
    # exit()
    # cleaned = subtract_wdh_components(science_frame, rotated_components, coefficients)
    subtraction_coefficients = coefficients / component_l1_norms
    cleaned = subtract_wdh_components(
        science_for_fit,
        rotated_components,
        subtraction_coefficients,
        psf_scaling,
        median_psf_estimate,
    )
    cleaned, _ = subtract_median_profile_np(cleaned)
    header_out = add_history(header, science_path, component_files, coefficients, psf_scaling)
    header_out["RADPROF"] = (True, "Median radial profile subtracted after WDH subtraction")
    header_out["WDHITER"] = (int(iteration_idx), "WDH subtraction iteration for this output")
    header_out["WDHNITER"] = (int(n_iterations), "Total WDH subtraction iterations requested")
    header_out["WDHRANK"] = (int(rank), "Rank of WDH least-squares design matrix")
    if residuals.size:
        header_out["WDHSSR"] = (float(residuals[0]), "WDH least-squares residual sum of squares")
    if singular_values.size and singular_values[-1] != 0:
        header_out["WDHSCOND"] = (
            float(singular_values[0] / singular_values[-1]),
            "WDH design matrix condition estimate",
        )

    output_path = output_filename(science_path, output_dir)
    if save_frame:
        fits.writeto(output_path, science_frame.astype(np.float32), header=header, overwrite=True)
    return output_path, coefficients, psf_scaling, date_obs, weighted_coefficients, cleaned


def run_subtraction(args):
    """Run WDH subtraction for every science frame in BAND_DIR."""
    params, yaml_path = load_params(args.param_file)
    band_dir = resolve_band_dir(params)
    output_dir = band_dir / "wdh_subtracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    subtract_median_profile = params.get("RPROFSUB", False)
    n_iterations = int(args.n_iterations)
    if n_iterations < 1:
        raise ValueError("--n-iterations must be at least 1")

    science_frames = find_science_frames(band_dir)
    component_files = find_component_files(band_dir)
    components = read_components(component_files)
    noise_map, noise_path = load_noise_map(params, band_dir, components[0].shape)
    # noise_map = make_noise_map(median_psf_estimate)
    speckle_ellipses = load_speckle_ellipses(args.sparkle_coords_file)
    static_fitting_mask, disk_exclusion_mask = build_fitting_masks(
        params,
        components[0].shape,
        speckle_ellipses,
    )
    fits.writeto(
        output_dir / "wdh_static_fit_mask.fits",
        static_fitting_mask.astype(np.float32),
        overwrite=True,
    )
    fits.writeto(
        output_dir / "wdh_disk_exclusion_mask.fits",
        disk_exclusion_mask.astype(np.float32),
        overwrite=True,
    )
    print(f"Read parameter file: {yaml_path}")
    print(f"Science frames: {len(science_frames)}")
    print(f"WDH components: {len(component_files)}")
    print(f"Noise map: {noise_path}")
    print(f"Output directory: {output_dir}")
    print(f"WDH subtraction iterations: {n_iterations}")

    outputs = []
    date_obs_values = []
    accumulated_coefficients_by_frame = None
    current_science_frames = [read_science_frame(science_path)[0] for science_path in science_frames]
    for iteration_idx in range(1, n_iterations + 1):
        print(f"\nWDH subtraction iteration {iteration_idx}/{n_iterations}")
        median_psf_estimate = build_median_psf_estimate_from_frames(
            current_science_frames,
            components[0].shape,
        )
        fits.writeto(
            output_dir / f"median_science_psf_estimate_iter{iteration_idx}.fits",
            median_psf_estimate.astype(np.float32),
            overwrite=True,
        )
        fits.writeto(
            output_dir / "median_science_psf_estimate.fits",
            median_psf_estimate.astype(np.float32),
            overwrite=True,
        )

        outputs = []
        next_science_frames = []
        iteration_coefficients_by_frame = []
        if iteration_idx == 1:
            date_obs_values = []
        for frame_idx, science_path in enumerate(science_frames):
            save_frame = (iteration_idx == n_iterations)
            output_path, coeffs, psf_scaling, date_obs, weighted_coeffs, cleaned = process_science_frame(
                science_path,
                components,
                component_files,
                output_dir,
                static_fitting_mask,
                disk_exclusion_mask,
                noise_map,
                median_psf_estimate,
                save_frame=save_frame,
                subtract_median_profile=subtract_median_profile,
                science_frame_override=current_science_frames[frame_idx],
                iteration_idx=iteration_idx,
                n_iterations=n_iterations,
            )
            if iteration_idx == 1:
                date_obs_values.append(date_obs)
            iteration_coefficients_by_frame.append(weighted_coeffs)
            next_science_frames.append(cleaned)
            coeff_str = ", ".join(f"{value:.6g}" for value in coeffs)
            print(
                f"{science_path.name} -> {output_path.name}; "
                f"coeffs=[{coeff_str}], psf_scaling={psf_scaling:.6g}"
            )
            outputs.append(output_path)

        iteration_coefficients_by_frame = np.asarray(iteration_coefficients_by_frame, dtype=np.float64)
        if accumulated_coefficients_by_frame is None:
            accumulated_coefficients_by_frame = iteration_coefficients_by_frame
        else:
            accumulated_coefficients_by_frame += iteration_coefficients_by_frame
        current_science_frames = next_science_frames
    
    band_name = params.get("BAND_NAME")
    file_prefix = params.get("FILE_PREFIX")
    # Normalize the coefficients by the global peak
    normalized_coefficients_by_frame = accumulated_coefficients_by_frame / np.max(accumulated_coefficients_by_frame)
    # normalized_coefficients_by_frame = accumulated_coefficients_by_frame

    plot_path = plot_coefficients(
        normalized_coefficients_by_frame,
        date_obs_values,
        output_dir,
        components,
        component_files,
        band_name,
        file_prefix,
    )
    if plot_path is not None:
        print(f"Coefficient plot: {plot_path}")

    return outputs


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Subtract best-fit WDH model components from coronagraphic science frames."
    )
    parser.add_argument(
        "-p",
        "--param_file",
        required=True,
        help="YAML parameter filename or path containing BAND_DIR.",
    )
    parser.add_argument(
        "--sparkle-coords-file",
        required=False,
        default=None,
        help="Optional text file with rows: x y major_axis minor_axis theta_deg.",
    )
    parser.add_argument(
        "--n-iterations",
        type=int,
        required=False,
        default=1,
        help="Number of iterative WDH fit/subtraction passes to run.",
    )
    return parser.parse_args()


def main():
    """Command-line entry point."""
    args = parse_args()
    run_subtraction(args)


if __name__ == "__main__":
    main()
