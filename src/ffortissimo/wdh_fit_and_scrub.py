"""Subtract best-fit WDH components from individual science frames."""
import argparse
import glob
import os
import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import scipy.ndimage as ndi
import yaml
from astropy.io import fits
from ffortissimo.utils.improc_tools import subtract_median_profile_np


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
    """Return 1-3 best-fit WDH model component FITS paths from windfit_MCMC."""
    mcmc_dir = band_dir / "windfit_MCMC"
    if not mcmc_dir.is_dir():
        raise FileNotFoundError(f"Could not find windfit_MCMC directory: {mcmc_dir}")

    component_files = sorted(
        mcmc_dir.glob("*_BestModel_comp*.fits"),
        key=component_number,
    )
    if not component_files:
        raise FileNotFoundError(f"Could not find '*_BestModel_comp*.fits' files in {mcmc_dir}")
    if len(component_files) > 3:
        raise ValueError(
            f"Expected at most 3 WDH component files in {mcmc_dir}, found {len(component_files)}"
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


def rotate_components(components, parang):
    """Rotate WDH components into one science frame's parallactic angle."""
    return [
        ndi.rotate(
            component,
            angle=float(-parang),
            reshape=False,
            order=1,
            mode="constant",
            cval=0.0,
        )
        for component in components
    ]


def fit_wdh_components(science_frame, rotated_components):
    """Fit rotated WDH components to one science frame with linear least squares."""
    science_vector = science_frame.ravel()
    component_vectors = [component.ravel() for component in rotated_components]
    finite_mask = np.isfinite(science_vector)
    for vector in component_vectors:
        finite_mask &= np.isfinite(vector)

    if not np.any(finite_mask):
        raise ValueError("No finite pixels are available for the WDH component fit.")

    design_matrix = np.column_stack([vector[finite_mask] for vector in component_vectors])
    target_vector = science_vector[finite_mask]
    return np.linalg.lstsq(design_matrix, target_vector, rcond=None)


def zero_negative_coefficients(coefficients):
    """Treat negative component fits as non-detections for subtraction and reporting."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    return np.where(coefficients < 0.0, 0.0, coefficients)


def subtract_wdh_components(science_frame, rotated_components, coefficients):
    """Subtract the fitted WDH component model from one science frame."""
    model = np.zeros_like(science_frame, dtype=np.float64)
    for coefficient, component in zip(coefficients, rotated_components):
        model += coefficient * component

    cleaned = science_frame - model
    cleaned[~np.isfinite(science_frame)] = science_frame[~np.isfinite(science_frame)]
    return cleaned


def output_filename(science_path, output_dir):
    """Return the destination filename for one WDH-subtracted science frame."""
    return output_dir / f"{science_path.stem}_wdhsub.fits"


def add_history(header, science_path, component_files, coefficients):
    """Annotate an output FITS header with WDH subtraction provenance."""
    header = header.copy()
    header["WDHSUB"] = (True, "WDH components subtracted")
    header["WDHNCOMP"] = (len(component_files), "Number of fitted WDH components")
    header["WDHSRC"] = (science_path.name[:68], "Input science frame")
    for idx, coefficient in enumerate(coefficients, start=1):
        header[f"WDHCO{idx}"] = (
            float(coefficient),
            f"Least-squares coefficient for WDH component {idx}",
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
    subgrid = grid_slot.subgridspec(len(components), 1, hspace=0.35)
    for idx, (component, component_file) in enumerate(zip(components, component_files)):
        ax_key = fig.add_subplot(subgrid[idx, 0])
        ax_key.imshow(
            # image_for_thumbnail(component),
            component+0.1,
            origin="lower", cmap="bone",
            norm=LogNorm())
        ax_key.set_xticks([])
        ax_key.set_yticks([])
        ax_key.set_title(
            f"Component {idx + 1}",
            color=colors[idx],
            fontsize=7,
        )


def plot_coefficients(coefficients_by_frame, date_obs_values, output_dir, components, component_files):
    """Plot fitted WDH component coefficients versus minutes from observation start."""
    if not coefficients_by_frame:
        return None

    minutes_since_obs_start = date_obs_to_minutes_since_start(date_obs_values)
    coefficient_array = np.asarray(coefficients_by_frame, dtype=np.float64)
    output_path = output_dir / "wdh_component_coefficients.png"
    n_frames, n_components = coefficient_array.shape
    frame_positions = np.arange(n_frames, dtype=float)
    bar_width = min(0.9 / max(n_components, 1), 0.5)
    # bar_width = 0.5
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig = plt.figure(figsize=(12, 5.5))
    grid = fig.add_gridspec(1, 2, width_ratios=[5.0, 1.2], wspace=0.18)
    ax = fig.add_subplot(grid[0, 0])
    for idx in range(n_components):
        offset = (idx - (n_components - 1) / 2.0) * bar_width
        ax.bar(
            frame_positions + offset,
            coefficient_array[:, idx],
            width=bar_width,
            color=colors[idx],
            label=f"Component {idx + 1}",
        )
    tick_step = max(1, int(np.ceil(n_frames / 12)))
    tick_indices = np.arange(0, n_frames, tick_step)
    ax.set_xticks(frame_positions[tick_indices])
    ax.set_xticklabels([str(minutes_since_obs_start[idx]) for idx in tick_indices], rotation=45)
    ax.set_xlabel("Minutes since observation start")
    ax.set_ylabel("Coefficient")
    ax.set_title("WDH Model Components Through Observation")
    ax.legend(loc="upper right")
    plot_component_key(fig, grid[0, 1], components, component_files, colors)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path

def determine_relative_weights(components):
    """Determine the relative weights of the WDH components."""
    components_arr = np.asarray(components)
    weights = []
    for component in components:
        weights.append(np.sum(component) / np.sum(components_arr))
    return weights

def process_science_frame(
    science_path,
    components,
    component_files,
    output_dir,
    subtract_median_profile=False,
):
    """Fit and subtract WDH components from one science frame."""
    science_frame, header = read_science_frame(science_path)
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

    rotated_components = rotate_components(components, parang)
    subtracted_components = []
    for component in rotated_components:
        if subtract_median_profile:
            subtracted_component, _ = subtract_median_profile_np(component)
        else:
            subtracted_component = component
        subtracted_components.append(subtracted_component)
    rel_weights = determine_relative_weights(rotated_components)
    coefficients, residuals, rank, singular_values = fit_wdh_components(
        science_frame,
        rotated_components,
    )
    # coefficients = zero_negative_coefficients(coefficients)
    weighted_coefficients = np.asarray(coefficients) * np.asarray(rel_weights)
    # print(f"Relative weights: {rel_weights}")
    # print(f"Weighted coefficients: {weighted_coefficients}")
    # print(f"Sum of model component 1: {np.sum(rotated_components[0])}")
    # print(f"Sum of model component 2: {np.sum(rotated_components[1])}")
    # print(f"Sum of model component 3: {np.sum(rotated_components[2])}")
    # exit()
    cleaned = subtract_wdh_components(science_frame, rotated_components, coefficients)
    header_out = add_history(header, science_path, component_files, coefficients)
    header_out["WDHRANK"] = (int(rank), "Rank of WDH least-squares design matrix")
    if residuals.size:
        header_out["WDHSSR"] = (float(residuals[0]), "WDH least-squares residual sum of squares")
    if singular_values.size and singular_values[-1] != 0:
        header_out["WDHSCOND"] = (
            float(singular_values[0] / singular_values[-1]),
            "WDH design matrix condition estimate",
        )

    output_path = output_filename(science_path, output_dir)
    fits.writeto(output_path, cleaned.astype(np.float32), header=header_out, overwrite=True)
    return output_path, coefficients, date_obs, weighted_coefficients


def run_subtraction(param_file):
    """Run WDH subtraction for every science frame in BAND_DIR."""
    params, yaml_path = load_params(param_file)
    band_dir = resolve_band_dir(params)
    output_dir = band_dir / "wdh_subtracted"
    output_dir.mkdir(parents=True, exist_ok=True)
    subtract_median_profile = params.get("RPROFSUB", False)

    science_frames = find_science_frames(band_dir)
    component_files = find_component_files(band_dir)
    components = read_components(component_files)

    print(f"Read parameter file: {yaml_path}")
    print(f"Science frames: {len(science_frames)}")
    print(f"WDH components: {len(component_files)}")
    print(f"Output directory: {output_dir}")

    outputs = []
    date_obs_values = []
    coefficients_by_frame = []
    for science_path in science_frames:
        output_path, coeffs, date_obs, weighted_coeffs = process_science_frame(
            science_path,
            components,
            component_files,
            output_dir,
            subtract_median_profile=subtract_median_profile,
        )
        date_obs_values.append(date_obs)
        coefficients_by_frame.append(weighted_coeffs)
        coeff_str = ", ".join(f"{value:.6g}" for value in coeffs)
        print(f"{science_path.name} -> {output_path.name}; coeffs=[{coeff_str}]")
        outputs.append(output_path)

    plot_path = plot_coefficients(
        coefficients_by_frame,
        date_obs_values,
        output_dir,
        components,
        component_files,
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
    return parser.parse_args()


def main():
    """Command-line entry point."""
    args = parse_args()
    run_subtraction(args.param_file)


if __name__ == "__main__":
    main()
