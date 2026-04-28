#!/usr/bin/env python3
"""Run pyKLIP on all FITS files in a directory."""

import argparse
import glob
import os

import astropy.io.fits as fits
import numpy as np
import yaml

from ffortissimo.dev.pyklip.instruments.Instrument import GenericData
import ffortissimo.dev.pyklip.parallelized as parallelized


def _load_params(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _fits_files_in_dir(input_dir):
    patterns = ["*.fits", "*.FITS", "*.fit", "*.FIT"]
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(os.path.join(input_dir, pat)))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No FITS files found in: {input_dir}")
    return paths


def _read_frames_and_parangs(fits_paths):
    frames = []
    parangs = []
    for path in fits_paths:
        data, hdr = fits.getdata(path, header=True)
        arr = np.asarray(data)
        if arr.ndim > 2:
            # Collapse any leading dimensions and keep first image plane.
            arr = arr.reshape((-1, arr.shape[-2], arr.shape[-1]))[0]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D image in {path}, got shape {arr.shape}")
        frames.append(arr.astype(np.float32, copy=False))
        parangs.append(float(hdr.get("PARANG", 0.0)))
    return np.asarray(frames), np.asarray(parangs, dtype=float)


def _center_for_shape(params, image_shape):
    aligned_center = params.get("ALIGNED_CENTER")
    if aligned_center is not None and len(aligned_center) == 2:
        return np.asarray(aligned_center, dtype=float)
    y, x = image_shape
    return np.asarray([(x - 1) / 2.0, (y - 1) / 2.0], dtype=float)


def run_pyklip(param_file, fits_dir):
    params = _load_params(param_file)
    fits_paths = _fits_files_in_dir(fits_dir)
    frames, parangs = _read_frames_and_parangs(fits_paths)

    aligned_center = _center_for_shape(params, frames[0].shape)
    centers = np.repeat(aligned_center[None, :], frames.shape[0], axis=0)

    iwa = float(params.get("IWA", 0))
    owa_default = min(frames.shape[1], frames.shape[2]) / 2.0
    owa = float(params.get("OWA", owa_default))

    dataset = GenericData(
        frames,
        centers=centers,
        parangs=parangs,
        IWA=iwa,
        filenames=np.asarray(fits_paths),
    )
    dataset.OWA = owa

    klmode_number = int(params.get("KLMODE_NUMBER", 1))
    annuli = int(params.get("ANNULI", 1))
    subsections = int(params.get("SUBSECTIONS", 1))
    mode = str(params.get("MODE", "ADI"))
    minrot = float(params.get("MOVE_HERE", params.get("MINROT", 0)))
    hp_filter = params.get("HP_FILTER", False)
    file_prefix = str(params.get("FILE_PREFIX", "pyklip"))

    print(f"Fits directory: {fits_dir}")
    print(f"KL mode number: {klmode_number}")
    print(f"Annuli: {annuli}")
    print(f"Subsections: {subsections}")
    print(f"Mode: {mode}")
    print(f"Minrot: {minrot}")
    print(f"HP filter: {hp_filter}")
    print(f"File prefix: {file_prefix}")

    output_dir = os.path.join(fits_dir, "pyklip_results")
    os.makedirs(output_dir, exist_ok=True)

    parallelized.klip_dataset(
        dataset,
        mode=mode,
        outputdir=output_dir,
        fileprefix=file_prefix,
        annuli=annuli,
        subsections=subsections,
        numbasis=[klmode_number],
        minrot=minrot,
        # highpass=hp_filter,
        calibrate_flux=False,
        aligned_center=aligned_center,
        time_collapse="median",
    )

    print(f"Processed {len(fits_paths)} FITS files.")
    print(f"Saved pyKLIP outputs to: {output_dir}")
    print(f"Reduced image file prefix: {file_prefix}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run pyKLIP on FITS files in a directory.")
    parser.add_argument(
        "-p",
        "--param_file",
        required=True,
        help="Path to YAML parameter file with pyKLIP settings.",
    )
    parser.add_argument(
        "-d",
        "--fits_dir",
        required=True,
        help="Directory containing FITS files to reduce.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    run_pyklip(cli_args.param_file, cli_args.fits_dir)
