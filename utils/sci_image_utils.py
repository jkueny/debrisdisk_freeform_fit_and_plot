import re
import numpy as np
import pandas as pd
from astropy.io import fits

def parang_sort(filename):
    """Extracts the float value between '2x2bin_' and '_parang'."""
    match = re.search(r'2x2bin_([-+]?\d*\.\d+|\d+)_parang', filename)
    if match:
        return float(match.group(1))  # Convert extracted string to float
    return float('inf')  # Assign an arbitrary large value if no match is found

import numpy as np
import pandas as pd
from astropy.io import fits



def prep_image_frames_parangs(filelist,
                              csv_file,
                              n_models: int = 1,
                              time_bin_sz: str = "1s"):
    """
    Parameters
    ----------
    filelist : list[str]
        Paths to science thumbnails (FITS files).
    parquet_file : str | Path
        Parquet with timestamp, direction_1 … direction_3 columns.
    n_models : int, optional
        How many WDH models to pull (1-3).  Clamped internally.
    time_bin_sz : str, optional
        Pandas‐style bin size, e.g. "min", "5s", "10min".

    Returns
    -------
    frames            : (M, …)   ndarray   science images
    derot_angs        : (M,)     ndarray   parallactic angles
    wind_dirs         : (n, M)   ndarray   padded wind PAs
    valid_mask        : (n, M)   bool      True where PA is real
    """
    SENTINEL = -1.0          # <-- safe “empty” value for JAX
    VALID_RANGE = (0.0, 360.0)
    # ------- sanity ---------------------------------------------------------
    n_models = int(n_models)
    if not (1 <= n_models <= 3):
        raise ValueError("n_models must be 1, 2, or 3")

    # ------- load & pre-process wind table ---------------------------------
    # df = pd.read_parquet(parquet_file)
    df_long = pd.read_csv(csv_file)
    # df_long = df_long.reset_index()
    # print(df.index)
    ts_short = df_long["timestamp"].str[:20]     # “YYYYmmddHHMMSSffffff”
    df_long["ts_dt"] = pd.to_datetime(ts_short, format="%Y%m%d%H%M%S%f", utc=True)
    # df_long = df_long.set_index("ts_dt").sort_index()

    df_wide = (df_long
            .pivot_table(index="ts_dt",
                         columns="layer",
                         values="direction",
                         aggfunc="first")     # first→because each (bin,layer) pair is unique
            .sort_index())                   # <— index is now unique!

    # df_wide.columns are Int64Index([1,2,3]); rename if desired:
    df_wide.columns = [f"direction_{i}" for i in df_wide.columns]

    # ------- storage -------------------------------------------------------
    wind_lists = [[] for _ in range(n_models)]
    derot_angs, frames = [], []

    # ------- loop through science frames -----------------------------------
    for each, fname in enumerate(filelist):
        data, hdr = fits.getdata(fname, header=True)
        frames.append(data)
        derot_angs.append(hdr["PARANG"])

        obs_ts  = pd.to_datetime(hdr["DATE-OBS"], utc=True)
        obs_bin = obs_ts.round(time_bin_sz)
        pos = df_wide.index.get_indexer([obs_bin], method="nearest")[0]
        row = df_wide.iloc[pos]
        # if each < 5:
        # #     print(obs_bin)
        #     print(row)

        for j in range(n_models):
            col = f"direction_{j}"
            val = row.get(col, np.nan)
            if pd.isna(val) or not (VALID_RANGE[0] <= val <= VALID_RANGE[1]):
                val = SENTINEL
            wind_lists[j].append(val)

    # pack up the outputs
    # wind_dirs shape (n_models, n_images)
    wind_dirs  = np.stack([np.asarray(lst, dtype=float) for lst in wind_lists]) 
    # valid mask same shape; (n_models, n_images)
    valid_mask = wind_dirs != SENTINEL

    return (np.asarray(frames),
            np.asarray(derot_angs, dtype=float),
            wind_dirs,
            valid_mask)
