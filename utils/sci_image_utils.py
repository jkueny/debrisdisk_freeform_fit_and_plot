import re
import pandas as pd
from astropy.io import fits

def parang_sort(filename):
    """Extracts the float value between '2x2bin_' and '_parang'."""
    match = re.search(r'2x2bin_([-+]?\d*\.\d+|\d+)_parang', filename)
    if match:
        return float(match.group(1))  # Convert extracted string to float
    return float('inf')  # Assign an arbitrary large value if no match is found

def prep_image_frames_parangs(filelist, parquet_file, time_bin_sz="min"):
    wind_directions1 = []
    wind_directions2 = []
    derot_angs = []
    frames = []
    df = pd.read_parquet(parquet_file)
    df = df.reset_index()
    for ea,name in enumerate(filelist):
        dat_unit, hdr_unit = fits.getdata(name,header=True)
        # print(name, hdr_unit['PARANG'])
        derot_angs.append(hdr_unit['PARANG'])
        frames.append(dat_unit)
        df["ts6"] = df["timestamp"].str[:20]
        df["ts_dt"] = pd.to_datetime(df["ts6"], format="%Y%m%d%H%M%S%f", utc=True)
        obs = hdr_unit["DATE-OBS"]
        obs_ts = pd.to_datetime(obs, utc=True)
        df = df.set_index("ts_dt").sort_index()
        # TODO generalize this for any temporal bin size (within reason)
        obs_round = obs_ts.round(time_bin_sz) #round to nearest minute, for now
        nearest = df.reindex([obs_round], method="nearest").iloc[0]
        # print(f"Science frame at {obs_ts} → matched wind at {obs_round}")
        # one or both directions may be np.nan, meaning no wind detected
        winddir1 = nearest["direction_1"]
        winddir2 = nearest["direction_2"]
        if ea < 5:
            print(obs, nearest["timestamp"], winddir1, winddir2)
        wind_directions1.append(winddir1)
        wind_directions2.append(winddir2)
    return np.asarray(frames), np.asarray(derot_angs),  \
        np.asarray(wind_directions1), np.asarray(wind_directions2)