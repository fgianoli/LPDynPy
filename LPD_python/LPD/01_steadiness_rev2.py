### la rev 2 rispetto alla vecchia versione è molto più veloce e permette di calcolare anche più serie temporali alla volta

import os
import numpy as np
import rasterio
from numba import jit

@jit(nopython=True)
def slp_lm(x, yrs):
    valid_mask = ~np.isnan(x)
    valid_yrs = yrs[valid_mask]
    valid_x = x[valid_mask]
    if len(valid_x) == 0:
        return np.nan
    elif len(valid_x) == 1:
        return 0
    x_mean = valid_yrs.mean()
    y_mean = valid_x.mean()
    num = np.sum((valid_yrs - x_mean) * (valid_x - y_mean))
    den = np.sum((valid_yrs - x_mean) ** 2)
    if den == 0:
        return 0
    return num / den

@jit(nopython=True)
def mtid_function(x):
    valid_mask = ~np.isnan(x)
    if np.all(~valid_mask):
        return np.nan
    elif np.sum(valid_mask) == 1:
        return 0
    years1 = np.max(np.where(valid_mask)[0])
    return np.nansum(x[years1] - x[valid_mask])

def compute_steadiness(data, yrs):
    bands, height, width = data.shape
    slope = np.full((height, width), np.nan)
    mtid = np.full((height, width), np.nan)

    for i in range(height):
        for j in range(width):
            ts = data[:, i, j]
            slope[i, j] = slp_lm(ts, yrs)
            mtid[i, j] = mtid_function(ts)

    SteadInd = np.full_like(slope, np.nan)
    SteadInd[(slope < 0) & (mtid > 0)] = 1
    SteadInd[(slope < 0) & (mtid < 0)] = 2
    SteadInd[(slope > 0) & (mtid < 0)] = 3
    SteadInd[(slope > 0) & (mtid > 0)] = 4
    SteadInd[(slope == 0) | (mtid == 0)] = 0
    return SteadInd

# === PARAMETRI ===
input_raster = "/scratch/gianofe/SumNDVI_rev2/sumndvi_newcorrection.tif"
output_folder = "/home/gianofe/Documents/corrected/steadiness_intervalli/"
start_year_in_raster = 1999
year_ranges = [(2000, 2015), (2008, 2023), (2004, 2019)] ## cambiare questi intervalli di anni

# === LETTURA RASTER UNA VOLTA SOLA ===
with rasterio.open(input_raster) as src:
    full_data = src.read()
    profile = src.profile

# === CICLO SUGLI INTERVALLI TEMPORALI ===
for (start, end) in year_ranges:
    print(f"\n🕒 Calcolo per intervallo {start}–{end}...")
    selected_years = np.arange(start, end + 1)
    band_indices = selected_years - start_year_in_raster
    data = full_data[band_indices, :, :]

    result = compute_steadiness(data, selected_years)

    output_path = os.path.join(output_folder, f"SteadInd_{start}_{end}.tif")
    os.makedirs(output_folder, exist_ok=True)

    profile.update(count=1, dtype='float32')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(result.astype('float32'), 1)

    print(f"✅ Salvato: {output_path}")

print("\n🎉 Tutti gli intervalli elaborati!")
