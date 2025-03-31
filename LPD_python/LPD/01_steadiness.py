import os
import numpy as np
import rasterio
import xarray as xr
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
    valid_indices = np.where(valid_mask)[0]
    first = valid_indices[0]
    last = valid_indices[-1]
    return x[last] - x[first]

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
    return SteadInd, slope, mtid

if __name__ == "__main__":
    input_raster = "/scratch/gianofe/SumNDVI_correction_rev4/NDVI_SUM_multiband_rev4.tif"
    output_raster = "/scratch/gianofe/SumNDVI_correction_rev4/outputs/SteadInd_2008_2023.tif"
    slope_raster = "/scratch/gianofe/SumNDVI_correction_rev4/outputs/slope_2008_20123.tif"
    mtid_raster = "/scratch/gianofe/SumNDVI_correction_rev4/outputs/mtid_2008_2023.tif"

    with rasterio.open(input_raster) as src:
        data = src.read()
        profile = src.profile

    start_year = 1999
    selected_years = np.arange(2004, 2019 + 1)
    band_indices = selected_years - start_year
    data = data[band_indices, :, :]

    print("Calcolo della Steadiness Index...")
    result, slope, mtid = compute_steadiness(data, selected_years)

    profile.update(count=1, dtype='float32')

    print(f"Scrittura dello Steadiness Index su: {output_raster}")
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(result.astype('float32'), 1)

    print(f"Scrittura dello slope su: {slope_raster}")
    with rasterio.open(slope_raster, 'w', **profile) as dst:
        dst.write(slope.astype('float32'), 1)

    print(f"Scrittura del MTID su: {mtid_raster}")
    with rasterio.open(mtid_raster, 'w', **profile) as dst:
        dst.write(mtid.astype('float32'), 1)

    print("Completato.")
