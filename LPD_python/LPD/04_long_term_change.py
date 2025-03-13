import os
import numpy as np
import xarray as xr
import rasterio


def long_term_change(steadiness_index, baseline_levels, state_change, filename=""):
    print("Starting Long Term Change classification...")

    if isinstance(steadiness_index, str):
        print(f"Opening steadiness index raster: {steadiness_index}")
        with rasterio.open(steadiness_index) as src:
            steadiness_index = xr.DataArray(src.read(1), dims=("y", "x"))
            profile = src.profile

    if isinstance(baseline_levels, str):
        print(f"Opening baseline levels raster: {baseline_levels}")
        with rasterio.open(baseline_levels) as src:
            baseline_levels = xr.DataArray(src.read(1), dims=("y", "x"))

    if isinstance(state_change, str):
        print(f"Opening state change raster: {state_change}")
        with rasterio.open(state_change) as src:
            state_change = xr.DataArray(src.read(1), dims=("y", "x"))

    print("Reclassifying Steadiness Index and Baseline Levels...")
    steadiness_baseline = np.full_like(steadiness_index, np.nan)

    mapping_baseline = {
        (1, 1): 1, (1, 2): 2, (1, 3): 3,
        (2, 1): 4, (2, 2): 5, (2, 3): 6,
        (3, 1): 7, (3, 2): 8, (3, 3): 9,
        (4, 1): 10, (4, 2): 11, (4, 3): 12
    }

    for (si, bl), val in mapping_baseline.items():
        steadiness_baseline[(steadiness_index == si) & (baseline_levels == bl)] = val

    print("Reclassifying Long Term Change categories...")
    long_term_change_map = np.full_like(state_change, np.nan)

    mapping_change = {
        (1, 1): 1, (1, 2): 2, (1, 3): 3,
        (2, 1): 4, (2, 2): 5, (2, 3): 6,
        (3, 1): 7, (3, 2): 8, (3, 3): 9,
        (4, 1): 10, (4, 2): 10, (4, 3): 10,
        (5, 1): 11, (5, 2): 11, (5, 3): 11,
        (6, 1): 12, (6, 2): 12, (6, 3): 12,
        (7, 1): 13, (7, 2): 13, (7, 3): 13,
        (8, 1): 14, (8, 2): 14, (8, 3): 14,
        (9, 1): 15, (9, 2): 15, (9, 3): 15,
        (10, 1): 16, (10, 2): 17, (10, 3): 18,
        (11, 1): 19, (11, 2): 20, (11, 3): 21,
        (12, 1): 22, (12, 2): 22, (12, 3): 22
    }

    for (sb, sc), val in mapping_change.items():
        long_term_change_map[(steadiness_baseline == sb) & (state_change == sc)] = val

    if filename:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        print(f"Saving long-term change raster to {filename}")
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(long_term_change_map, 1)

    print("Long Term Change classification complete!")
    return long_term_change_map


if __name__ == "__main__":
    steadiness_index_raster = "/home/gianofe/Documents/corrected/SteadIndxxl.tif"
    baseline_levels_raster = "/home/gianofe/Documents/corrected/baselinexxl.tif"
    state_change_raster = "/home/gianofe/Documents/corrected/state_changexxl.tif"
    output_raster = "/home/gianofe/Documents/corrected/long_term_change.tif"

    print("Running Long Term Change function...")
    Long_Term_Change_Map = long_term_change(
        steadiness_index=steadiness_index_raster,
        baseline_levels=baseline_levels_raster,
        state_change=state_change_raster,
        filename=output_raster
    )
    print("Process completed successfully!")
