import os
import numpy as np
import xarray as xr
import rasterio


def mean_years_function(x, years):
    print(f"Computing mean for indices: {years}")
    if np.isnan(x[years]).all():
        print("Warning: All selected bands contain only NaN values. Using global fallback.")
        fallback_value = np.nanmean(x)  # Media globale di tutto il dataset
        mean_values = np.full_like(x[0], fallback_value) if not np.isnan(fallback_value) else np.zeros_like(x[0])
    else:
        mean_values = np.nanmean(np.where(np.isnan(x[years]), 0, x[years]), axis=0, keepdims=False)
    if np.isnan(mean_values).all():
        raise ValueError("Computed mean contains only NaN values. Check input raster data.")
    return mean_values


def state_change(obj2process, yearsBaseline=3, changeNclass=1, filename=""):
    print("Starting state change classification...")

    if isinstance(obj2process, str):
        print(f"Opening raster file: {obj2process}")
        with rasterio.open(obj2process) as src:
            obj2process = xr.DataArray(src.read(), dims=("band", "y", "x"))
            profile = src.profile

    print(f"Raster shape: {obj2process.shape}")
    start_year = 1999
    selected_years = np.arange(2008, 2024)  # Anni dal 2008 al 2023
    band_indices = selected_years - start_year  # Converti anni in indici (base 0)

    print(f"Selected band indices: {band_indices}")
    if obj2process.shape[0] < max(band_indices) + 1:
        raise ValueError(f"Not enough bands ({obj2process.shape[0]}) to process years {yearsBaseline}.")

    obj2process = obj2process[band_indices, :, :]

    print("Computing average raster...")
    print(f"NaN count in selected bands: {np.isnan(obj2process).sum()}")
    print(f"Available bands: {obj2process.shape[0]}")

    avg_first = mean_years_function(obj2process, np.arange(yearsBaseline))
    avg_last = mean_years_function(obj2process, np.arange(-yearsBaseline, 0))

    print("Computing difference raster...")
    diff_raster = avg_first - avg_last

    print("Computing quantiles...")
    thresholds = np.nanquantile(diff_raster, [0.33, 0.66])
    print(f"Thresholds: {thresholds}")

    classified_raster = np.zeros_like(diff_raster)
    classified_raster[diff_raster <= thresholds[0]] = 3
    classified_raster[(diff_raster > thresholds[0]) & (diff_raster <= thresholds[1])] = 2
    classified_raster[diff_raster > thresholds[1]] = 1

    if filename:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        print(f"Saving classified raster to {filename}")
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(classified_raster, 1)

    print("State change classification complete!")
    return classified_raster


if __name__ == "__main__":
    input_raster = "/home/gianofe/Documents/corrected/sumndvi_multiband_xxl.tif"
    output_raster = "/home/gianofe/Documents/corrected/state_changexxl.tif"

    print(f"Opening input raster: {input_raster}")
    with rasterio.open(input_raster) as src:
        data = src.read()
        profile = src.profile

    print(f"Raster shape: {data.shape}")
    obj2process = xr.DataArray(data, dims=("band", "y", "x"))

    print("Running state change function...")
    State_Change = state_change(obj2process, yearsBaseline=3, changeNclass=1)

    print(f"Saving output raster to {output_raster}")
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(State_Change, 1)

    print("Process completed successfully!")
