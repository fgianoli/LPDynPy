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


def baseline_lev(obj2process, yearsBaseline=3, drylandProp=0.4, highprodProp=0.1, filename=""):
    print("Starting baseline level classification...")

    if isinstance(obj2process, str):
        print(f"Opening raster file: {obj2process}")
        with rasterio.open(obj2process) as src:
            obj2process = xr.DataArray(src.read(), dims=("band", "y", "x"))
            profile = src.profile

    print(f"Raster shape: {obj2process.shape}")
    start_year = 1999
    selected_years = np.arange(2008, 2024)  # Anni dal 2008 al 2023
    band_indices = selected_years - start_year  # Converti anni in indici (base 0)

    print(f"Selected band indices: {band_indices[:yearsBaseline]}")
    if obj2process.shape[0] < max(band_indices[:yearsBaseline]) + 1:
        raise ValueError(f"Not enough bands ({obj2process.shape[0]}) to process years {yearsBaseline}.")

    obj2process = obj2process[band_indices[:yearsBaseline], :, :]

    print("Computing average raster...")
    print(f"NaN count in selected bands: {np.isnan(obj2process).sum()}")
    print(f"Available bands: {obj2process.shape[0]}")
    print(f"Selected band indices for processing: {band_indices[:yearsBaseline]}")
    valid_bands = [b for b in range(obj2process.shape[0]) if not np.isnan(obj2process[b]).all()]
    if len(valid_bands) == 0:
        raise ValueError("No valid bands available for averaging!")
    avg_raster = mean_years_function(obj2process, valid_bands)

    print("Computing quantiles...")
    quantiles = np.nanquantile(avg_raster, np.linspace(0, 1, 11))
    print(f"Quantiles: {quantiles}")

    if drylandProp > 1:
        drylandProp /= 100
    if highprodProp > 1:
        highprodProp /= 100
    if drylandProp + highprodProp > 1:
        raise ValueError("The sum of drylandProp and highprodProp must be <= 1")

    thresholds = {
        "low": int(drylandProp * 10),
        "high": int(highprodProp * 10),
        "medium": 10 - int(drylandProp * 10) - int(highprodProp * 10)
    }
    print(f"Thresholds: {thresholds}")

    print("Classifying raster...")
    classified_raster = np.digitize(avg_raster, bins=quantiles, right=True)
    classified_raster[classified_raster <= thresholds['low']] = 1
    classified_raster[classified_raster > thresholds['low']] = 2
    classified_raster[classified_raster > (thresholds['low'] + thresholds['medium'])] = 3

    if filename:
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        print(f"Saving classified raster to {filename}")
        with rasterio.open(filename, 'w', **profile) as dst:
            dst.write(classified_raster, 1)

    print("Baseline classification complete!")
    return classified_raster


if __name__ == "__main__":
    input_raster = "/home/gianofe/Documents/corrected/sumndvi_multiband_xxl.tif"
    output_raster = "/home/gianofe/Documents/corrected/baselinexxl.tif"

    print(f"Opening input raster: {input_raster}")
    with rasterio.open(input_raster) as src:
        data = src.read()
        profile = src.profile

    print(f"Raster shape: {data.shape}")
    obj2process = xr.DataArray(data, dims=("band", "y", "x"))

    print("Running baseline level function...")
    Baseline_Level = baseline_lev(obj2process, yearsBaseline=3, drylandProp=0.4, highprodProp=0.1)

    print(f"Saving output raster to {output_raster}")
    with rasterio.open(output_raster, 'w', **profile) as dst:
        dst.write(Baseline_Level, 1)

    print("Process completed successfully!")
