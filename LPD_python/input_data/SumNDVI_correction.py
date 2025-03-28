import os
import numpy as np
import pandas as pd
from netCDF4 import Dataset
import rasterio
from rasterio.transform import from_origin
from osgeo import gdal
from joblib import Parallel, delayed
from numba import jit
import time

# Parametri configurabili
CSV_PATH = "/home/gianofe/Documents/ndvi1km_wgt_avg_total.csv"  # <-- MODIFICA QUI
OUTPUT_DIR = "/scratch/gianofe/SumNDVI_correction_rev3/"           # <-- MODIFICA QUI
VALID_THRESHOLD = 0.40
NUM_CORES = 6
GTIFF_COMPRESSION = "ZSTD"

# Carica il CSV
df = pd.read_csv(CSV_PATH)
years = sorted(df["year"].unique())
os.makedirs(OUTPUT_DIR, exist_ok=True)


@jit(nopython=True)
def interpolate_temporal(ndvi_stack):
    """Interpolazione lineare temporale per ogni pixel (ottimizzata con Numba)."""
    n_images, rows, cols = ndvi_stack.shape
    filled = ndvi_stack.copy()

    for i in range(rows):
        for j in range(cols):
            ts = ndvi_stack[:, i, j]
            if np.all(np.isnan(ts)):
                continue
            valid = ~np.isnan(ts)
            if np.sum(valid) < 2:
                continue
            x = np.arange(n_images)
            x_valid = x[valid]
            y_valid = ts[valid]
            filled[:, i, j] = np.interp(x, x_valid, y_valid)

    return filled


def process_file(row):
    file_path = row["path"]
    satellite = row["satellite"]

    if not os.path.exists(file_path):
        print(f"❌ File non trovato: {file_path}")
        return None, None, None

    try:
        if satellite in ["VGT", "PROBAV"]:
            with Dataset(file_path, "r") as nc_file:
                if "NDVI" not in nc_file.variables:
                    print(f"❌ NDVI mancante nel file NetCDF: {file_path}")
                    return None, None, None

                ndvi_data = np.squeeze(nc_file.variables["NDVI"][:].astype(np.float32))
                ndvi_data = (ndvi_data - 0.013) / 0.958
                ndvi_data[(ndvi_data < -1.0) | (ndvi_data > 1.0)] = np.nan

                lat = nc_file.variables["lat"][:]
                lon = nc_file.variables["lon"][:]
                transform = from_origin(lon[0], lat[0], lon[1] - lon[0], lat[0] - lat[1])

        elif satellite == "OLCI":
            with rasterio.open(file_path) as src:
                ndvi_data = src.read(1).astype(np.float32)
                ndvi_data[(ndvi_data == 2) | (ndvi_data > 1.0) | (ndvi_data < -1.0)] = np.nan
                transform = src.transform

        else:
            print(f"❌ Satellite sconosciuto: {satellite}")
            return None, None, None

        return ndvi_data, transform

    except Exception as e:
        print(f"❌ Errore in {file_path}: {e}")
        return None, None, None


def process_year(selected_year):
    print(f"\n📆 Anno {selected_year}")
    year_data = df[df["year"] == selected_year]
    start_time = time.time()

    results = Parallel(n_jobs=NUM_CORES)(
        delayed(process_file)(row) for _, row in year_data.iterrows()
    )

    ndvi_stack = [np.asarray(r[0]) for r in results if r[0] is not None]
    transform = next((r[1] for r in results if r[1] is not None), None)

    if not ndvi_stack:
        print(f"⚠️ Nessun dato valido per {selected_year}")
        return None

    print("🔄 Interpolazione temporale sui dekadi...")
    ndvi_array = np.stack(ndvi_stack)
    ndvi_interp = interpolate_temporal(ndvi_array)
    annual_sum = np.nansum(ndvi_interp, axis=0)

    valid_count = np.sum(~np.isnan(ndvi_array), axis=0)
    min_valid = max(1, len(ndvi_stack) * VALID_THRESHOLD)
    valid_pixels = valid_count >= min_valid
    annual_sum[~valid_pixels] = np.nan

    nan_perc = np.isnan(annual_sum).sum() / annual_sum.size * 100
    print(f"📊 NoData: {nan_perc:.2f}%")

    output_path = os.path.join(OUTPUT_DIR, f"NDVI_SUM_{selected_year}.tif")
    profile = {
        "driver": "GTiff",
        "height": annual_sum.shape[0],
        "width": annual_sum.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -9999,
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(np.where(np.isnan(annual_sum), -9999, annual_sum).astype("float32"), 1)

    temp_path = output_path.replace(".tif", "_tmp.tif")
    os.rename(output_path, temp_path)
    gdal.Translate(output_path, temp_path, creationOptions=[f"COMPRESS={GTIFF_COMPRESSION}", "PREDICTOR=2"])
    os.remove(temp_path)

    print(f"✅ Anno {selected_year} completato in {time.time() - start_time:.1f}s")
    return selected_year, output_path


# Esecuzione principale in batch di 5 anni
year_paths = []
BATCH_SIZE = 5
for i in range(0, len(years), BATCH_SIZE):
    batch = years[i:i + BATCH_SIZE]
    print(f"\n🚀 Elaborazione batch: {batch}")
    results = Parallel(n_jobs=min(len(batch), NUM_CORES))(
        delayed(process_year)(year) for year in batch
    )
    for res in results:
        if res:
            year_paths.append(res)

