import os
import numpy as np
import rasterio
import rioxarray
import xarray as xr
import dask.array as da
from rasterio.transform import from_origin
from datetime import datetime
import pandas as pd
from joblib import Parallel, delayed
import time
import multiprocessing
from osgeo import gdal
from scipy.interpolate import griddata
import numba

# 📌 Parametri configurabili
csv_path = "/home/gianofe/Documents/ndvi1km_wgt_avg_total.csv"
output_dir = "/scratch/gianofe/season_variables_rev2"
start_year = 1999       # 🔁 Anno di partenza
n_years = 3            # 🔁 Numero di anni consecutivi da elaborare
VALID_THRESHOLD = 0.40
NUM_CORES = multiprocessing.cpu_count()

# 📁 Crea cartella di output
os.makedirs(output_dir, exist_ok=True)

# 📄 Leggi il CSV
data_df = pd.read_csv(csv_path)
data_df['date'] = pd.to_datetime(data_df['path'].str.extract(r'(\d{8})')[0], format='%Y%m%d', errors='coerce')

# ⚙️ Correzione NDVI con Numba
@numba.njit(parallel=True)
def correct_ndvi(ndvi_data):
    for i in numba.prange(ndvi_data.shape[0]):
        for j in range(ndvi_data.shape[1]):
            val = ndvi_data[i, j]
            if not np.isnan(val):
                ndvi_data[i, j] = (val - 0.013) / 0.958
    return ndvi_data

# 📥 Caricamento singolo file NDVI
def load_ndvi(row):
    file_path = row['path']
    if not os.path.exists(file_path):
        print(f"❌ File non trovato: {file_path}")
        return None, None

    try:
        if file_path.endswith('.tif'):
            with rioxarray.open_rasterio(file_path) as ds:
                ndvi_data = ds[0].values.astype(np.float32)
                nc_transform = ds.rio.transform()
        else:
            with xr.open_dataset(file_path) as ds:
                if "NDVI" not in ds.variables:
                    print(f"⚠️ NDVI non trovato nel file: {file_path}")
                    return None, None
                ndvi_data = ds["NDVI"].values.astype(np.float32)
                if ndvi_data.ndim == 3:
                    ndvi_data = ndvi_data[0, :, :]
                lat = ds["lat"].values
                lon = ds["lon"].values
                res_x = lon[1] - lon[0]
                res_y = lat[0] - lat[1]
                nc_transform = from_origin(lon[0], lat[0], res_x, res_y)

        ndvi_data[(ndvi_data == 250) | (ndvi_data == 255)] = np.nan
        ndvi_data = correct_ndvi(ndvi_data)
        return ndvi_data, nc_transform

    except Exception as e:
        print(f"❌ Errore nel caricamento {file_path}: {e}")
        return None, None

# 🧩 Interpolazione dei valori NaN
def fast_interpolate(ndvi_stack):
    print("🔄 Interpolazione dei NaN...")
    x, y = np.meshgrid(np.arange(ndvi_stack.shape[2]), np.arange(ndvi_stack.shape[1]))
    for i in range(ndvi_stack.shape[0]):
        nan_mask = np.isnan(ndvi_stack[i])
        if np.any(nan_mask):
            valid_points = ~nan_mask
            ndvi_stack[i][nan_mask] = griddata(
                (x[valid_points], y[valid_points]),
                ndvi_stack[i][valid_points],
                (x[nan_mask], y[nan_mask]), method='nearest')
    return ndvi_stack

# 📊 Calcolo metriche stagionali
def calculate_season_metrics_optimized(ndvi_stack, valid_count, dates):
    print(f"🔢 Calcolo metriche stagionali per {len(dates)} date...")

    min_valid_count = max(1, len(dates) * VALID_THRESHOLD)
    valid_pixels = valid_count >= min_valid_count
    ndvi_stack[:, ~valid_pixels] = np.nan

    nan_percentage = np.isnan(ndvi_stack).mean() * 100
    print(f"📉 Percentuale di NaN dopo il filtraggio: {nan_percentage:.2f}%")

    if nan_percentage > 95:
        print("⚠️ Troppi NaN. Anno ignorato.")
        return None, None, None

    ndvi_stack = fast_interpolate(ndvi_stack)

    min_ndvi = np.nanmin(ndvi_stack, axis=0)
    max_ndvi = np.nanmax(ndvi_stack, axis=0)
    bg2 = min_ndvi + (max_ndvi - min_ndvi) * 0.1

    ndvi_above_bg2 = ndvi_stack >= bg2
    changes = np.diff(ndvi_above_bg2.astype(int), axis=0)

    sos_idx = np.argmax(changes == 1, axis=0)
    eos_idx = np.argmax(changes == -1, axis=0)

    valid_sos = (sos_idx > 0) & (sos_idx < len(dates))
    valid_eos = (eos_idx > 0) & (eos_idx < len(dates))

    sos = np.full(ndvi_stack.shape[1:], np.nan, dtype=np.float32)
    eos = np.full(ndvi_stack.shape[1:], np.nan, dtype=np.float32)

    sos[valid_sos] = np.array([dates[idx].toordinal() for idx in sos_idx[valid_sos]])
    eos[valid_eos] = np.array([dates[idx].toordinal() for idx in eos_idx[valid_eos]])

    gsl = eos - sos
    return sos, eos, gsl

# 🔁 Loop su N anni a partire da start_year
for year in range(start_year, start_year + n_years):
    print(f"\n📌 Elaborazione per l'anno {year}")
    year_data = data_df[data_df['date'].dt.year == year]
    dates = year_data['date'].tolist()

    if year_data.empty:
        print("⚠️ Nessun dato disponibile.")
        continue

    start_time = time.time()
    results = Parallel(n_jobs=NUM_CORES)(delayed(load_ndvi)(row) for _, row in year_data.iterrows())
    print(f"⏳ Tempo di caricamento: {time.time() - start_time:.2f} sec")

    ndvi_stack = []
    valid_count = None
    nc_transform = None

    for ndvi_data, transform in results:
        if ndvi_data is None or transform is None:
            continue
        ndvi_stack.append(ndvi_data)
        if valid_count is None:
            valid_count = np.zeros_like(ndvi_data, dtype=np.int32)
        valid_count += ~np.isnan(ndvi_data)
        nc_transform = transform

    if not ndvi_stack:
        print("⚠️ Nessun dato NDVI valido.")
        continue

    ndvi_stack = da.stack(ndvi_stack, axis=0).compute()
    print(f"📏 Stack NDVI: {ndvi_stack.shape}")

    sos_map, eos_map, gsl_map = calculate_season_metrics_optimized(ndvi_stack, valid_count, dates)
    if sos_map is None:
        continue

    output_path = os.path.join(output_dir, f"seasonal_variables_{year}.tif")

    with rasterio.open(output_path, 'w', **{
        'driver': 'GTiff',
        'height': sos_map.shape[0],
        'width': sos_map.shape[1],
        'count': 3,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': nc_transform,
        'nodata': -9999,
        'compress': 'LZW'  # ✅ Compressione attiva
    }) as dst:
        dst.write(sos_map, 1)
        dst.write(eos_map, 2)
        dst.write(gsl_map, 3)

    print(f"✅ Output salvato in: {output_path}")

print("\n🎯 Tutti gli anni elaborati con successo.")
