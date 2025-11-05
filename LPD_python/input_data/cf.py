import os
import shutil
import numpy as np
import rasterio
from rasterio.transform import from_origin
from netCDF4 import Dataset
import pandas as pd
from numba import njit

# ======= PARAMETRI =======
csv_path = '/home/gianofe/Documents/ndvi1km_wgt_avg_total.csv'
output_dir = '/scratch/gianofe/cf_rev2'

CORRECT_VGT_PROBAV = True
VALID_RANGE = (-0.1, 1.0)  # NDVI expected range
CF_VALID_THRESHOLD = 0.5  # percentuale minima di osservazioni valide per calcolo
NUM_CORES = 5  # numero di core da usare per elaborazione parallela
# =========================

# Caricamento CSV
data = pd.read_csv(csv_path)
cf_output_dir = os.path.join(output_dir, "cf")
os.makedirs(cf_output_dir, exist_ok=True)

# Anno di partenza
try:
    start_year = int(input("Inserisci l'anno di inizio analisi (premi Invio per analizzare tutti gli anni): ").strip() or min(data['year']))
except ValueError:
    print("Errore: anno non valido")
    exit()

years = sorted([y for y in data['year'].unique() if y >= start_year])

def compute_cf(stack):
    mean = np.nanmean(stack, axis=0)
    std = np.nanstd(stack, axis=0)
    result = np.zeros_like(mean, dtype=np.float32)
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            if not np.isnan(mean[i, j]) and mean[i, j] != 0:
                result[i, j] = std[i, j] / mean[i, j]
            else:
                result[i, j] = np.nan
    return result

def check_disk_space(path, min_space=10):  # GB
    stat = shutil.disk_usage(path)
    return stat.free / (1024**3) > min_space

from joblib import Parallel, delayed

def process_year(year):
    year_data = data[data['year'] == year]
    ndvi_stack = []
    transform = None

    print(f"Elaborazione anno: {year}")

    for _, row in year_data.iterrows():
        file_path = row['path']
        satellite = row['satellite']

        if not check_disk_space(cf_output_dir):
            print("Spazio insufficiente.")
            return

        if not os.path.exists(file_path):
            print(f"File non trovato: {file_path}")
            continue

        if satellite not in ['VGT', 'PROBAV', 'OLCI']:
            print(f"Satellite non supportato o saltato: {satellite}")
            continue

        try:
            if satellite in ['VGT', 'PROBAV']:
                with Dataset(file_path, 'r') as nc:
                    if 'NDVI' not in nc.variables:
                        print(f"NDVI mancante: {file_path}")
                        return

                    ndvi = nc.variables['NDVI'][:]
                    ndvi = np.squeeze(ndvi)
                    ndvi = np.ma.filled(ndvi, np.nan).astype(np.float32)
                    ndvi[(ndvi > VALID_RANGE[1]) | (ndvi < VALID_RANGE[0])] = np.nan

                    if CORRECT_VGT_PROBAV:
                        ndvi = (ndvi - 0.013) / 0.958

                    if transform is None:
                        lat = nc.variables['lat'][:]
                        lon = nc.variables['lon'][:]
                        res_x = lon[1] - lon[0]
                        res_y = lat[0] - lat[1]
                        transform = from_origin(lon[0], lat[0], res_x, res_y)

            elif satellite == 'OLCI':
                with rasterio.open(file_path) as src:
                    ndvi = src.read(1).astype(np.float32)
                    ndvi[ndvi == 2] = np.nan
                    ndvi[(ndvi > VALID_RANGE[1]) | (ndvi < VALID_RANGE[0])] = np.nan
                    if transform is None:
                        transform = src.transform

            ndvi_stack.append(np.asarray(ndvi))
            print(f"Caricato: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"Errore in {file_path}: {e}")
            continue

    if not ndvi_stack:
        print(f"Nessun dato valido per {year}")
        return

    stack = np.stack(ndvi_stack)
    valid_count = np.sum(~np.isnan(stack), axis=0)
    min_valid = int(len(ndvi_stack) * CF_VALID_THRESHOLD)
    valid_mask = valid_count >= min_valid

    cf = compute_cf(stack)
    cf[~valid_mask] = np.nan

    print(f"Pixel validi dopo filtro: {np.sum(valid_mask)} su {cf.size}")

    output_path = os.path.join(cf_output_dir, f'CF_{year}.tif')
    if os.path.exists(output_path):
        print(f"File esistente sovrascritto: {output_path}")
        os.remove(output_path)

    profile = {
        'driver': 'GTiff',
        'height': cf.shape[0],
        'width': cf.shape[1],
        'count': 1,
        'dtype': 'float32',
        'crs': 'EPSG:4326',
        'transform': transform,
        'compress': 'ZSTD'
    }

    try:
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(np.where(np.isnan(cf), -9999, cf).astype(np.float32), 1)
        print(f"Salvato: {output_path}")
    except Exception as e:
        print(f"Scrittura fallita: {e}")

Parallel(n_jobs=NUM_CORES)(delayed(process_year)(year) for year in years)

print("Elaborazione completata.")
