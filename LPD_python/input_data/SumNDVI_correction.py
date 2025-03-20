### This code applies the value correction based on https://land.copernicus.eu/en/technical-library/quality-assessment-report-normalised-difference-vegetation-index-333-m-version-2/@@download/file page 86

import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from netCDF4 import Dataset
import pandas as pd
from osgeo import gdal
from scipy.ndimage import generic_filter
from joblib import Parallel, delayed
import time

# Percorsi e parametri
csv_path = '/home/gianofe/Documents/ndvi1km_wgt_avg_total.csv'
output_dir = '/scratch/gianofe/SumNDVI_rev2/'
VALID_THRESHOLD = 0.40
NUM_CORES = 30

# Carica i dati
print("📌 Caricamento del dataset...")
data = pd.read_csv(csv_path)
years = data['year'].unique()
os.makedirs(output_dir, exist_ok=True)


# Funzione per elaborare un singolo file
def process_file(row):
    file_path = row['path']
    satellite = row['satellite']

    if not os.path.exists(file_path):
        print(f"❌ File non trovato: {file_path}")
        return None, None, None

    try:
        if satellite in ['VGT', 'PROBAV']:
            with Dataset(file_path, 'r') as nc_file:
                if 'NDVI' not in nc_file.variables:
                    print(f"❌ Layer NDVI non trovato nel file NetCDF: {file_path}")
                    return None, None, None
                ndvi_data = (nc_file.variables['NDVI'][:].astype(np.float32) - 0.013) / 0.958
                ndvi_data = np.squeeze(ndvi_data)
                lat = nc_file.variables['lat'][:]
                lon = nc_file.variables['lon'][:]
                resolution_x, resolution_y = lon[1] - lon[0], lat[0] - lat[1]
                transform = from_origin(lon[0], lat[0], resolution_x, resolution_y)
        elif satellite == 'OLCI':
            with rasterio.open(file_path) as src:
                ndvi_data = src.read(1).astype(np.float32)
                ndvi_data[ndvi_data == src.nodata] = np.nan
                transform = src.transform

        ndvi_data[(ndvi_data == 250) | (ndvi_data == 255)] = np.nan
        valid_mask = ~np.isnan(ndvi_data)
        valid_count = np.zeros_like(ndvi_data, dtype=np.int32)
        valid_count[valid_mask] += 1

        print(f"✅ File processato correttamente: {file_path}")
        return ndvi_data, valid_count, transform
    except Exception as e:
        print(f"❌ Errore nella lettura del file: {file_path}, {e}")
        return None, None, None


# Funzione per elaborare un intero anno
def process_year(selected_year):
    year_data = data[data['year'] == selected_year]
    print(f"📌 Inizio elaborazione per l'anno {selected_year} con {len(year_data)} file...")
    start_time = time.time()

    results = Parallel(n_jobs=NUM_CORES)(delayed(process_file)(row) for _, row in year_data.iterrows())

    annual_sum, valid_count, transform = None, None, None
    for ndvi_data, v_count, file_transform in results:
        if ndvi_data is None or v_count is None:
            continue
        if annual_sum is None:
            annual_sum = np.zeros_like(ndvi_data, dtype=np.float32)
            valid_count = np.zeros_like(ndvi_data, dtype=np.int32)
            transform = file_transform
        annual_sum += ndvi_data
        valid_count += v_count

    if annual_sum is not None:
        min_valid_count = max(1, len(year_data) * VALID_THRESHOLD)
        valid_pixels = valid_count >= min_valid_count

        def interpolate_nn(data):
            mask = np.isnan(data)
            return np.nan if np.all(mask) else np.nanmean(data)

        print(f"🔄 Interpolazione per i pixel mancanti per l'anno {selected_year}...")
        interpolated_image = generic_filter(annual_sum, interpolate_nn, size=3, mode='nearest')
        annual_sum[valid_pixels & np.isnan(annual_sum)] = interpolated_image[valid_pixels & np.isnan(annual_sum)]
        annual_sum[~valid_pixels] = np.nan

        nan_percentage = np.isnan(annual_sum).sum() / annual_sum.size * 100
        print(f"📊 Percentuale di NoData per {selected_year}: {nan_percentage:.2f}%")

        temp_path = os.path.join(output_dir, f'NDVI_SUM_{selected_year}_temp.tif')
        output_path = os.path.join(output_dir, f'NDVI_SUM_{selected_year}.tif')

        profile = {
            'driver': 'GTiff',
            'height': annual_sum.shape[0],
            'width': annual_sum.shape[1],
            'count': 1,
            'dtype': 'float32',
            'crs': 'EPSG:4326',
            'transform': transform,
            'nodata': -9999,
        }

        print(f"💾 Salvando file raster per {selected_year}...")
        with rasterio.open(temp_path, 'w', **profile) as dst:
            dst.write(annual_sum, 1)

        gdal.Translate(output_path, temp_path, creationOptions=["COMPRESS=ZSTD", "PREDICTOR=2"])
        os.remove(temp_path)
        print(f"✅ Elaborazione per {selected_year} completata in {time.time() - start_time:.2f} secondi.")


# Processa tutti gli anni in parallelo
Parallel(n_jobs=NUM_CORES)(delayed(process_year)(year) for year in years)
