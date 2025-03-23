import os
import numpy as np
import rasterio
from rasterio.transform import from_origin
from netCDF4 import Dataset
import pandas as pd
from osgeo import gdal
from scipy.ndimage import uniform_filter
from joblib import Parallel, delayed
import time

# Parameters
csv_path = '/home/gianofe/Documents/ndvi1km_wgt_avg_total.csv'
output_dir = '/scratch/gianofe/SumNDVI_rev2/'
VALID_THRESHOLD = 0.40
NUM_CORES = 5
START_YEAR = 1999

# Load data
print("📌 Loading dataset...")
data = pd.read_csv(csv_path)
years = sorted([y for y in data['year'].unique() if y >= START_YEAR])
os.makedirs(output_dir, exist_ok=True)

def fast_interpolation(data):
    mask = np.isnan(data)
    filled = data.copy()
    data_zeroed = np.nan_to_num(data)
    local_sum = uniform_filter(data_zeroed, size=3)
    count = uniform_filter((~np.isnan(data)).astype(float), size=3)
    with np.errstate(invalid='ignore'):
        local_mean = np.where(count > 0, local_sum / count, np.nan)
    filled[mask] = local_mean[mask]
    return filled

def process_file(row):
    file_path = row['path']
    satellite = row['satellite']

    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None, None, None

    try:
        if satellite in ['VGT', 'PROBAV']:
            with Dataset(file_path, 'r') as nc_file:
                if 'NDVI' not in nc_file.variables:
                    print(f"❌ NDVI missing in NetCDF: {file_path}")
                    return None, None, None
                ndvi_data = nc_file.variables['NDVI'][:].astype(np.float32)
                ndvi_data = np.squeeze(ndvi_data)
                lat = nc_file.variables['lat'][:]
                lon = nc_file.variables['lon'][:]
                res_x = lon[1] - lon[0]
                res_y = lat[0] - lat[1]
                transform = from_origin(lon[0], lat[0], res_x, res_y)
                ndvi_data[(ndvi_data == 250) | (ndvi_data == 255)] = np.nan
                ndvi_data = (ndvi_data - 0.013) / 0.958

        elif satellite == 'OLCI':
            with rasterio.open(file_path) as src:
                ndvi_data = src.read(1).astype(np.float32)
                print(f"👉 {file_path} | nodata: {src.nodata}")
                print(f"👉 Min/Max values: {np.nanmin(ndvi_data)} / {np.nanmax(ndvi_data)}")

                if src.nodata is not None:
                    ndvi_data[ndvi_data == src.nodata] = np.nan
                ndvi_data[(ndvi_data > 1.0) | (ndvi_data < -1.0)] = np.nan
                ndvi_data[ndvi_data == 2] = np.nan
                transform = src.transform

        valid_mask = ~np.isnan(ndvi_data)
        valid_count = np.zeros_like(ndvi_data, dtype=np.int32)
        valid_count[valid_mask] += 1

        print(f"✅ Processed: {file_path}")
        return ndvi_data, valid_count, transform

    except Exception as e:
        print(f"❌ Error in file {file_path}: {e}")
        return None, None, None

def process_year(selected_year):
    year_data = data[data['year'] == selected_year]
    print(f"\n📆 Processing {selected_year} ({len(year_data)} files)...")
    start = time.time()

    results = Parallel(n_jobs=NUM_CORES)(
        delayed(process_file)(row) for _, row in year_data.iterrows()
    )

    ndvi_stack = [res[0] for res in results if res[0] is not None]
    vcount_stack = [res[1] for res in results if res[1] is not None]
    transform = next((res[2] for res in results if res[2] is not None), None)

    if ndvi_stack:
        print(f"📈 Annual stack: {len(ndvi_stack)} rasters")
        annual_sum = np.nansum(np.stack(ndvi_stack), axis=0)
        valid_count = np.sum(np.stack(vcount_stack), axis=0)

        min_valid = max(1, len(year_data) * VALID_THRESHOLD)
        valid_pixels = valid_count >= min_valid

        print(f"📊 Valid pixels >= {int(VALID_THRESHOLD * 100)}%: {(valid_pixels.sum() / valid_pixels.size) * 100:.2f}%")

        print("⚡ Fast interpolation...")
        interpolated = fast_interpolation(annual_sum)
        mask_to_interpolate = valid_pixels & np.isnan(annual_sum)
        annual_sum[mask_to_interpolate] = interpolated[mask_to_interpolate]
        annual_sum[~valid_pixels] = np.nan

        nan_perc = np.isnan(annual_sum).sum() / annual_sum.size * 100
        print(f"📊 NoData {selected_year}: {nan_perc:.2f}%")

        temp_path = f"/dev/shm/NDVI_SUM_{selected_year}_temp.tif"
        final_path = os.path.join(output_dir, f'NDVI_SUM_{selected_year}.tif')

        for path in [temp_path, final_path]:
            if os.path.exists(path):
                print(f"🗑️ Removed existing: {path}")
                os.remove(path)

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

        print(f"💾 Writing raster {selected_year}...")
        with rasterio.open(temp_path, 'w', **profile) as dst:
            dst.write(annual_sum, 1)

        gdal.Translate(final_path, temp_path, creationOptions=["COMPRESS=ZSTD", "PREDICTOR=2"])
        os.remove(temp_path)

        print(f"✅ Done {selected_year} in {time.time() - start:.2f} sec.")

for i in range(0, len(years), 8):
    batch = years[i:i + 8]
    print(f"\n🔹 Processing batch: {batch}")
    Parallel(n_jobs=NUM_CORES)(delayed(process_year)(y) for y in batch)

print("\n🏁 All years processed.")
