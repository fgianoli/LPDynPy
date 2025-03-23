# NDVI Annual Sum Processor

## Description
This Python script processes NDVI images from various satellite sensors (VGT, PROBAV, OLCI) and produces an **annual NDVI sum** for each pixel, using:

- Sensor-specific corrections
- Automatic handling of `NoData` values
- 40% validity threshold to retain a pixel
- Interpolation of missing pixels using the 3x3 neighborhood mean
- Writing final compressed GeoTIFF rasters

## Required Inputs
- **CSV with raster metadata**: contains file paths (`.tif` or `.nc`), satellite name, and acquisition year
- **NDVI images**: one raster per date in the year

## Workflow

### 1. **CSV Reading**
Loads a CSV file containing:
- `path`: raster file path
- `satellite`: sensor name (e.g., `OLCI`)
- `year`: acquisition year

### 2. **Single File Processing (`process_file`)**
- Opens the raster or NetCDF file
- Extracts the NDVI band
- Applies:
  - Replacement of special values (`250`, `255`, `2`, out-of-range values)
  - Correction for VGT/PROBAV (`(val - 0.013) / 0.958`)
  - NoData masking
- Returns:
  - Cleaned NDVI image
  - Binary mask of valid pixels (for counting)
  - Spatial transform

### 3. **Annual Processing (`process_year`)**
- Filters the dataset for the current year
- Runs `process_file` in parallel on all files using `joblib.Parallel`
- Computes:
  - Sum of valid NDVI values (`np.nansum`)
  - Count of valid observations per pixel
- Applies a **40% validity threshold**:
  - Pixels with fewer observations are discarded
  - Missing pixels above the threshold are **interpolated** using local mean (3x3 window)

### 4. **Fast Interpolation (`fast_interpolation`)**
- Uses `scipy.ndimage.uniform_filter` to compute local mean of valid pixels
- Fills `NaN` values only if surrounding data is sufficient

### 5. **Raster Writing**
- Temporarily saves raster in `/dev/shm` (RAM disk)
- Writes final GeoTIFF raster:
  - CRS: WGS84 (`EPSG:4326`)
  - ZSTD compression
  - `nodata`: -9999

## Key Parameters
- `VALID_THRESHOLD`: minimum % of valid observations to keep a pixel (`0.40` = 40%)
- `NUM_CORES`: number of parallel jobs (`5`)
- `START_YEAR`: processing start year (`2020`)
- `output_dir`: directory for annual NDVI sum rasters

## Output
For each year, the following file is generated:
```
NDVI_SUM_<YEAR>.tif
```
Containing the annual NDVI sum with interpolation applied where appropriate.

## Dependencies
- `numpy`, `pandas`, `rasterio`, `scipy`, `netCDF4`, `joblib`, `osgeo.gdal`
