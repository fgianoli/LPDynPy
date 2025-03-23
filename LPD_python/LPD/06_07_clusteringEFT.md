
# 🧬 Ecosystem Functional Types (EFT) Clustering

This script performs clustering of **Ecosystem Functional Types (EFTs)** based on NDVI time series and optional Land Cover classification. It allows for two clustering modes:

- **Global Clustering**: Clustering is applied across the entire raster, ignoring land cover classes.
- **Land Cover–specific Clustering**: NDVI values are clustered separately within each land cover class.

---

## 🗺️ 1. Input Data

| Raster         | Description                                              |
|----------------|----------------------------------------------------------|
| `NDVI raster`  | Multi-band raster with NDVI-derived variables (averaged) |
| `Land Cover`   | Single-band land cover classification raster             |

---

## ⚙️ 2. Parameters

| Parameter         | Description                                           | Default                 |
|-------------------|-------------------------------------------------------|-------------------------|
| `CLUSTER_MODE`     | `"global_clustering"` or `"landcover_eft"`          | `"landcover_eft"`       |
| `NUM_CORES`        | Number of CPU cores for parallel processing          | `40`                    |
| `NUM_CLUSTERS`     | Number of clusters (used in global mode)             | `25`                    |
| `OUTPUT_DIR`       | Output directory for saving results                  | `/.../new_correction/`  |

---

## 🧪 3. Workflow

### 🔹 A. Data Preparation

- Loads both NDVI and Land Cover rasters using Dask
- Flattens and aligns them into a 2D array
- Removes NoData pixels
- Converts data to a clean Pandas DataFrame

### 🔹 B. Clustering Modes

#### ✅ Global Clustering (`global_clustering`)
- Standardizes NDVI values across all pixels
- Applies **MiniBatchKMeans** clustering globally
- Outputs a single-layer raster of cluster IDs

#### ✅ Land Cover–based Clustering (`landcover_eft`)
- Iterates over unique land cover classes
- Applies clustering **within each class**
- Cluster count scales with the number of pixels per class
- More ecologically meaningful clustering

---

## 💾 4. Output

- A single-band classified raster:
  - Each pixel holds an integer cluster ID
  - `255` is used for NoData

File name depends on clustering mode:
- `EFTs_clusters.tif` → global clustering
- `EFTs_landcover.tif` → land cover–specific clustering

---

## 📎 Example Use Case

```bash
NDVI raster: no_col_2023.tif
Land Cover raster: LC1000.tif
Mode: landcover_eft
Output: EFTs_landcover_2023_rev2.tif
```

---

## 🧠 Use Cases

- Deriving **Ecosystem Functional Types (EFTs)**
- Supporting **landscape classification** and **ecosystem monitoring**
- Identifying **spatial patterns** in productivity and land use

---

## 🧰 Libraries Used

- NumPy, Pandas
- Rasterio, Dask
- Scikit-learn (MiniBatchKMeans, StandardScaler)
- Joblib, Numba

---

## 📘 Related Concepts

- Ecosystem Functional Types (EFTs)
- Land Cover–based stratification
- Unsupervised classification (clustering)
- NDVI trajectory analysis
