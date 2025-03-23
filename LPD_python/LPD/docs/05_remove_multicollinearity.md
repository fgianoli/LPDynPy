
# 🧪 Multicollinearity Removal for Raster Time Series

This script removes multicollinearity from a set of **multi-band raster files** (e.g., NDVI, phenological variables like SOS/EOS/GSL), retaining only those variables that are **not highly correlated** with each other, based on a Pearson correlation threshold.

It is useful for **dimensionality reduction**, **input preparation for machine learning**, or **multivariate statistical analysis**.

---

## 📊 1. What It Does

The process follows these steps:

### 🔹 A. Input Raster Stack

- Accepts a list of `.tif` raster files, each containing a time series as multiple bands.
- Selects only the bands corresponding to a defined period (`yrs2use`, e.g., 2008–2023).
- Computes the **mean value across selected years** for each raster (per pixel).
- Stacks the mean rasters into a 3D array: `(n_variables, height, width)`.

### 🔹 B. Correlation Analysis

- Reshapes the stack to 2D: `(n_pixels, n_variables)`
- Computes the **Pearson correlation matrix**
- Filters out variables that are **highly correlated** with others (default threshold = 0.7)
- Retains only uncorrelated or weakly correlated variables

---

## 🧠 2. Why Remove Multicollinearity?

Highly correlated layers provide redundant information and can:

- Bias statistical models (e.g., regression, PCA)
- Inflate variance
- Reduce model interpretability

This step ensures that only independent spatial layers are used in further analysis.

---

## ⚙️ Parameters

| Parameter         | Description                                           | Default        |
|-------------------|-------------------------------------------------------|----------------|
| `file_paths`      | List of input `.tif` raster file paths                | (required)     |
| `yrs2use`         | List of band indices to use for averaging             | `None`         |
| `multicol_cutoff` | Pearson correlation threshold for filtering variables | `0.7`          |
| `filename`        | Output path for the filtered stack                    | `""`           |

---

## 🗺️ Output

- A raster stack with reduced number of bands (variables), saved to the specified `filename`
- Each band corresponds to one retained variable (mean over selected years)

---

## 📎 Example Use Case

```bash
Input files:
- sumndvi_newcorrection.tif
- cf_multiband.tif
- SOS_multiband.tif
- EOS_multiband.tif
- GSL_multiband.tif

Years used: 2008–2023 (bands 9–24)
Output: no_col_2023.tif
```

---

## 📘 Related Concepts

- Variance Inflation Factor (VIF)
- Feature selection
- Dimensionality reduction
- Principal Component Analysis (PCA)
