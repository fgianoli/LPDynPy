
# 🔄 State Change Classification

This script computes and classifies the **state change** in land productivity by comparing early and late periods in a vegetation index time series (e.g., NDVI). It provides a simple, interpretable 3-class raster that highlights trends in productivity levels over time.

---

## 📊 1. Method Overview

The approach compares the **average productivity** during two periods:

- **Early period** (baseline): e.g., 2008–2010
- **Late period** (final): e.g., 2020–2022

By subtracting the late-period average from the early-period average, the script derives a **difference raster** that captures long-term change in productivity.

---

## 🧮 2. Calculation Steps

### 🔹 A. Input Raster

The raster is expected to contain multiple bands, where each band corresponds to a year of vegetation productivity (e.g., SumNDVI from 1999 to 2023).

- Example: Band 0 = 1999, Band 1 = 2000, ..., Band 24 = 2023

### 🔹 B. Select Analysis Period

The script selects bands corresponding to years 2008–2022 and defines the number of years to average (`yearsBaseline`, default = 3):

```python
avg_first = mean over [2008, 2009, 2010]
avg_last  = mean over [2020, 2021, 2022]
```

### 🔹 C. Compute Difference

```python
diff_raster = avg_first - avg_last
```

This produces a pixel-wise difference map, which can show:
- Positive values → degradation
- Negative values → improvement
- Near zero → stable areas

### 🔹 D. Quantile Classification

The difference raster is classified into 3 classes using quantile thresholds (33% and 66%):

| Class Value | Meaning         |
|-------------|------------------|
| `1`         | Improvement      |
| `2`         | Stable           |
| `3`         | Degradation      |

---

## ⚙️ Parameters

| Parameter        | Description                                           | Default  |
|------------------|-------------------------------------------------------|----------|
| `yearsBaseline`  | Number of years to average for early/late periods     | `3`      |
| `changeNclass`   | Number of quantile thresholds (unused, kept for future use) | `1`      |
| `filename`       | Output raster file path                               | `""`     |

---

## 🗺️ Output

A single-band classified raster with pixel values:

- `1` = Improved productivity
- `2` = No major change (stable)
- `3` = Decreased productivity (degradation)

This product is useful for:

- Detecting **long-term changes** in vegetation or productivity
- Mapping **degraded** or **restored** areas
- Supporting **ecosystem monitoring**

---

## 📎 Example Use Case

```bash
Input raster: sumndvi_newcorrection.tif (1999–2023)
Early years: 2008–2010
Late years: 2020–2022
Output: state_change_2008_2023.tif
```

---

## 📘 Related Concepts

- Land Productivity Dynamics (LPD)
- Land Degradation Neutrality (LDN)
- NDVI Trend Analysis
