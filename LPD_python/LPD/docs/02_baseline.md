
# 🌾 Baseline Productivity Level Classification

This method classifies land areas based on their **baseline productivity level** using a set of early years in a vegetation time series (e.g., NDVI). It is inspired by methods used in **land productivity dynamics** and **land degradation assessment** (e.g., UNCCD/LADA frameworks).

---

## 📊 1. Method Overview

The script performs the following steps:

### 🔹 A. Input Raster

The input raster contains multiple bands, each representing **annual vegetation productivity** (e.g., SumNDVI) from 1999 onwards.

- Example: Band 0 = 1999, Band 1 = 2000, ..., Band 24 = 2023
- The script uses bands corresponding to **early years** to compute a productivity baseline (default: 2008–2010)

---

### 🔹 B. Baseline Period Selection

A baseline is built by averaging the productivity values over the first `yearsBaseline` bands within the selected range (default = 3 years):

```python
selected_years = np.arange(2008, 2023)
baseline_years = selected_years[:yearsBaseline]  # e.g., [2008, 2009, 2010]
```

---

### 🔹 C. Mean Productivity Map

For each pixel:
- If all values are NaN, fallback to global mean
- Else, compute the pixel-wise **mean across baseline years**

This results in a **mean baseline productivity raster**.

---

### 🔹 D. Quantile-Based Classification

The averaged productivity values are divided into **10 quantile classes** (deciles), and then grouped into 3 classes:

| Class Name   | Quantile Range                    | Default Proportion |
|--------------|------------------------------------|---------------------|
| 1 - Low      | Bottom `drylandProp` fraction      | 40% (`drylandProp=0.4`) |
| 2 - Medium   | Middle fraction                    | Remaining           |
| 3 - High     | Top `highprodProp` fraction        | 10% (`highprodProp=0.1`) |

These thresholds can be customized.

---

## 🧠 2. Classification Logic

Each pixel is assigned a class using the following logic:

```python
if quantile_class <= low_threshold:
    class = 1  # Low productivity
elif quantile_class > (low_threshold + medium_threshold):
    class = 3  # High productivity
else:
    class = 2  # Medium productivity
```

Result: a classified raster where each pixel has a value:

- `1` = Low productivity
- `2` = Medium productivity
- `3` = High productivity

---

## 🧪 Parameters

| Parameter       | Description                                      | Default      |
|-----------------|--------------------------------------------------|--------------|
| `yearsBaseline` | Number of baseline years to average              | `3`          |
| `drylandProp`   | Proportion of area considered low productivity   | `0.4` (40%)  |
| `highprodProp`  | Proportion of area considered high productivity  | `0.1` (10%)  |
| `filename`      | Output file path (optional)                      | `""`         |

---

## 🗺️ Output

- A single-band raster with values:
  - `1` → Low productivity
  - `2` → Medium productivity
  - `3` → High productivity

This classified baseline can be used for:

- Long-term productivity monitoring
- Degradation or improvement assessments
- Ecosystem functioning analysis

---

## 📎 Example Use Case

```bash
Input raster: sumndvi_newcorrection.tif (1999–2023)
Baseline years: 2008, 2009, 2010
Output: baseline_2008_2023.tif
```

---

## 📘 Related Concepts

- Land Degradation Neutrality (LDN)
- UNCCD Land Productivity Dynamics (LPD)
- Baseline condition mapping
