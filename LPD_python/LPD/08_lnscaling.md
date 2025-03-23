
# 🌿 Local Net Productivity Scaling (LNScaling)

This script computes the **Local Net Productivity Scaling (LNS)** index, a measure of how current productivity compares to the **local potential** productivity. This is a key component in land degradation assessments, especially within the **Land Productivity Dynamics (LPD)** framework.

---

## 📊 1. Purpose

LNScaling expresses the **scaled current productivity** of each pixel as a **percentage** of the best productivity observed in similar areas (i.e., within the same Ecosystem Functional Type, EFT). This method highlights areas that are **underperforming relative to their ecological potential**.

---

## 🧮 2. Method Overview

### 🔹 A. Input Rasters

| Raster         | Description                                      |
|----------------|--------------------------------------------------|
| `EFTs raster`  | Single-band classified raster with cluster IDs (Ecosystem Functional Types) |
| `ProdVar`      | Multi-band productivity variable raster (e.g., corrected NDVI sum) |

### 🔹 B. Processing Steps

1. **Read input rasters**
   - Loads EFTs and productivity time series (ProdVar)

2. **Compute average productivity**
   - Averages bands corresponding to a given year range (e.g., 2008–2023)

3. **Calculate potential productivity**
   - Computes the **90th percentile** of productivity within each EFT class

4. **Cap observed productivity**
   - Any observed productivity higher than the local potential is set to the potential

5. **Compute LNS**
   - `LNS = (current_productivity / potential_productivity) * 100`
   - Expresses current productivity as a percentage of the best local value

6. **Save result**
   - Output is saved as a single-band raster

---

## ⚙️ Parameters

| Parameter        | Description                                      | Example                           |
|------------------|--------------------------------------------------|-----------------------------------|
| `EFTs_path`      | Path to the EFT raster                           | `EFTs_landcover_2023_rev2.tif`    |
| `ProdVar_path`   | Path to productivity raster                      | `cf_multiband.tif`                |
| `years`          | List of years matching the bands in `ProdVar`    | `list(range(1999, 2025))`         |
| `start_year`     | Start of the evaluation window                   | `2008`                            |
| `end_year`       | End of the evaluation window                     | `2023`                            |
| `cores`          | Number of cores for parallel computation         | (currently not used)              |
| `filename`       | Output path for the LNS raster                   | `lns_2023_rev2.tif`               |

---

## 💾 Output

- A single-band raster where:
  - Each pixel holds the **LNS value** as a percentage (0–100+)
  - `NaN` indicates no data or excluded pixels
- Highlights areas with **underperformance or degradation**

---

## 📎 Example Use Case

```bash
EFTs raster: EFTs_landcover_2023_rev2.tif
Productivity: cf_multiband.tif
Years used: 2008–2023
Output: lns_2023_rev2.tif
```

---

## 📘 Related Concepts

- Land Productivity Dynamics (LPD)
- Local potential estimation
- Ecosystem Functional Types (EFTs)
- Land Degradation Neutrality (LDN) indicators
