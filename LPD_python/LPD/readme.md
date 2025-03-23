# 🌍 Land Productivity Dynamics (LPD) Toolkit

This repository provides a comprehensive set of Python scripts to assess **Land Productivity Dynamics (LPD)** using remote sensing time series (e.g., NDVI). The methodology is aligned with the standards of the **UNCCD** and supports reporting on **Land Degradation Neutrality (LDN)** under **SDG Indicator 15.3.1**.

Each script in the toolkit represents a key step in the analytical workflow, enabling users to:

- Analyze vegetation trends and variability
- Identify areas of degradation or improvement
- Derive ecosystem functional classifications
- Assess land productivity status and changes
- Integrate indicators into combined assessments for reporting

---

## 📘 Background

The LPD framework is a core component of the **land degradation indicator system** developed under the **United Nations Convention to Combat Desertification (UNCCD)**. It includes the following key indicators:

- **Long-Term Change** (trend, baseline, deviation)
- **Current Status** (relative to local potential)
- **Combined Assessment** (integration of above for classification)

This toolkit facilitates reproducible, transparent, and customizable LPD analyses using raster-based time series such as **NDVI**, phenological metrics, or other proxies of land productivity.

---

## 🔁 Workflow Overview

Below is the recommended sequence of scripts to perform a full LPD analysis:

1. [`01_steadiness.md`](docs/01_steadiness.md) – Calculate **Steadiness Index**: trend & deviation from time series.
2. [`02_baseline.md`](docs/02_baseline.md) – Classify **Baseline Productivity Level** from early years.
3. [`03_state_change.md`](docs/03_state_change.md) – Identify **State Change** between early and recent periods.
4. [`04_long_term_change.md`](docs/04_long_term_change.md) – Combine indicators to derive **22-class Long-Term Change**.
5. [`05_remove_multicollinearity.md`](docs/05_remove_multicollinearity.md) – Remove **multicollinearity** in input layers.
6. [`06_07_clusteringEFT.md`](docs/06_07_clusteringEFT.md) – Perform **clustering** of Ecosystem Functional Types (EFTs).
7. [`08_lnscaling.md`](docs/08_lnscaling.md) – Compute **Local Net Scaling (LNS)**: current productivity vs. potential.
8. [`09_lpd_combassess.md`](docs/09_lpd_combassess.md) – Generate the **Combined LPD Assessment** map (5 classes).

---

## 🛠️ Requirements

- Python 3.8+
- Libraries:
  - `rasterio`, `numpy`, `pandas`, `scikit-learn`
  - `dask`, `joblib`, `numba`, `matplotlib`
- Input data:
  - Multi-band NDVI or productivity rasters (e.g., 1999–2023)
  - Land cover map (optional, for clustering)
  - Preprocessed raster stacks for clustering/LNS

---

## 🧭 References

- UNCCD (2016). *Good Practice Guidance for SDG Indicator 15.3.1*
- Bai et al. (2008). *Global Assessment of Land Degradation and Improvement.*
- Ivits & Cherlet (2016). *Land Productivity Dynamics: Towards integrated assessment*

---

## 🤝 Contributing

If you would like to contribute new modules, improve existing logic, or suggest enhancements, feel free to open a pull request or start a discussion in the Issues section.

---

