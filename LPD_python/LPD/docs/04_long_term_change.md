
# 🌍 Long-Term Change Classification

This script computes a comprehensive **Long-Term Change (LTC)** map by integrating three separate indicators:

- **Steadiness Index**: Captures long-term trends and temporal deviations.
- **Baseline Productivity Level**: Classifies pixels based on their initial productivity.
- **State Change**: Identifies recent changes in productivity.

By combining these layers, the method generates a **22-class categorical map** that synthesizes the long-term dynamics of each pixel.

---

## 🔗 1. Inputs

The script requires three raster inputs:

| Input               | Description                                      |
|---------------------|--------------------------------------------------|
| `Steadiness Index`  | A raster with values 0–4 (from the steadiness analysis) |
| `Baseline Levels`   | A raster with values 1–3 (low, medium, high productivity) |
| `State Change`      | A raster with values 1–3 (improvement, stable, degradation) |

---

## 🧮 2. Step-by-Step Processing

### 🔹 A. Steadiness + Baseline Reclassification

The first step combines the **Steadiness Index** (values 1–4) with **Baseline Level** (values 1–3), resulting in a composite raster (`steadiness_baseline`) with 12 unique classes.

Example mapping:

| Steadiness | Baseline | Combined Class |
|------------|----------|----------------|
| 1          | 1        | 1              |
| 2          | 3        | 6              |
| 4          | 2        | 11             |
| etc.       | ...      | ...            |

### 🔹 B. Combine with State Change

Each combination of the 12 `steadiness_baseline` classes and 3 `state_change` classes is then reclassified into a **final Long-Term Change** class (1–22) using a defined mapping.

Examples:

| Steadiness+Baseline | State Change | LTC Class |
|---------------------|--------------|-----------|
| 1                   | 1            | 1         |
| 4                   | 3            | 18        |
| 12                  | 2            | 22        |
| etc.                | ...          | ...       |

---

## 🗂️ 3. Output

- A single-band raster with **22 categorical values**
- Each value represents a unique combination of:
  - Long-term trend
  - Initial condition
  - Recent change

This classification allows for in-depth spatio-temporal analysis of **land productivity dynamics**.

---

## ⚙️ Parameters

| Parameter         | Description                                |
|-------------------|--------------------------------------------|
| `filename`        | Output file path (GeoTIFF)                 |
| `profile`         | Raster metadata profile (copied from input) |

---

## 🗺️ 4. Use Cases

- **UNCCD Land Degradation Neutrality (LDN) indicator framework**
- Monitoring **ecosystem changes** over long periods
- Identifying **priority areas** for restoration or conservation

---

## 📎 Example Use Case

```bash
Input Steadiness Index: SteadInd_2008_2023.tif
Input Baseline Level: baseline_2008_2023.tif
Input State Change: state_change_2008_2023.tif
Output: long_term_change_2008_2023.tif
```

---

## 📘 Related Concepts

- Land Productivity Dynamics (LPD)
- Ecosystem Functional Types
- Steadiness and State Change indicators
