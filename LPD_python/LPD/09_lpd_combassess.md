
# 🌍 Land Productivity Dynamics (LPD) – Combined Assessment

This script performs a **combined assessment of land productivity** by integrating two key layers:

- **Land Productivity Change (LPC)**: Derived from long-term trends (e.g., Steadiness Index + State Change)
- **Current Land Productivity Status (LPS)**: Based on recent productivity vs. potential (e.g., from LNScaling)

The result is a single **five-class map** used for **land degradation assessments** in frameworks like **UNCCD’s Land Degradation Neutrality (LDN)** indicator system.

---

## 📊 1. Method Overview

### 🔹 Inputs

| Layer                | Description                                                    |
|----------------------|----------------------------------------------------------------|
| `LandProd_change`    | Raster with values from 1–22 (e.g., Long-Term Change classes)  |
| `LandProd_current`   | Raster with current productivity percentage (e.g., LNS output) |
| `local_prod_threshold` | % threshold to define “adequate” current productivity         |

---

## 🧠 2. Classification Logic

### 🔸 With Both Inputs

If both change and current productivity are provided, classification follows this logic:

| Combined Condition                                                              | Class | Interpretation                      |
|----------------------------------------------------------------------------------|-------|--------------------------------------|
| Decline in trend **and** current < threshold                                     | 1     | Declining productivity              |
| Moderate decline **or** early warning signs                                      | 2     | Early signs of decline              |
| Fluctuations or stressed areas                                                   | 3     | Stressed (temporary decline)        |
| Positive fluctuation or recent improvement, but < threshold                     | 4     | Not stressed (recovery signs)       |
| Increasing trend and current ≥ threshold                                         | 5     | Increasing productivity             |

### 🔸 With Only Long-Term Change

If only `LandProd_change` is available:

| Long-Term Change Class     | Final Class |
|----------------------------|-------------|
| 1–6, 8–9                   | 1           |
| 7                          | 2           |
| 10–12                      | 3           |
| 13–15                      | 4           |
| 16–22                      | 5           |

---

## ⚙️ Parameters

| Parameter              | Description                                         | Default  |
|------------------------|-----------------------------------------------------|----------|
| `LandProd_change_path` | Path to the long-term change raster                 | required |
| `LandProd_current_path`| Path to the current productivity raster (optional)  | `None`   |
| `local_prod_threshold` | Threshold (%) for adequate productivity             | `50`     |
| `filename`             | Output file path                                    | `""`     |

---

## 💾 Output

- A single-band raster (float32) with values from 1 to 5:
  - `1` → Declining productivity
  - `2` → Early warning signs
  - `3` → Temporary or stressed decline
  - `4` → Improving but not yet recovered
  - `5` → Increasing and stable productivity

---

## 📎 Example Use Case

```bash
LandProd_change: long_term_change_2008_2023.tif
LandProd_current: lns_2023_rev2.tif
Output: LPD_finalMap_2023_new_correction.tif
```

---

## 📘 Related Concepts

- UNCCD Land Productivity Dynamics (LPD)
- Land Degradation Neutrality (LDN)
- Combined land condition assessment
- SDG Indicator 15.3.1
