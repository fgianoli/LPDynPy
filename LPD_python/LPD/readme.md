# LPD Computation

# 🌿 Steadiness Index: Concept and Calculation

The **Steadiness Index** is a synthetic indicator that summarizes the **temporal behavior** of a pixel over a given time period (e.g., annual NDVI between 2000 and 2023). It is commonly used in **land degradation** and **vegetation dynamics** analysis to classify each pixel according to its **trend and stability**.

---

## 📊 1. What calculations are involved?

For each pixel, the time series is analyzed using **two main metrics**:

---

### 🔸 A. Slope (Linear Trend)

This measures the **overall trend** of the time series:

- Computed as the **slope of the least-squares linear regression** between `year` and `value`.
- Interpretation:
  - 📈 **Positive slope** → Increasing trend (e.g., vegetation improvement)
  - 📉 **Negative slope** → Decreasing trend (e.g., land degradation)
  - ➖ **Zero slope** → No clear trend

---

### 🔸 B. MTID (Mean Temporal Indicator of Deviation)

This measures the **temporal deviation** relative to the most recent valid value in the series. It estimates **how much the current value differs from the historical behavior**.

- Computed as:

MTID = sum of (last_valid_value - each_valid_value)

  - Interpretation:
- ➕ **Positive MTID** → Current value is **higher** than past values
- ➖ **Negative MTID** → Current value is **lower** than past values
- 0 → No significant deviation

---

## 🧠 2. How are they combined?

The **Steadiness Index** raster assigns each pixel a **discrete class value** (0 to 4), based on the combination of `slope` and `mtid`:

| Slope       | MTID       | Steadiness Index | Meaning                                        |
|-------------|------------|------------------|------------------------------------------------|
| < 0         | > 0        | 1                | Degrading, but still above past average        |
| < 0         | < 0        | 2                | Degrading and below past average               |
| > 0         | < 0        | 3                | Improving, but currently below average         |
| > 0         | > 0        | 4                | Improving and above average                    |
| 0 or MTID=0 | any        | 0                | No clear trend / stable                        |

---

## 📌 3. What is it used for?

The Steadiness Index is helpful for:

- Assessing **land degradation** or **restoration**
- Monitoring **land productivity dynamics**
- **Zoning** landscapes based on temporal patterns
- Supporting **environmental or agricultural decisions**

---

## 💬 Real-world example

In an agricultural region:

- A pixel with **negative slope** and **positive MTID** → degradation just started, but still above long-term average.
- A pixel with **negative slope** and **negative MTID** → in advanced degradation, performing worse than historical values.

---

## 🗺️ Steadiness Index Legend (Classes)

```text
0 - Stable or No trend
1 - Degrading but still above average
2 - Degrading and below average
3 - Improving but still below average
4 - Improving and above average

