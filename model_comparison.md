# LSTM vs ARIMA: Model Comparison Report

A comprehensive comparison of the **LSTM deep learning model** and the **ARIMA statistical baseline** for day-ahead electricity load forecasting on the Calgary dataset (2011–2024).

---

## 1. Full Test Set Performance

Average metrics across **all test set days** (15% holdout, 2022–2024):

| Metric | LSTM | ARIMA (2,1,2) Baseline | ARIMA (2,0,2) Optimized | LSTM Improvement vs Best ARIMA |
| :--- | :---: | :---: | :---: | :---: |
| **MAE (MW)** | **31.33** | 131.22 | 119.27 | **73.7% lower** |
| **RMSE (MW)** | **38.70** | 155.12 | 143.03 | **72.9% lower** |
| **MAPE (%)** | **2.56%** | 11.47% | 10.89% | **76.5% lower** |
| **R² Score** | **0.9201** | — | — | — |

> The LSTM achieves approximately **4× lower error** than the best-tuned ARIMA configuration across all metrics.

---

## 2. ARIMA Hyperparameter Optimization Summary

A 51-configuration grid search was run in parallel via SLURM Array Jobs across cluster compute nodes. The top 5 ARIMA configurations on the validation set:

| Rank | Order | History (Hours) | MAE (MW) | RMSE (MW) | MAPE (%) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | ARIMA(2, 0, 2) | 168 (7 days) | 119.27 | 143.03 | 10.89% |
| 2 | ARIMA(2, 0, 2) | 336 (14 days) | 119.39 | 143.70 | 10.95% |
| 3 | ARIMA(2, 0, 2) | 72 (3 days) | 121.30 | 147.03 | 10.84% |
| 4 | ARIMA(2, 0, 0) | 336 (14 days) | 130.88 | 157.59 | 12.11% |
| 5 | ARIMA(2, 0, 1) | 336 (14 days) | 132.46 | 159.65 | 12.33% |

**Key finding**: Removing differencing ($d=0$) improved ARIMA by ~12% over the $d=1$ baseline, since electricity load is cyclic and mean-reverting rather than trend-dominated.

---

## 3. Specific Event Day Comparison

Performance on challenging weather events and calendar anomalies:

| Date | Event | LSTM MAE | ARIMA MAE | LSTM MAPE | ARIMA MAPE | Winner |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 2024-01-10 | Temperature Drop | 73.00 | 201.85 | 5.00% | 13.15% | LSTM |
| 2024-01-12 | Extreme Cold | 33.00 | 140.00 | 2.15% | 8.62% | LSTM |
| 2024-10-12 | Weekend (Long Holiday) | 29.27 | 56.66 | 2.64% | 5.58% | LSTM |
| 2024-10-14 | Public Holiday | 90.84 | 101.68 | 8.25% | 9.23% | LSTM |
| 2024-10-19 | Typical Weekend | 21.69 | 64.06 | 1.97% | 6.11% | LSTM |
| 2024-10-21 | Typical Weekday | 37.42 | 167.02 | 2.98% | 12.60% | LSTM |
| 2024-05-07 | High Wind Conditions | 23.58 | 286.77 | 1.96% | 21.98% | LSTM |

> LSTM wins on **every single event day**. The gap is largest during extreme weather (temperature drops, high wind), where ARIMA's linear assumptions break down but the LSTM's learned non-linear feature interactions remain robust.

---

## 4. Weekend vs Weekday Breakdown

| Day Type | LSTM MAE | ARIMA MAE | LSTM MAPE | ARIMA MAPE |
| :--- | :---: | :---: | :---: | :---: |
| **Weekday** | 31.00 | 135.84 | 2.49% | 11.62% |
| **Weekend** | 32.13 | 119.74 | 2.75% | 11.09% |

Both models perform slightly worse on weekends due to less regular demand patterns. However, the LSTM degrades gracefully (MAE +1.1 MW) while ARIMA actually improves on weekends—likely because weekend load profiles are smoother and more amenable to linear modelling.

---

## 5. Why LSTM Outperforms ARIMA

| Factor | ARIMA | LSTM |
| :--- | :--- | :--- |
| **Multi-variate inputs** | ❌ Univariate (load only) | ✅ Temperature, wind, calendar features |
| **Non-linear patterns** | ❌ Linear model | ✅ Non-linear activation functions |
| **Long-range dependencies** | ❌ Limited by order $(p, q)$ | ✅ Gated memory cells |
| **Extreme events** | ❌ Poor generalization | ✅ Learns from weather covariates |
| **Computational cost** | ✅ Fast per-model fit | ❌ GPU training required |
| **Interpretability** | ✅ Transparent coefficients | ❌ Black-box |

---

## 6. Conclusion

The LSTM model achieves a **MAPE of 2.56%** compared to ARIMA's best of **10.89%**, representing a **76.5% relative improvement**. The LSTM's ability to incorporate exogenous weather and calendar features, combined with its capacity to learn non-linear temporal dependencies, makes it substantially more accurate for day-ahead electricity load forecasting. ARIMA remains a useful interpretable baseline but cannot match deep learning performance on this multi-variate forecasting task.
