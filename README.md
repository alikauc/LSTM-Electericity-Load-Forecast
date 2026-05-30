# 🔋 Electricity Load Forecasting using LSTM
---
The objective was to **predict day-ahead electricity load** using a Long Short-Term Memory (LSTM) deep learning model, leveraging historical time-series data for accurate forecasting.


## 📌 Project Overview
- **Goal:** Forecast electricity load 24 hours in advance.
- **Approach:** Applied LSTM neural networks to capture temporal dependencies in load demand.
- **Dataset:** [Hourly electricity consumption data](https://www.aeso.ca/market/market-and-system-reporting/data-requests/hourly-load-by-area-and-region) together with weather conditions ( the [provided dataset](https://github.com/alikauc/LSTM-Electericity-Load-Forecast/blob/main/An-LSTM-Based-Approach-to-Day-Ahead-Electricity-Load-Forecasting-main/data_complete.csv) is preprocessed and normalized).
- **Evaluation Metric:** Normalized Root Mean Square Error (NRMSE).

---

## 👥 Team Members
- **Ali Karimi** <!-- – Data preprocessing, LSTM model development, evaluation, visualization.-->
- Noureldin Amer
- Yuhao Huang
- Mohammad Alhashem

<!--📎 **Original Repository** (uploaded by teammate): [username/project-name](https://github.com/username/project-name)-->

---

## 📊 My Contributions
- Designed and implemented **data cleaning and preprocessing** scripts([code](https://github.com/alikauc/LSTM-Electericity-Load-Forecast/blob/main/An-LSTM-Based-Approach-to-Day-Ahead-Electricity-Load-Forecasting-main/data_cleaning.ipynb)).
- Developed and trained the **LSTM neural network** for load forecasting ([code](https://github.com/alikauc/LSTM-Electericity-Load-Forecast/blob/main/An-LSTM-Based-Approach-to-Day-Ahead-Electricity-Load-Forecasting-main/training%20and%20validating%20LSTM%20model%20for%20load%20forcasting.ipynb)).
- Conducted **hyperparameter tuning** to optimize performance.
- Calculated and analyzed **NRMSE** for model evaluation.
- Created **plots and visualizations** comparing actual vs predicted loads.
- Contributed to writing the project report.

---

## 📈 Results & Model Comparison

We benchmark the LSTM against an optimized ARIMA statistical baseline. ARIMA hyperparameters were tuned via a 51-configuration parallel grid search on the cluster.

### Full Test Set Performance

| Metric | LSTM | ARIMA (2,0,2) Best | LSTM Improvement |
| :--- | :---: | :---: | :---: |
| **MAE (MW)** | **31.33** | 119.27 | 73.7% lower |
| **RMSE (MW)** | **38.70** | 143.03 | 72.9% lower |
| **MAPE** | **2.56%** | 10.89% | 76.5% lower |
| **R² Score** | **0.9201** | — | — |

### Challenging Event Days

| Date | Event | LSTM MAPE | ARIMA MAPE |
| :--- | :--- | :---: | :---: |
| 2024-01-10 | Temperature Drop | 5.00% | 13.15% |
| 2024-01-12 | Extreme Cold | 2.15% | 8.62% |
| 2024-05-07 | High Wind | 1.96% | 21.98% |
| 2024-10-14 | Public Holiday | 8.25% | 9.23% |

> LSTM wins on **every event day**, with the largest margins during extreme weather where ARIMA's linear assumptions fail.

📄 **Full comparison report**: [model_comparison.md](model_comparison.md)

---

## 🛠 Tech Stack
- **Programming:** Python
- **Libraries:** PyTorch, Pandas, NumPy, Matplotlib, scikit-learn, statsmodels, joblib
- **Tools:** Jupyter Notebook, Git, HPC cluster (SLURM) for training and ARIMA optimization

<!--## 🚀 How to Run
```bash
# Clone this repository
git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
cd REPO_NAME

# Install dependencies
pip install -r requirements.txt

# Run the model
python main.py-->
For more details, look at the [report](https://github.com/alikauc/LSTM-Electericity-Load-Forecast/blob/main/An-LSTM-Based-Approach-to-Day-Ahead-Electricity-Load-Forecasting-main/Final_Report_ENEL645_group7.pdf).

