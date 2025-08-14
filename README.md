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

## 🛠 Tech Stack
- **Programming:** Python
- **Libraries:** PyTorch, Pandas, NumPy, Matplotlib, scikit-learn
- **Tools:** Jupyter Notebook, Git, HPC cluster for training

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
