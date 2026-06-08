# ⚡ LSTM Electricity Load Forecasting — Production Pipeline

A production-hardened, configurable pipeline for **day-ahead electricity load forecasting** using LSTM neural networks, benchmarked against an optimized ARIMA statistical baseline.

Built on the [Calgary AESO dataset](https://www.aeso.ca/market/market-and-system-reporting/data-requests/hourly-load-by-area-and-region) (2011–2024) with weather covariates. Designed for reproducible training on HPC clusters (SLURM / Compute Canada).

> 📓 **Looking for the original research project?** See [`An-LSTM-Based-Approach-to-Day-Ahead-Electricity-Load-Forecasting-main/README.md`](An-LSTM-Based-Approach-to-Day-Ahead-Electricity-Load-Forecasting-main/README.md) for the Jupyter notebooks, report, and academic context.

---

## 📊 Model Performance

| Metric | LSTM | ARIMA (2,0,2) Best | LSTM Improvement |
| :--- | :---: | :---: | :---: |
| **MAE (MW)** | **31.33** | 119.27 | 73.7% lower |
| **RMSE (MW)** | **38.70** | 143.03 | 72.9% lower |
| **MAPE** | **2.56%** | 10.89% | 76.5% lower |
| **R² Score** | **0.9201** | — | — |

📄 Full benchmark with event-day breakdown and ARIMA grid search results: [`model_comparison.md`](model_comparison.md)

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    config.yaml                          │
│              (all hyperparameters)                      │
└────────────────────┬────────────────────────────────────┘
                     │
              config_loader.py   ← YAML + CLI overrides
                     │
              ┌──────▼──────┐
              │   train.py   │   ← main() orchestrator
              │  (pipeline)  │
              └──┬───┬───┬──┘
                 │   │   │
    ┌────────────┘   │   └────────────┐
    ▼                ▼                ▼
data_loader.py  model_architecture.py  arima_model.py
(load, scale,    (LSTMModel nn.Module)  (ARIMA baseline
 split, batch)                           with fallback)
    │                                     │
    └──────────┬──────────────────────────┘
               ▼
         log_config.py
    (dual-handler logging:
     console INFO + file DEBUG)
```

---

## 📁 Project Structure

```
├── train.py                  # Main training & evaluation pipeline
├── data_loader.py            # Data ingestion, scaling, train/val/test split
├── model_architecture.py     # LSTM model definition (PyTorch nn.Module)
├── arima_model.py            # ARIMA baseline with input validation & fallback
├── config.yaml               # All hyperparameters and paths (YAML)
├── config_loader.py          # YAML config + CLI argument override support
├── log_config.py             # Centralized dual-handler logging setup
├── requirements.txt          # Python dependencies with minimum version pins
│
├── tests/                    # Test suite (24 tests)
│   ├── test_data_loader.py   # Data pipeline shape & dimension tests
│   ├── test_arima_model.py   # ARIMA input validation tests
│   ├── test_config_loader.py # Config loading & merging tests
│   └── test_log_config.py    # Logging setup tests
│
├── submit_job.sh             # SLURM job script (H100 GPU)
├── submit_job_mig.sh         # SLURM job script (MIG partition)
├── submit_arima_array.sh     # SLURM array job for ARIMA grid search
├── test_arima_only.py        # Standalone ARIMA evaluation script
├── aggregate_arima_results.py# Aggregates ARIMA grid search results
│
├── model_comparison.md       # Full LSTM vs ARIMA benchmark report
├── models/                   # Saved model checkpoints (.pth)
├── logs/                     # Timestamped training log files
│
└── An-LSTM-Based-Approach-to-Day-Ahead-Electricity-Load-Forecasting-main/
    ├── README.md             # Original research project README
    ├── data_complete.csv     # Preprocessed dataset (hourly, 2011–2024)
    ├── data_cleaning.ipynb   # Data preprocessing notebook
    ├── training and validating LSTM model for load forcasting.ipynb
    └── Final_Report_ENEL645_group7.pdf
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- CUDA-capable GPU (tested on NVIDIA H100 / A100 MIG)
- Dataset: `data_complete.csv` (included in the research subfolder)

### Local Setup

```bash
# Clone the repository
git clone https://github.com/alikauc/LSTM-Electericity-Load-Forecast.git
cd LSTM-Electericity-Load-Forecast

# Create virtual environment
python -m venv env
source env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run training with default config
python train.py

# Override hyperparameters via CLI
python train.py --epochs 50 --lr 0.0005 --batch-size 128
```

### HPC / SLURM Deployment

```bash
# Load required modules (Compute Canada)
module load python/3.11.5 scipy-stack

# Activate virtualenv and install extras
source env/bin/activate
pip install pyyaml pytest

# Submit training job to GPU node
sbatch submit_job.sh

# Or use a MIG partition
sbatch submit_job_mig.sh
```

---

## ⚙️ Configuration

All hyperparameters are externalized in [`config.yaml`](config.yaml). Any value can be overridden via CLI flags:

| Config Key | CLI Flag | Default | Description |
| :--- | :--- | :---: | :--- |
| `training.epochs` | `--epochs` | 100 | Maximum training epochs |
| `training.batch_size` | `--batch-size` | 64 | Mini-batch size |
| `training.learning_rate` | `--lr` | 0.001 | Adam learning rate |
| `training.patience` | `--patience` | 5 | Early stopping patience |
| `training.seed` | `--seed` | 42 | Random seed for reproducibility |
| `model.hidden_size` | `--hidden-size` | 64 | LSTM hidden dimension |
| `model.num_layers` | `--num-layers` | 2 | Number of stacked LSTM layers |
| `data.csv_path` | `--csv-file` | `data_complete.csv` | Path to dataset |

---

## 🧪 Testing

The test suite covers data loading, configuration, logging, and ARIMA validation:

```bash
# Run all tests
python -m pytest tests/ -v

# On Compute Canada (system packages required)
bash -c 'module load scipy-stack 2>/dev/null; source env/bin/activate && python -m pytest tests/ -v'
```

**Current status: 24/24 tests passing ✅**

---

## 🔧 Production Hardening (Changelog)

This pipeline was systematically hardened from the original research prototype. Key improvements:

| Area | What Changed | Why |
| :--- | :--- | :--- |
| **Data Leakage** | Scaler fit on training split only | Prevents information leakage from test/val data into normalization |
| **Reproducibility** | Deterministic seeding (torch, numpy, random, cudnn) | Bit-for-bit reproducible runs on same hardware |
| **Configuration** | YAML config + CLI overrides | No more editing source code for hyperparameter sweeps |
| **Logging** | Dual-handler (console INFO + file DEBUG) | Structured, timestamped logs instead of print statements |
| **Validation** | Input guards on ARIMA functions | Fail-fast on invalid orders, empty history, bad parameters |
| **Modularity** | `main()` refactored from ~280 to ~95 lines | 5 extracted functions with clear single responsibilities |
| **Testing** | 24 unit tests across 4 modules | Regression safety net for future changes |
| **Dependencies** | `requirements.txt` with version pins | Reproducible environment setup |

---

## 📜 License

[MIT License](LICENSE)

---

## 👤 Author

**Ali Karimi** — [GitHub](https://github.com/alikauc)
