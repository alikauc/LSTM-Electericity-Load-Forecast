# Walkthrough: Restructuring the LSTM Load Forecasting Pipeline

Welcome! This walkthrough details how we restructured your monolithic electricity load forecasting Jupyter Notebook into a production-grade, modular Python codebase. By applying fundamental software engineering principles and introducing automated unit testing, we transitioned the pipeline into a highly maintainable, readable, and robust system.

---

## 📖 Key Software Engineering Principles Applied

### 1. Separation of Concerns (SoC)
**Separation of Concerns** is a design principle that dictates that a program should be split into distinct sections, where each section addresses a single, isolated aspect of the program's overall functionality. 

In a standard data science Jupyter Notebook, we often mix:
* File loading, parsing, and cleaning
* Neural network architecture definitions
* Training loops and state updates
* Plotting and report generations

This makes debugging, editing, and version controlling extremely difficult. To apply SoC, we extracted these concerns into three dedicated, single-responsibility files:
* [data_loader.py](file:///c:/Users/alikc/OneDrive/Documents/LSTM-Electericity-Load-Forecast/data_loader.py) handles raw ingestion, normalization, window indexing, and PyTorch `DataLoader` preparation.
* [model_architecture.py](file:///c:/Users/alikc/OneDrive/Documents/LSTM-Electericity-Load-Forecast/model_architecture.py) holds the PyTorch layers, dimensions, and forward propagation logic for the `LSTMModel`.
* [train.py](file:///c:/Users/alikc/OneDrive/Documents/LSTM-Electericity-Load-Forecast/train.py) orchestrates training, early stopping validation checkpoints, comparison forecasting (with ARIMA), and result reporting/plotting.

---

### 2. Don't Repeat Yourself (DRY)
The **DRY (Don't Repeat Yourself)** principle states that "every piece of knowledge must have a single, unambiguous, authoritative representation within a system." 

In the original notebook, forecasting logic, dataset slicing, and metric calculations were copied multiple times. We eliminated this duplication:
* **Central Feature Specifications**: Stored `DEFAULT_FEATURES` in a single place (`data_loader.py`) to prevent sync mismatches.
* **Unified Metrics**: Used standard scikit-learn metrics (`mean_absolute_error`, etc.) in a consolidated flow inside `train.py`.
* **Standardized Forecasting**: Centralized LSTM prediction windows and inverse scaling calculations inside functional, parameterized subroutines.

---

## 🛠️ The Restructured Modules

Let's review the architectural details of each new module we created.

### 📊 1. Data Loader Module
File: [data_loader.py](file:///c:/Users/alikc/OneDrive/Documents/LSTM-Electericity-Load-Forecast/data_loader.py)

This module handles the data pipeline. It contains:
* **`LoadForecastDataset(Dataset)`**: Implements PyTorch's custom Dataset class.
  * In `__init__`, it accepts a numpy array of scaled features.
  * It iterates using a sliding window:
    ```python
    for i in range(len(data) - input_len - output_len):
        x_window = data[i : i + input_len] # Features input context
        y_window = data[i + input_len : i + input_len + output_len, 0] # Calgary Load Target
    ```
  * Converts the extracted windows into highly optimized `float32` tensors.
* **`prepare_data(...)`**: Orchestrates data loading.
  * Ingests the complete CSV and filters specified columns.
  * Scales columns to `[0, 1]` using scikit-learn's `MinMaxScaler`.
  * Splits the dataset chronologically: **70% for Training, 15% for Validation, and 15% for Testing**. Chronological splitting is critical in time-series forecasting to prevent data leakage from future timestamps into the past.
  * Wraps splits in `Subset` objects and creates PyTorch `DataLoader` objects. Only the training loader uses `shuffle=True` to help model generalization.

---

### 🧠 2. Model Architecture Module
File: [model_architecture.py](file:///c:/Users/alikc/OneDrive/Documents/LSTM-Electericity-Load-Forecast/model_architecture.py)

This module defines the deep learning model:
* **`LSTMModel(nn.Module)`**: Subclasses `nn.Module` to encapsulate a multi-layer LSTM network.
* Uses `batch_first=True` so that the model expects input tensors of shape `[batch_size, sequence_length, features]`.
* In `forward()`, it routes inputs through the LSTM layers:
  ```python
  out, _ = self.lstm(x) # Out shape: [batch_size, sequence_length, hidden_size]
  ```
* Rather than mapping the output at all time-steps, it extracts the final sequence element's hidden state (`out[:, -1, :]`) and passes it into a linear layer to predict the 24-hour horizon.

---

### 🚀 3. Orchestration & Training Module
File: [train.py](file:///c:/Users/alikc/OneDrive/Documents/LSTM-Electericity-Load-Forecast/train.py)

This module runs the entire workflow. It houses:
* **`train_model(...)`**: Executes training loops and backpropagation updates (`optimizer.step()`). It tracks validation losses and uses **Early Stopping** to prevent overfitting by restoring the best weights state when the validation loss fails to decrease for a sequential number of epochs (`patience`).
* **ARIMA Comparison**: Integrates classical `ARIMA(2, 1, 2)` baseline forecasting to evaluate modern deep learning vs traditional statistical time-series methods. To boost performance, we implemented parallel processing via `joblib.Parallel` so ARIMA calculates in parallel across the full test set.
* **IEEE Format Plotting**: Prepares multi-panel subplots conforming to standard paper templates (axes scaling, serialization, legend formatting, grids) and exports outputs as SVG/PNG and high-resolution vector PDF plots.
* **Archival compression**: Packs all outputs (reports, CSV comparisons, loss curves, subplots) into a structured `forecast_results.zip`.

---

## 🧪 Learning the Basics of `pytest`

Testing is a fundamental pillar of software engineering. Jupyter Notebooks rely on visual cell checks, but in production, we need **automated, repeatable unit tests**.

### What is `pytest`?
`pytest` is a popular Python testing framework. It automatically discovers testing files (matching `test_*.py` or `*_test.py`) and executes functions starting with `test_`.

### Our Test Suite
File: [test_data_loader.py](file:///c:/Users/alikc/OneDrive/Documents/LSTM-Electericity-Load-Forecast/tests/test_data_loader.py)

We wrote three simple, yet robust unit tests that isolate and verify our modules:
1. **`test_dataset_shapes`**:
   * Generates a random array mimicking our multi-variable dataset.
   * Asserts that `LoadForecastDataset` correctly slices the sliding windows (i.e. length is exactly `num_samples - input_len - output_len`).
   * Fetches an item and asserts that features `x` are of shape `(24, 6)` and target `y` is of shape `(24,)` and both are PyTorch Tensors.
2. **`test_prepare_data_pipeline`**:
   * Uses `pytest`'s built-in `tmp_path` fixture to dynamically write a small temporary mock dataset CSV.
   * Runs the complete `prepare_data` pipeline.
   * Asserts that MinMaxScaler scales data exactly between 0 and 1.
   * Verifies that the custom data loader partitions the split subsets according to ratios, and that the dataloader outputs expected batch tensor sizes.
3. **`test_model_dimensions`**:
   * Instantiates an `LSTMModel`.
   * Feeds dummy tensor inputs simulating batched variables.
   * Asserts that the network computes cleanly and produces outputs of shape `(batch_size, 24)`.

---

## 🏃 How to Run the Tests

To run these tests on your machine, first ensure you install `pytest` in your environment:
```bash
pip install pytest
```

Then, navigate to the project directory and execute:
```bash
pytest tests/
```

`pytest` will automatically discover the `tests/test_data_loader.py` file, execute all three test suites, and print a clean summary output verifying that all assertions passed.
