import os
import math
import random
import logging
import zipfile
import warnings
from datetime import timedelta
from typing import Tuple, List, Dict, Any

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import matplotlib as mpl
from tqdm import tqdm

from data_loader import prepare_data, DEFAULT_FEATURES
from model_architecture import LSTMModel
from arima_model import forecast_day_arima, process_single_forecast_arima
from log_config import setup_logging
from config_loader import get_train_config

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default random seed for reproducibility
DEFAULT_SEED = 42


def set_seeds(seed: int = DEFAULT_SEED) -> None:
    """
    Sets random seeds across all libraries for deterministic reproducibility.

    Must be called BEFORE any data loading, model initialization, or
    DataLoader shuffling to guarantee identical results across runs.

    Args:
        seed (int): The random seed value. Default is 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info("Random seeds set to %d (torch, numpy, random, cudnn deterministic)", seed)


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int = 100,
    patience: int = 5,
    device: torch.device = DEVICE,
) -> Tuple[List[float], List[float]]:
    """
    Trains the PyTorch model with Early Stopping validation verification.

    Args:
        model (nn.Module): The model to train.
        train_loader (DataLoader): PyTorch training dataloader.
        val_loader (DataLoader): PyTorch validation dataloader.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimization algorithm.
        epochs (int): Max number of epochs. Default 100.
        patience (int): Early stopping patience threshold. Default 5.
        device (torch.device): GPU/CPU device pointer.

    Returns:
        Tuple[List[float], List[float]]: Train losses and validation losses across epochs.
    """
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                output = model(X_batch)
                val_loss += criterion(output, y_batch).item()

        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        logger.info("Epoch %d/%d | Train Loss: %.4f | Val Loss: %.4f", epoch + 1, epochs, avg_train, avg_val)

        # --- EARLY STOPPING CHECK ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                logger.info("Early stopping triggered at epoch %d", epoch + 1)
                if best_model_state is not None:
                    model.load_state_dict(best_model_state)
                break

    return train_losses, val_losses


def forecast_day_lstm(
    model: nn.Module,
    df_scaled: pd.DataFrame,
    df_unscaled: pd.DataFrame,
    features: List[str],
    scaler: Any,
    target_date: str,
    device: torch.device = DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forecasting helper for a specific day using the trained LSTM.

    Args:
        model (nn.Module): Trained LSTM model.
        df_scaled (pd.DataFrame): Scaled feature dataframe.
        df_unscaled (pd.DataFrame): Original unscaled dataframe.
        features (List[str]): List of column features.
        scaler (MinMaxScaler): Scaler used to inverse scale targets.
        target_date (str): Timestamp/date string starting the 24h prediction window.
        device (torch.device): Device execution target.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted electricity load (24h) and actual load (24h).
    """
    forecast_start = pd.Timestamp(target_date)
    input_start = forecast_start - timedelta(hours=24)
    input_end = forecast_start - timedelta(hours=1)
    target_end = forecast_start + timedelta(hours=23)

    # Historical context values
    X_input = df_scaled.loc[input_start:input_end, features].values
    X_tensor = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(X_tensor).cpu().numpy().flatten()

    # Inverse transform. Note: the scaler expects the same feature dimension,
    # so we pad predictions with zeros for all other features before inverting.
    padded_pred = np.c_[y_pred_scaled, np.zeros((24, len(features) - 1))]
    y_pred = scaler.inverse_transform(padded_pred)[:, 0]

    y_true = df_unscaled.loc[forecast_start:target_end, "Load_Calgary"].values
    return y_pred, y_true




def run_test_evaluation_lstm(
    model: nn.Module,
    df_scaled: pd.DataFrame,
    df_unscaled: pd.DataFrame,
    features: List[str],
    scaler: Any,
    valid_test_indices: List[int],
    full_timestamps: pd.DatetimeIndex,
    input_len: int,
    device: torch.device = DEVICE,
) -> List[Dict[str, Any]]:
    """
    Evaluates the trained LSTM model across all test set sequences.
    """
    forecast_results = []
    logger.info("Evaluating LSTM on test set...")
    for idx in tqdm(valid_test_indices, desc="LSTM Testing"):
        forecast_start = full_timestamps[idx + input_len]
        forecast_date = forecast_start.strftime("%Y-%m-%d")
        try:
            X_pred = df_scaled.loc[
                forecast_start - pd.Timedelta(hours=24) : forecast_start - pd.Timedelta(hours=1), features
            ].values
            X_pred_tensor = torch.tensor(X_pred, dtype=torch.float32).unsqueeze(0).to(device)

            model.eval()
            with torch.no_grad():
                y_pred_scaled = model(X_pred_tensor).cpu().numpy().flatten()

            padded_pred = np.c_[y_pred_scaled, np.zeros((24, len(features) - 1))]
            y_pred = scaler.inverse_transform(padded_pred)[:, 0]

            actual_day_data = df_unscaled.loc[
                forecast_start : forecast_start + pd.Timedelta(hours=23), "Load_Calgary"
            ]
            if len(actual_day_data) < 24:
                continue

            y_true = actual_day_data.values

            mae = mean_absolute_error(y_true, y_pred)
            rmse = math.sqrt(mean_squared_error(y_true, y_pred))
            mape = mean_absolute_percentage_error(y_true, y_pred) * 100
            r2 = r2_score(y_true, y_pred)

            forecast_results.append({"date": forecast_date, "MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2})
        except Exception as e:
            logger.warning("LSTM skipped %s due to error: %s", forecast_date, e)

    return forecast_results


def plot_loss_curve(train_losses: List[float], val_losses: List[float], output_path: str = "loss_curve.png"):
    """
    Plots and saves the training vs validation loss curve.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved loss curve to %s", output_path)


def main():
    # Initialize logging before anything else
    setup_logging()

    # Load configuration from YAML + CLI overrides
    cfg = get_train_config()

    # Set reproducibility seeds before any random operations
    set_seeds(cfg["training"]["seed"])

    # Resolve dataset path with fallback
    csv_file = cfg["data"]["csv_path"]
    if not os.path.exists(csv_file):
        csv_file = cfg["data"].get("csv_path_fallback", csv_file)
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset CSV not found at {cfg['data']['csv_path']} or fallback")

    # Extract config values
    input_len = cfg["training"]["input_len"]
    output_len = cfg["training"]["output_len"]
    batch_size = cfg["training"]["batch_size"]
    epochs = cfg["training"]["epochs"]
    patience = cfg["training"]["patience"]
    learning_rate = cfg["training"]["learning_rate"]
    features = cfg["data"]["features"]

    logger.info("Ingesting dataset from: %s", csv_file)
    logger.info("Config: epochs=%d, batch_size=%d, lr=%.4f, patience=%d",
                epochs, batch_size, learning_rate, patience)

    # 1. Load Data
    train_loader, val_loader, test_loader, scaler, df_unscaled, df_scaled = prepare_data(
        csv_path=csv_file,
        features=features,
        input_len=input_len,
        output_len=output_len,
        batch_size=batch_size,
    )

    # 2. Build Model
    model = LSTMModel(
        input_size=len(features),
        hidden_size=cfg["model"]["hidden_size"],
        num_layers=cfg["model"]["num_layers"],
        output_size=output_len,
    ).to(DEVICE)
    logger.info("Model successfully loaded to device: %s", DEVICE)

    # 3. Train Model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    logger.info("Beginning LSTM Model Training...")
    train_losses, val_losses = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=epochs,
        patience=patience,
        device=DEVICE,
    )

    # 4. Save model checkpoints
    model_checkpoint_path = cfg["output"]["model_checkpoint"]
    os.makedirs(os.path.dirname(model_checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), model_checkpoint_path)
    logger.info("Trained model saved to %s", model_checkpoint_path)

    # 5. Plot loss curve
    plot_loss_curve(train_losses, val_losses)

    # 6. Specific Days Comparisons (IEEE Subplots format)
    comparison_results = evaluate_event_days(
        model=model, df_scaled=df_scaled, df_unscaled=df_unscaled,
        features=features, scaler=scaler, device=DEVICE,
    )

    # 7. Full Test Set Evaluation
    lstm_forecast_results, arima_forecast_results = run_full_test_evaluation(
        model=model, df_scaled=df_scaled, df_unscaled=df_unscaled,
        features=features, scaler=scaler, input_len=input_len,
        output_len=output_len, device=DEVICE,
    )

    # 8. Performance Report
    write_performance_report(lstm_forecast_results, arima_forecast_results)

    # 9. Weekend vs Weekday analysis
    analyze_weekend_weekday(lstm_forecast_results, arima_forecast_results, df_unscaled)

    # 10. Archive results
    archive_results(comparison_results)


def evaluate_event_days(
    model: nn.Module,
    df_scaled: pd.DataFrame,
    df_unscaled: pd.DataFrame,
    features: List[str],
    scaler: Any,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """
    Evaluates LSTM and ARIMA forecasts on specific weather and holiday event days.
    Generates IEEE-formatted comparison subplots and per-day CSVs.

    Returns:
        List of comparison result dictionaries for each event day.
    """
    start_dates = [
        "2024-01-10", "2024-01-12", "2024-10-12", "2024-10-14",
        "2024-10-19", "2024-10-21", "2024-05-07",
    ]
    date_description = {
        "2024-01-10": "Temperature Drop Event",
        "2024-01-12": "Extreme Cold Day",
        "2024-10-12": "Weekend (Long Holiday)",
        "2024-10-14": "Public Holiday",
        "2024-10-19": "Typical Weekend",
        "2024-10-21": "Typical Weekday",
        "2024-05-07": "High Wind Conditions",
    }
    subplot_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

    mpl.rcParams.update({
        "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
        "axes.titlesize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "figure.figsize": (7, 9), "figure.dpi": 300,
    })

    fig, axs = plt.subplots(4, 2, figsize=(7, 8))
    axs = axs.flatten()
    y_min, y_max = 800, 1800
    comparison_results = []

    logger.info("Evaluating specific weather and holiday event days...")
    for i, date in enumerate(start_dates):
        lstm_pred, y_true = forecast_day_lstm(
            model=model, df_scaled=df_scaled, df_unscaled=df_unscaled,
            features=features, scaler=scaler, target_date=date, device=device,
        )
        arima_pred, _ = forecast_day_arima(df_unscaled=df_unscaled, target_date=date)

        pd.DataFrame({
            "Hour": range(24), "Actual_Load": y_true,
            "LSTM_Prediction": lstm_pred, "ARIMA_Prediction": arima_pred,
        }).to_csv(f"forecast_comparison_{date}.csv", index=False)

        lstm_mae = mean_absolute_error(y_true, lstm_pred)
        arima_mae = mean_absolute_error(y_true, arima_pred)
        lstm_mape = mean_absolute_percentage_error(y_true, lstm_pred) * 100
        arima_mape = mean_absolute_percentage_error(y_true, arima_pred) * 100

        ax = axs[i]
        ax.plot(range(24), y_true, color="black", linewidth=1.2, label="Actual")
        ax.plot(range(24), lstm_pred, linestyle="--", color="blue", linewidth=1, label="LSTM")
        ax.plot(range(24), arima_pred, linestyle=":", color="red", linewidth=1, label="ARIMA")
        ax.text(0.03, 0.97, subplot_labels[i], transform=ax.transAxes,
                fontsize=9, fontweight="bold", verticalalignment="top")
        ax.set_ylim(y_min, y_max)
        ax.set_title(date_description.get(date, ""))
        ax.text(0.03, 0.03,
                f"LSTM: MAE={lstm_mae:.0f}, MAPE={lstm_mape:.1f}%\n"
                f"ARIMA: MAE={arima_mae:.0f}, MAPE={arima_mape:.1f}%",
                transform=ax.transAxes, fontsize=7)
        ax.legend(loc="upper right", framealpha=0.0, fontsize=7)
        ax.grid(True, linestyle=":", alpha=0.5, color="lightgray")
        if i % 2 == 0:
            ax.set_ylabel("Load (MW)")
        if i >= 4 or i == len(start_dates) - 1:
            ax.set_xlabel("Hour of Day")

        comparison_results.append({
            "Date": date, "LSTM_MAE": lstm_mae, "ARIMA_MAE": arima_mae,
            "LSTM_RMSE": math.sqrt(mean_squared_error(y_true, lstm_pred)),
            "ARIMA_RMSE": math.sqrt(mean_squared_error(y_true, arima_pred)),
            "LSTM_MAPE": lstm_mape, "ARIMA_MAPE": arima_mape,
        })

    for j in range(len(start_dates), 8):
        axs[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("forecast_comparison_ieee.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved IEEE-formatted subplots.")
    pd.DataFrame(comparison_results).to_csv("lstm_vs_arima_comparison.csv", index=False)
    return comparison_results


def run_full_test_evaluation(
    model: nn.Module, df_scaled: pd.DataFrame, df_unscaled: pd.DataFrame,
    features: List[str], scaler: Any, input_len: int, output_len: int,
    device: torch.device,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Runs full chronological test set evaluation for both LSTM and ARIMA.

    Returns:
        Tuple of (lstm_forecast_results, arima_forecast_results).
    """
    dataset_length = len(df_scaled) - input_len - output_len
    train_size = int(dataset_length * 0.7)
    val_size = int(dataset_length * 0.15)
    valid_test_indices = list(range(train_size + val_size, dataset_length))
    full_timestamps = df_scaled.index

    lstm_forecast_results = run_test_evaluation_lstm(
        model=model, df_scaled=df_scaled, df_unscaled=df_unscaled,
        features=features, scaler=scaler, valid_test_indices=valid_test_indices,
        full_timestamps=full_timestamps, input_len=input_len, device=device,
    )
    pd.DataFrame(lstm_forecast_results).to_csv("daily_forecast_metrics.csv", index=False)

    logger.info("Evaluating ARIMA on test set in parallel...")
    arima_forecast_results = Parallel(n_jobs=-1)(
        delayed(process_single_forecast_arima)(idx, df_unscaled, full_timestamps, input_len)
        for idx in tqdm(valid_test_indices, desc="ARIMA Testing")
    )
    arima_forecast_results = [r for r in arima_forecast_results if r is not None]

    return lstm_forecast_results, arima_forecast_results


def write_performance_report(
    lstm_results: List[Dict], arima_results: List[Dict],
    output_path: str = "model_performance.txt",
) -> None:
    """Writes aggregate performance metrics for both models to a text file."""
    if not lstm_results or not arima_results:
        logger.warning("Skipping performance report: insufficient results")
        return

    lstm_df = pd.DataFrame(lstm_results)
    arima_df = pd.DataFrame(arima_results)

    with open(output_path, "w") as f:
        f.write("Average Performance on Full Test Set\n\n")
        f.write("LSTM:\n")
        f.write(f"   MAE: {lstm_df['MAE'].mean():.2f}\n")
        f.write(f"   RMSE: {lstm_df['RMSE'].mean():.2f}\n")
        f.write(f"   MAPE: {lstm_df['MAPE'].mean():.2f}%\n")
        f.write(f"   R2 Score: {lstm_df['R2'].mean():.4f}\n\n")
        f.write("ARIMA:\n")
        f.write(f"   MAE: {arima_df['MAE'].mean():.2f}\n")
        f.write(f"   RMSE: {arima_df['RMSE'].mean():.2f}\n")
        f.write(f"   MAPE: {arima_df['MAPE'].mean():.2f}%\n")
    logger.info("Performance report saved to %s", output_path)


def analyze_weekend_weekday(
    lstm_results: List[Dict], arima_results: List[Dict],
    df_unscaled: pd.DataFrame,
    output_path: str = "weekend_weekday_comparison.csv",
) -> None:
    """Compares model performance on weekends vs weekdays."""
    logger.info("Saving weekend vs weekday analysis comparison...")

    def _get_weekend_flag(date_val: pd.Timestamp) -> bool:
        row = df_unscaled.loc[date_val.floor("D")]["Is_Weekend"]
        return row.iloc[0] if isinstance(row, pd.Series) else row

    lstm_by_day = pd.DataFrame(lstm_results).assign(
        date=lambda x: pd.to_datetime(x["date"]),
        is_weekend=lambda x: x["date"].apply(_get_weekend_flag),
    ).groupby("is_weekend").agg({"MAE": "mean", "MAPE": "mean"})

    arima_by_day = pd.DataFrame(arima_results).assign(
        date=lambda x: pd.to_datetime(x["date"]),
        is_weekend=lambda x: x["date"].apply(_get_weekend_flag),
    ).groupby("is_weekend").agg({"MAE": "mean", "MAPE": "mean"})

    comparison = pd.DataFrame(index=[False, True])
    comparison["LSTM_MAE"] = lstm_by_day["MAE"]
    comparison["LSTM_MAPE"] = lstm_by_day["MAPE"]
    comparison["ARIMA_MAE"] = arima_by_day["MAE"]
    comparison["ARIMA_MAPE"] = arima_by_day["MAPE"]
    comparison = comparison.reset_index().rename(columns={"index": "Is_Weekend"})
    comparison["Day_Type"] = comparison["Is_Weekend"].map({False: "Weekday", True: "Weekend"})
    comparison = comparison[["Day_Type", "LSTM_MAE", "LSTM_MAPE", "ARIMA_MAE", "ARIMA_MAPE"]]
    comparison.to_csv(output_path, index=False)
    logger.info("Weekend vs Weekday comparison saved to %s", output_path)


def archive_results(
    comparison_results: List[Dict],
    output_path: str = "forecast_results.zip",
) -> None:
    """Compresses all output files into a single ZIP archive."""
    start_dates = [r["Date"] for r in comparison_results]
    output_files = (
        ["daily_forecast_metrics.csv", "model_performance.txt", "loss_curve.png", "lstm_vs_arima_comparison.csv"]
        + [f"forecast_comparison_{date}.csv" for date in start_dates]
        + ["forecast_comparison_ieee.png", "weekend_weekday_comparison.csv"]
    )
    logger.info("Compressing output files into %s...", output_path)
    with zipfile.ZipFile(output_path, "w") as zipf:
        for file in output_files:
            if os.path.exists(file):
                zipf.write(file)
    logger.info("All result files compressed into %s", output_path)


if __name__ == "__main__":
    main()

