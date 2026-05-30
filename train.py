import os
import math
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

warnings.filterwarnings("ignore")

# Setup device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

        # --- EARLY STOPPING CHECK ---
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            counter = 0
            best_model_state = model.state_dict()
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
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
    print("Evaluating LSTM on test set...")
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
            print(f"⚠️ LSTM skipped {forecast_date} due to error: {e}")

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
    print(f"Saved loss curve to {output_path}")


def main():
    # Define execution parameters
    csv_file = os.path.join(
        "An-LSTM-Based-Approach-to-Day-Ahead-Electricity-Load-Forecasting-main", "data_complete.csv"
    )
    if not os.path.exists(csv_file):
        csv_file = "data_complete.csv"  # Fallback

    print(f"Ingesting dataset from: {csv_file}")
    input_len = 24
    output_len = 24
    batch_size = 64
    epochs = 100
    patience = 5

    # 1. Load Data
    train_loader, val_loader, test_loader, scaler, df_unscaled, df_scaled = prepare_data(
        csv_path=csv_file,
        features=DEFAULT_FEATURES,
        input_len=input_len,
        output_len=output_len,
        batch_size=batch_size,
    )

    # 2. Build Model
    model = LSTMModel(input_size=len(DEFAULT_FEATURES), hidden_size=64, num_layers=2, output_size=output_len).to(
        DEVICE
    )
    print(f"Model successfully loaded to device: {DEVICE}")

    # 3. Train Model
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Beginning LSTM Model Training...")
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
    os.makedirs("models", exist_ok=True)
    model_checkpoint_path = "models/lstm_model.pth"
    torch.save(model.state_dict(), model_checkpoint_path)
    print(f"✅ Trained model saved to {model_checkpoint_path}")

    # 5. Plot loss curve
    plot_loss_curve(train_losses, val_losses)

    # 6. Specific Days Comparisons (IEEE Subplots format)
    start_dates = [
        "2024-01-10",  # major temp drop
        "2024-01-12",  # coldest day
        "2024-10-12",  # long weekend Saturday
        "2024-10-14",  # holiday Monday
        "2024-10-19",  # normal Saturday
        "2024-10-21",  # normal Monday
        "2024-05-07",  # max wind speed
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

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.figsize": (7, 9),
            "figure.dpi": 300,
        }
    )

    fig, axs = plt.subplots(4, 2, figsize=(7, 8))
    axs = axs.flatten()

    y_min, y_max = 800, 1800
    comparison_results = []

    print("Evaluating specific weather and holiday event days...")
    for i, date in enumerate(start_dates):
        lstm_pred, y_true = forecast_day_lstm(
            model=model,
            df_scaled=df_scaled,
            df_unscaled=df_unscaled,
            features=DEFAULT_FEATURES,
            scaler=scaler,
            target_date=date,
            device=DEVICE,
        )
        arima_pred, _ = forecast_day_arima(df_unscaled=df_unscaled, target_date=date)

        # Save single day tabular data to CSV
        comparison_df = pd.DataFrame(
            {"Hour": range(24), "Actual_Load": y_true, "LSTM_Prediction": lstm_pred, "ARIMA_Prediction": arima_pred}
        )
        comparison_df.to_csv(f"forecast_comparison_{date}.csv", index=False)

        # Calculate metrics
        lstm_mae = mean_absolute_error(y_true, lstm_pred)
        arima_mae = mean_absolute_error(y_true, arima_pred)
        lstm_mape = mean_absolute_percentage_error(y_true, lstm_pred) * 100
        arima_mape = mean_absolute_percentage_error(y_true, arima_pred) * 100

        # Plot into subplots
        ax = axs[i]
        ax.plot(range(24), y_true, color="black", linewidth=1.2, label="Actual")
        ax.plot(range(24), lstm_pred, linestyle="--", color="blue", linewidth=1, label="LSTM")
        ax.plot(range(24), arima_pred, linestyle=":", color="red", linewidth=1, label="ARIMA")

        ax.text(
            0.03,
            0.97,
            subplot_labels[i],
            transform=ax.transAxes,
            fontsize=9,
            fontweight="bold",
            verticalalignment="top",
        )
        ax.set_ylim(y_min, y_max)
        ax.set_title(date_description.get(date, ""))

        metrics_text = (
            f"LSTM: MAE={lstm_mae:.0f}, MAPE={lstm_mape:.1f}%\n" f"ARIMA: MAE={arima_mae:.0f}, MAPE={arima_mape:.1f}%"
        )
        ax.text(0.03, 0.03, metrics_text, transform=ax.transAxes, fontsize=7)
        ax.legend(loc="upper right", framealpha=0.0, fontsize=7)
        ax.grid(True, linestyle=":", alpha=0.5, color="lightgray")

        if i % 2 == 0:
            ax.set_ylabel("Load (MW)")
        if i >= 4 or i == len(start_dates) - 1:
            ax.set_xlabel("Hour of Day")

        comparison_results.append(
            {
                "Date": date,
                "LSTM_MAE": lstm_mae,
                "ARIMA_MAE": arima_mae,
                "LSTM_RMSE": math.sqrt(mean_squared_error(y_true, lstm_pred)),
                "ARIMA_RMSE": math.sqrt(mean_squared_error(y_true, arima_pred)),
                "LSTM_MAPE": lstm_mape,
                "ARIMA_MAPE": arima_mape,
            }
        )

    # Hide unused subplot (since we have 7 days in a 4x2 grid)
    if len(start_dates) < 8:
        for j in range(len(start_dates), 8):
            axs[j].set_visible(False)

    plt.tight_layout()
    plt.savefig("forecast_comparison_ieee.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved IEEE-formatted subplots.")

    pd.DataFrame(comparison_results).to_csv("lstm_vs_arima_comparison.csv", index=False)

    # 7. Complete Test Set Evaluation (Chronological Sequences)
    # The split ranges indices matching Subset splits
    dataset_length = len(df_scaled) - input_len - output_len
    train_size = int(dataset_length * 0.7)
    val_size = int(dataset_length * 0.15)
    valid_test_indices = list(range(train_size + val_size, dataset_length))

    full_timestamps = df_scaled.index

    # LSTM Full Evaluation
    lstm_forecast_results = run_test_evaluation_lstm(
        model=model,
        df_scaled=df_scaled,
        df_unscaled=df_unscaled,
        features=DEFAULT_FEATURES,
        scaler=scaler,
        valid_test_indices=valid_test_indices,
        full_timestamps=full_timestamps,
        input_len=input_len,
        device=DEVICE,
    )
    lstm_results_df = pd.DataFrame(lstm_forecast_results)
    lstm_results_df.to_csv("daily_forecast_metrics.csv", index=False)

    # ARIMA Full Evaluation in parallel (using joblib for high efficiency)
    print("Evaluating ARIMA on test set in parallel...")
    arima_forecast_results = Parallel(n_jobs=-1)(
        delayed(process_single_forecast_arima)(idx, df_unscaled, full_timestamps, input_len)
        for idx in tqdm(valid_test_indices, desc="ARIMA Testing")
    )
    arima_forecast_results = [r for r in arima_forecast_results if r is not None]
    arima_df = pd.DataFrame(arima_forecast_results)

    # Write performance aggregates text summary
    if not lstm_results_df.empty and len(arima_forecast_results) > 0:
        avg_mae_lstm = lstm_results_df["MAE"].mean()
        avg_rmse_lstm = lstm_results_df["RMSE"].mean()
        avg_mape_lstm = lstm_results_df["MAPE"].mean()
        avg_r2_lstm = lstm_results_df["R2"].mean()

        avg_mae_arima = arima_df["MAE"].mean()
        avg_rmse_arima = arima_df["RMSE"].mean()
        avg_mape_arima = arima_df["MAPE"].mean()

        performance_file = "model_performance.txt"
        with open(performance_file, "w") as f:
            f.write("📊 Average Performance on Full Test Set\n\n")
            f.write("🔹 LSTM:\n")
            f.write(f"   MAE: {avg_mae_lstm:.2f}\n")
            f.write(f"   RMSE: {avg_rmse_lstm:.2f}\n")
            f.write(f"   MAPE: {avg_mape_lstm:.2f}%\n")
            f.write(f"   R² Score: {avg_r2_lstm:.4f}\n\n")
            f.write("🔹 ARIMA:\n")
            f.write(f"   MAE: {avg_mae_arima:.2f}\n")
            f.write(f"   RMSE: {avg_rmse_arima:.2f}\n")
            f.write(f"   MAPE: {avg_mape_arima:.2f}%\n")
        print(f"Performance report saved to {performance_file}")

    # Weekend vs Weekday analysis
    print("Saving weekend vs weekday analysis comparison...")
    lstm_by_day = pd.DataFrame(lstm_forecast_results).assign(
        date=lambda x: pd.to_datetime(x["date"]),
        is_weekend=lambda x: x["date"].apply(lambda d: df_unscaled.loc[d.floor("D")]["Is_Weekend"].iloc[0] if isinstance(df_unscaled.loc[d.floor("D")]["Is_Weekend"], pd.Series) else df_unscaled.loc[d.floor("D")]["Is_Weekend"]),
    ).groupby("is_weekend").agg({"MAE": "mean", "MAPE": "mean"})

    arima_by_day = pd.DataFrame(arima_forecast_results).assign(
        date=lambda x: pd.to_datetime(x["date"]),
        is_weekend=lambda x: x["date"].apply(lambda d: df_unscaled.loc[d.floor("D")]["Is_Weekend"].iloc[0] if isinstance(df_unscaled.loc[d.floor("D")]["Is_Weekend"], pd.Series) else df_unscaled.loc[d.floor("D")]["Is_Weekend"]),
    ).groupby("is_weekend").agg({"MAE": "mean", "MAPE": "mean"})

    weekend_weekday_comparison = pd.DataFrame(index=[False, True])
    weekend_weekday_comparison["LSTM_MAE"] = lstm_by_day["MAE"]
    weekend_weekday_comparison["LSTM_MAPE"] = lstm_by_day["MAPE"]
    weekend_weekday_comparison["ARIMA_MAE"] = arima_by_day["MAE"]
    weekend_weekday_comparison["ARIMA_MAPE"] = arima_by_day["MAPE"]

    weekend_weekday_comparison = weekend_weekday_comparison.reset_index().rename(columns={"index": "Is_Weekend"})
    weekend_weekday_comparison["Day_Type"] = weekend_weekday_comparison["Is_Weekend"].map(
        {False: "Weekday", True: "Weekend"}
    )
    weekend_weekday_comparison = weekend_weekday_comparison[
        ["Day_Type", "LSTM_MAE", "LSTM_MAPE", "ARIMA_MAE", "ARIMA_MAPE"]
    ]

    weekend_weekday_comparison.to_csv("weekend_weekday_comparison.csv", index=False)
    print("Weekend vs Weekday comparison saved to weekend_weekday_comparison.csv")

    # 8. Create compressed results archive
    output_files = (
        ["daily_forecast_metrics.csv", "model_performance.txt", "loss_curve.png", "lstm_vs_arima_comparison.csv"]
        + [f"forecast_comparison_{date}.csv" for date in start_dates]
        + ["forecast_comparison_ieee.png", "weekend_weekday_comparison.csv"]
    )

    print("Compressing output files into forecast_results.zip...")
    with zipfile.ZipFile("forecast_results.zip", "w") as zipf:
        for file in output_files:
            if os.path.exists(file):
                zipf.write(file)
    print("📦 All result files compressed into forecast_results.zip")


if __name__ == "__main__":
    main()
