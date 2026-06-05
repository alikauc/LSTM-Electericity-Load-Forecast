import os
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import MinMaxScaler

# Centralized features specification
DEFAULT_FEATURES = ["Load_Calgary", "Temperature_C", "Wind_Speed_mps", "Is_Weekend", "Is_Holiday", "Day_of_Week_Num"]


class LoadForecastDataset(Dataset):
    """
    A custom PyTorch Dataset for electricity load forecasting.
    Uses sliding windows to extract sequences of features and corresponding targets.
    """
    def __init__(self, data: np.ndarray, input_len: int = 24, output_len: int = 24):
        """
        Args:
            data (np.ndarray): Scaled numerical values, shape (num_samples, num_features).
                               The target feature (Load_Calgary) must be at index 0.
            input_len (int): Sequence length of the input historical context (hours).
            output_len (int): Length of the forecast horizon to predict (hours).
        """
        self.X: List[np.ndarray] = []
        self.y: List[np.ndarray] = []

        # Sliding window extraction
        for i in range(len(data) - input_len - output_len):
            x_window = data[i : i + input_len]
            # Target is the electricity load (index 0) over the future forecast horizon
            y_window = data[i + input_len : i + input_len + output_len, 0]
            self.X.append(x_window)
            self.y.append(y_window)

        # Convert to float32 tensors
        self.X_tensor = torch.tensor(np.array(self.X), dtype=torch.float32)
        self.y_tensor = torch.tensor(np.array(self.y), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X_tensor)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X_tensor[idx], self.y_tensor[idx]


def prepare_data(
    csv_path: str,
    features: Optional[List[str]] = None,
    input_len: int = 24,
    output_len: int = 24,
    batch_size: int = 64,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[DataLoader, DataLoader, DataLoader, MinMaxScaler, pd.DataFrame, pd.DataFrame]:
    """
    Loads raw CSV data, filters features, fits MinMaxScaler, defines LoadForecastDataset,
    and returns PyTorch DataLoaders split into train, validation, and test subsets.

    Args:
        csv_path (str): Path to the complete CSV dataset.
        features (List[str]): List of features to include. Default is DEFAULT_FEATURES.
        input_len (int): Input sequence length (hours). Default is 24.
        output_len (int): Prediction horizon length (hours). Default is 24.
        batch_size (int): Batch size for loaders. Default is 64.
        train_ratio (float): Fraction of dataset used for training. Default is 0.70.
        val_ratio (float): Fraction of dataset used for validation. Default is 0.15.

    Returns:
        Tuple containing:
            - train_loader (DataLoader)
            - val_loader (DataLoader)
            - test_loader (DataLoader)
            - scaler (MinMaxScaler)
            - df (pd.DataFrame): Unscaled parsed dataframe
            - df_scaled (pd.DataFrame): Scaled dataframe
    """
    if features is None:
        features = DEFAULT_FEATURES

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset CSV not found at {csv_path}")

    # Load data, parsing 'Datetime' as time index
    df = pd.read_csv(csv_path, parse_dates=["Datetime"], index_col="Datetime")

    # Ensure chronological order — unsorted CSVs would corrupt sliding windows
    df = df.sort_index()

    # Validate requested features exist in the dataset
    missing_cols = set(features) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Features not found in CSV columns: {missing_cols}. "
            f"Available columns: {list(df.columns)}"
        )
    df = df[features]

    # Guard against NaN contamination — NaNs propagate through scaler and cause NaN loss
    nan_counts = df.isna().sum()
    if nan_counts.any():
        raise ValueError(
            f"Dataset contains NaN values that would corrupt training:\n{nan_counts[nan_counts > 0]}"
        )

    # Guard against insufficient data length
    min_rows = input_len + output_len + 1  # At least 1 sliding window
    if len(df) < min_rows:
        raise ValueError(
            f"Dataset has {len(df)} rows but needs at least {min_rows} "
            f"(input_len={input_len} + output_len={output_len} + 1) to create a single window."
        )

    # Calculate partition boundaries BEFORE scaling to prevent data leakage.
    # The scaler must be fitted on training data only — never on val/test rows.
    total_windows = len(df) - input_len - output_len
    train_size = int(total_windows * train_ratio)
    val_size = int(total_windows * val_ratio)

    # The last training window (index train_size-1) accesses raw rows up to
    # (train_size - 1) + input_len + output_len - 1 = train_size + input_len + output_len - 2.
    # df.iloc[:end] is exclusive, so end = that index + 1.
    train_row_end = train_size + input_len + output_len - 1

    # Fit scaler on TRAINING rows only, then transform the full dataset
    scaler = MinMaxScaler()
    scaler.fit(df.iloc[:train_row_end])
    df_scaled = pd.DataFrame(scaler.transform(df), columns=features, index=df.index)

    # Initialize sliding window dataset
    dataset = LoadForecastDataset(df_scaled.values, input_len, output_len)

    # Verify partition math is consistent
    total = len(dataset)
    if train_size + val_size > total:
        raise ValueError(
            f"Split sizes ({train_size} + {val_size} = {train_size + val_size}) exceed dataset length ({total})"
        )

    # Define subsets chronologically to preserve temporal order in validation and testing
    train_set = Subset(dataset, list(range(0, train_size)))
    val_set = Subset(dataset, list(range(train_size, train_size + val_size)))
    test_set = Subset(dataset, list(range(train_size + val_size, total)))

    # Create PyTorch DataLoaders
    # Shuffle only training set to assist generalization; keep validation/test ordered for chronological plotting
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, scaler, df, df_scaled
