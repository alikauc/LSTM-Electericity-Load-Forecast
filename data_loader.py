import os
from typing import Tuple, List
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
    features: List[str] = None,
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
    df = df[features]

    # Preprocessing scaling
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=features, index=df.index)

    # Initialize sliding window dataset
    dataset = LoadForecastDataset(df_scaled.values, input_len, output_len)

    # Calculate partition indices
    total = len(dataset)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)

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
