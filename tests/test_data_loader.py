import os
import pytest
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler

from data_loader import LoadForecastDataset, prepare_data, DEFAULT_FEATURES
from model_architecture import LSTMModel


def test_dataset_shapes():
    """
    Tests that the LoadForecastDataset slides windows correctly and produces
    tensors with precise, expected dimensions.
    """
    # 100 timesteps, 6 features
    num_samples = 100
    num_features = 6
    dummy_data = np.random.rand(num_samples, num_features)

    input_len = 24
    output_len = 24

    dataset = LoadForecastDataset(dummy_data, input_len=input_len, output_len=output_len)

    # Expected dataset length is: num_samples - input_len - output_len
    expected_len = num_samples - input_len - output_len
    assert len(dataset) == expected_len, f"Dataset length should be {expected_len}"

    # Verify tensor shape of a single item
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor), "Input X must be a PyTorch Tensor"
    assert isinstance(y, torch.Tensor), "Target y must be a PyTorch Tensor"

    assert x.shape == (input_len, num_features), f"Expected X shape {(input_len, num_features)}, got {x.shape}"
    assert y.shape == (output_len,), f"Expected y shape {(output_len,)}, got {y.shape}"

    # Verify target y represents the first column (Load_Calgary) in the future window
    # Let's check that y matches the slice of the first feature (index 0)
    expected_y = dummy_data[input_len : input_len + output_len, 0]
    np.testing.assert_allclose(y.numpy(), expected_y, rtol=1e-5)


def test_prepare_data_pipeline(tmp_path):
    """
    Creates a temporary dummy CSV to test the complete prepare_data data-pipeline,
    validating the scaling, DataLoaders, and train/val/test splits.
    """
    # 1. Generate a mock dataframe with Datetime index
    dates = pd.date_range(start="2026-01-01", periods=100, freq="H")
    dummy_df = pd.DataFrame(
        {
            "Load_Calgary": np.random.randint(800, 1800, size=100),
            "Temperature_C": np.random.uniform(-30, 30, size=100),
            "Wind_Speed_mps": np.random.uniform(0, 20, size=100),
            "Is_Weekend": np.random.choice([0, 1], size=100),
            "Is_Holiday": np.random.choice([0, 1], size=100),
            "Day_of_Week_Num": np.random.randint(0, 7, size=100),
        },
        index=dates,
    )
    dummy_df.index.name = "Datetime"

    # Save to a temporary CSV path
    temp_csv = tmp_path / "mock_data.csv"
    dummy_df.to_csv(temp_csv)

    input_len = 24
    output_len = 24
    batch_size = 8

    # 2. Run preparation pipeline
    train_loader, val_loader, test_loader, scaler, df, df_scaled = prepare_data(
        csv_path=str(temp_csv),
        features=DEFAULT_FEATURES,
        input_len=input_len,
        output_len=output_len,
        batch_size=batch_size,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    # 3. Assert correct types and properties
    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)
    assert isinstance(test_loader, DataLoader)
    assert isinstance(scaler, MinMaxScaler)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df_scaled, pd.DataFrame)

    # Scaled values must lie in the range [0, 1]
    assert df_scaled.values.min() >= -1e-7
    assert df_scaled.values.max() <= 1.0 + 1e-7

    # Validate split allocations
    # Dataset total length is 100 - 24 - 24 = 52
    # Train split (70%) = int(52 * 0.70) = 36
    # Val split (15%) = int(52 * 0.15) = 7
    # Test split = 52 - 36 - 7 = 9
    assert len(train_loader.dataset) == 36
    assert len(val_loader.dataset) == 7
    assert len(test_loader.dataset) == 9

    # Verify a batch shape from dataloader
    batch_x, batch_y = next(iter(train_loader))
    assert batch_x.shape == (batch_size, input_len, len(DEFAULT_FEATURES))
    assert batch_y.shape == (batch_size, output_len)


def test_model_dimensions():
    """
    Validates that the PyTorch LSTMModel processes batched feature inputs
    and propagates outputs of the correct forecasting dimension cleanly.
    """
    batch_size = 16
    input_len = 24
    num_features = len(DEFAULT_FEATURES)
    output_len = 24

    model = LSTMModel(input_size=num_features, hidden_size=32, num_layers=2, output_size=output_len)

    # Dummy batched inputs: [batch_size, sequence_length, features]
    dummy_input = torch.randn(batch_size, input_len, num_features)

    # Forward propagation
    output = model(dummy_input)

    # Output shape should be [batch_size, output_size]
    assert output.shape == (batch_size, output_len), f"Expected shape {(batch_size, output_len)}, got {output.shape}"
