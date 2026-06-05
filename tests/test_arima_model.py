"""
Tests for ARIMA model input validation.
Validates that invalid inputs are properly rejected with clear error messages.
"""

import pytest
import pandas as pd
import numpy as np

from arima_model import _fit_arima, forecast_day_arima


class TestFitArimaValidation:
    """Tests for _fit_arima input validation."""

    def test_invalid_order_length(self):
        """ARIMA order must be a 3-tuple."""
        history = pd.Series(np.random.rand(100))
        with pytest.raises(ValueError, match="must be a 3-tuple"):
            _fit_arima(history, order=(1, 1))

    def test_negative_order_values(self):
        """ARIMA order values must be non-negative."""
        history = pd.Series(np.random.rand(100))
        with pytest.raises(ValueError, match="must be non-negative"):
            _fit_arima(history, order=(-1, 1, 1))

    def test_negative_fallback_order(self):
        """Fallback order values must also be non-negative."""
        history = pd.Series(np.random.rand(100))
        with pytest.raises(ValueError, match="must be non-negative"):
            _fit_arima(history, order=(1, 1, 1), fallback_order=(1, -1, 1))

    def test_empty_history_raises(self):
        """Fitting ARIMA on empty series must raise ValueError."""
        empty_series = pd.Series([], dtype=float)
        with pytest.raises(ValueError, match="empty history"):
            _fit_arima(empty_series)

    def test_valid_order_succeeds(self):
        """Valid ARIMA fit should return without error."""
        np.random.seed(42)
        history = pd.Series(np.cumsum(np.random.randn(200)))
        result = _fit_arima(history, order=(1, 0, 0), fallback_order=(0, 0, 1))
        assert result is not None


class TestForecastDayArimaValidation:
    """Tests for forecast_day_arima input validation."""

    def test_negative_history_hours(self):
        """history_hours must be positive."""
        dummy_df = pd.DataFrame(
            {"Load_Calgary": np.random.rand(500)},
            index=pd.date_range("2024-01-01", periods=500, freq="h"),
        )
        with pytest.raises(ValueError, match="must be positive"):
            forecast_day_arima(dummy_df, "2024-01-15", history_hours=-10)

    def test_zero_history_hours(self):
        """history_hours=0 must raise ValueError."""
        dummy_df = pd.DataFrame(
            {"Load_Calgary": np.random.rand(500)},
            index=pd.date_range("2024-01-01", periods=500, freq="h"),
        )
        with pytest.raises(ValueError, match="must be positive"):
            forecast_day_arima(dummy_df, "2024-01-15", history_hours=0)
