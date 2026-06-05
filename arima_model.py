import logging
import math
from datetime import timedelta
from typing import Tuple, Dict, Any

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA

# Default ARIMA configuration
logger = logging.getLogger(__name__)
DEFAULT_ORDER = (2, 1, 2)
FALLBACK_ORDER = (1, 1, 1)
HISTORY_HOURS = 168  # 7 days of hourly history
FORECAST_HORIZON = 24  # 24-hour ahead forecast


def _fit_arima(history: pd.Series, order: Tuple[int, int, int] = DEFAULT_ORDER,
               fallback_order: Tuple[int, int, int] = FALLBACK_ORDER) -> Any:
    """
    Fits an ARIMA model with relaxed constraints and automatic fallback.

    Attempts to fit the primary ARIMA order first. If the solver encounters
    numerical instability (e.g., LU decomposition errors from near-singular
    covariance matrices), it falls back to a simpler model order.

    Args:
        history (pd.Series): Historical time-series data to fit.
        order (Tuple[int, int, int]): Primary ARIMA(p, d, q) order.
        fallback_order (Tuple[int, int, int]): Fallback ARIMA(p, d, q) order.

    Returns:
        Fitted ARIMA model result object.

    Raises:
        ValueError: If order tuples contain negative values or history is empty.
    """
    # Validate inputs
    for name, o in [("order", order), ("fallback_order", fallback_order)]:
        if len(o) != 3:
            raise ValueError(f"{name} must be a 3-tuple (p, d, q), got {o}")
        if any(v < 0 for v in o):
            raise ValueError(f"{name} values must be non-negative, got {o}")

    if len(history) == 0:
        raise ValueError("Cannot fit ARIMA on empty history series")

    try:
        model = ARIMA(history, order=order,
                      enforce_stationarity=False, enforce_invertibility=False)
        return model.fit()
    except Exception:
        logger.debug("Primary ARIMA%s failed, falling back to %s", order, fallback_order)
        model = ARIMA(history, order=fallback_order,
                      enforce_stationarity=False, enforce_invertibility=False)
        return model.fit()


def forecast_day_arima(
    df_unscaled: pd.DataFrame,
    target_date: str,
    order: Tuple[int, int, int] = DEFAULT_ORDER,
    fallback_order: Tuple[int, int, int] = FALLBACK_ORDER,
    history_hours: int = HISTORY_HOURS
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Forecasting helper for a specific day using a classical ARIMA model.

    Uses history_hours (default 168 hours) of historical data to fit an ARIMA model,
    then forecasts the next 24 hours of electricity load.

    Args:
        df_unscaled (pd.DataFrame): Original unscaled dataframe with datetime index.
        target_date (str): Timestamp/date string starting the 24h prediction window.
        order (Tuple[int, int, int]): Primary ARIMA order to fit.
        fallback_order (Tuple[int, int, int]): Fallback ARIMA order if primary fails.
        history_hours (int): Number of hours of history to fit ARIMA on.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Predicted electricity load (24h) and actual load (24h).

    Raises:
        ValueError: If history_hours is not positive.
    """
    if history_hours <= 0:
        raise ValueError(f"history_hours must be positive, got {history_hours}")

    forecast_start = pd.Timestamp(target_date)
    history_start = forecast_start - timedelta(hours=history_hours)
    history_end = forecast_start - timedelta(hours=1)
    target_end = forecast_start + timedelta(hours=FORECAST_HORIZON - 1)

    history = df_unscaled.loc[history_start:history_end, "Load_Calgary"]

    model_fit = _fit_arima(history, order=order, fallback_order=fallback_order)
    forecast = model_fit.forecast(steps=FORECAST_HORIZON)

    y_true = df_unscaled.loc[forecast_start:target_end, "Load_Calgary"].values
    return forecast.values, y_true


def process_single_forecast_arima(
    idx: int,
    df_unscaled: pd.DataFrame,
    full_timestamps: pd.DatetimeIndex,
    input_len: int,
    order: Tuple[int, int, int] = DEFAULT_ORDER,
    fallback_order: Tuple[int, int, int] = FALLBACK_ORDER,
    history_hours: int = HISTORY_HOURS
) -> Dict[str, Any]:
    """
    Thread-safe ARIMA execution helper for parallel evaluation on the test set.

    Designed to be called via joblib.Parallel for efficient batch evaluation
    across all test set sequences.

    Args:
        idx (int): Index into the dataset for the current evaluation window.
        df_unscaled (pd.DataFrame): Original unscaled dataframe with datetime index.
        full_timestamps (pd.DatetimeIndex): Complete datetime index of the dataset.
        input_len (int): Length of the input context window (hours).
        order (Tuple[int, int, int]): Primary ARIMA order to fit.
        fallback_order (Tuple[int, int, int]): Fallback ARIMA order if primary fails.
        history_hours (int): Number of hours of history to fit ARIMA on.

    Returns:
        Dict[str, Any]: Dictionary with date, MAE, RMSE, and MAPE metrics,
                        or None if insufficient actual data is available.
    """
    forecast_start = full_timestamps[idx + input_len]
    forecast_date = forecast_start.strftime("%Y-%m-%d")
    try:
        history = df_unscaled.loc[
            forecast_start - pd.Timedelta(hours=history_hours) : forecast_start - pd.Timedelta(hours=1),
            "Load_Calgary"
        ]

        model_fit = _fit_arima(history, order=order, fallback_order=fallback_order)
        forecast = model_fit.forecast(steps=FORECAST_HORIZON)

        actual_day_data = df_unscaled.loc[
            forecast_start : forecast_start + pd.Timedelta(hours=FORECAST_HORIZON - 1), "Load_Calgary"
        ]
        if len(actual_day_data) < FORECAST_HORIZON:
            return None

        y_true = actual_day_data.values
        y_pred = forecast.values

        mae = mean_absolute_error(y_true, y_pred)
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mape = mean_absolute_percentage_error(y_true, y_pred) * 100

        return {"date": forecast_date, "MAE": mae, "RMSE": rmse, "MAPE": mape}
    except Exception as e:
        logger.warning("ARIMA skipped %s due to error: %s", forecast_date, e)
        return None
