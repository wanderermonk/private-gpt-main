"""Lightweight, offline forecasting helpers (numpy only).

Deliberately dependency-light: no statsmodels / prophet / sklearn, so this
adds nothing that could phone home and runs on an air-gapped box. Two classic
methods are implemented by hand:

  * linear  - ordinary least-squares trend line.
  * holt    - Holt's linear (double exponential smoothing): level + trend,
              good when the series trends but isn't purely straight.

``choose_and_forecast`` picks between them using one-step backtest error, so
the caller doesn't have to know which fits better.

These functions operate on plain numbers. Turning messy document text into a
clean series is the job of the LLM extraction step in predict_service.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ForecastResult:
    method: str
    history: list[float]
    forecast: list[float]
    # Simple quality signals (not guarantees):
    fit_error: float  # backtest MAPE-like error, lower is better
    note: str = ""


def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean absolute percentage error, robust to zeros (falls back to MAE
    scale when actuals are ~0)."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    denom = np.where(np.abs(actual) < 1e-9, 1.0, np.abs(actual))
    return float(np.mean(np.abs(actual - predicted) / denom))


def linear_forecast(history: list[float], horizon: int) -> tuple[list[float], float]:
    """Least-squares straight line extrapolated `horizon` steps ahead."""
    y = np.asarray(history, dtype=float)
    n = len(y)
    x = np.arange(n, dtype=float)
    # slope/intercept via polyfit degree 1
    slope, intercept = np.polyfit(x, y, 1)
    future_x = np.arange(n, n + horizon, dtype=float)
    forecast = slope * future_x + intercept
    # in-sample fit error
    fitted = slope * x + intercept
    return forecast.tolist(), _mape(y, fitted)


def holt_forecast(
    history: list[float],
    horizon: int,
    alpha: float = 0.5,
    beta: float = 0.3,
) -> tuple[list[float], float]:
    """Holt's linear trend method (double exponential smoothing)."""
    y = np.asarray(history, dtype=float)
    n = len(y)
    if n < 2:
        # Not enough to estimate a trend; flat-line the last value.
        last = float(y[-1]) if n else 0.0
        return [last] * horizon, 0.0

    level = y[0]
    trend = y[1] - y[0]
    fitted = [level + trend]
    for t in range(1, n):
        prev_level = level
        level = alpha * y[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev_level) + (1 - beta) * trend
        fitted.append(level + trend)
    # Forecast: last level + h * trend
    forecast = [float(level + (h + 1) * trend) for h in range(horizon)]
    # Backtest error on the one-step-ahead fitted values (skip the first).
    err = _mape(y[1:], np.asarray(fitted[:-1]))
    return forecast, err


def choose_and_forecast(history: list[float], horizon: int = 3) -> ForecastResult:
    """Run both methods and return whichever backtests better.

    Raises ValueError if there isn't enough data to forecast.
    """
    clean = [float(v) for v in history if v is not None and np.isfinite(float(v))]
    if len(clean) < 2:
        raise ValueError(
            "Need at least 2 numeric data points to forecast; "
            f"got {len(clean)}."
        )
    horizon = max(1, int(horizon))

    lin_fc, lin_err = linear_forecast(clean, horizon)
    holt_fc, holt_err = holt_forecast(clean, horizon)

    if holt_err <= lin_err:
        method, forecast, err = "holt", holt_fc, holt_err
    else:
        method, forecast, err = "linear", lin_fc, lin_err

    note = ""
    if len(clean) < 4:
        note = (
            "Very short history (<4 points): treat this projection as a rough "
            "directional estimate, not a precise forecast."
        )
    return ForecastResult(
        method=method,
        history=clean,
        forecast=[round(v, 4) for v in forecast],
        fit_error=round(err, 4),
        note=note,
    )
