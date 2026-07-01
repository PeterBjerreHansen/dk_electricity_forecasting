from __future__ import annotations

import math

import pandas as pd

from dkenergy_forecast.types import require_columns


def pinball_loss(
    predictions: pd.DataFrame,
    *,
    quantile: float,
    y_col: str = "y",
    pred_col: str | None = None,
) -> float:
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    pred_col = pred_col or _default_quantile_column(quantile)
    require_columns(predictions, [y_col, pred_col], "predictions")
    values = predictions[[y_col, pred_col]].dropna()
    if values.empty:
        return math.nan

    error = values[y_col] - values[pred_col]
    loss = error.map(lambda value: max(quantile * value, (quantile - 1) * value))
    return float(loss.mean())


def interval_coverage(
    predictions: pd.DataFrame,
    *,
    y_col: str = "y",
    lower_col: str = "q10",
    upper_col: str = "q90",
) -> float:
    require_columns(predictions, [y_col, lower_col, upper_col], "predictions")
    values = predictions[[y_col, lower_col, upper_col]].dropna()
    if values.empty:
        return math.nan
    covered = (values[y_col] >= values[lower_col]) & (values[y_col] <= values[upper_col])
    return float(covered.mean())


def average_interval_width(
    predictions: pd.DataFrame,
    *,
    lower_col: str = "q10",
    upper_col: str = "q90",
) -> float:
    require_columns(predictions, [lower_col, upper_col], "predictions")
    values = predictions[[lower_col, upper_col]].dropna()
    if values.empty:
        return math.nan
    return float((values[upper_col] - values[lower_col]).mean())


def _default_quantile_column(quantile: float) -> str:
    return f"q{int(round(quantile * 100))}"
