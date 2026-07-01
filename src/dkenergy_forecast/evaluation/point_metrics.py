from __future__ import annotations

import math

import pandas as pd

from dkenergy_forecast.types import require_columns


def mae(predictions: pd.DataFrame, *, y_col: str = "y", pred_col: str = "y_pred") -> float:
    values = _valid_pairs(predictions, y_col, pred_col)
    if values.empty:
        return math.nan
    return float((values[pred_col] - values[y_col]).abs().mean())


def rmse(predictions: pd.DataFrame, *, y_col: str = "y", pred_col: str = "y_pred") -> float:
    values = _valid_pairs(predictions, y_col, pred_col)
    if values.empty:
        return math.nan
    return float(((values[pred_col] - values[y_col]) ** 2).mean() ** 0.5)


def bias(predictions: pd.DataFrame, *, y_col: str = "y", pred_col: str = "y_pred") -> float:
    values = _valid_pairs(predictions, y_col, pred_col)
    if values.empty:
        return math.nan
    return float((values[pred_col] - values[y_col]).mean())


def _valid_pairs(predictions: pd.DataFrame, y_col: str, pred_col: str) -> pd.DataFrame:
    require_columns(predictions, [y_col, pred_col], "predictions")
    return predictions[[y_col, pred_col]].dropna()
