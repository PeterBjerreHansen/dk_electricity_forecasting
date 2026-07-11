from __future__ import annotations

import math
from collections.abc import Iterable

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


def interval_score(
    predictions: pd.DataFrame,
    *,
    alpha: float = 0.20,
    y_col: str = "y",
    lower_col: str = "q10",
    upper_col: str = "q90",
) -> float:
    """Return the mean central prediction-interval score.

    Lower values are better. The score rewards narrow intervals and adds a
    distance-weighted penalty when the observation falls outside the interval.
    ``alpha=0.20`` corresponds to the central q10--q90 interval.
    """

    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    require_columns(predictions, [y_col, lower_col, upper_col], "predictions")
    values = predictions[[y_col, lower_col, upper_col]].dropna()
    if values.empty:
        return math.nan
    _require_ordered_interval(values, lower_col=lower_col, upper_col=upper_col)

    width = values[upper_col] - values[lower_col]
    below_penalty = (2.0 / alpha) * (values[lower_col] - values[y_col]).clip(lower=0)
    above_penalty = (2.0 / alpha) * (values[y_col] - values[upper_col]).clip(lower=0)
    return float((width + below_penalty + above_penalty).mean())


def weighted_interval_score(
    predictions: pd.DataFrame,
    *,
    y_col: str = "y",
    lower_col: str = "q10",
    median_col: str = "q50",
    upper_col: str = "q90",
) -> float:
    """Return WIS for q10, q50, and q90 forecasts.

    This is the standard one-interval WIS: median pinball loss plus the
    alpha-weighted 80% interval score, normalized by ``1.5``. Lower values are
    better. Rows missing any required quantile are excluded as one unit.
    """

    columns = [y_col, lower_col, median_col, upper_col]
    require_columns(predictions, columns, "predictions")
    values = predictions[columns].dropna()
    if values.empty:
        return math.nan
    _require_ordered_quantiles(
        values,
        lower_col=lower_col,
        median_col=median_col,
        upper_col=upper_col,
    )

    median_pinball = pinball_loss(
        values,
        quantile=0.50,
        y_col=y_col,
        pred_col=median_col,
    )
    central_interval_score = interval_score(
        values,
        alpha=0.20,
        y_col=y_col,
        lower_col=lower_col,
        upper_col=upper_col,
    )
    return float((median_pinball + 0.10 * central_interval_score) / 1.5)


def quantile_calibration_error(
    predictions: pd.DataFrame,
    *,
    quantile: float,
    y_col: str = "y",
    pred_col: str | None = None,
) -> float:
    """Return signed empirical quantile coverage minus nominal coverage."""

    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    pred_col = pred_col or _default_quantile_column(quantile)
    require_columns(predictions, [y_col, pred_col], "predictions")
    values = predictions[[y_col, pred_col]].dropna()
    if values.empty:
        return math.nan
    empirical_coverage = (values[y_col] <= values[pred_col]).mean()
    return float(empirical_coverage - quantile)


def mean_absolute_calibration_error(
    predictions: pd.DataFrame,
    *,
    quantiles: Iterable[float] = (0.10, 0.50, 0.90),
    y_col: str = "y",
) -> float:
    """Return mean absolute calibration error across requested quantiles."""

    requested_quantiles = tuple(quantiles)
    if any(not 0 < quantile < 1 for quantile in requested_quantiles):
        raise ValueError("quantiles must be between 0 and 1")
    columns = [_default_quantile_column(quantile) for quantile in requested_quantiles]
    if any(column not in predictions.columns for column in columns):
        return math.nan
    require_columns(predictions, [y_col, *columns], "predictions")
    values = predictions[[y_col, *columns]].dropna()
    if values.empty:
        return math.nan

    errors = []
    for quantile, column in zip(requested_quantiles, columns):
        error = quantile_calibration_error(
            values,
            quantile=quantile,
            y_col=y_col,
            pred_col=column,
        )
        if math.isnan(error):
            return math.nan
        errors.append(abs(error))
    return float(sum(errors) / len(errors)) if errors else math.nan


def _default_quantile_column(quantile: float) -> str:
    return f"q{int(round(quantile * 100))}"


def _require_ordered_interval(
    values: pd.DataFrame,
    *,
    lower_col: str,
    upper_col: str,
) -> None:
    crossed = values[lower_col] > values[upper_col]
    if crossed.any():
        raise ValueError(f"Prediction intervals contain {int(crossed.sum())} crossed row(s)")


def _require_ordered_quantiles(
    values: pd.DataFrame,
    *,
    lower_col: str,
    median_col: str,
    upper_col: str,
) -> None:
    crossed = (values[lower_col] > values[median_col]) | (
        values[median_col] > values[upper_col]
    )
    if crossed.any():
        raise ValueError(f"Predictions contain {int(crossed.sum())} crossed quantile row(s)")
