from __future__ import annotations

import math
from typing import Any

import pandas as pd

from dkenergy_forecast.evaluation.point_metrics import bias, mae, rmse
from dkenergy_forecast.evaluation.probabilistic_metrics import (
    average_interval_width,
    interval_score,
    interval_coverage,
    mean_absolute_calibration_error,
    pinball_loss,
    weighted_interval_score,
)
from dkenergy_forecast.types import require_columns


def add_prediction_diagnostics(
    predictions: pd.DataFrame,
    *,
    y_col: str = "y",
    pred_col: str = "y_pred",
) -> pd.DataFrame:
    """Add reusable point-error columns when actuals are available."""

    output = predictions.copy()
    require_columns(output, [pred_col], "predictions")
    if y_col in output.columns:
        output["error"] = output[pred_col] - output[y_col]
        output["abs_error"] = output["error"].abs()
        output["squared_error"] = output["error"] ** 2
    return output


def model_score_table(
    predictions: pd.DataFrame,
    *,
    model_label_col: str = "model_label",
    include_all_area: bool = True,
) -> pd.DataFrame:
    """Build a model score table for production and experiment diagnostics."""

    require_columns(
        predictions,
        [model_label_col, "area", "y", "y_pred"],
        "predictions",
    )
    rows: list[dict[str, Any]] = []
    identity_columns = [model_label_col]
    if "model_release_id" in predictions.columns:
        identity_columns.append("model_release_id")

    if include_all_area:
        groups = [
            (("ALL", *_as_tuple(identity)), frame)
            for identity, frame in predictions.groupby(identity_columns, dropna=False)
        ]
    else:
        groups = []
    groups.extend(
        [
            (_as_tuple(identity), frame)
            for identity, frame in predictions.groupby(
                ["area", *identity_columns],
                dropna=False,
            )
        ]
    )

    for key, frame in groups:
        area, model_label, *release_values = key
        identity = {"model_label": model_label}
        if release_values:
            identity["model_release_id"] = release_values[0]
        rows.append(
            {
                **identity,
                "area": area,
                "rows": int(len(frame)),
                "evaluated_rows": int(frame[["y", "y_pred"]].dropna().shape[0]),
                "mae": mae(frame),
                "rmse": rmse(frame),
                "bias": bias(frame),
                "coverage": _coverage_or_nan(frame),
                "interval_width": _interval_width_or_nan(frame),
                "pinball_q10": _pinball_or_nan(frame, quantile=0.10),
                "pinball_q50": _pinball_or_nan(frame, quantile=0.50),
                "pinball_q90": _pinball_or_nan(frame, quantile=0.90),
                "interval_score_80": _interval_score_or_nan(frame),
                "weighted_interval_score": _wis_or_nan(frame),
                "calibration_error": _calibration_error_or_nan(frame),
                "missing_rate": float(frame["y_pred"].isna().mean()),
            }
        )

    sort_columns = ["area", "mae", "model_label"]
    if "model_release_id" in predictions.columns:
        sort_columns.append("model_release_id")
    return pd.DataFrame(rows).sort_values(sort_columns).reset_index(drop=True)


def probabilistic_metric_table(
    predictions: pd.DataFrame,
    *,
    model_label_col: str = "model_label",
) -> pd.DataFrame:
    """Build long-form probabilistic metrics grouped by model label."""

    require_columns(predictions, [model_label_col], "predictions")
    rows: list[dict[str, Any]] = []
    identity_columns = [model_label_col]
    if "model_release_id" in predictions.columns:
        identity_columns.append("model_release_id")
    for group_key, frame in predictions.groupby(identity_columns, dropna=False):
        model_label, *release_values = _as_tuple(group_key)
        identity = {"model_label": model_label}
        if release_values:
            identity["model_release_id"] = release_values[0]
        rows.extend(
            [
                {
                    **identity,
                    "metric": "pinball_q10",
                    "value": _pinball_or_nan(frame, quantile=0.10),
                },
                {
                    **identity,
                    "metric": "pinball_q50",
                    "value": _pinball_or_nan(frame, quantile=0.50),
                },
                {
                    **identity,
                    "metric": "pinball_q90",
                    "value": _pinball_or_nan(frame, quantile=0.90),
                },
                {
                    **identity,
                    "metric": "p10_p90_coverage",
                    "value": _coverage_or_nan(frame),
                },
                {
                    **identity,
                    "metric": "p10_p90_avg_width",
                    "value": _interval_width_or_nan(frame),
                },
            ]
        )
    return pd.DataFrame(rows)


def _as_tuple(value: object) -> tuple[Any, ...]:
    return value if isinstance(value, tuple) else (value,)


def _coverage_or_nan(frame: pd.DataFrame) -> float:
    if {"y", "q10", "q90"}.issubset(frame.columns):
        return interval_coverage(frame)
    return math.nan


def _interval_width_or_nan(frame: pd.DataFrame) -> float:
    if {"q10", "q90"}.issubset(frame.columns):
        return average_interval_width(frame)
    return math.nan


def _pinball_or_nan(frame: pd.DataFrame, *, quantile: float) -> float:
    column = f"q{int(round(quantile * 100))}"
    if {"y", column}.issubset(frame.columns):
        return pinball_loss(frame, quantile=quantile, pred_col=column)
    return math.nan


def _interval_score_or_nan(frame: pd.DataFrame) -> float:
    if {"y", "q10", "q90"}.issubset(frame.columns):
        return interval_score(frame)
    return math.nan


def _wis_or_nan(frame: pd.DataFrame) -> float:
    if {"y", "q10", "q50", "q90"}.issubset(frame.columns):
        return weighted_interval_score(frame)
    return math.nan


def _calibration_error_or_nan(frame: pd.DataFrame) -> float:
    if {"y", "q10", "q50", "q90"}.issubset(frame.columns):
        return mean_absolute_calibration_error(frame)
    return math.nan
