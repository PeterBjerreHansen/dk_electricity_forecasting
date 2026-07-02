from __future__ import annotations

import math
from typing import Any

import pandas as pd

from dkenergy_forecast.evaluation.point_metrics import bias, mae, rmse
from dkenergy_forecast.evaluation.probabilistic_metrics import (
    average_interval_width,
    interval_coverage,
    pinball_loss,
)
from dkenergy_forecast.evaluation.value_metrics import cheapest_k_hit_rate
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
    """Build the production-facing model score table."""

    require_columns(
        predictions,
        [model_label_col, "area", "y", "y_pred"],
        "predictions",
    )
    rows: list[dict[str, Any]] = []

    if include_all_area:
        groups = [
            (("ALL", model_label), frame)
            for model_label, frame in predictions.groupby(model_label_col, dropna=False)
        ]
    else:
        groups = []
    groups.extend(
        [
            ((area, model_label), frame)
            for (area, model_label), frame in predictions.groupby(
                ["area", model_label_col],
                dropna=False,
            )
        ]
    )

    for (area, model_label), frame in groups:
        rows.append(
            {
                "model_label": model_label,
                "area": area,
                "rows": int(len(frame)),
                "evaluated_rows": int(frame[["y", "y_pred"]].dropna().shape[0]),
                "mae": mae(frame),
                "rmse": rmse(frame),
                "bias": bias(frame),
                "coverage": _coverage_or_nan(frame),
                "interval_width": _interval_width_or_nan(frame),
                "missing_rate": float(frame["y_pred"].isna().mean()),
            }
        )

    return pd.DataFrame(rows).sort_values(["area", "mae", "model_label"]).reset_index(drop=True)


def probabilistic_metric_table(
    predictions: pd.DataFrame,
    *,
    model_label_col: str = "model_label",
) -> pd.DataFrame:
    """Build long-form probabilistic metrics grouped by model label."""

    require_columns(predictions, [model_label_col], "predictions")
    rows: list[dict[str, Any]] = []
    for model_label, frame in predictions.groupby(model_label_col, dropna=False):
        rows.extend(
            [
                {
                    "model_label": model_label,
                    "metric": "pinball_q10",
                    "value": _pinball_or_nan(frame, quantile=0.10),
                },
                {
                    "model_label": model_label,
                    "metric": "pinball_q50",
                    "value": _pinball_or_nan(frame, quantile=0.50),
                },
                {
                    "model_label": model_label,
                    "metric": "pinball_q90",
                    "value": _pinball_or_nan(frame, quantile=0.90),
                },
                {
                    "model_label": model_label,
                    "metric": "p10_p90_coverage",
                    "value": _coverage_or_nan(frame),
                },
                {
                    "model_label": model_label,
                    "metric": "p10_p90_avg_width",
                    "value": _interval_width_or_nan(frame),
                },
            ]
        )
    return pd.DataFrame(rows)


def cheapest_k_table(predictions: pd.DataFrame, *, k: int) -> pd.DataFrame:
    frames = []
    require_columns(predictions, ["model_label"], "predictions")
    for model_label, frame in predictions.groupby("model_label", dropna=False):
        values = cheapest_k_hit_rate(frame, k=k)
        values["model_label"] = model_label
        frames.append(values)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


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
