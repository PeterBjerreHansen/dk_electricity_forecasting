from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pandas as pd

from dkenergy_forecast.evaluation.point_metrics import mae
from dkenergy_forecast.evaluation.probabilistic_metrics import (
    mean_absolute_calibration_error,
    weighted_interval_score,
)
from dkenergy_forecast.types import (
    COPENHAGEN_TZ,
    TARGET_REGIME_BOUNDARY_LOCAL,
    require_columns,
)


MARKET_REGIME_BOUNDARY_LOCAL = TARGET_REGIME_BOUNDARY_LOCAL

DEFAULT_STRATA_COLUMNS: dict[str, str] = {
    "month": "evaluation_month",
    "area": "area",
    "hour": "evaluation_hour",
    "dst": "evaluation_dst",
    "negative_price": "evaluation_price_sign",
    "extreme_price": "evaluation_extreme_price",
    "market_regime": "market_regime",
}


def prepare_evaluation_strata(
    predictions: pd.DataFrame,
    *,
    y_col: str = "y",
    extreme_threshold: float | None = None,
    extreme_quantile: float = 0.95,
) -> tuple[pd.DataFrame, float]:
    """Add stable, human-readable subgroup columns for model diagnostics."""

    require_columns(predictions, ["ds_utc", "area", y_col], "predictions")
    if not 0 < extreme_quantile < 1:
        raise ValueError("extreme_quantile must be between 0 and 1")

    output = predictions.copy()
    output["ds_utc"] = pd.to_datetime(output["ds_utc"], utc=True)
    local = output["ds_utc"].dt.tz_convert(COPENHAGEN_TZ)
    output["evaluation_month"] = local.dt.strftime("%Y-%m")
    output["evaluation_hour"] = local.dt.strftime("%H")

    is_dst = local.map(lambda timestamp: bool(timestamp.dst()))
    output["evaluation_dst"] = is_dst.map(
        {True: "dst", False: "standard_time"}
    )

    actuals = pd.to_numeric(output[y_col], errors="coerce")
    if actuals.isna().any():
        raise ValueError("predictions contain missing or non-numeric actual target values")
    output["evaluation_price_sign"] = actuals.lt(0).map(
        {True: "negative", False: "non_negative"}
    )

    threshold = _resolve_extreme_threshold(
        actuals,
        extreme_threshold=extreme_threshold,
        extreme_quantile=extreme_quantile,
    )
    output["evaluation_extreme_price"] = actuals.abs().ge(threshold).map(
        {True: "extreme", False: "typical"}
    )
    output["market_regime"] = _market_regime(output, local)
    return output, threshold


def stratified_score_table(
    predictions: pd.DataFrame,
    *,
    strata: Mapping[str, str] = DEFAULT_STRATA_COLUMNS,
    model_label_col: str = "model_label",
    extreme_threshold: float | None = None,
    extreme_quantile: float = 0.95,
) -> pd.DataFrame:
    """Return long-form model scores for each evaluation subgroup."""

    require_columns(predictions, [model_label_col, "forecast_origin_utc"], "predictions")
    prepared, threshold = prepare_evaluation_strata(
        predictions,
        extreme_threshold=extreme_threshold,
        extreme_quantile=extreme_quantile,
    )
    missing = [column for column in strata.values() if column not in prepared.columns]
    if missing:
        raise ValueError(f"Unknown stratification columns: {missing}")

    rows: list[dict[str, Any]] = []
    for stratum, column in strata.items():
        grouped = prepared.groupby([column, model_label_col], dropna=False, sort=True)
        for (value, model_label), frame in grouped:
            rows.append(
                {
                    "stratum": stratum,
                    "stratum_value": str(value),
                    "model_label": model_label,
                    "rows": int(len(frame)),
                    "origin_count": int(frame["forecast_origin_utc"].nunique()),
                    "mae": mae(frame),
                    "weighted_interval_score": _wis_or_nan(frame),
                    "calibration_error": _calibration_or_nan(frame),
                    "extreme_price_threshold": threshold,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["stratum", "stratum_value", "model_label"]
    ).reset_index(drop=True)


def _resolve_extreme_threshold(
    actuals: pd.Series,
    *,
    extreme_threshold: float | None,
    extreme_quantile: float,
) -> float:
    if extreme_threshold is not None:
        threshold = float(extreme_threshold)
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("extreme_threshold must be a finite non-negative number")
        return threshold
    return float(actuals.dropna().abs().quantile(extreme_quantile))


def _market_regime(frame: pd.DataFrame, local: pd.Series) -> pd.Series:
    derived = pd.Series(
        "native_hourly",
        index=frame.index,
        dtype="object",
    )
    derived.loc[local.ge(MARKET_REGIME_BOUNDARY_LOCAL)] = (
        "quarter_hour_aggregated_to_hourly"
    )
    if "source_resolution_minutes" in frame.columns:
        resolution = pd.to_numeric(frame["source_resolution_minutes"], errors="coerce")
        derived.loc[resolution.le(15)] = "quarter_hour_aggregated_to_hourly"
        derived.loc[resolution.gt(15)] = "native_hourly"
    if "market_regime" in frame.columns:
        provided = frame["market_regime"].astype("string")
        derived.loc[provided.notna()] = provided.loc[provided.notna()].astype(str)
    return derived


def _wis_or_nan(frame: pd.DataFrame) -> float:
    if not {"y", "q10", "q50", "q90"}.issubset(frame.columns):
        return math.nan
    return weighted_interval_score(frame)


def _calibration_or_nan(frame: pd.DataFrame) -> float:
    if not {"y", "q10", "q50", "q90"}.issubset(frame.columns):
        return math.nan
    return mean_absolute_calibration_error(frame)
