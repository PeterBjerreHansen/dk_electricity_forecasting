from __future__ import annotations

import math
import random
from collections.abc import Iterable
from typing import Any

import pandas as pd

from dkenergy_forecast.evaluation.point_metrics import bias, mae, rmse
from dkenergy_forecast.evaluation.probabilistic_metrics import (
    average_interval_width,
    interval_coverage,
    interval_score,
    mean_absolute_calibration_error,
    pinball_loss,
    weighted_interval_score,
)
from dkenergy_forecast.evaluation.splits import (
    EvaluationInterval,
    filter_evaluation_interval,
)
from dkenergy_forecast.evaluation.stratification import (
    prepare_evaluation_strata,
    stratified_score_table,
)
from dkenergy_forecast.types import require_columns


PAIR_KEY_COLUMNS = ["forecast_origin_utc", "unique_id", "ds_utc", "area"]
QUANTILE_COLUMNS = ["q10", "q50", "q90"]
MODEL_COLUMN = "model_label"
DIFFERENCE_METRICS = [
    "mae",
    "rmse",
    "bias",
    "weighted_interval_score",
    "calibration_error",
]


def pair_model_predictions(
    predictions: pd.DataFrame,
    *,
    reference_model: str,
    comparison_model: str,
    reference_release: str | None = None,
    comparison_release: str | None = None,
) -> pd.DataFrame:
    """Return exactly aligned rows for two models.

    Pairing is deliberately strict: both models must have one row for every
    observed forecast key, and the recorded actual must agree. A comparison
    with missing or duplicated rows is rejected instead of silently changing
    the evaluation sample.
    """

    if reference_model == comparison_model:
        raise ValueError("reference_model and comparison_model must be different")
    require_columns(
        predictions,
        [MODEL_COLUMN, "forecast_origin_utc", "ds_utc", "area", "y", "y_pred"],
        "predictions",
    )
    key_columns = [column for column in PAIR_KEY_COLUMNS if column in predictions]
    selected = predictions[
        predictions[MODEL_COLUMN].isin([reference_model, comparison_model])
    ].copy()
    selected = _select_releases(
        selected,
        reference_model=reference_model,
        comparison_model=comparison_model,
        reference_release=reference_release,
        comparison_release=comparison_release,
    )
    selected["forecast_origin_utc"] = pd.to_datetime(
        selected["forecast_origin_utc"], utc=True
    )
    selected["ds_utc"] = pd.to_datetime(selected["ds_utc"], utc=True)
    selected = _normalize_numeric_predictions(selected)
    _validate_selected_models(
        selected,
        model_labels=(reference_model, comparison_model),
        key_columns=key_columns,
    )

    reference = selected[selected[MODEL_COLUMN].eq(reference_model)]
    comparison = selected[selected[MODEL_COLUMN].eq(comparison_model)]
    _require_identical_keys(reference, comparison, key_columns=key_columns)

    value_columns = [
        "y",
        "y_pred",
        *[column for column in QUANTILE_COLUMNS if column in selected],
    ]
    reference_values = reference[key_columns + value_columns].rename(
        columns={column: f"reference_{column}" for column in value_columns}
    )
    comparison_values = comparison[key_columns + value_columns].rename(
        columns={column: f"comparison_{column}" for column in value_columns}
    )
    paired = reference_values.merge(
        comparison_values,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    actual_difference = (
        paired["reference_y"] - paired["comparison_y"]
    ).abs()
    if actual_difference.gt(1e-9).any():
        raise ValueError("The paired model rows disagree on actual target values")
    paired["y"] = paired.pop("reference_y")
    paired = paired.drop(columns=["comparison_y"])
    return paired.sort_values(key_columns).reset_index(drop=True)


def origin_metric_differences(
    predictions: pd.DataFrame,
    *,
    reference_model: str,
    comparison_model: str,
    reference_release: str | None = None,
    comparison_release: str | None = None,
) -> pd.DataFrame:
    """Return comparison-minus-reference metrics for each forecast origin."""

    paired = pair_model_predictions(
        predictions,
        reference_model=reference_model,
        comparison_model=comparison_model,
        reference_release=reference_release,
        comparison_release=comparison_release,
    )
    return _origin_differences_from_paired(paired)


def moving_block_bootstrap_mean_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    block_length: int = 7,
    n_resamples: int = 2_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Estimate a mean CI with a deterministic circular block bootstrap.

    Input order is significant and should be chronological. Sampling adjacent
    origins in blocks retains short-range serial dependence that an ordinary
    row bootstrap would discard.
    """

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    try:
        clean = [float(value) for value in values]
    except (TypeError, ValueError) as error:
        raise ValueError("values must contain only finite numbers") from error
    if not clean:
        raise ValueError("values must not be empty")
    if any(not math.isfinite(value) for value in clean):
        raise ValueError("values must contain only finite numbers")

    sample_size = len(clean)
    effective_block_length = min(block_length, sample_size)
    rng = random.Random(seed)
    bootstrap_means: list[float] = []
    for _ in range(n_resamples):
        sampled: list[float] = []
        while len(sampled) < sample_size:
            start = rng.randrange(sample_size)
            sampled.extend(
                clean[(start + offset) % sample_size]
                for offset in range(effective_block_length)
            )
        bootstrap_means.append(sum(sampled[:sample_size]) / sample_size)

    tail = (1.0 - confidence) / 2.0
    bootstrap_series = pd.Series(bootstrap_means, dtype="float64")
    return {
        "mean": float(sum(clean) / sample_size),
        "lower": float(bootstrap_series.quantile(tail)),
        "upper": float(bootstrap_series.quantile(1.0 - tail)),
        "confidence": confidence,
        "origin_count": sample_size,
        "block_length": effective_block_length,
        "n_resamples": n_resamples,
        "seed": seed,
        "method": "circular_moving_block_bootstrap",
    }


def build_model_comparison(
    predictions: pd.DataFrame,
    *,
    reference_model: str,
    comparison_model: str,
    reference_release: str | None = None,
    comparison_release: str | None = None,
    interval: EvaluationInterval,
    confidence: float = 0.95,
    block_length: int = 7,
    n_resamples: int = 2_000,
    seed: int = 2026,
    extreme_threshold: float | None = None,
    extreme_quantile: float = 0.95,
    split_provenance: dict[str, Any] | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a descriptive, deterministic comparison of two forecast models.

    The report contains evidence only. It deliberately has no acceptance
    thresholds or model-selection side effects.
    """

    interval_predictions = filter_evaluation_interval(predictions, interval)
    require_columns(interval_predictions, [MODEL_COLUMN], "predictions")
    selected = interval_predictions[
        interval_predictions[MODEL_COLUMN].isin(
            [reference_model, comparison_model]
        )
    ].copy()
    selected = _select_releases(
        selected,
        reference_model=reference_model,
        comparison_model=comparison_model,
        reference_release=reference_release,
        comparison_release=comparison_release,
    )
    if "model_release_id" in selected.columns:
        reference_release = _single_release(selected, reference_model)
        comparison_release = _single_release(selected, comparison_model)
    selected = _normalize_numeric_predictions(selected)
    paired = pair_model_predictions(
        selected,
        reference_model=reference_model,
        comparison_model=comparison_model,
        reference_release=reference_release,
        comparison_release=comparison_release,
    )
    per_origin = _origin_differences_from_paired(paired)
    reference_metrics = _metric_summary(paired, prefix="reference")
    comparison_metrics = _metric_summary(paired, prefix="comparison")

    bootstrap = {
        metric: _optional_bootstrap(
            per_origin[f"{metric}_difference"],
            confidence=confidence,
            block_length=block_length,
            n_resamples=n_resamples,
            seed=seed,
        )
        for metric in ("mae", "weighted_interval_score", "calibration_error")
    }
    overall_differences = {
        metric: _difference_or_nan(
            comparison_metrics[metric], reference_metrics[metric]
        )
        for metric in DIFFERENCE_METRICS
    }

    _, resolved_extreme_threshold = prepare_evaluation_strata(
        paired.rename(columns={"comparison_y_pred": "y_pred"}),
        extreme_threshold=extreme_threshold,
        extreme_quantile=extreme_quantile,
    )
    subgroup_scores = stratified_score_table(
        selected,
        extreme_threshold=resolved_extreme_threshold,
        extreme_quantile=extreme_quantile,
    )
    subgroup_differences = _stratum_differences(
        subgroup_scores,
        reference_model=reference_model,
        comparison_model=comparison_model,
    )

    return {
        "schema_version": "model_comparison_v1",
        "reference_model": reference_model,
        "comparison_model": comparison_model,
        "reference_release": reference_release,
        "comparison_release": comparison_release,
        "difference_definition": "comparison_minus_reference",
        "evaluation_interval": interval.as_dict(),
        "split_provenance": split_provenance,
        "source_sha256": source_sha256,
        "pairing": {
            "key_columns": [
                column for column in PAIR_KEY_COLUMNS if column in selected
            ],
            "paired_rows": int(len(paired)),
            "origin_count": int(per_origin["forecast_origin_utc"].nunique()),
            "model_keys_identical": True,
        },
        "overall": {
            "reference": {
                "model_label": reference_model,
                "metrics": reference_metrics,
            },
            "comparison": {
                "model_label": comparison_model,
                "metrics": comparison_metrics,
            },
            "differences": overall_differences,
        },
        "per_origin_differences": per_origin.to_dict(orient="records"),
        "bootstrap_confidence_intervals": bootstrap,
        "stratification": {
            "extreme_quantile": extreme_quantile,
            "extreme_price_threshold": resolved_extreme_threshold,
            "scores": subgroup_scores.to_dict(orient="records"),
            "differences": subgroup_differences.to_dict(orient="records"),
        },
    }


def _origin_differences_from_paired(paired: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for origin, frame in paired.groupby("forecast_origin_utc", sort=True):
        reference = _metric_summary(frame, prefix="reference")
        comparison = _metric_summary(frame, prefix="comparison")
        row: dict[str, Any] = {
            "forecast_origin_utc": origin,
            "rows": int(len(frame)),
        }
        for metric in DIFFERENCE_METRICS:
            row[f"reference_{metric}"] = reference[metric]
            row[f"comparison_{metric}"] = comparison[metric]
            row[f"{metric}_difference"] = _difference_or_nan(
                comparison[metric], reference[metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_selected_models(
    selected: pd.DataFrame,
    *,
    model_labels: tuple[str, str],
    key_columns: list[str],
) -> None:
    labels = set(selected[MODEL_COLUMN].dropna())
    missing_labels = sorted(set(model_labels) - labels)
    if missing_labels:
        raise ValueError(f"Predictions are missing selected model(s): {missing_labels}")
    missing_key_count = int(selected[key_columns].isna().any(axis=1).sum())
    if missing_key_count:
        raise ValueError(
            f"Selected predictions contain {missing_key_count} row(s) with missing pairing keys"
        )
    for label, frame in selected.groupby(MODEL_COLUMN, sort=False):
        duplicate_count = int(frame.duplicated(key_columns).sum())
        if duplicate_count:
            raise ValueError(
                f"Model {label!r} has {duplicate_count} duplicate prediction key row(s)"
            )
        for column in ("y", "y_pred"):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.isna().any() or not numeric.map(math.isfinite).all():
                raise ValueError(
                    f"Model {label!r} has missing or non-finite {column} values"
                )
        _validate_quantile_completeness(frame, label=str(label))


def _select_releases(
    selected: pd.DataFrame,
    *,
    reference_model: str,
    comparison_model: str,
    reference_release: str | None,
    comparison_release: str | None,
) -> pd.DataFrame:
    if "model_release_id" not in selected.columns:
        if reference_release is not None or comparison_release is not None:
            raise ValueError("Predictions have no model_release_id column")
        return selected

    requested = {
        reference_model: reference_release,
        comparison_model: comparison_release,
    }
    frames: list[pd.DataFrame] = []
    for model_label, release_id in requested.items():
        model_rows = selected[selected[MODEL_COLUMN].eq(model_label)]
        available = sorted(model_rows["model_release_id"].dropna().astype(str).unique())
        if release_id is None:
            if len(available) != 1:
                raise ValueError(
                    f"Model {model_label!r} has multiple releases {available}; "
                    "select one explicitly"
                )
            release_id = available[0]
        release_rows = model_rows[
            model_rows["model_release_id"].astype(str).eq(str(release_id))
        ]
        if release_rows.empty:
            raise ValueError(
                f"Model {model_label!r} has no predictions for release {release_id!r}; "
                f"available={available}"
            )
        frames.append(release_rows)
    return pd.concat(frames, ignore_index=True)


def _single_release(selected: pd.DataFrame, model_label: str) -> str:
    releases = selected.loc[
        selected[MODEL_COLUMN].eq(model_label), "model_release_id"
    ].dropna().astype(str).unique()
    if len(releases) != 1:
        raise ValueError(f"Expected exactly one release for model {model_label!r}")
    return str(releases[0])


def _validate_quantile_completeness(frame: pd.DataFrame, *, label: str) -> None:
    present = [column for column in QUANTILE_COLUMNS if column in frame]
    if not present or not any(frame[column].notna().any() for column in present):
        return
    missing_columns = sorted(set(QUANTILE_COLUMNS) - set(present))
    if missing_columns:
        raise ValueError(
            f"Model {label!r} has incomplete quantile columns: {missing_columns}"
        )
    if any(frame[column].isna().any() for column in QUANTILE_COLUMNS):
        raise ValueError(f"Model {label!r} has partially missing quantile forecasts")
    if not all(frame[column].map(math.isfinite).all() for column in QUANTILE_COLUMNS):
        raise ValueError(f"Model {label!r} has non-finite quantile forecasts")
    weighted_interval_score(frame[["y", *QUANTILE_COLUMNS]])


def _require_identical_keys(
    reference: pd.DataFrame,
    comparison: pd.DataFrame,
    *,
    key_columns: list[str],
) -> None:
    key_check = reference[key_columns].merge(
        comparison[key_columns],
        on=key_columns,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    unmatched = key_check["_merge"].ne("both")
    if unmatched.any():
        counts = key_check.loc[unmatched, "_merge"].value_counts().to_dict()
        raise ValueError(
            "Reference and comparison prediction keys differ; refusing an unpaired "
            f"comparison: {counts}"
        )


def _normalize_numeric_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    for column in ["y", "y_pred", *QUANTILE_COLUMNS]:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _metric_summary(paired: pd.DataFrame, *, prefix: str) -> dict[str, Any]:
    frame = pd.DataFrame(
        {
            "y": paired["y"],
            "y_pred": paired[f"{prefix}_y_pred"],
        }
    )
    for column in QUANTILE_COLUMNS:
        source = f"{prefix}_{column}"
        if source in paired:
            frame[column] = paired[source]
    return {
        "rows": int(len(frame)),
        "origin_count": int(paired["forecast_origin_utc"].nunique()),
        "mae": mae(frame),
        "rmse": rmse(frame),
        "bias": bias(frame),
        "pinball_q10": _pinball_or_nan(frame, 0.10),
        "pinball_q50": _pinball_or_nan(frame, 0.50),
        "pinball_q90": _pinball_or_nan(frame, 0.90),
        "coverage_80": _coverage_or_nan(frame),
        "interval_width_80": _interval_width_or_nan(frame),
        "interval_score_80": _interval_score_or_nan(frame),
        "weighted_interval_score": _wis_or_nan(frame),
        "calibration_error": _calibration_or_nan(frame),
    }


def _stratum_differences(
    scores: pd.DataFrame,
    *,
    reference_model: str,
    comparison_model: str,
) -> pd.DataFrame:
    reference = scores[scores[MODEL_COLUMN].eq(reference_model)].drop(
        columns=[MODEL_COLUMN]
    )
    comparison = scores[scores[MODEL_COLUMN].eq(comparison_model)].drop(
        columns=[MODEL_COLUMN]
    )
    paired = reference.merge(
        comparison,
        on=["stratum", "stratum_value"],
        how="outer",
        suffixes=("_reference", "_comparison"),
        indicator=True,
        validate="one_to_one",
    )
    if not paired["_merge"].eq("both").all():
        raise ValueError("Reference and comparison stratum keys differ")
    if not paired["rows_reference"].eq(paired["rows_comparison"]).all():
        raise ValueError("Reference and comparison stratum row counts differ")

    rows: list[dict[str, Any]] = []
    for record in paired.to_dict(orient="records"):
        row = {
            "stratum": record["stratum"],
            "stratum_value": record["stratum_value"],
            "rows": int(record["rows_reference"]),
            "origin_count": int(record["origin_count_reference"]),
        }
        for metric in ("mae", "weighted_interval_score", "calibration_error"):
            reference_value = record[f"{metric}_reference"]
            comparison_value = record[f"{metric}_comparison"]
            row[f"reference_{metric}"] = reference_value
            row[f"comparison_{metric}"] = comparison_value
            row[f"{metric}_difference"] = _difference_or_nan(
                comparison_value, reference_value
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["stratum", "stratum_value"]
    ).reset_index(drop=True)


def _optional_bootstrap(
    values: pd.Series,
    **kwargs: Any,
) -> dict[str, Any] | None:
    if not values.map(_finite).all():
        return None
    return moving_block_bootstrap_mean_ci(values, **kwargs)


def _pinball_or_nan(frame: pd.DataFrame, quantile: float) -> float:
    column = f"q{int(quantile * 100)}"
    if column not in frame:
        return math.nan
    return pinball_loss(frame, quantile=quantile, pred_col=column)


def _coverage_or_nan(frame: pd.DataFrame) -> float:
    return interval_coverage(frame) if {"q10", "q90"}.issubset(frame) else math.nan


def _interval_width_or_nan(frame: pd.DataFrame) -> float:
    if not {"q10", "q90"}.issubset(frame):
        return math.nan
    return average_interval_width(frame)


def _interval_score_or_nan(frame: pd.DataFrame) -> float:
    return interval_score(frame) if {"q10", "q90"}.issubset(frame) else math.nan


def _wis_or_nan(frame: pd.DataFrame) -> float:
    if not set(QUANTILE_COLUMNS).issubset(frame):
        return math.nan
    return weighted_interval_score(frame)


def _calibration_or_nan(frame: pd.DataFrame) -> float:
    if not set(QUANTILE_COLUMNS).issubset(frame):
        return math.nan
    return mean_absolute_calibration_error(frame)


def _difference_or_nan(comparison: object, reference: object) -> float:
    if not _finite(comparison) or not _finite(reference):
        return math.nan
    return float(comparison) - float(reference)


def _finite(value: object) -> bool:
    try:
        return pd.notna(value) and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
