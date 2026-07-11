from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from typing import Any, Iterable

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


@dataclass(frozen=True)
class PromotionPolicy:
    """Small, explicit set of champion-promotion guardrails."""

    min_mae_relative_improvement: float = 0.01
    max_wis_relative_degradation: float = 0.0
    max_calibration_error_increase: float = 0.02
    max_calibration_error: float = 0.10
    max_subgroup_mae_relative_degradation: float = 0.10
    min_subgroup_rows: int = 24
    require_probabilistic_comparison: bool = True
    require_mae_ci_improvement: bool = True

    def __post_init__(self) -> None:
        numeric_non_negative = {
            "min_mae_relative_improvement": self.min_mae_relative_improvement,
            "max_wis_relative_degradation": self.max_wis_relative_degradation,
            "max_calibration_error_increase": self.max_calibration_error_increase,
            "max_calibration_error": self.max_calibration_error,
            "max_subgroup_mae_relative_degradation": (
                self.max_subgroup_mae_relative_degradation
            ),
        }
        invalid = [name for name, value in numeric_non_negative.items() if value < 0]
        if invalid:
            raise ValueError(f"Promotion policy values must be non-negative: {invalid}")
        if self.min_mae_relative_improvement >= 1:
            raise ValueError("min_mae_relative_improvement must be below 1")
        if self.min_subgroup_rows < 1:
            raise ValueError("min_subgroup_rows must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def paired_model_predictions(
    predictions: pd.DataFrame,
    *,
    candidate_label: str,
    champion_label: str,
    model_label_col: str = "model_label",
) -> pd.DataFrame:
    """Align two models exactly, rejecting duplicates and omitted forecast rows."""

    if candidate_label == champion_label:
        raise ValueError("candidate_label and champion_label must be different")
    require_columns(
        predictions,
        [model_label_col, "forecast_origin_utc", "ds_utc", "area", "y", "y_pred"],
        "predictions",
    )
    key_columns = [column for column in PAIR_KEY_COLUMNS if column in predictions.columns]
    selected = predictions[predictions[model_label_col].isin([candidate_label, champion_label])].copy()
    selected["forecast_origin_utc"] = pd.to_datetime(
        selected["forecast_origin_utc"], utc=True
    )
    selected["ds_utc"] = pd.to_datetime(selected["ds_utc"], utc=True)
    selected = _normalize_numeric_predictions(selected)
    _validate_selected_models(
        selected,
        candidate_label=candidate_label,
        champion_label=champion_label,
        model_label_col=model_label_col,
        key_columns=key_columns,
    )

    candidate = selected[selected[model_label_col].eq(candidate_label)].copy()
    champion = selected[selected[model_label_col].eq(champion_label)].copy()
    _require_identical_keys(candidate, champion, key_columns=key_columns)

    value_columns = ["y", "y_pred", *[column for column in QUANTILE_COLUMNS if column in selected]]
    candidate_values = candidate[key_columns + value_columns].rename(
        columns={column: f"candidate_{column}" for column in value_columns}
    )
    champion_values = champion[key_columns + value_columns].rename(
        columns={column: f"champion_{column}" for column in value_columns}
    )
    paired = candidate_values.merge(
        champion_values,
        on=key_columns,
        how="inner",
        validate="one_to_one",
    )
    actual_difference = (
        pd.to_numeric(paired["candidate_y"], errors="coerce")
        - pd.to_numeric(paired["champion_y"], errors="coerce")
    ).abs()
    if actual_difference.isna().any():
        raise ValueError("Selected predictions contain missing or non-numeric actual values")
    if actual_difference.gt(1e-9).any():
        raise ValueError("Candidate and champion rows disagree on actual target values")
    paired["y"] = pd.to_numeric(paired.pop("candidate_y"), errors="raise")
    paired = paired.drop(columns=["champion_y"])
    return paired.sort_values(key_columns).reset_index(drop=True)


def paired_origin_comparison(
    predictions: pd.DataFrame,
    *,
    candidate_label: str,
    champion_label: str,
    model_label_col: str = "model_label",
) -> pd.DataFrame:
    """Return candidate-minus-champion metrics for each forecast origin."""

    paired = paired_model_predictions(
        predictions,
        candidate_label=candidate_label,
        champion_label=champion_label,
        model_label_col=model_label_col,
    )
    rows: list[dict[str, Any]] = []
    for origin, frame in paired.groupby("forecast_origin_utc", sort=True):
        candidate = _metric_summary(frame, prefix="candidate")
        champion = _metric_summary(frame, prefix="champion")
        candidate_mae = candidate["mae"]
        champion_mae = champion["mae"]
        rows.append(
            {
                "forecast_origin_utc": origin,
                "rows": int(len(frame)),
                "candidate_mae": candidate_mae,
                "champion_mae": champion_mae,
                "mae_difference": candidate_mae - champion_mae,
                "mae_winner": _winner(
                    candidate_mae,
                    champion_mae,
                    candidate_label=candidate_label,
                    champion_label=champion_label,
                ),
                "candidate_weighted_interval_score": candidate[
                    "weighted_interval_score"
                ],
                "champion_weighted_interval_score": champion[
                    "weighted_interval_score"
                ],
                "weighted_interval_score_difference": _difference_or_nan(
                    candidate["weighted_interval_score"],
                    champion["weighted_interval_score"],
                ),
                "candidate_calibration_error": candidate["calibration_error"],
                "champion_calibration_error": champion["calibration_error"],
                "calibration_error_difference": _difference_or_nan(
                    candidate["calibration_error"],
                    champion["calibration_error"],
                ),
            }
        )
    return pd.DataFrame(rows)


def block_bootstrap_mean_ci(
    values: Iterable[float],
    *,
    confidence: float = 0.95,
    block_length: int = 7,
    n_resamples: int = 2_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Circular moving-block bootstrap CI for an origin-level mean.

    Input order is significant and should be chronological. Whole blocks of
    adjacent origins are sampled, retaining short-range serial dependence.
    """

    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    if n_resamples < 1:
        raise ValueError("n_resamples must be positive")

    raw_values = list(values)
    if not raw_values:
        raise ValueError("values must not be empty")
    try:
        clean = [float(value) for value in raw_values]
    except (TypeError, ValueError) as error:
        raise ValueError("values must contain only finite numbers") from error
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


def build_evaluation_report(
    predictions: pd.DataFrame,
    *,
    candidate_label: str,
    champion_label: str,
    interval: EvaluationInterval,
    policy: PromotionPolicy | None = None,
    confidence: float = 0.95,
    block_length: int = 7,
    n_resamples: int = 2_000,
    seed: int = 2026,
    extreme_threshold: float | None = None,
    extreme_quantile: float = 0.95,
    split_provenance: dict[str, Any] | None = None,
    source_sha256: str | None = None,
) -> dict[str, Any]:
    """Build the complete deterministic arena report as plain Python data."""

    policy = policy or PromotionPolicy()
    interval_predictions = filter_evaluation_interval(predictions, interval)
    require_columns(interval_predictions, ["model_label"], "predictions")
    selected = interval_predictions[
        interval_predictions["model_label"].isin([candidate_label, champion_label])
    ].copy()
    selected = _normalize_numeric_predictions(selected)
    paired = paired_model_predictions(
        selected,
        candidate_label=candidate_label,
        champion_label=champion_label,
    )
    per_origin = paired_origin_comparison(
        selected,
        candidate_label=candidate_label,
        champion_label=champion_label,
    )
    overall = {
        candidate_label: _metric_summary(paired, prefix="candidate"),
        champion_label: _metric_summary(paired, prefix="champion"),
    }

    bootstrap = {
        "mae_difference": block_bootstrap_mean_ci(
            per_origin["mae_difference"],
            confidence=confidence,
            block_length=block_length,
            n_resamples=n_resamples,
            seed=seed,
        ),
        "weighted_interval_score_difference": _optional_bootstrap(
            per_origin["weighted_interval_score_difference"],
            confidence=confidence,
            block_length=block_length,
            n_resamples=n_resamples,
            seed=seed,
        ),
        "calibration_error_difference": _optional_bootstrap(
            per_origin["calibration_error_difference"],
            confidence=confidence,
            block_length=block_length,
            n_resamples=n_resamples,
            seed=seed,
        ),
    }

    _, resolved_extreme_threshold = prepare_evaluation_strata(
        paired.rename(columns={"candidate_y_pred": "y_pred"}),
        extreme_threshold=extreme_threshold,
        extreme_quantile=extreme_quantile,
    )
    subgroup_scores = stratified_score_table(
        selected,
        extreme_threshold=resolved_extreme_threshold,
        extreme_quantile=extreme_quantile,
    )
    guardrails = _subgroup_guardrails(
        subgroup_scores,
        candidate_label=candidate_label,
        champion_label=champion_label,
        policy=policy,
    )
    promotion = _promotion_decision(
        overall=overall,
        bootstrap=bootstrap,
        guardrails=guardrails,
        candidate_label=candidate_label,
        champion_label=champion_label,
        policy=policy,
    )

    report = {
        "schema_version": "evaluation_arena_v1",
        "candidate_label": candidate_label,
        "champion_label": champion_label,
        "evaluation_interval": interval.as_dict(),
        "split_provenance": split_provenance,
        "source_sha256": source_sha256,
        "pairing": {
            "key_columns": [
                column for column in PAIR_KEY_COLUMNS if column in selected.columns
            ],
            "paired_rows": int(len(paired)),
            "origin_count": int(per_origin["forecast_origin_utc"].nunique()),
            "candidate_and_champion_keys_identical": True,
        },
        "overall": overall,
        "paired_origin_comparisons": per_origin.to_dict(orient="records"),
        "bootstrap_confidence_intervals": bootstrap,
        "stratification": {
            "extreme_quantile": extreme_quantile,
            "extreme_price_threshold": resolved_extreme_threshold,
            "scores": subgroup_scores.to_dict(orient="records"),
            "guardrails": guardrails.to_dict(orient="records"),
        },
        "promotion_policy": policy.as_dict(),
        "promotion": promotion,
    }
    return report


def _validate_selected_models(
    selected: pd.DataFrame,
    *,
    candidate_label: str,
    champion_label: str,
    model_label_col: str,
    key_columns: list[str],
) -> None:
    labels = set(selected[model_label_col].dropna().astype(str))
    missing_labels = sorted({candidate_label, champion_label} - labels)
    if missing_labels:
        raise ValueError(f"Predictions are missing selected model(s): {missing_labels}")
    missing_key_count = int(selected[key_columns].isna().any(axis=1).sum())
    if missing_key_count:
        raise ValueError(
            f"Selected predictions contain {missing_key_count} row(s) with missing pairing keys"
        )
    for label, frame in selected.groupby(model_label_col, sort=False):
        duplicate_count = int(frame.duplicated(key_columns).sum())
        if duplicate_count:
            raise ValueError(
                f"Model {label!r} has {duplicate_count} duplicate prediction key row(s)"
            )
        for column in ["y", "y_pred"]:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if numeric.isna().any() or not numeric.map(math.isfinite).all():
                raise ValueError(
                    f"Model {label!r} has missing or non-finite {column} values"
                )
        _validate_quantile_completeness(frame, label=str(label))


def _validate_quantile_completeness(frame: pd.DataFrame, *, label: str) -> None:
    present = [column for column in QUANTILE_COLUMNS if column in frame.columns]
    if not present:
        return
    any_values = any(frame[column].notna().any() for column in present)
    if not any_values:
        return
    missing_columns = sorted(set(QUANTILE_COLUMNS) - set(present))
    if missing_columns:
        raise ValueError(f"Model {label!r} has incomplete quantile columns: {missing_columns}")
    if any(frame[column].isna().any() for column in QUANTILE_COLUMNS):
        raise ValueError(f"Model {label!r} has partially missing quantile forecasts")
    if not all(frame[column].map(math.isfinite).all() for column in QUANTILE_COLUMNS):
        raise ValueError(f"Model {label!r} has non-finite quantile forecasts")
    values = frame[["y", *QUANTILE_COLUMNS]].copy()
    weighted_interval_score(values)


def _require_identical_keys(
    candidate: pd.DataFrame,
    champion: pd.DataFrame,
    *,
    key_columns: list[str],
) -> None:
    key_check = candidate[key_columns].merge(
        champion[key_columns],
        on=key_columns,
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    unmatched = key_check["_merge"].ne("both")
    if unmatched.any():
        counts = key_check.loc[unmatched, "_merge"].value_counts().to_dict()
        raise ValueError(
            "Candidate and champion prediction keys differ; refusing an unpaired "
            f"comparison: {counts}"
        )


def _normalize_numeric_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    for column in ["y", "y_pred", *QUANTILE_COLUMNS]:
        if column in output.columns:
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
        if source in paired.columns:
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


def _optional_bootstrap(values: pd.Series, **kwargs: Any) -> dict[str, Any] | None:
    if not values.map(lambda value: pd.notna(value) and math.isfinite(float(value))).all():
        return None
    return block_bootstrap_mean_ci(values, **kwargs)


def _subgroup_guardrails(
    subgroup_scores: pd.DataFrame,
    *,
    candidate_label: str,
    champion_label: str,
    policy: PromotionPolicy,
) -> pd.DataFrame:
    candidate = subgroup_scores[subgroup_scores["model_label"].eq(candidate_label)].drop(
        columns=["model_label"]
    )
    champion = subgroup_scores[subgroup_scores["model_label"].eq(champion_label)].drop(
        columns=["model_label"]
    )
    paired = candidate.merge(
        champion,
        on=["stratum", "stratum_value"],
        suffixes=("_candidate", "_champion"),
        validate="one_to_one",
    )
    if len(paired) != len(candidate) or len(paired) != len(champion):
        raise ValueError("Candidate and champion subgroup keys differ")
    rows = []
    for record in paired.to_dict(orient="records"):
        candidate_mae = float(record["mae_candidate"])
        champion_mae = float(record["mae_champion"])
        paired_rows = min(int(record["rows_candidate"]), int(record["rows_champion"]))
        eligible = paired_rows >= policy.min_subgroup_rows
        allowed_mae = champion_mae * (
            1.0 + policy.max_subgroup_mae_relative_degradation
        )
        passed = candidate_mae <= allowed_mae if eligible else None
        rows.append(
            {
                "stratum": record["stratum"],
                "stratum_value": record["stratum_value"],
                "rows": paired_rows,
                "origin_count": min(
                    int(record["origin_count_candidate"]),
                    int(record["origin_count_champion"]),
                ),
                "candidate_mae": candidate_mae,
                "champion_mae": champion_mae,
                "mae_relative_change": _relative_change(candidate_mae, champion_mae),
                "max_allowed_relative_degradation": (
                    policy.max_subgroup_mae_relative_degradation
                ),
                "eligible": eligible,
                "passed": passed,
            }
        )
    return pd.DataFrame(rows).sort_values(["stratum", "stratum_value"]).reset_index(
        drop=True
    )


def _promotion_decision(
    *,
    overall: dict[str, dict[str, Any]],
    bootstrap: dict[str, dict[str, Any] | None],
    guardrails: pd.DataFrame,
    candidate_label: str,
    champion_label: str,
    policy: PromotionPolicy,
) -> dict[str, Any]:
    candidate = overall[candidate_label]
    champion = overall[champion_label]
    allowed_candidate_mae = champion["mae"] * (
        1.0 - policy.min_mae_relative_improvement
    )
    checks: list[dict[str, Any]] = [
        {
            "name": "overall_mae",
            "passed": candidate["mae"] <= allowed_candidate_mae,
            "candidate": candidate["mae"],
            "champion": champion["mae"],
            "required_relative_improvement": policy.min_mae_relative_improvement,
        }
    ]

    mae_ci = bootstrap["mae_difference"]
    checks.append(
        {
            "name": "paired_mae_confidence_interval",
            "passed": (
                mae_ci["upper"] <= 0 if policy.require_mae_ci_improvement else None
            ),
            "candidate_minus_champion_upper": mae_ci["upper"],
            "required": policy.require_mae_ci_improvement,
        }
    )

    candidate_wis = candidate["weighted_interval_score"]
    champion_wis = champion["weighted_interval_score"]
    candidate_wis_available = _finite(candidate_wis)
    champion_wis_available = _finite(champion_wis)
    wis_available = candidate_wis_available and champion_wis_available
    checks.append(
        {
            "name": "weighted_interval_score",
            "passed": (
                candidate_wis
                <= champion_wis * (1.0 + policy.max_wis_relative_degradation)
                if wis_available
                else (
                    None
                    if candidate_wis_available
                    and not policy.require_probabilistic_comparison
                    else False
                )
            ),
            "candidate": candidate_wis,
            "champion": champion_wis,
            "max_relative_degradation": policy.max_wis_relative_degradation,
            "available": wis_available,
            "candidate_available": candidate_wis_available,
            "champion_available": champion_wis_available,
        }
    )

    candidate_calibration = candidate["calibration_error"]
    champion_calibration = champion["calibration_error"]
    candidate_calibration_available = _finite(candidate_calibration)
    champion_calibration_available = _finite(champion_calibration)
    calibration_available = (
        candidate_calibration_available and champion_calibration_available
    )
    checks.extend(
        [
            {
                "name": "calibration_vs_champion",
                "passed": (
                    candidate_calibration
                    <= champion_calibration + policy.max_calibration_error_increase
                    if calibration_available
                    else (
                        None
                        if candidate_calibration_available
                        and not policy.require_probabilistic_comparison
                        else False
                    )
                ),
                "candidate": candidate_calibration,
                "champion": champion_calibration,
                "max_absolute_increase": policy.max_calibration_error_increase,
                "available": calibration_available,
                "candidate_available": candidate_calibration_available,
                "champion_available": champion_calibration_available,
            },
            {
                "name": "absolute_calibration",
                "passed": (
                    candidate_calibration <= policy.max_calibration_error
                    if candidate_calibration_available
                    else False
                ),
                "candidate": candidate_calibration,
                "maximum": policy.max_calibration_error,
                "available": candidate_calibration_available,
            },
        ]
    )

    eligible_guardrails = guardrails[guardrails["eligible"]]
    subgroup_passed = bool(eligible_guardrails["passed"].all())
    failed_guardrails = eligible_guardrails[~eligible_guardrails["passed"]]
    checks.append(
        {
            "name": "subgroup_mae_guardrails",
            "passed": subgroup_passed,
            "eligible_subgroup_count": int(len(eligible_guardrails)),
            "failed_subgroup_count": int(len(failed_guardrails)),
        }
    )

    failed = [check["name"] for check in checks if check["passed"] is False]
    return {
        "decision": "promote_candidate" if not failed else "retain_champion",
        "passed": not failed,
        "failed_checks": failed,
        "checks": checks,
    }


def _pinball_or_nan(frame: pd.DataFrame, quantile: float) -> float:
    column = f"q{int(quantile * 100)}"
    if column not in frame:
        return math.nan
    return pinball_loss(frame, quantile=quantile, pred_col=column)


def _coverage_or_nan(frame: pd.DataFrame) -> float:
    return interval_coverage(frame) if {"q10", "q90"}.issubset(frame) else math.nan


def _interval_width_or_nan(frame: pd.DataFrame) -> float:
    return average_interval_width(frame) if {"q10", "q90"}.issubset(frame) else math.nan


def _interval_score_or_nan(frame: pd.DataFrame) -> float:
    return interval_score(frame) if {"q10", "q90"}.issubset(frame) else math.nan


def _wis_or_nan(frame: pd.DataFrame) -> float:
    return weighted_interval_score(frame) if set(QUANTILE_COLUMNS).issubset(frame) else math.nan


def _calibration_or_nan(frame: pd.DataFrame) -> float:
    return (
        mean_absolute_calibration_error(frame)
        if set(QUANTILE_COLUMNS).issubset(frame)
        else math.nan
    )


def _winner(
    candidate_value: float,
    champion_value: float,
    *,
    candidate_label: str,
    champion_label: str,
) -> str:
    if math.isclose(candidate_value, champion_value, rel_tol=1e-12, abs_tol=1e-12):
        return "tie"
    return candidate_label if candidate_value < champion_value else champion_label


def _difference_or_nan(candidate: float, champion: float) -> float:
    return candidate - champion if _finite(candidate) and _finite(champion) else math.nan


def _relative_change(candidate: float, champion: float) -> float | None:
    if champion == 0:
        return 0.0 if candidate == 0 else None
    return (candidate - champion) / champion


def _finite(value: object) -> bool:
    return pd.notna(value) and math.isfinite(float(value))
