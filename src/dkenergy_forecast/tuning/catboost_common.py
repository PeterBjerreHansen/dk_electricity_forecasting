from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class CatBoostTuningResult:
    feature_set: str
    best_value: float
    best_params: dict[str, Any]
    best_trial_number: int
    feature_columns: list[str]
    validation_origin_count: int
    trial_count: int
    trials: pd.DataFrame


def suggest_catboost_params(
    trial: Any,
    *,
    feature_count: int,
    random_seed: int,
    max_iterations: int,
    task_type: str | None = None,
    has_time: bool = False,
    search_profile: str = "default",
) -> dict[str, Any]:
    """Suggest a CatBoost search space tuned for time-series tabular regression."""

    if max_iterations < 300:
        raise ValueError("max_iterations must be at least 300")
    if search_profile not in {"default", "conservative"}:
        raise ValueError("search_profile must be 'default' or 'conservative'")

    bootstrap_type = trial.suggest_categorical(
        "bootstrap_type",
        ["Bayesian", "Bernoulli", "MVS"],
    )
    if search_profile == "conservative":
        depth_range = (3, 6)
        learning_rate_range = (0.01, 0.08)
        l2_range = (10.0, 200.0)
        random_strength_range = (2.0, 20.0)
        border_choices = [32, 64, 128]
        leaf_iterations_range = (1, 4)
        subsample_range = (0.65, 0.95)
        bagging_temperature_range = (0.5, 5.0)
        rsm_range = (0.60, 0.95)
    else:
        depth_range = (4, 10)
        learning_rate_range = (0.01, 0.20)
        l2_range = (1.0, 50.0)
        random_strength_range = (0.0, 10.0)
        border_choices = [64, 128, 254]
        leaf_iterations_range = (1, 8)
        subsample_range = (0.55, 1.0)
        bagging_temperature_range = (0.0, 3.0)
        rsm_range = (0.60, 1.0)

    params: dict[str, Any] = {
        "loss_function": "Quantile:alpha=0.5",
        "eval_metric": "MAE",
        "iterations": trial.suggest_int("iterations", 300, max_iterations, step=100),
        "depth": trial.suggest_int("depth", *depth_range),
        "learning_rate": trial.suggest_float("learning_rate", *learning_rate_range, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", *l2_range, log=True),
        "random_strength": trial.suggest_float("random_strength", *random_strength_range),
        "bootstrap_type": bootstrap_type,
        "border_count": trial.suggest_categorical("border_count", border_choices),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Plain", "Ordered"]),
        "leaf_estimation_iterations": trial.suggest_int("leaf_estimation_iterations", *leaf_iterations_range),
        "random_seed": random_seed,
        "verbose": False,
        "allow_writing_files": False,
    }
    if has_time:
        params["has_time"] = True

    if bootstrap_type == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float(
            "bagging_temperature",
            *bagging_temperature_range,
        )
    else:
        params["subsample"] = trial.suggest_float("subsample", *subsample_range)

    if feature_count >= 40 or search_profile == "conservative":
        params["rsm"] = trial.suggest_float("rsm", *rsm_range)

    if task_type:
        params["task_type"] = task_type

    return params


def trials_to_frame(study: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row: dict[str, Any] = {
            "number": trial.number,
            "state": trial.state.name,
            "value": trial.value,
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        row.update({f"user_{key}": value for key, value in trial.user_attrs.items()})
        rows.append(row)
    return pd.DataFrame(rows)


def recency_sample_weights(
    frame: pd.DataFrame,
    *,
    reference_origin: pd.Timestamp,
    half_life_days: float | None,
    floor: float | None = None,
) -> pd.Series | None:
    """Return exponential recency weights for rows before a validation origin."""

    if half_life_days is None:
        return None
    validate_sample_weight_config(half_life_days, floor)
    time_column = "forecast_origin_utc" if "forecast_origin_utc" in frame.columns else "ds_utc"
    reference = pd.Timestamp(reference_origin)
    if reference.tzinfo is None:
        reference = reference.tz_localize("UTC")
    else:
        reference = reference.tz_convert("UTC")
    timestamps = pd.to_datetime(frame[time_column], utc=True)
    age_days = (reference - timestamps) / pd.Timedelta(days=1)
    weights = 0.5 ** (age_days.clip(lower=0) / float(half_life_days))
    if floor is not None:
        weights = float(floor) + (1 - float(floor)) * weights
    return weights.astype(float)


def validate_sample_weight_config(
    half_life_days: float | None,
    floor: float | None,
) -> None:
    if half_life_days is not None and half_life_days <= 0:
        raise ValueError("sample_weight_half_life_days must be positive")
    if floor is not None and not 0 <= floor <= 1:
        raise ValueError("sample_weight_floor must be between 0 and 1")
