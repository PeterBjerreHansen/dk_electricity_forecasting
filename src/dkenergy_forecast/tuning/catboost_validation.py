from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import pandas as pd

from dkenergy_forecast.backtesting.horizons import make_danish_delivery_day_horizon
from dkenergy_forecast.backtesting.rolling_origin import rolling_origin_backtest
from dkenergy_forecast.evaluation.summary import add_prediction_diagnostics, model_score_table
from dkenergy_forecast.features import tabular_feature_columns_for_set
from dkenergy_forecast.models.registry import baseline_model_factories
from dkenergy_forecast.publishing import json_safe
from dkenergy_forecast.tuning.catboost_common import (
    CatBoostTuningResult,
    recency_sample_weights,
    suggest_catboost_params,
    trials_to_frame,
)
from dkenergy_forecast.types import (
    PRICE_AVAILABILITY_COLUMN,
    ensure_price_availability,
    normalize_utc_column,
    price_available_before_mask,
    require_columns,
)


RESIDUAL_BASELINE_COLUMN = "baseline_wdwe_weighted_median_y_pred"
DEFAULT_CATBOOST_MODEL_PREFIX = "catboost"
FEATURE_FRAME_BASELINE_COLUMNS = {
    "same_hour_last_week": "lag_168h",
    "rolling_median_hour_weekend_56d": "seasonal_median_hour_weekend",
    "weighted_median_v1": RESIDUAL_BASELINE_COLUMN,
}


@dataclass(frozen=True)
class RecencyWeightSpec:
    label: str
    half_life_days: float | None = None
    floor: float | None = None


@dataclass(frozen=True)
class CatBoostCandidateSpec:
    feature_set: str
    target_mode: str
    search_profile: str = "conservative"
    recency: RecencyWeightSpec = RecencyWeightSpec("unweighted")

    @property
    def label(self) -> str:
        return candidate_label_for(
            feature_set=self.feature_set,
            target_mode=self.target_mode,
            search_profile=self.search_profile,
            recency_label=self.recency.label,
        )


@dataclass(frozen=True)
class CatBoostValidationConfig:
    validation_months: tuple[str, ...] = (
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
        "2026-06",
    )
    retune_every_months: int = 6
    retrain_every_days: int = 7
    n_trials: int = 12
    training_origin_days: int | None = None
    eval_origin_count: int = 14
    min_train_rows: int = 1000
    max_iterations: int = 1000
    early_stopping_rounds: int = 80
    random_seed: int = 42
    has_time: bool = True
    task_type: str | None = None
    timeout_seconds: int | None = None
    residual_baseline_column: str = RESIDUAL_BASELINE_COLUMN
    model_prefix: str = DEFAULT_CATBOOST_MODEL_PREFIX
    replay_all_candidates: bool = True


@dataclass(frozen=True)
class CatBoostValidationResult:
    candidate_tuning_scores: pd.DataFrame
    selected_validation_configs: pd.DataFrame
    catboost_predictions: pd.DataFrame
    catboost_replay_metadata: pd.DataFrame
    feature_importance: pd.DataFrame
    combined_model_scores: pd.DataFrame
    outer_month_model_scores: pd.DataFrame
    per_origin_model_scores: pd.DataFrame
    per_origin_deltas: pd.DataFrame

ProgressFn = Callable[[str], None]
CatBoostArtifactLevel = Literal["summary", "diagnostic", "audit"]
CATBOOST_ARTIFACT_LEVELS: tuple[CatBoostArtifactLevel, ...] = ("summary", "diagnostic", "audit")


def build_candidate_grid(
    *,
    feature_sets: list[str],
    target_modes: list[str],
    recency_specs: list[RecencyWeightSpec],
    search_profiles: list[str] | None = None,
) -> list[CatBoostCandidateSpec]:
    """Build the notebook-facing CatBoost candidate grid."""

    profiles = search_profiles or ["conservative"]
    return [
        CatBoostCandidateSpec(
            feature_set=feature_set,
            target_mode=target_mode,
            search_profile=search_profile,
            recency=recency,
        )
        for feature_set in feature_sets
        for target_mode in target_modes
        for search_profile in profiles
        for recency in recency_specs
    ]


def make_retune_month_blocks(
    months: list[str] | tuple[str, ...],
    *,
    retune_every_months: int = 6,
) -> list[tuple[str, ...]]:
    """Split validation months into the periods covered by one tuning decision."""

    if retune_every_months <= 0:
        raise ValueError("retune_every_months must be positive")
    ordered_months = sorted(dict.fromkeys(months))
    return [
        tuple(ordered_months[index : index + retune_every_months])
        for index in range(0, len(ordered_months), retune_every_months)
    ]


def run_catboost_validation(
    frame: pd.DataFrame,
    *,
    candidates: list[CatBoostCandidateSpec],
    config: CatBoostValidationConfig,
    baseline_predictions: pd.DataFrame | None = None,
    progress: ProgressFn | None = print,
) -> CatBoostValidationResult:
    """Retune on schedule, retrain on schedule, and score every validation origin."""

    validate_operation_config(config)
    prepared = prepare_nested_frame(frame)
    validation_months = [month for month in config.validation_months if month in set(prepared["outer_month"])]
    if not validation_months:
        raise ValueError(
            "None of config.validation_months are present in frame: "
            f"{list(config.validation_months)}"
        )

    candidate_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    replay_rows: list[dict[str, Any]] = []
    importance_frames: list[pd.DataFrame] = []

    for block_months in make_retune_month_blocks(
        validation_months,
        retune_every_months=config.retune_every_months,
    ):
        block_label = validation_block_label(block_months)
        retune_at = pd.Timestamp(f"{block_months[0]}-01T00:00:00Z")
        block_frame = prepared[prepared["outer_month"].isin(block_months)].copy()
        if progress:
            progress(
                f"Retune {block_label}: "
                f"{block_frame['forecast_origin_utc'].nunique()} validation origin(s), "
                f"retrain every {config.retrain_every_days} day(s)."
            )

        block_candidates = tune_candidates_for_retune(
            prepared[prepared["forecast_origin_utc"] < retune_at].copy(),
            candidates=candidates,
            config=config,
            retune_at=retune_at,
            validation_block=block_label,
            validation_months=block_months,
            progress=progress,
        )
        candidate_rows.extend(block_candidates)
        viable = [
            row
            for row in block_candidates
            if row["status"] == "ok" and math.isfinite(row["tuning_mae"])
        ]
        if not viable:
            raise ValueError(f"No viable CatBoost candidates for validation block {block_label}.")

        selected = min(viable, key=lambda row: (row["tuning_mae"], row["candidate_label"]))
        selected_rows.append(selected)

        replay_candidates = viable if config.replay_all_candidates else [selected]
        for candidate in replay_candidates:
            predictions, metadata, importance = replay_candidate_on_schedule(
                prepared,
                validation_origins=block_frame[["forecast_origin_utc"]].drop_duplicates(),
                candidate=candidate,
                selected_candidate_label=selected["candidate_label"],
                config=config,
            )
            prediction_frames.append(predictions)
            replay_rows.extend(metadata)
            if not importance.empty:
                importance_frames.append(importance)

    return validation_result(
        candidate_rows=candidate_rows,
        selected_rows=selected_rows,
        prediction_frames=prediction_frames,
        replay_rows=replay_rows,
        importance_frames=importance_frames,
        baseline_predictions=baseline_predictions,
        model_prefix=config.model_prefix,
    )


def tune_candidates_for_retune(
    frame: pd.DataFrame,
    *,
    candidates: list[CatBoostCandidateSpec],
    config: CatBoostValidationConfig,
    retune_at: pd.Timestamp,
    validation_block: str,
    validation_months: tuple[str, ...],
    progress: ProgressFn | None = print,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tuning_start = retune_at - pd.DateOffset(months=config.retune_every_months)
    tuning_end = retune_at

    for candidate in candidates:
        feature_columns = tabular_feature_columns_for_set(frame, candidate.feature_set)
        row: dict[str, Any] = {
            "retune_at_utc": retune_at,
            "validation_block": validation_block,
            "validation_months": tuple(validation_months),
            "validation_start_month": validation_months[0],
            "validation_end_month": validation_months[-1],
            "validation_month_count": len(validation_months),
            "tuning_validation_start_utc": tuning_start,
            "tuning_validation_end_utc": tuning_end,
            "candidate_label": candidate.label,
            "feature_set": candidate.feature_set,
            "target_mode": candidate.target_mode,
            "search_profile": candidate.search_profile,
            "recency_label": candidate.recency.label,
            "sample_weight_half_life_days": candidate.recency.half_life_days,
            "sample_weight_floor": candidate.recency.floor,
            "feature_count": len(feature_columns),
            "status": "ok",
            "tuning_mae": math.nan,
        }
        if not feature_columns:
            row.update(status="skipped", error_message="no feature columns")
            rows.append(row)
            continue
        if (
            candidate.target_mode == "residual_baseline"
            and config.residual_baseline_column not in frame.columns
        ):
            row.update(status="skipped", error_message=f"missing {config.residual_baseline_column}")
            rows.append(row)
            continue

        if progress:
            progress(
                f"  Tuning {candidate.label} on "
                f"{tuning_start:%Y-%m-%d}..{tuning_end:%Y-%m-%d} "
                f"({len(feature_columns)} features)..."
            )
        try:
            result = tune_candidate_on_block(
                frame,
                candidate=candidate,
                feature_columns=feature_columns,
                config=config,
                tuning_start=tuning_start,
                tuning_end=tuning_end,
            )
        except Exception as exc:
            row.update(status="failed", error_message=repr(exc))
            rows.append(row)
            if progress:
                progress(f"    failed: {exc!r}")
            continue

        row.update(
            tuning_mae=float(result.best_value),
            best_trial_number=int(result.best_trial_number),
            trial_count=int(result.trial_count),
            tuning_validation_origin_count=int(result.validation_origin_count),
            feature_columns=result.feature_columns,
            catboost_params=result.best_params,
            trials=result.trials,
        )
        rows.append(row)
    return rows


def tune_candidate_on_block(
    frame: pd.DataFrame,
    *,
    candidate: CatBoostCandidateSpec,
    feature_columns: list[str],
    config: CatBoostValidationConfig,
    tuning_start: pd.Timestamp,
    tuning_end: pd.Timestamp,
) -> CatBoostTuningResult:
    optuna = load_optuna()
    CatBoostRegressor, Pool = load_catboost()
    cat_features = [column for column in ["area"] if column in feature_columns]
    train = drop_unusable_target_rows(
        training_rows_for_origin(
            frame,
            origin=tuning_start,
            training_origin_days=config.training_origin_days,
        ),
        target_mode=candidate.target_mode,
        residual_baseline_column=config.residual_baseline_column,
    )
    validate = drop_unusable_target_rows(
        frame[
            (frame["forecast_origin_utc"] >= tuning_start)
            & (frame["forecast_origin_utc"] < tuning_end)
        ].copy(),
        target_mode=candidate.target_mode,
        residual_baseline_column=config.residual_baseline_column,
    )
    if len(train) < config.min_train_rows or validate.empty:
        raise ValueError(
            "Blocked tuning fold has insufficient data: "
            f"train_rows={len(train)}, validation_rows={len(validate)}"
        )

    sampler = optuna.samplers.TPESampler(seed=config.random_seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=min(5, max(1, config.n_trials // 3)),
        n_warmup_steps=1,
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    def objective(trial: Any) -> float:
        params = suggest_catboost_params(
            trial,
            feature_count=len(feature_columns),
            random_seed=config.random_seed,
            max_iterations=config.max_iterations,
            task_type=config.task_type,
            has_time=config.has_time,
            search_profile=candidate.search_profile,
        )
        try:
            model, train_fit, train_eval = fit_catboost_model(
                CatBoostRegressor,
                Pool,
                train,
                feature_columns=feature_columns,
                cat_features=cat_features,
                params=params,
                target_mode=candidate.target_mode,
                residual_baseline_column=config.residual_baseline_column,
                eval_origin_count=config.eval_origin_count,
                early_stopping_rounds=config.early_stopping_rounds,
                sample_weight_half_life_days=candidate.recency.half_life_days,
                sample_weight_floor=candidate.recency.floor,
                weight_reference_origin=tuning_start,
            )
            prediction = predict_catboost(
                model,
                Pool,
                validate,
                feature_columns=feature_columns,
                cat_features=cat_features,
                target_mode=candidate.target_mode,
                residual_baseline_column=config.residual_baseline_column,
            )
        except Exception as exc:
            trial.set_user_attr("failed_exception", repr(exc))
            return math.inf

        score = float((pd.Series(prediction, index=validate.index) - validate["y"]).abs().mean())
        trial.set_user_attr("validation_origin_count", int(validate["forecast_origin_utc"].nunique()))
        trial.set_user_attr("validation_rows", int(len(validate)))
        trial.set_user_attr("train_rows", int(len(train)))
        trial.set_user_attr("train_fit_rows", int(len(train_fit)))
        trial.set_user_attr("train_eval_rows", int(len(train_eval)))
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    study.optimize(objective, n_trials=config.n_trials, timeout=config.timeout_seconds)
    best_params = dict(study.best_trial.params)
    best_params["random_seed"] = config.random_seed
    if config.task_type:
        best_params["task_type"] = config.task_type

    return CatBoostTuningResult(
        feature_set=candidate.label,
        best_value=float(study.best_value),
        best_params=best_params,
        best_trial_number=int(study.best_trial.number),
        feature_columns=feature_columns,
        validation_origin_count=int(validate["forecast_origin_utc"].nunique()),
        trial_count=len(study.trials),
        trials=trials_to_frame(study),
    )


def replay_candidate_on_schedule(
    frame: pd.DataFrame,
    *,
    validation_origins: pd.DataFrame,
    candidate: dict[str, Any],
    selected_candidate_label: str,
    config: CatBoostValidationConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]], pd.DataFrame]:
    CatBoostRegressor, Pool = load_catboost()
    feature_columns = list(candidate["feature_columns"])
    cat_features = [column for column in ["area"] if column in feature_columns]
    params = catboost_replay_params(
        candidate["catboost_params"],
        random_seed=config.random_seed,
        has_time=config.has_time,
    )
    origins = [
        pd.Timestamp(origin).tz_convert("UTC")
        for origin in validation_origins["forecast_origin_utc"].sort_values().drop_duplicates()
    ]
    retrain_origins = scheduled_retrain_origins(origins, retrain_every_days=config.retrain_every_days)
    selected_by_tuning = candidate["candidate_label"] == selected_candidate_label
    model_label = f"{config.model_prefix}__{candidate['candidate_label']}"
    outputs: list[pd.DataFrame] = []
    meta_rows: list[dict[str, Any]] = []
    importance_rows: list[dict[str, Any]] = []

    for index, retrain_origin in enumerate(retrain_origins):
        next_retrain = retrain_origins[index + 1] if index + 1 < len(retrain_origins) else None
        segment_origins = [
            origin
            for origin in origins
            if origin >= retrain_origin and (next_retrain is None or origin < next_retrain)
        ]
        predict = frame[frame["forecast_origin_utc"].isin(segment_origins)].copy()
        predict = drop_unusable_prediction_rows(
            predict,
            target_mode=str(candidate["target_mode"]),
            residual_baseline_column=config.residual_baseline_column,
        )
        train = drop_unusable_target_rows(
            training_rows_for_origin(
                frame,
                origin=retrain_origin,
                training_origin_days=config.training_origin_days,
            ),
            target_mode=str(candidate["target_mode"]),
            residual_baseline_column=config.residual_baseline_column,
        )
        if len(train) < config.min_train_rows or predict.empty:
            raise ValueError(
                f"Not enough rows to replay {candidate['candidate_label']} at "
                f"{retrain_origin.isoformat()}: train={len(train)}, predict={len(predict)}"
            )

        model, train_fit, train_eval = fit_catboost_model(
            CatBoostRegressor,
            Pool,
            train,
            feature_columns=feature_columns,
            cat_features=cat_features,
            params=params,
            target_mode=str(candidate["target_mode"]),
            residual_baseline_column=config.residual_baseline_column,
            eval_origin_count=config.eval_origin_count,
            early_stopping_rounds=config.early_stopping_rounds,
            sample_weight_half_life_days=candidate["sample_weight_half_life_days"],
            sample_weight_floor=candidate["sample_weight_floor"],
            weight_reference_origin=retrain_origin,
        )
        prediction = predict_catboost(
            model,
            Pool,
            predict,
            feature_columns=feature_columns,
            cat_features=cat_features,
            target_mode=str(candidate["target_mode"]),
            residual_baseline_column=config.residual_baseline_column,
        )
        origin_predictions = predict[prediction_metadata_columns(predict)].copy()
        origin_predictions["y_pred"] = prediction
        origin_predictions["model_label"] = model_label
        origin_predictions["candidate_label"] = candidate["candidate_label"]
        origin_predictions["feature_set"] = candidate["feature_set"]
        origin_predictions["target_mode"] = candidate["target_mode"]
        origin_predictions["search_profile"] = candidate["search_profile"]
        origin_predictions["recency_label"] = candidate["recency_label"]
        origin_predictions["selected_by_tuning"] = selected_by_tuning
        origin_predictions["validation_block"] = candidate["validation_block"]
        origin_predictions["retrain_origin_utc"] = retrain_origin
        outputs.append(origin_predictions)

        train_fit_mae = replay_mae(
            model,
            Pool,
            train_fit,
            feature_columns=feature_columns,
            cat_features=cat_features,
            target_mode=str(candidate["target_mode"]),
            residual_baseline_column=config.residual_baseline_column,
        )
        train_eval_mae = replay_mae(
            model,
            Pool,
            train_eval,
            feature_columns=feature_columns,
            cat_features=cat_features,
            target_mode=str(candidate["target_mode"]),
            residual_baseline_column=config.residual_baseline_column,
        )
        rows_by_origin = predict.groupby("forecast_origin_utc").size().to_dict()
        for origin in segment_origins:
            meta_rows.append(
                {
                    "forecast_origin_utc": origin,
                    "outer_month": origin.strftime("%Y-%m"),
                    "retune_at_utc": candidate["retune_at_utc"],
                    "retrain_origin_utc": retrain_origin,
                    "validation_block": candidate["validation_block"],
                    "validation_months": candidate["validation_months"],
                    "model_label": model_label,
                    "candidate_label": candidate["candidate_label"],
                    "selected_by_tuning": selected_by_tuning,
                    "feature_set": candidate["feature_set"],
                    "target_mode": candidate["target_mode"],
                    "search_profile": candidate["search_profile"],
                    "recency_label": candidate["recency_label"],
                    "tuning_validation_mae": candidate["tuning_mae"],
                    "train_rows": int(len(train)),
                    "train_fit_rows": int(len(train_fit)),
                    "train_eval_rows": int(len(train_eval)),
                    "train_fit_mae": train_fit_mae,
                    "train_eval_mae": train_eval_mae,
                    "predict_rows": int(rows_by_origin.get(origin, 0)),
                    "best_iteration": int(model.get_best_iteration())
                    if model.get_best_iteration() is not None
                    else None,
                }
            )

        if hasattr(model, "get_feature_importance"):
            for feature, importance in zip(feature_columns, model.get_feature_importance()):
                importance_rows.append(
                    {
                        "model_label": model_label,
                        "candidate_label": candidate["candidate_label"],
                        "validation_block": candidate["validation_block"],
                        "retrain_origin_utc": retrain_origin,
                        "feature": feature,
                        "importance": float(importance),
                    }
                )

    return pd.concat(outputs, ignore_index=True), meta_rows, pd.DataFrame(importance_rows)


def fit_catboost_model(
    CatBoostRegressor: Any,
    Pool: Any,
    train: pd.DataFrame,
    *,
    feature_columns: list[str],
    cat_features: list[str],
    params: dict[str, Any],
    target_mode: str,
    residual_baseline_column: str,
    eval_origin_count: int,
    early_stopping_rounds: int,
    sample_weight_half_life_days: float | None,
    sample_weight_floor: float | None,
    weight_reference_origin: pd.Timestamp,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    train_fit, train_eval = split_train_eval(train, eval_origin_count=eval_origin_count)
    train_fit = sort_for_catboost_time(train_fit)
    train_eval = sort_for_catboost_time(train_eval)
    model = CatBoostRegressor(**params)
    fit_pool = Pool(
        train_fit[feature_columns],
        label=target_values(
            train_fit,
            target_mode=target_mode,
            residual_baseline_column=residual_baseline_column,
        ),
        weight=recency_sample_weights(
            train_fit,
            reference_origin=weight_reference_origin,
            half_life_days=sample_weight_half_life_days,
            floor=sample_weight_floor,
        ),
        cat_features=cat_features,
    )
    fit_kwargs: dict[str, Any] = {}
    if not train_eval.empty and early_stopping_rounds > 0:
        fit_kwargs = {
            "eval_set": Pool(
                train_eval[feature_columns],
                label=target_values(
                    train_eval,
                    target_mode=target_mode,
                    residual_baseline_column=residual_baseline_column,
                ),
                cat_features=cat_features,
            ),
            "early_stopping_rounds": early_stopping_rounds,
            "use_best_model": True,
        }
    model.fit(fit_pool, **fit_kwargs)
    return model, train_fit, train_eval


def predict_catboost(
    model: Any,
    Pool: Any,
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    cat_features: list[str],
    target_mode: str,
    residual_baseline_column: str,
) -> Any:
    prediction = model.predict(Pool(frame[feature_columns], cat_features=cat_features))
    if target_mode == "residual_baseline":
        prediction = prediction + frame[residual_baseline_column].to_numpy()
    return prediction


def validation_result(
    *,
    candidate_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    prediction_frames: list[pd.DataFrame],
    replay_rows: list[dict[str, Any]],
    importance_frames: list[pd.DataFrame],
    baseline_predictions: pd.DataFrame | None,
    model_prefix: str,
) -> CatBoostValidationResult:
    catboost_predictions = add_prediction_diagnostics(pd.concat(prediction_frames, ignore_index=True))
    candidate_scores = pd.DataFrame(candidate_rows)
    selected_configs = pd.DataFrame(selected_rows)
    replay_metadata = pd.DataFrame(replay_rows)
    feature_importance = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame(
            columns=[
                "model_label",
                "candidate_label",
                "validation_block",
                "retrain_origin_utc",
                "feature",
                "importance",
            ]
        )
    )

    combined = catboost_predictions
    if baseline_predictions is not None and not baseline_predictions.empty:
        base = baseline_predictions.copy()
        base["forecast_origin_utc"] = pd.to_datetime(base["forecast_origin_utc"], utc=True)
        if "outer_month" not in base.columns:
            base["outer_month"] = origin_month_labels(base["forecast_origin_utc"])
        combined = pd.concat([base, catboost_predictions], ignore_index=True)

    per_origin_scores = per_origin_model_score_table(combined)
    return CatBoostValidationResult(
        candidate_tuning_scores=candidate_scores,
        selected_validation_configs=selected_configs,
        catboost_predictions=catboost_predictions,
        catboost_replay_metadata=replay_metadata,
        feature_importance=feature_importance,
        combined_model_scores=model_score_table(combined),
        outer_month_model_scores=outer_month_score_table(combined),
        per_origin_model_scores=per_origin_scores,
        per_origin_deltas=per_origin_delta_table(
            per_origin_scores,
            replay_metadata=replay_metadata,
            catboost_model_prefix=model_prefix,
        ),
    )


def run_baseline_comparator(
    panel: pd.DataFrame,
    *,
    origins: pd.DataFrame,
    min_train_days: int,
) -> pd.DataFrame:
    """Run notebook comparison baselines, including optional feature-frame baselines."""

    frames = []
    factories = baseline_model_factories(include_optional=True)
    for model_label in FEATURE_FRAME_BASELINE_COLUMNS:
        factory = factories[model_label]
        predictions = rolling_origin_backtest(
            model_factory=factory,
            panel=panel,
            origins=origins,
            horizon_builder=lambda panel_arg, origin_arg: make_danish_delivery_day_horizon(
                panel_arg,
                origin_arg,
                days_ahead=1,
            ),
            min_train_rows=min_train_days * 24 * panel["area"].nunique(),
        )
        predictions["model_label"] = model_label
        frames.append(predictions)
    output = pd.concat(frames, ignore_index=True)
    output["outer_month"] = origin_month_labels(output["forecast_origin_utc"])
    return add_prediction_diagnostics(output)


def baseline_predictions_from_feature_frame(
    frame: pd.DataFrame,
    *,
    origins: pd.DataFrame | None = None,
    baseline_columns: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Build notebook baseline predictions from an existing policy feature frame.

    This is equivalent to the default production baselines when the frame was
    built with the default ``PriceFeatureConfig``. It avoids rerunning rolling
    baseline models over thousands of origins in notebook policy comparisons.
    """

    prepared = prepare_nested_frame(frame)
    mapping = baseline_columns or FEATURE_FRAME_BASELINE_COLUMNS
    missing_columns = sorted(set(mapping.values()) - set(prepared.columns))
    if missing_columns:
        raise ValueError(
            "Feature-frame baseline comparison is missing required column(s): "
            f"{missing_columns}"
        )

    if origins is not None:
        require_columns(origins, ["forecast_origin_utc"], "origins")
        selected_origins = set(
            pd.to_datetime(origins["forecast_origin_utc"], utc=True).drop_duplicates()
        )
        prepared = prepared[prepared["forecast_origin_utc"].isin(selected_origins)].copy()

    metadata_columns = [
        column
        for column in [
            "unique_id",
            "ds_utc",
            "forecast_origin_utc",
            "horizon",
            "y",
            "area",
            "ds_local",
            "local_date",
            "local_hour",
            "local_day_of_week",
            "local_month",
            "is_weekend",
            "is_dst",
            "utc_offset_hours",
            PRICE_AVAILABILITY_COLUMN,
            "dataset_version",
        ]
        if column in prepared.columns
    ]
    frames: list[pd.DataFrame] = []
    for model_label, feature_column in mapping.items():
        predictions = prepared[metadata_columns].copy()
        predictions["model_name"] = model_label
        predictions["model_version"] = "feature_frame_v1"
        predictions["y_pred"] = prepared[feature_column].to_numpy()
        predictions["model_label"] = model_label
        frames.append(predictions)

    output = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if output.empty:
        return output
    output = add_prediction_diagnostics(output)
    output["outer_month"] = origin_month_labels(output["forecast_origin_utc"])
    return output


def write_catboost_validation_artifacts(
    output_dir: Path,
    result: CatBoostValidationResult,
    *,
    manifest: dict[str, Any] | None = None,
    artifact_level: CatBoostArtifactLevel = "diagnostic",
) -> None:
    if artifact_level not in CATBOOST_ARTIFACT_LEVELS:
        raise ValueError(f"artifact_level must be one of {CATBOOST_ARTIFACT_LEVELS}; got {artifact_level!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_safe_frame(result.candidate_tuning_scores).to_parquet(
        output_dir / "candidate_tuning_scores.parquet",
        index=False,
    )
    parquet_safe_frame(result.selected_validation_configs).to_parquet(
        output_dir / "selected_configs.parquet",
        index=False,
    )
    result.combined_model_scores.to_parquet(output_dir / "model_scores.parquet", index=False)
    result.outer_month_model_scores.to_parquet(output_dir / "outer_month_scores.parquet", index=False)

    if artifact_level in {"diagnostic", "audit"}:
        selected_predictions = result.catboost_predictions[
            result.catboost_predictions.get("selected_by_tuning", pd.Series(False, index=result.catboost_predictions.index))
            .fillna(False)
            .astype(bool)
        ].copy()
        selected_predictions.to_parquet(output_dir / "selected_predictions.parquet", index=False)
        result.feature_importance.to_parquet(output_dir / "feature_importance.parquet", index=False)
        result.per_origin_model_scores.to_parquet(output_dir / "per_origin_scores.parquet", index=False)
        result.per_origin_deltas.to_parquet(output_dir / "per_origin_deltas.parquet", index=False)
        write_tuning_trials_jsonl(
            output_dir / "tuning_trials.jsonl",
            result.candidate_tuning_scores,
        )

    if artifact_level == "audit":
        parquet_safe_frame(result.catboost_replay_metadata).to_parquet(
            output_dir / "replay_metadata.parquet",
            index=False,
        )
        result.catboost_predictions.to_parquet(output_dir / "all_predictions.parquet", index=False)

    if manifest is not None:
        (output_dir / "run_manifest.json").write_text(
            json_safe_json(manifest),
            encoding="utf-8",
        )


def write_tuning_trials_jsonl(path: Path, candidate_tuning_scores: pd.DataFrame) -> None:
    """Write nested Optuna trial frames as one line-delimited JSON log."""

    records = list(tuning_trial_records(candidate_tuning_scores))
    if not records:
        path.write_text("", encoding="utf-8")
        return
    lines = [
        json.dumps(json_safe(record), sort_keys=True, allow_nan=False)
        for record in records
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tuning_trial_records(candidate_tuning_scores: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    metadata_columns = [
        "family",
        "policy_label",
        "validation_block",
        "validation_months",
        "retune_at_utc",
        "candidate_label",
        "feature_set",
        "target_mode",
        "search_profile",
        "recency_label",
        "sample_weight_half_life_days",
        "sample_weight_floor",
        "feature_count",
        "status",
    ]
    for candidate_row in candidate_tuning_scores.to_dict(orient="records"):
        trials = candidate_row.get("trials")
        if not isinstance(trials, pd.DataFrame) or trials.empty:
            continue
        base = {
            column: jsonl_value(candidate_row.get(column))
            for column in metadata_columns
            if column in candidate_row
        }
        for trial_row in trials.to_dict(orient="records"):
            params = {
                key.removeprefix("param_"): jsonl_value(value)
                for key, value in trial_row.items()
                if key.startswith("param_") and jsonl_value(value) is not None
            }
            user_attrs = {
                key.removeprefix("user_"): jsonl_value(value)
                for key, value in trial_row.items()
                if key.startswith("user_") and jsonl_value(value) is not None
            }
            records.append(
                {
                    **base,
                    "trial_number": jsonl_value(trial_row.get("number")),
                    "state": jsonl_value(trial_row.get("state")),
                    "value": jsonl_value(trial_row.get("value")),
                    "params": params,
                    "user_attrs": user_attrs,
                }
            )
    return records


def jsonl_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    try:
        if bool(missing):
            return None
    except (TypeError, ValueError):
        pass
    return value


def parquet_safe_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.drop(columns=["trials"], errors="ignore").copy()
    for column in output.columns:
        if output[column].map(is_nested_metadata_value).any():
            output[column] = output[column].map(json_metadata_value)
    return output


def is_nested_metadata_value(value: Any) -> bool:
    return isinstance(value, dict | list | tuple)


def json_metadata_value(value: Any) -> Any:
    if not is_nested_metadata_value(value):
        return value
    return json.dumps(json_safe(value), sort_keys=True)


def prepare_nested_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["forecast_origin_utc", "ds_utc", "y", "area"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"CatBoost validation frame is missing required columns: {missing}")
    prepared = frame.copy()
    prepared = normalize_utc_column(prepared, "forecast_origin_utc")
    prepared = normalize_utc_column(prepared, "ds_utc")
    prepared = ensure_price_availability(prepared)
    prepared["outer_month"] = origin_month_labels(prepared["forecast_origin_utc"])
    return prepared.sort_values(["forecast_origin_utc", "unique_id", "ds_utc"]).reset_index(drop=True)


def outer_origins_for_months(frame: pd.DataFrame, months: list[str] | tuple[str, ...]) -> pd.DataFrame:
    prepared = prepare_nested_frame(frame)
    return (
        prepared.loc[prepared["outer_month"].isin(months), ["forecast_origin_utc"]]
        .drop_duplicates()
        .sort_values("forecast_origin_utc")
        .reset_index(drop=True)
    )


def per_origin_model_score_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (origin, outer_month, model_label), frame in predictions.groupby(
        ["forecast_origin_utc", "outer_month", "model_label"],
        dropna=False,
    ):
        valid = frame[["y", "y_pred"]].dropna()
        rows.append(
            {
                "forecast_origin_utc": origin,
                "outer_month": outer_month,
                "model_label": model_label,
                "rows": int(len(frame)),
                "evaluated_rows": int(len(valid)),
                "mae": float(frame["abs_error"].mean()) if "abs_error" in frame else math.nan,
                "rmse": float((frame["squared_error"].mean()) ** 0.5)
                if "squared_error" in frame
                else math.nan,
                "bias": float(frame["error"].mean()) if "error" in frame else math.nan,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["forecast_origin_utc", "mae", "model_label"])
        .reset_index(drop=True)
    )


def outer_month_score_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (outer_month, model_label), frame in predictions.groupby(
        ["outer_month", "model_label"],
        dropna=False,
    ):
        rows.append(
            {
                "outer_month": outer_month,
                "model_label": model_label,
                "rows": int(len(frame)),
                "evaluated_rows": int(frame[["y", "y_pred"]].dropna().shape[0]),
                "mae": float(frame["abs_error"].mean()),
                "rmse": float((frame["squared_error"].mean()) ** 0.5),
                "bias": float(frame["error"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["outer_month", "mae", "model_label"]).reset_index(drop=True)


def per_origin_delta_table(
    per_origin_scores: pd.DataFrame,
    *,
    replay_metadata: pd.DataFrame,
    catboost_model_prefix: str,
) -> pd.DataFrame:
    is_catboost = per_origin_scores["model_label"].str.startswith(f"{catboost_model_prefix}__")
    cat = per_origin_scores[is_catboost].copy()
    baselines = per_origin_scores[~is_catboost].copy()
    if cat.empty or baselines.empty:
        return pd.DataFrame()
    best_baseline = (
        baselines.sort_values(["forecast_origin_utc", "mae", "model_label"])
        .groupby("forecast_origin_utc", as_index=False)
        .head(1)
        .rename(
            columns={
                "model_label": "best_baseline_label",
                "mae": "best_baseline_mae",
                "rmse": "best_baseline_rmse",
                "bias": "best_baseline_bias",
            }
        )
    )
    replay_columns = [
        "forecast_origin_utc",
        "model_label",
        "retune_at_utc",
        "retrain_origin_utc",
        "validation_block",
        "validation_months",
        "candidate_label",
        "selected_by_tuning",
        "feature_set",
        "target_mode",
        "search_profile",
        "recency_label",
        "tuning_validation_mae",
        "train_rows",
        "train_fit_rows",
        "train_eval_rows",
        "train_fit_mae",
        "train_eval_mae",
        "predict_rows",
        "best_iteration",
    ]
    replay = (
        replay_metadata.reindex(columns=replay_columns)
        if not replay_metadata.empty
        else pd.DataFrame(columns=replay_columns)
    )
    output = (
        cat.rename(columns={"mae": "catboost_mae", "rmse": "catboost_rmse", "bias": "catboost_bias"})
        .merge(
            best_baseline[
                [
                    "forecast_origin_utc",
                    "best_baseline_label",
                    "best_baseline_mae",
                    "best_baseline_rmse",
                    "best_baseline_bias",
                ]
            ],
            on="forecast_origin_utc",
            how="left",
        )
        .merge(replay, on=["forecast_origin_utc", "model_label"], how="left")
    )
    output["catboost_minus_best_baseline_mae"] = output["catboost_mae"] - output["best_baseline_mae"]
    output["validation_minus_tuning_mae"] = output["catboost_mae"] - output["tuning_validation_mae"]
    output["validation_minus_train_fit_mae"] = output["catboost_mae"] - output["train_fit_mae"]
    output["validation_minus_train_eval_mae"] = output["catboost_mae"] - output["train_eval_mae"]
    return (
        output.sort_values(["forecast_origin_utc", "catboost_minus_best_baseline_mae"])
        .reset_index(drop=True)
    )


def origin_month_labels(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, utc=True).dt.strftime("%Y-%m")


def training_rows_for_origin(
    frame: pd.DataFrame,
    *,
    origin: pd.Timestamp,
    training_origin_days: int | None,
) -> pd.DataFrame:
    origin = pd.Timestamp(origin).tz_convert("UTC")
    prepared = ensure_price_availability(frame)
    mask = (
        (prepared["forecast_origin_utc"] < origin)
        & price_available_before_mask(prepared, origin)
        & prepared["y"].notna()
    )
    if training_origin_days is not None:
        mask &= prepared["forecast_origin_utc"] >= origin - pd.Timedelta(days=training_origin_days)
    return prepared[mask].copy()


def split_train_eval(
    train: pd.DataFrame,
    *,
    eval_origin_count: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = train["forecast_origin_utc"].sort_values().drop_duplicates().tolist()
    if eval_origin_count <= 0 or len(origins) <= eval_origin_count + 1:
        return train, train.iloc[0:0].copy()
    eval_origins = set(origins[-eval_origin_count:])
    return (
        train[~train["forecast_origin_utc"].isin(eval_origins)].copy(),
        train[train["forecast_origin_utc"].isin(eval_origins)].copy(),
    )


def scheduled_retrain_origins(
    origins: list[pd.Timestamp],
    *,
    retrain_every_days: int,
) -> list[pd.Timestamp]:
    if retrain_every_days <= 0:
        raise ValueError("retrain_every_days must be positive")
    retrain_origins: list[pd.Timestamp] = []
    for origin in origins:
        if not retrain_origins or origin >= retrain_origins[-1] + pd.Timedelta(days=retrain_every_days):
            retrain_origins.append(origin)
    return retrain_origins


def replay_mae(
    model: Any,
    pool_cls: Any,
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    cat_features: list[str],
    target_mode: str,
    residual_baseline_column: str,
) -> float:
    if frame.empty:
        return math.nan
    prediction = predict_catboost(
        model,
        pool_cls,
        frame,
        feature_columns=feature_columns,
        cat_features=cat_features,
        target_mode=target_mode,
        residual_baseline_column=residual_baseline_column,
    )
    return float((pd.Series(prediction, index=frame.index) - frame["y"]).abs().mean())


def drop_unusable_target_rows(
    frame: pd.DataFrame,
    *,
    target_mode: str,
    residual_baseline_column: str,
) -> pd.DataFrame:
    if target_mode == "direct":
        return frame.dropna(subset=["y"])
    if residual_baseline_column not in frame.columns:
        raise ValueError(f"residual replay requires {residual_baseline_column}")
    return frame.dropna(subset=["y", residual_baseline_column])


def drop_unusable_prediction_rows(
    frame: pd.DataFrame,
    *,
    target_mode: str,
    residual_baseline_column: str,
) -> pd.DataFrame:
    if target_mode == "direct":
        return frame
    if residual_baseline_column not in frame.columns:
        raise ValueError(f"residual replay requires {residual_baseline_column}")
    missing = frame[residual_baseline_column].isna()
    if bool(missing.any()):
        raise ValueError(
            f"residual replay requires complete {residual_baseline_column}; "
            f"missing_rows={int(missing.sum())}"
        )
    return frame


def target_values(
    frame: pd.DataFrame,
    *,
    target_mode: str,
    residual_baseline_column: str = RESIDUAL_BASELINE_COLUMN,
) -> pd.Series:
    if target_mode == "direct":
        return frame["y"]
    return frame["y"] - frame[residual_baseline_column]


def sort_for_catboost_time(frame: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [
        column
        for column in ["forecast_origin_utc", "unique_id", "ds_utc"]
        if column in frame.columns
    ]
    return frame.sort_values(sort_columns).reset_index(drop=True) if sort_columns else frame


def catboost_replay_params(
    params: dict[str, Any],
    *,
    random_seed: int,
    has_time: bool,
) -> dict[str, Any]:
    ignored = {"loss_function", "eval_metric", "verbose", "allow_writing_files"}
    output = {key: value for key, value in params.items() if key not in ignored and value is not None}
    output.update(
        {
            "loss_function": "Quantile:alpha=0.5",
            "eval_metric": "MAE",
            "random_seed": output.get("random_seed", random_seed),
            "verbose": False,
            "allow_writing_files": False,
        }
    )
    if has_time:
        output["has_time"] = True
    return output


def candidate_label_for(
    *,
    feature_set: str,
    target_mode: str,
    search_profile: str,
    recency_label: str,
) -> str:
    return f"{feature_set}__{target_mode}__{search_profile}__{recency_label}"


def validation_block_label(months: tuple[str, ...]) -> str:
    if not months:
        raise ValueError("validation block cannot be empty")
    return months[0] if len(months) == 1 else f"{months[0]}..{months[-1]}"


def prediction_metadata_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in [
            "unique_id",
            "area",
            "ds_utc",
            "ds_local",
            "local_date",
            "local_hour",
            "forecast_origin_utc",
            "horizon",
            "y",
            "dataset_version",
            PRICE_AVAILABILITY_COLUMN,
            "outer_month",
        ]
        if column in frame.columns
    ]


def validate_operation_config(config: CatBoostValidationConfig) -> None:
    if config.retune_every_months <= 0:
        raise ValueError("retune_every_months must be positive")
    if config.retrain_every_days <= 0:
        raise ValueError("retrain_every_days must be positive")
    if config.training_origin_days is not None and config.training_origin_days <= 0:
        raise ValueError("training_origin_days must be positive when provided")


def json_safe_json(value: dict[str, Any]) -> str:
    return json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n"


def load_catboost() -> tuple[Any, Any]:
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise ImportError(
            "CatBoost validation requires CatBoost. Install it with "
            '`pip install -e ".[tuning]"` or `pip install catboost>=1.2 optuna>=4`.'
        ) from exc
    return CatBoostRegressor, Pool


def load_optuna() -> Any:
    try:
        import optuna
    except ImportError as exc:
        raise ImportError(
            "CatBoost validation requires Optuna. Install it with "
            '`pip install -e ".[tuning]"` or `pip install optuna>=4`.'
        ) from exc
    return optuna
