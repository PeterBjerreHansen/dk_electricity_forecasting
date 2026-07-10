from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

from dkenergy_forecast.backtesting.horizons import make_danish_delivery_day_horizon
from dkenergy_forecast.backtesting.origins import choose_recent_complete_daily_origins
from dkenergy_forecast.backtesting.rolling_origin import rolling_origin_backtest
from dkenergy_forecast.evaluation.summary import add_prediction_diagnostics, model_score_table
from dkenergy_forecast.io import load_price_panel
from dkenergy_forecast.layout import CHRONOS_LORA_WEATHER_MODEL_LABEL, PROJECT_ROOT
from dkenergy_forecast.models.chronos_production import (
    PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG,
    load_lora_artifact_manifest,
    weather_artifact_summary,
)
from dkenergy_forecast.models.registry import (
    latest_publish_model_factories,
    production_model_specs,
)
from dkenergy_forecast.publishing import (
    build_dashboard_payload,
    build_published_forecast_history,
    build_published_forecast_scores,
    git_commit,
    make_forecast_run_manifest,
    unique_run_id,
    update_latest_exports,
    write_forecast_run_artifacts,
    write_published_forecast_history,
)
from dkenergy_forecast.types import (
    COPENHAGEN_TZ,
    DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
    PRICE_AVAILABILITY_COLUMN,
    copenhagen_timestamp,
    parse_local_time,
    to_utc_timestamp,
)


@dataclass(frozen=True)
class PublishedForecastResult:
    run_id: str
    forecast_origin_utc: pd.Timestamp
    paths: dict[str, Path]


def run_publish_forecast(args: Any, *, project_root: Path = PROJECT_ROOT) -> PublishedForecastResult:
    factories = latest_publish_model_factories(
        args.models,
        weather_features_long_path=args.weather_features_long_path,
        chronos_model_artifact_path=args.chronos_model_artifact_path,
    )

    panel_path = Path(args.panel_path)
    qa_path = Path(args.qa_path) if args.qa_path else None
    panel = load_price_panel(
        panel_path,
        qa_path if qa_path and qa_path.exists() else None,
        require_final_historical=not args.allow_incomplete_panel,
    )
    forecast_origin = resolve_forecast_origin(
        panel,
        args.forecast_origin_utc,
        args.at_hour_utc,
        args.forecast_local_time,
    )

    model_labels = list(factories)
    predictions = publish_predictions_for_origins(
        panel=panel,
        origins=pd.DataFrame({"forecast_origin_utc": [forecast_origin]}),
        factories=factories,
        min_train_days=args.min_train_days,
    )
    predictions = add_prediction_diagnostics(predictions)
    run_id = args.run_id or unique_run_id("forecast")
    predictions["run_id"] = run_id
    score_origins = choose_recent_complete_daily_origins(
        panel,
        days=args.score_days,
        at_hour_utc=args.at_hour_utc,
        forecast_local_time=args.forecast_local_time,
        max_origins=args.score_max_origins,
        min_history_days=args.min_train_days,
        holdout_days=args.score_holdout_days,
    )
    score_predictions = publish_predictions_for_origins(
        panel=panel,
        origins=score_origins,
        factories=factories,
        min_train_days=args.min_train_days,
    )
    score_predictions = add_prediction_diagnostics(score_predictions)
    score_predictions["run_id"] = run_id
    scores = model_score_table(score_predictions)
    published_history_predictions = build_published_forecast_history(args.artifact_root, panel)
    published_history_scores = build_published_forecast_scores(published_history_predictions)
    run_dir = Path(args.artifact_root) / run_id
    artifact_paths = {
        "predictions": str(run_dir / "predictions.parquet"),
        "score_predictions": str(run_dir / "score_predictions.parquet"),
        "model_scores": str(run_dir / "model_scores.parquet"),
        "manifest": str(run_dir / "manifest.json"),
        "dashboard": str(run_dir / "forecast_dashboard.json"),
        "published_history_predictions": str(Path(args.published_history_dir) / "predictions.parquet"),
        "published_history_scores": str(Path(args.published_history_dir) / "model_scores.parquet"),
    }
    manifest = make_forecast_run_manifest(
        run_id=run_id,
        forecast_origin_utc=forecast_origin,
        predictions=predictions,
        scores=scores,
        artifact_paths=artifact_paths,
        dataset_version=sorted(panel["dataset_version"].dropna().unique().tolist()),
        git_commit_value=git_commit(project_root),
        extra={
            "panel_path": str(panel_path),
            "qa_path": str(qa_path) if qa_path else None,
            "at_hour_utc": None if args.at_hour_utc is None else int(args.at_hour_utc),
            "forecast_local_time": args.forecast_local_time,
            "price_availability_policy": {
                "column": PRICE_AVAILABILITY_COLUMN,
                "publication_local_time": DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
                "timezone": COPENHAGEN_TZ,
                "eligibility_operator": "< forecast_origin_utc",
            },
            "min_train_days": int(args.min_train_days),
            "score_origin_min_utc": score_origins["forecast_origin_utc"].min(),
            "score_origin_max_utc": score_origins["forecast_origin_utc"].max(),
            "score_origin_count": int(len(score_origins)),
            "score_prediction_row_count": int(len(score_predictions)),
            "published_history_prediction_row_count": int(len(published_history_predictions)),
            "published_history_score_row_count": int(len(published_history_scores)),
            "score_days": int(args.score_days),
            "score_max_origins": int(args.score_max_origins),
            "score_holdout_days": int(args.score_holdout_days),
            "published_history_score_source": "published_forecast_history",
            "model_registry_labels": model_labels,
            "model_registry": selected_model_registry_metadata(model_labels),
            **selected_chronos_metadata(model_labels, args.chronos_model_artifact_path),
            **selected_weather_metadata(model_labels, args.weather_features_long_path),
        },
    )
    dashboard = build_dashboard_payload(
        predictions=predictions,
        scores=scores,
        manifest=manifest,
        score_predictions=score_predictions,
        published_history_predictions=published_history_predictions,
        published_history_scores=published_history_scores,
    )

    written = write_forecast_run_artifacts(
        run_dir,
        predictions=predictions,
        scores=scores,
        manifest=manifest,
        score_predictions=score_predictions,
        dashboard=dashboard,
    )
    latest = update_latest_exports(
        latest_forecast_dir=args.latest_forecast_dir,
        recent_scores_dir=args.recent_scores_dir,
        dashboard_path=args.dashboard_path,
        predictions=predictions,
        scores=scores,
        manifest=manifest,
        dashboard=dashboard,
        score_predictions=score_predictions,
    )
    published_history = write_published_forecast_history(
        args.published_history_dir,
        predictions=published_history_predictions,
        scores=published_history_scores,
    )
    return PublishedForecastResult(
        run_id=run_id,
        forecast_origin_utc=forecast_origin,
        paths={**written, **latest, **published_history},
    )


def publish_predictions_for_origins(
    *,
    panel: pd.DataFrame,
    origins: pd.DataFrame,
    factories,
    min_train_days: int,
) -> pd.DataFrame:
    prediction_frames = []
    min_train_rows = min_train_days * 24 * panel["area"].nunique()
    for model_label, factory in factories.items():
        predictions = rolling_origin_backtest(
            model_factory=factory,
            panel=panel,
            origins=origins,
            horizon_builder=lambda panel_arg, origin_arg: make_danish_delivery_day_horizon(
                panel_arg,
                origin_arg,
                days_ahead=1,
            ),
            min_train_rows=min_train_rows,
        )
        predictions["model_label"] = model_label
        prediction_frames.append(predictions)

    return pd.concat(prediction_frames, ignore_index=True)


def print_model_registry() -> None:
    specs = production_model_specs()
    print("Registered production models:")
    for label, spec in specs.items():
        default_marker = "default" if spec.default_enabled else "optional"
        publish_marker = "latest-publish" if spec.supports_latest_publish else "backtest-only"
        extra_marker = f"extra={spec.required_extra}" if spec.required_extra else "extra=base"
        quantile_marker = "quantiles" if spec.emits_quantiles else "point"
        weather_marker = ", weather" if spec.requires_weather else ""
        print(
            f"- {label}: {spec.family}, {default_marker}, {publish_marker}, "
            f"{extra_marker}, {quantile_marker}{weather_marker}"
        )
        print(f"  {spec.description}")


def selected_model_registry_metadata(labels: list[str]) -> dict[str, dict[str, object]]:
    specs = production_model_specs()
    return {
        label: {
            "family": specs[label].family,
            "default_enabled": specs[label].default_enabled,
            "supports_latest_publish": specs[label].supports_latest_publish,
            "required_extra": specs[label].required_extra,
            "emits_quantiles": specs[label].emits_quantiles,
            "requires_weather": specs[label].requires_weather,
        }
        for label in labels
    }


def selected_weather_metadata(labels: list[str], weather_features_long_path: str) -> dict[str, object]:
    specs = production_model_specs()
    if not any(specs[label].requires_weather for label in labels):
        return {}
    return {
        "weather": weather_artifact_summary(weather_features_long_path),
    }


def selected_chronos_metadata(labels: list[str], chronos_model_artifact_path: str | Path) -> dict[str, object]:
    if CHRONOS_LORA_WEATHER_MODEL_LABEL not in labels:
        return {}
    config = replace(
        PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG,
        model_artifact_path=chronos_model_artifact_path,
    )
    manifest = load_lora_artifact_manifest(config.model_artifact_path)
    covariates = manifest.get("covariates", [])
    return {
        "chronos": {
            CHRONOS_LORA_WEATHER_MODEL_LABEL: {
                "model_artifact_path": str(config.model_artifact_path),
                "base_model_id": config.base_model_id,
                "context_length": int(config.context_length),
                "prediction_length": int(config.prediction_length),
                "artifact_schema_version": manifest.get("artifact_schema_version"),
                "covariate_count": len(covariates) if isinstance(covariates, list) else None,
            }
        }
    }


def resolve_forecast_origin(
    panel: pd.DataFrame,
    supplied_origin: str | None,
    at_hour_utc: int | None,
    forecast_local_time: str,
) -> pd.Timestamp:
    if supplied_origin:
        return to_utc_timestamp(supplied_origin)
    if at_hour_utc is not None:
        if not 0 <= at_hour_utc <= 23:
            raise ValueError("at_hour_utc must be between 0 and 23")
        return panel["ds_utc"].max().normalize() + pd.Timedelta(hours=at_hour_utc)

    local_time = parse_local_time(forecast_local_time)
    latest_local_date = panel["ds_utc"].max().tz_convert(COPENHAGEN_TZ).date()
    origin_local = copenhagen_timestamp(latest_local_date, local_time)
    return origin_local.tz_convert("UTC")
