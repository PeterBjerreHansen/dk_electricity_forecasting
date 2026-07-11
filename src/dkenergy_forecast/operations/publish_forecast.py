from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dkenergy_forecast.backtesting.horizons import make_danish_delivery_day_horizon
from dkenergy_forecast.backtesting.rolling_origin import rolling_origin_backtest
from dkenergy_forecast.evaluation.summary import add_prediction_diagnostics
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
    file_sha256,
    git_commit,
    make_forecast_run_manifest,
    unique_run_id,
    update_latest_exports,
    write_forecast_run_artifacts,
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
    generated_at = to_utc_timestamp(
        getattr(args, "generated_at_utc", None) or datetime.now(timezone.utc)
    )
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
        reference_time_utc=generated_at,
    )
    run_kind = resolve_run_kind(
        getattr(args, "run_kind", None),
        supplied_origin=args.forecast_origin_utc,
    )
    decision_cutoff = to_utc_timestamp(
        getattr(args, "decision_cutoff_utc", None) or forecast_origin
    )
    validate_live_deadline(
        run_kind=run_kind,
        generated_at=generated_at,
        decision_cutoff=decision_cutoff,
    )
    if run_kind == "live":
        validate_live_price_context(panel, forecast_origin)

    model_labels = list(factories)
    predictions = publish_predictions_for_origins(
        panel=panel,
        origins=pd.DataFrame({"forecast_origin_utc": [forecast_origin]}),
        factories=factories,
        min_train_days=args.min_train_days,
    )
    predictions = add_prediction_diagnostics(predictions)
    run_id = args.run_id or default_run_id(run_kind, forecast_origin)
    predictions["run_id"] = run_id
    scores = _read_optional_parquet(
        Path(args.recent_scores_dir) / "model_scores.parquet",
        columns=_empty_score_columns(),
    )
    score_predictions = _read_optional_parquet(
        Path(args.recent_scores_dir) / "predictions.parquet",
    )
    published_history_predictions = _read_optional_parquet(
        Path(args.published_history_dir) / "predictions.parquet",
    )
    published_history_scores = _read_optional_parquet(
        Path(args.published_history_dir) / "model_scores.parquet",
        columns=_empty_score_columns(),
    )
    published_at = to_utc_timestamp(datetime.now(timezone.utc))
    score_eligible = bool(
        run_kind in {"live", "shadow"} and published_at <= decision_cutoff
    )
    run_scores = pd.DataFrame(columns=_empty_score_columns())
    idempotency_key = (
        f"{run_kind}:{forecast_origin.isoformat()}:{','.join(sorted(model_labels))}"
    )
    run_dir = Path(args.artifact_root) / run_id
    artifact_paths = {
        "predictions": str(run_dir / "predictions.parquet"),
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
        scores=run_scores,
        artifact_paths=artifact_paths,
        dataset_version=sorted(panel["dataset_version"].dropna().unique().tolist()),
        git_commit_value=git_commit(project_root),
        idempotency_key=idempotency_key,
        extra={
            "panel_path": str(panel_path),
            "panel_sha256": file_sha256(panel_path),
            "qa_path": str(qa_path) if qa_path else None,
            "qa_sha256": file_sha256(qa_path) if qa_path and qa_path.exists() else None,
            "at_hour_utc": None if args.at_hour_utc is None else int(args.at_hour_utc),
            "forecast_local_time": args.forecast_local_time,
            "price_availability_policy": {
                "column": PRICE_AVAILABILITY_COLUMN,
                "publication_local_time": DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
                "timezone": COPENHAGEN_TZ,
                "eligibility_operator": "< forecast_origin_utc",
            },
            "min_train_days": int(args.min_train_days),
            "run_kind": run_kind,
            "decision_cutoff_utc": decision_cutoff,
            "generated_at_utc": generated_at,
            "published_at_utc": published_at,
            "score_eligible": score_eligible,
            "score_ineligibility_reason": (
                None
                if score_eligible
                else _score_ineligibility_reason(
                    run_kind=run_kind,
                    published_at=published_at,
                    decision_cutoff=decision_cutoff,
                )
            ),
            "diagnostics_source": "separate_recent_diagnostics_artifacts",
            "score_prediction_row_count": int(len(score_predictions)),
            "published_history_prediction_row_count": int(len(published_history_predictions)),
            "published_history_score_row_count": int(len(published_history_scores)),
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
        score_predictions=score_predictions if not score_predictions.empty else None,
        published_history_predictions=published_history_predictions,
        published_history_scores=published_history_scores,
    )

    written = write_forecast_run_artifacts(
        run_dir,
        predictions=predictions,
        scores=run_scores,
        manifest=manifest,
        score_predictions=None,
        dashboard=dashboard,
    )
    latest: dict[str, Path] = {}
    if run_kind == "live" and score_eligible:
        latest = update_latest_exports(
            latest_forecast_dir=args.latest_forecast_dir,
            recent_scores_dir=args.recent_scores_dir,
            dashboard_path=args.dashboard_path,
            predictions=predictions,
            scores=scores,
            manifest=manifest,
            dashboard=dashboard,
            score_predictions=None,
            write_recent_scores=False,
        )
    return PublishedForecastResult(
        run_id=run_id,
        forecast_origin_utc=forecast_origin,
        paths={**written, **latest},
    )


def resolve_run_kind(value: str | None, *, supplied_origin: str | None) -> str:
    run_kind = value or ("replay" if supplied_origin else "live")
    if run_kind not in {"live", "shadow", "replay"}:
        raise ValueError(f"Unsupported run_kind: {run_kind!r}")
    return run_kind


def default_run_id(run_kind: str, forecast_origin: object) -> str:
    if run_kind == "live":
        stamp = to_utc_timestamp(forecast_origin).strftime("%Y%m%dT%H%M%SZ")
        return f"live_{stamp}"
    return unique_run_id(run_kind)


def validate_live_deadline(
    *,
    run_kind: str,
    generated_at: object,
    decision_cutoff: object,
) -> None:
    generated = to_utc_timestamp(generated_at)
    cutoff = to_utc_timestamp(decision_cutoff)
    if run_kind == "live" and generated > cutoff:
        raise ValueError(
            "Live forecast generation started after its decision cutoff: "
            f"generated_at_utc={generated.isoformat()}, "
            f"decision_cutoff_utc={cutoff.isoformat()}. "
            "Use --run-kind replay for retrospective forecasts."
        )


def validate_live_price_context(panel: pd.DataFrame, forecast_origin: object) -> None:
    """Require a complete current delivery day before a live day-ahead run."""

    origin = to_utc_timestamp(forecast_origin)
    expected = make_danish_delivery_day_horizon(panel, origin, days_ahead=0)[
        ["unique_id", "ds_utc"]
    ]
    observed = panel[["unique_id", "ds_utc"]].copy()
    observed["ds_utc"] = pd.to_datetime(observed["ds_utc"], utc=True)
    missing = expected.merge(
        observed.drop_duplicates(),
        on=["unique_id", "ds_utc"],
        how="left",
        indicator=True,
    )
    missing = missing[missing["_merge"].eq("left_only")]
    if not missing.empty:
        sample = missing[["unique_id", "ds_utc"]].head(5).to_dict(orient="records")
        raise ValueError(
            "Live price context does not cover the complete current Danish delivery day: "
            f"missing_rows={len(missing)}, sample={sample}"
        )


def _score_ineligibility_reason(
    *,
    run_kind: str,
    published_at: pd.Timestamp,
    decision_cutoff: pd.Timestamp,
) -> str:
    if run_kind not in {"live", "shadow"}:
        return f"run_kind_{run_kind}_is_not_scoreable"
    if published_at > decision_cutoff:
        return "published_after_decision_cutoff"
    return "not_score_eligible"


def _read_optional_parquet(
    path: Path,
    *,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=columns)
    return pd.read_parquet(path)


def _empty_score_columns() -> list[str]:
    return [
        "model_label",
        "area",
        "rows",
        "evaluated_rows",
        "mae",
        "rmse",
        "bias",
        "coverage",
        "interval_width",
        "pinball_q10",
        "pinball_q50",
        "pinball_q90",
        "interval_score_80",
        "weighted_interval_score",
        "calibration_error",
        "missing_rate",
    ]


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
                "artifact_content_sha256": manifest.get("artifact_content_sha256"),
                "base_model_revision": manifest.get("base_model_revision"),
                "training_code_commit": manifest.get("git_commit"),
                "training_data_sha256": manifest.get("training_data_sha256"),
                "random_seed": manifest.get("random_seed"),
                "covariate_count": len(covariates) if isinstance(covariates, list) else None,
            }
        }
    }


def resolve_forecast_origin(
    panel: pd.DataFrame,
    supplied_origin: str | None,
    at_hour_utc: int | None,
    forecast_local_time: str,
    *,
    reference_time_utc: object | None = None,
) -> pd.Timestamp:
    if supplied_origin:
        return to_utc_timestamp(supplied_origin)
    if at_hour_utc is not None:
        if not 0 <= at_hour_utc <= 23:
            raise ValueError("at_hour_utc must be between 0 and 23")
        reference = to_utc_timestamp(reference_time_utc or panel["ds_utc"].max())
        return reference.normalize() + pd.Timedelta(hours=at_hour_utc)

    local_time = parse_local_time(forecast_local_time)
    reference = to_utc_timestamp(reference_time_utc or panel["ds_utc"].max())
    decision_date_local = reference.tz_convert(COPENHAGEN_TZ).date()
    origin_local = copenhagen_timestamp(decision_date_local, local_time)
    return origin_local.tz_convert("UTC")
