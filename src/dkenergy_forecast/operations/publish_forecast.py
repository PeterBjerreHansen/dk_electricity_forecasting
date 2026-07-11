from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dkenergy_forecast.backtesting.horizons import make_danish_delivery_day_horizon
from dkenergy_forecast.backtesting.rolling_origin import rolling_origin_backtest
from dkenergy_forecast.evaluation.summary import add_prediction_diagnostics
from dkenergy_forecast.io import load_price_panel
from dkenergy_forecast.layout import CHRONOS_LORA_WEATHER_MODEL_LABEL, PROJECT_ROOT
from dkenergy_forecast.models.chronos_production import (
    load_lora_artifact_manifest,
    weather_artifact_summary,
)
from dkenergy_forecast.models.registry import (
    WEIGHTED_MEDIAN_MODEL_LABEL,
    latest_publish_model_factories,
    production_model_specs,
)
from dkenergy_forecast.operations.contracts import (
    ForecastRequest,
    ProductionConfig,
    load_production_config,
    parse_delivery_date,
)
from dkenergy_forecast.publishing import (
    build_dashboard_payload,
    file_sha256,
    git_commit,
    make_forecast_run_manifest,
    unique_run_id,
    write_forecast_run_artifacts,
    write_latest_pointer,
)
from dkenergy_forecast.types import (
    COPENHAGEN_TZ,
    DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
    PRICE_AVAILABILITY_COLUMN,
    copenhagen_timestamp,
    to_utc_timestamp,
)


@dataclass(frozen=True)
class PublishedForecastResult:
    run_id: str
    request: ForecastRequest
    forecast_status: str
    published_model: str
    paths: dict[str, Path]

    @property
    def forecast_origin_utc(self) -> pd.Timestamp:
        return self.request.forecast_origin_utc


def run_publish_forecast(
    args: Any,
    *,
    project_root: Path = PROJECT_ROOT,
) -> PublishedForecastResult:
    """Publish one configured Chronos release, with one explicit fixed fallback."""

    generated_at = to_utc_timestamp(
        getattr(args, "generated_at_utc", None) or datetime.now(timezone.utc)
    )
    request = resolve_forecast_request(args, generated_at=generated_at)
    runtime_root = Path(getattr(args, "runtime_root", None) or project_root)
    production = load_production_config(
        getattr(args, "production_config", project_root / "config" / "production.json"),
        runtime_root=runtime_root,
        artifact_path_override=getattr(args, "chronos_model_artifact_path", None),
    )
    _validate_production_models(production)

    panel_path = Path(args.panel_path)
    qa_path = Path(args.qa_path) if getattr(args, "qa_path", None) else None
    panel = load_price_panel(
        panel_path,
        qa_path if qa_path and qa_path.exists() else None,
        require_final_historical=not args.allow_incomplete_panel,
    )
    if request.run_kind == "live":
        validate_live_price_context(
            panel,
            request.information_cutoff_utc,
            delivery_date_local=request.delivery_date_local,
        )

    primary_error: dict[str, str] | None = None
    try:
        predictions = _predict_one_model(
            production.primary_model,
            request=request,
            panel=panel,
            min_train_days=args.min_train_days,
            weather_features_long_path=args.weather_features_long_path,
            chronos_model_artifact_path=production.primary_artifact_path,
        )
        release = model_release_metadata(
            production.primary_model,
            artifact_path=production.primary_artifact_path,
        )
        forecast_status = "primary"
        published_model = production.primary_model
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as exc:
        primary_error = {
            "type": type(exc).__name__,
            "message": str(exc)[:2000],
        }
        predictions = _predict_one_model(
            production.fallback_model,
            request=request,
            panel=panel,
            min_train_days=args.min_train_days,
            weather_features_long_path=args.weather_features_long_path,
            chronos_model_artifact_path=production.primary_artifact_path,
        )
        release = model_release_metadata(production.fallback_model)
        forecast_status = "degraded"
        published_model = production.fallback_model

    predictions = add_prediction_diagnostics(predictions)
    predictions = _attach_release_identity(
        predictions,
        release=release,
        requested_model=production.primary_model,
        forecast_status=forecast_status,
    )
    run_id = getattr(args, "run_id", None) or default_run_id(
        request.run_kind,
        request.forecast_origin_utc,
        delivery_date_local=request.delivery_date_local,
    )
    predictions["run_id"] = run_id

    empty_scores = _empty_scores()
    run_dir = Path(args.artifact_root) / run_id
    artifact_paths = {
        "predictions": str(run_dir / "predictions.parquet"),
        "manifest": str(run_dir / "manifest.json"),
        "completion": str(run_dir / "COMPLETED.json"),
        "dashboard": str(run_dir / "forecast_dashboard.json"),
    }
    idempotency_key = (
        f"{request.run_kind}:{request.delivery_date_local.isoformat()}:"
        f"{request.information_cutoff_utc.isoformat()}:{release['model_release_id']}"
    )
    model_metadata = {
        "primary_model": production.primary_model,
        "fallback_model": production.fallback_model,
        "published_model": published_model,
        **release,
    }
    manifest = make_forecast_run_manifest(
        run_id=run_id,
        forecast_origin_utc=request.forecast_origin_utc,
        predictions=predictions,
        scores=None,
        artifact_paths=artifact_paths,
        dataset_version=sorted(panel["dataset_version"].dropna().unique().tolist()),
        git_commit_value=git_commit(project_root),
        status="prepared",
        created_at_utc=request.generated_at_utc,
        run_kind=request.run_kind,
        idempotency_key=idempotency_key,
        extra={
            "forecast_status": forecast_status,
            "delivery_date_local": request.delivery_date_local.isoformat(),
            "information_cutoff_utc": request.information_cutoff_utc,
            "decision_deadline_utc": request.decision_deadline_utc,
            "generated_at_utc": request.generated_at_utc,
            "panel_path": str(panel_path),
            "panel_sha256": file_sha256(panel_path),
            "qa_path": str(qa_path) if qa_path else None,
            "qa_sha256": file_sha256(qa_path) if qa_path and qa_path.exists() else None,
            "price_availability_policy": {
                "column": PRICE_AVAILABILITY_COLUMN,
                "publication_local_time": DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
                "timezone": COPENHAGEN_TZ,
                "eligibility_operator": "< information_cutoff_utc",
            },
            "min_train_days": int(args.min_train_days),
            "model": model_metadata,
            "primary_failure": primary_error,
            **_weather_metadata(published_model, args.weather_features_long_path),
        },
    )
    dashboard = build_dashboard_payload(
        predictions=predictions,
        scores=None,
        manifest=manifest,
    )
    written = write_forecast_run_artifacts(
        run_dir,
        predictions=predictions,
        scores=empty_scores,
        manifest=manifest,
        dashboard=dashboard,
    )

    latest: dict[str, Path] = {}
    if request.run_kind == "live":
        completion = json.loads(written["completion"].read_text(encoding="utf-8"))
        committed_at = to_utc_timestamp(completion["committed_at_utc"])
        if committed_at > request.decision_deadline_utc:
            raise RuntimeError(
                "Forecast run completed after its decision deadline and was not published: "
                f"committed_at_utc={committed_at.isoformat()}, "
                f"decision_deadline_utc={request.decision_deadline_utc.isoformat()}"
            )
        pointer_path = Path(
            getattr(args, "latest_pointer_path", None)
            or Path(args.artifact_root).parent / "latest.json"
        )
        latest["latest_pointer"] = write_latest_pointer(pointer_path, run_dir=run_dir)

    return PublishedForecastResult(
        run_id=run_id,
        request=request,
        forecast_status=forecast_status,
        published_model=published_model,
        paths={**written, **latest},
    )


def resolve_forecast_request(args: Any, *, generated_at: object) -> ForecastRequest:
    generated = to_utc_timestamp(generated_at)
    supplied_cutoff = getattr(args, "information_cutoff_utc", None)
    legacy_origin = getattr(args, "forecast_origin_utc", None)
    run_kind = resolve_run_kind(
        getattr(args, "run_kind", None),
        supplied_origin=legacy_origin or supplied_cutoff,
    )
    if supplied_cutoff or legacy_origin:
        information_cutoff = to_utc_timestamp(supplied_cutoff or legacy_origin)
    elif run_kind == "live":
        information_cutoff = generated
    else:
        raise ValueError("Replay runs require --information-cutoff-utc")

    cutoff_local_date = information_cutoff.tz_convert(COPENHAGEN_TZ).date()
    delivery_value = getattr(args, "delivery_date_local", None)
    delivery_date = (
        parse_delivery_date(delivery_value)
        if delivery_value
        else cutoff_local_date + timedelta(days=1)
    )
    deadline_value = getattr(args, "decision_deadline_utc", None) or getattr(
        args, "decision_cutoff_utc", None
    )
    if deadline_value:
        deadline = to_utc_timestamp(deadline_value)
    else:
        deadline_time = getattr(args, "decision_deadline_local_time", None) or "12:00"
        deadline = copenhagen_timestamp(cutoff_local_date, deadline_time).tz_convert("UTC")

    return ForecastRequest(
        delivery_date_local=delivery_date,
        information_cutoff_utc=information_cutoff,
        decision_deadline_utc=deadline,
        generated_at_utc=generated,
        run_kind=run_kind,
    )


def _predict_one_model(
    model_label: str,
    *,
    request: ForecastRequest,
    panel: pd.DataFrame,
    min_train_days: int,
    weather_features_long_path: str | Path,
    chronos_model_artifact_path: str | Path,
) -> pd.DataFrame:
    factories = latest_publish_model_factories(
        [model_label],
        weather_features_long_path=weather_features_long_path,
        chronos_model_artifact_path=chronos_model_artifact_path,
    )
    return publish_predictions_for_origins(
        panel=panel,
        origins=request.origin_frame(),
        factories=factories,
        min_train_days=min_train_days,
        delivery_date_local=request.delivery_date_local,
    )


def publish_predictions_for_origins(
    *,
    panel: pd.DataFrame,
    origins: pd.DataFrame,
    factories: dict[str, Any],
    min_train_days: int,
    delivery_date_local: object | None = None,
) -> pd.DataFrame:
    prediction_frames: list[pd.DataFrame] = []
    min_train_rows = min_train_days * 24 * panel["area"].nunique()
    for model_label, factory in factories.items():
        predictions = rolling_origin_backtest(
            model_factory=factory,
            panel=panel,
            origins=origins,
            horizon_builder=lambda panel_arg, origin_arg: make_danish_delivery_day_horizon(
                panel_arg,
                origin_arg,
                delivery_date_local=delivery_date_local,
                days_ahead=1,
            ),
            min_train_rows=min_train_rows,
        )
        predictions["model_label"] = model_label
        if model_label == CHRONOS_LORA_WEATHER_MODEL_LABEL and "model_version" in predictions:
            predictions["model_release_id"] = predictions["model_version"].astype(str)
        else:
            predictions["model_release_id"] = model_label
        prediction_frames.append(predictions)
    if not prediction_frames:
        raise ValueError("No production model was selected")
    return pd.concat(prediction_frames, ignore_index=True)


def model_release_metadata(
    model_label: str,
    *,
    artifact_path: str | Path | None = None,
) -> dict[str, str | None]:
    if model_label == CHRONOS_LORA_WEATHER_MODEL_LABEL:
        if artifact_path is None:
            raise ValueError("Chronos production requires an artifact path")
        manifest = load_lora_artifact_manifest(artifact_path)
        digest = manifest.get("artifact_content_sha256")
        if not isinstance(digest, str) or not digest:
            raise ValueError("Chronos manifest is missing artifact_content_sha256")
        release_id = manifest.get("release_id") or Path(artifact_path).name
        return {
            "model_name": model_label,
            "model_release_id": str(release_id),
            "model_artifact_sha256": digest,
            "model_artifact_path": str(artifact_path),
        }
    if model_label == WEIGHTED_MEDIAN_MODEL_LABEL:
        contract = (
            "weekday:lookback=42,half_life=4,floor=.10;"
            "weekend:lookback=56,half_life=28,floor=.20;min_periods=4"
        )
        return {
            "model_name": model_label,
            "model_release_id": "weighted_median_v1",
            "model_artifact_sha256": hashlib.sha256(contract.encode()).hexdigest(),
            "model_artifact_path": None,
        }
    raise ValueError(f"Unsupported production model: {model_label!r}")


def _attach_release_identity(
    predictions: pd.DataFrame,
    *,
    release: dict[str, str | None],
    requested_model: str,
    forecast_status: str,
) -> pd.DataFrame:
    output = predictions.copy()
    output["model_label"] = release["model_name"]
    output["model_name"] = release["model_name"]
    output["model_version"] = release["model_release_id"]
    output["model_release_id"] = release["model_release_id"]
    output["model_artifact_sha256"] = release["model_artifact_sha256"]
    output["requested_model"] = requested_model
    output["forecast_status"] = forecast_status
    return output


def _weather_metadata(model_label: str, path: str | Path) -> dict[str, object]:
    if model_label != CHRONOS_LORA_WEATHER_MODEL_LABEL:
        return {}
    return {"weather": weather_artifact_summary(path)}


def _validate_production_models(config: ProductionConfig) -> None:
    available = production_model_specs()
    if config.primary_model != CHRONOS_LORA_WEATHER_MODEL_LABEL:
        raise ValueError("The primary production model must be chronos_weather")
    if config.fallback_model != WEIGHTED_MEDIAN_MODEL_LABEL:
        raise ValueError("The production fallback must be weighted_median_v1")
    missing = sorted({config.primary_model, config.fallback_model} - set(available))
    if missing:
        raise ValueError(f"Production configuration references unknown models: {missing}")


def resolve_run_kind(value: str | None, *, supplied_origin: str | None) -> str:
    run_kind = value or ("replay" if supplied_origin else "live")
    if run_kind not in {"live", "replay"}:
        raise ValueError(f"Unsupported run_kind: {run_kind!r}")
    return run_kind


def default_run_id(
    run_kind: str,
    forecast_origin: object,
    *,
    delivery_date_local: object | None = None,
) -> str:
    if run_kind == "live":
        stamp = to_utc_timestamp(forecast_origin).strftime("%Y%m%dT%H%M%SZ")
        delivery = (
            parse_delivery_date(delivery_date_local).strftime("%Y%m%d")
            if delivery_date_local is not None
            else "unknown"
        )
        return f"live_{delivery}_{stamp}"
    return unique_run_id(run_kind)


def validate_live_price_context(
    panel: pd.DataFrame,
    forecast_origin: object,
    *,
    delivery_date_local: object | None = None,
) -> None:
    """Require the complete delivery day immediately preceding the forecast day."""

    origin = to_utc_timestamp(forecast_origin)
    context_date = None
    if delivery_date_local is not None:
        context_date = parse_delivery_date(delivery_date_local) - timedelta(days=1)
    expected = make_danish_delivery_day_horizon(
        panel,
        origin,
        delivery_date_local=context_date,
        days_ahead=0,
    )[["unique_id", "ds_utc"]]
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


def print_model_registry() -> None:
    specs = production_model_specs()
    print("Production model contract:")
    for label, role in [
        (CHRONOS_LORA_WEATHER_MODEL_LABEL, "primary"),
        (WEIGHTED_MEDIAN_MODEL_LABEL, "fallback"),
    ]:
        spec = specs[label]
        print(f"- {label}: {role}, {spec.description}")


def _empty_scores() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "model_label",
            "area",
            "mae",
            "rmse",
            "bias",
            "coverage",
            "interval_width",
        ]
    )
