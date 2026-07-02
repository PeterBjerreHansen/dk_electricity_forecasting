#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.backtesting.horizons import make_danish_delivery_day_horizon  # noqa: E402
from dkenergy_forecast.backtesting.origins import choose_recent_complete_daily_origins  # noqa: E402
from dkenergy_forecast.backtesting.rolling_origin import rolling_origin_backtest  # noqa: E402
from dkenergy_forecast.evaluation.summary import (  # noqa: E402
    add_prediction_diagnostics,
    model_score_table,
)
from dkenergy_forecast.io import load_price_panel  # noqa: E402
from dkenergy_forecast.models.registry import (  # noqa: E402
    default_production_model_labels,
    latest_publish_model_factories,
    production_model_specs,
)
from dkenergy_forecast.publishing import (  # noqa: E402
    build_dashboard_payload,
    git_commit,
    make_forecast_run_manifest,
    unique_run_id,
    update_latest_exports,
    write_forecast_run_artifacts,
)
from dkenergy_forecast.types import to_utc_timestamp  # noqa: E402


def main() -> None:
    args = parse_args()
    if args.list_models:
        print_model_registry()
        return

    panel_path = Path(args.panel_path)
    qa_path = Path(args.qa_path) if args.qa_path else None
    panel = load_price_panel(
        panel_path,
        qa_path if qa_path and qa_path.exists() else None,
        require_final_historical=not args.allow_incomplete_panel,
    )
    forecast_origin = resolve_forecast_origin(panel, args.forecast_origin_utc, args.at_hour_utc)

    try:
        factories = latest_publish_model_factories(args.models)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

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
    run_dir = Path(args.artifact_root) / run_id
    artifact_paths = {
        "predictions": str(run_dir / "predictions.parquet"),
        "model_scores": str(run_dir / "model_scores.parquet"),
        "manifest": str(run_dir / "manifest.json"),
        "dashboard": str(run_dir / "forecast_dashboard.json"),
    }
    manifest = make_forecast_run_manifest(
        run_id=run_id,
        forecast_origin_utc=forecast_origin,
        predictions=predictions,
        scores=scores,
        artifact_paths=artifact_paths,
        dataset_version=sorted(panel["dataset_version"].dropna().unique().tolist()),
        git_commit_value=git_commit(ROOT),
        extra={
            "panel_path": str(panel_path),
            "qa_path": str(qa_path) if qa_path else None,
            "at_hour_utc": int(args.at_hour_utc),
            "min_train_days": int(args.min_train_days),
            "score_origin_min_utc": score_origins["forecast_origin_utc"].min(),
            "score_origin_max_utc": score_origins["forecast_origin_utc"].max(),
            "score_origin_count": int(len(score_origins)),
            "score_days": int(args.score_days),
            "score_max_origins": int(args.score_max_origins),
            "score_holdout_days": int(args.score_holdout_days),
            "model_registry_labels": model_labels,
        },
    )
    dashboard = build_dashboard_payload(
        predictions=predictions,
        scores=scores,
        manifest=manifest,
    )

    written = write_forecast_run_artifacts(
        run_dir,
        predictions=predictions,
        scores=scores,
        manifest=manifest,
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
    )

    print(f"Published forecast run: {run_id}")
    print(f"Forecast origin UTC: {forecast_origin.isoformat()}")
    for label, path in {**written, **latest}.items():
        print(f"Wrote {label}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a file-based forecast run for the dashboard API path.")
    parser.add_argument(
        "--panel-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.parquet"),
    )
    parser.add_argument(
        "--qa-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.qa.json"),
    )
    parser.add_argument(
        "--forecast-origin-utc",
        help="Forecast origin timestamp. Defaults to the latest panel UTC date at --at-hour-utc.",
    )
    parser.add_argument("--at-hour-utc", type=int, default=10)
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument(
        "--score-days",
        type=int,
        default=14,
        help="Lookback window for recent completed-origin scoring.",
    )
    parser.add_argument(
        "--score-max-origins",
        type=int,
        default=7,
        help="Maximum recent completed origins used for model scores.",
    )
    parser.add_argument(
        "--score-holdout-days",
        type=int,
        default=2,
        help="Holdout days between latest panel timestamp and scoring origins.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Production model labels to publish. Defaults to baseline models only: "
            f"{default_production_model_labels()}."
        ),
    )
    parser.add_argument("--list-models", action="store_true", help="Print registered production models and exit.")
    parser.add_argument("--run-id", help="Optional explicit immutable forecast run id.")
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "artifacts" / "forecast_runs"),
    )
    parser.add_argument(
        "--latest-forecast-dir",
        default=str(ROOT / "results" / "latest_forecast"),
    )
    parser.add_argument(
        "--recent-scores-dir",
        default=str(ROOT / "results" / "recent_scores"),
    )
    parser.add_argument(
        "--dashboard-path",
        default=str(ROOT / "app_data" / "forecast_dashboard.json"),
    )
    parser.add_argument(
        "--allow-incomplete-panel",
        action="store_true",
        help="Allow QA artifacts that are not marked final_historical.",
    )
    return parser.parse_args()


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
        print(f"- {label}: {spec.family}, {default_marker}, {publish_marker}")
        print(f"  {spec.description}")


def resolve_forecast_origin(
    panel: pd.DataFrame,
    supplied_origin: str | None,
    at_hour_utc: int,
) -> pd.Timestamp:
    if supplied_origin:
        return to_utc_timestamp(supplied_origin)
    if not 0 <= at_hour_utc <= 23:
        raise ValueError("at_hour_utc must be between 0 and 23")
    return panel["ds_utc"].max().normalize() + pd.Timedelta(hours=at_hour_utc)


if __name__ == "__main__":
    main()
