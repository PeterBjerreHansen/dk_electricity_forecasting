#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.backtesting.origins import choose_recent_complete_daily_origins  # noqa: E402
from dkenergy_forecast.features.weather_features import build_weather_experiment_frame  # noqa: E402
from dkenergy_forecast.io import load_price_panel  # noqa: E402
from dkenergy_forecast.publishing import git_commit, json_safe  # noqa: E402


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel_path)
    qa_path = Path(args.qa_path) if args.qa_path else None
    weather_path = Path(args.weather_features_long_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panel = load_price_panel(
        panel_path,
        qa_path if qa_path and qa_path.exists() else None,
        require_final_historical=not args.allow_incomplete_panel,
    )
    weather = pd.read_parquet(weather_path)
    origins = choose_recent_complete_daily_origins(
        panel,
        days=args.days,
        at_hour_utc=args.at_hour_utc,
        max_origins=args.max_origins,
    )
    experiment = build_weather_experiment_frame(
        panel,
        origins,
        weather,
        add_ensemble_features=not args.no_ensemble_features,
    )
    experiment.to_parquet(output_path, index=False)

    qa = make_qa(
        args=args,
        panel=panel,
        weather=weather,
        origins=origins,
        experiment=experiment,
        panel_path=panel_path,
        weather_path=weather_path,
    )
    qa_path_out = output_path.with_suffix(".qa.json")
    qa_path_out.write_text(json.dumps(json_safe(qa), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote weather experiment frame: {output_path}")
    print(f"Wrote QA report: {qa_path_out}")
    print(f"Rows: {len(experiment)}; columns: {len(experiment.columns)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an availability-masked price + weather experiment frame.")
    parser.add_argument(
        "--panel-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.parquet"),
    )
    parser.add_argument(
        "--qa-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.qa.json"),
    )
    parser.add_argument(
        "--weather-features-long-path",
        default=str(
            ROOT
            / "data"
            / "features"
            / "weather_open_meteo_area_hourly_long_open_meteo_previous_runs_v1.parquet"
        ),
    )
    parser.add_argument(
        "--output-path",
        default=str(ROOT / "data" / "features" / "weather_experiment_frame_v1.parquet"),
    )
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument("--at-hour-utc", type=int, default=10)
    parser.add_argument("--no-ensemble-features", action="store_true")
    parser.add_argument("--allow-incomplete-panel", action="store_true")
    return parser.parse_args()


def make_qa(
    *,
    args: argparse.Namespace,
    panel: pd.DataFrame,
    weather: pd.DataFrame,
    origins: pd.DataFrame,
    experiment: pd.DataFrame,
    panel_path: Path,
    weather_path: Path,
) -> dict[str, Any]:
    weather_feature_columns = [
        column
        for column in experiment.columns
        if column.startswith("weather_")
        and not column.endswith("_coverage_ratio")
        and not column.endswith("_available_at_utc")
    ]
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_path": str(panel_path),
        "weather_features_long_path": str(weather_path),
        "output_path": str(args.output_path),
        "panel_dataset_version": sorted(panel["dataset_version"].dropna().unique().tolist()),
        "forecast_origin_min_utc": origins["forecast_origin_utc"].min(),
        "forecast_origin_max_utc": origins["forecast_origin_utc"].max(),
        "forecast_origin_count": int(len(origins)),
        "row_count": int(len(experiment)),
        "column_count": int(len(experiment.columns)),
        "weather_long_row_count": int(len(weather)),
        "weather_feature_column_count": int(len(weather_feature_columns)),
        "weather_feature_columns": sorted(weather_feature_columns),
        "git_commit": git_commit(ROOT),
    }


if __name__ == "__main__":
    main()
