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
from dkenergy_forecast.features.price_features import PriceFeatureConfig  # noqa: E402
from dkenergy_forecast.features.weather_features import build_weather_experiment_frame  # noqa: E402
from dkenergy_forecast.io import load_price_panel  # noqa: E402
from dkenergy_forecast.layout import PROJECT_ROOT, runtime_layout  # noqa: E402
from dkenergy_forecast.publishing import git_commit, json_safe  # noqa: E402


DEFAULT_LAYOUT = runtime_layout(PROJECT_ROOT)
DEFAULT_RECENT_DAYS = 45
DEFAULT_BACKTEST_DAYS = 180
DEFAULT_RECENT_OUTPUT = DEFAULT_LAYOUT.features / "weather_experiment_frame_recent.parquet"
DEFAULT_BACKTEST_OUTPUT = DEFAULT_LAYOUT.features / "weather_experiment_frame_backtest.parquet"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    days = resolve_days(args)
    output_path = resolve_output_path(args)
    panel_path = Path(args.panel_path)
    qa_path = Path(args.qa_path) if args.qa_path else None
    weather_path = Path(args.weather_features_long_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    panel = load_price_panel(
        panel_path,
        qa_path if qa_path and qa_path.exists() else None,
        require_final_historical=not args.allow_incomplete_panel,
    )
    weather = pd.read_parquet(weather_path)
    origins = choose_recent_complete_daily_origins(
        panel,
        days=days,
        at_hour_utc=args.at_hour_utc,
        forecast_local_time=args.forecast_local_time,
        max_origins=args.max_origins,
    )
    frame = build_weather_experiment_frame(
        panel,
        origins,
        weather,
        price_feature_config=PriceFeatureConfig(
            include_weighted_median_baseline=args.include_weighted_median_baseline,
        ),
        add_ensemble_features=not args.no_ensemble_features,
        add_derived_features=not args.no_derived_features,
    )
    frame.to_parquet(output_path, index=False)

    qa = make_qa(
        args=args,
        days=days,
        output_path=output_path,
        panel=panel,
        weather=weather,
        origins=origins,
        frame=frame,
        panel_path=panel_path,
        weather_path=weather_path,
    )
    qa_path_out = output_path.with_suffix(".qa.json")
    qa_path_out.write_text(json.dumps(json_safe(qa), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote weather backtest frame: {output_path}")
    print(f"Wrote QA report: {qa_path_out}")
    print(
        "Rows: {rows}; columns: {columns}; origins: {origins}; frame_kind={kind}; days={days}".format(
            rows=len(frame),
            columns=len(frame.columns),
            origins=len(origins),
            kind=args.frame_kind,
            days=days,
        )
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an availability-masked price + weather frame for historical "
            "model backtests. This is not a live forecast publishing artifact."
        )
    )
    parser.add_argument(
        "--panel-path",
        default=str(DEFAULT_LAYOUT.price_panel),
    )
    parser.add_argument(
        "--qa-path",
        default=str(DEFAULT_LAYOUT.price_panel_qa),
    )
    parser.add_argument(
        "--weather-features-long-path",
        default=str(DEFAULT_LAYOUT.weather_features_long),
    )
    parser.add_argument(
        "--frame-kind",
        choices=["recent", "backtest", "custom"],
        default="backtest",
        help=(
            "recent writes a short diagnostic frame; backtest writes a standard "
            "offline comparison frame; custom requires explicit --days and --output-path."
        ),
    )
    parser.add_argument(
        "--output-path",
        help="Output parquet path. Defaults depend on --frame-kind.",
    )
    parser.add_argument(
        "--days",
        type=int,
        help="Number of recent complete daily origins to include. Defaults depend on --frame-kind.",
    )
    parser.add_argument("--max-origins", type=int, default=0)
    parser.add_argument(
        "--at-hour-utc",
        type=int,
        help="Legacy fixed UTC forecast hour. Omit to use --forecast-local-time.",
    )
    parser.add_argument("--forecast-local-time", default="12:00")
    parser.add_argument(
        "--include-weighted-median-baseline",
        action="store_true",
        help=(
            "Add the slower weighted-median baseline feature. Leave off for "
            "generic weather coverage/backtest frames; enable for residual CatBoost experiments."
        ),
    )
    parser.add_argument("--no-ensemble-features", action="store_true")
    parser.add_argument("--no-derived-features", action="store_true")
    parser.add_argument("--allow-incomplete-panel", action="store_true")
    return parser.parse_args(argv)


def resolve_days(args: argparse.Namespace) -> int:
    if args.days is not None:
        days = int(args.days)
    elif args.frame_kind == "recent":
        days = DEFAULT_RECENT_DAYS
    elif args.frame_kind == "backtest":
        days = DEFAULT_BACKTEST_DAYS
    else:
        raise SystemExit("--frame-kind custom requires explicit --days")
    if days <= 0:
        raise SystemExit("--days must be positive")
    return days


def resolve_output_path(args: argparse.Namespace) -> Path:
    if args.output_path:
        return Path(args.output_path)
    if args.frame_kind == "recent":
        return DEFAULT_RECENT_OUTPUT
    if args.frame_kind == "backtest":
        return DEFAULT_BACKTEST_OUTPUT
    raise SystemExit("--frame-kind custom requires explicit --output-path")


def make_qa(
    *,
    args: argparse.Namespace,
    days: int,
    output_path: Path,
    panel: pd.DataFrame,
    weather: pd.DataFrame,
    origins: pd.DataFrame,
    frame: pd.DataFrame,
    panel_path: Path,
    weather_path: Path,
) -> dict[str, Any]:
    weather_feature_columns = [
        column
        for column in frame.columns
        if column.startswith("weather_")
        and not column.endswith("_coverage_ratio")
        and not column.endswith("_available_at_utc")
    ]
    return {
        "artifact_type": "weather_backtest_frame",
        "frame_kind": args.frame_kind,
        "days": int(days),
        "max_origins": int(args.max_origins),
        "at_hour_utc": None if args.at_hour_utc is None else int(args.at_hour_utc),
        "forecast_local_time": args.forecast_local_time,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_path": str(panel_path),
        "weather_features_long_path": str(weather_path),
        "output_path": str(output_path),
        "panel_dataset_version": sorted(panel["dataset_version"].dropna().unique().tolist()),
        "forecast_origin_min_utc": origins["forecast_origin_utc"].min(),
        "forecast_origin_max_utc": origins["forecast_origin_utc"].max(),
        "forecast_origin_count": int(len(origins)),
        "row_count": int(len(frame)),
        "column_count": int(len(frame.columns)),
        "weather_long_row_count": int(len(weather)),
        "weather_feature_column_count": int(len(weather_feature_columns)),
        "weather_feature_columns": sorted(weather_feature_columns),
        "include_weighted_median_baseline": bool(args.include_weighted_median_baseline),
        "add_ensemble_features": not args.no_ensemble_features,
        "add_derived_features": not args.no_derived_features,
        "git_commit": git_commit(ROOT),
    }


if __name__ == "__main__":
    main()
