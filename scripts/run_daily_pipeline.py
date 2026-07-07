#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_WEATHER_FEATURES_LONG_PATH = (
    ROOT
    / "data"
    / "features"
    / "weather_open_meteo_area_hourly_long_v1.parquet"
)


def main() -> None:
    args = parse_args()
    commands = build_commands(args)
    for command in commands:
        run_command(command, dry_run=args.dry_run)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily file-based data refresh and forecast publish pipeline.")
    parser.add_argument("--eds-start", default=_env("EDS_START", "2024-07-01"))
    parser.add_argument("--eds-end", default=_env("EDS_END"))
    parser.add_argument("--open-meteo-start", default=_env("OPEN_METEO_START", "2024-07-01"))
    parser.add_argument("--open-meteo-end", default=_env("OPEN_METEO_END", _tomorrow_copenhagen()))
    parser.add_argument(
        "--at-hour-utc",
        type=int,
        default=_env_int("FORECAST_AT_HOUR_UTC"),
        help="Legacy fixed UTC forecast hour. Omit to use --forecast-local-time.",
    )
    parser.add_argument("--forecast-local-time", default=_env("FORECAST_LOCAL_TIME", "12:00"))
    parser.add_argument("--min-train-days", type=int, default=int(_env("MIN_TRAIN_DAYS", "60")))
    parser.add_argument("--score-days", type=int, default=int(_env("SCORE_DAYS", "14")))
    parser.add_argument("--score-max-origins", type=int, default=int(_env("SCORE_MAX_ORIGINS", "7")))
    parser.add_argument("--score-holdout-days", type=int, default=int(_env("SCORE_HOLDOUT_DAYS", "2")))
    parser.add_argument(
        "--with-weather",
        action="store_true",
        default=_env_bool("WITH_WEATHER", False),
        help="Explicitly fetch/build Open-Meteo source weather artifacts.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=_env_list("PUBLISH_MODELS"),
        help="Optional run_publish_forecast.py model labels. Defaults to the registry defaults.",
    )
    parser.add_argument(
        "--weather-features-long-path",
        default=_env("WEATHER_FEATURES_LONG_PATH", str(DEFAULT_WEATHER_FEATURES_LONG_PATH)),
        help="Open-Meteo long weather feature parquet passed to weather-aware publish models.",
    )
    parser.add_argument("--skip-price-ingest", action="store_true")
    parser.add_argument("--skip-weather", action="store_true", help="Disable weather even when WITH_WEATHER is set.")
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument(
        "--strict-panel",
        action="store_true",
        help="Require final_historical QA status instead of allowing incomplete live refreshes.",
    )
    parser.add_argument(
        "--allow-incomplete-panel",
        action="store_false",
        dest="strict_panel",
        help="Allow incomplete live refreshes. This is the default unless --strict-panel is supplied.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    python = sys.executable
    commands: list[list[str]] = []

    if not args.skip_price_ingest:
        fetch_prices = [
            python,
            str(ROOT / "scripts" / "fetch_eds_prices.py"),
            "--start",
            args.eds_start,
        ]
        if args.eds_end:
            fetch_prices.extend(["--end", args.eds_end])
        commands.append(fetch_prices)

    build_prices = [python, str(ROOT / "scripts" / "build_price_panel.py")]
    if not args.strict_panel:
        build_prices.append("--allow-incomplete-recent")
    commands.append(build_prices)

    refresh_weather = args.with_weather and not args.skip_weather

    if refresh_weather:
        commands.append(
            [
                python,
                str(ROOT / "scripts" / "fetch_open_meteo_previous_runs.py"),
                "--start",
                args.open_meteo_start,
                "--end",
                args.open_meteo_end,
            ]
        )
        commands.append([python, str(ROOT / "scripts" / "build_open_meteo_weather_features.py")])

    if not args.skip_backtest:
        baseline_backtest = [
            python,
            str(ROOT / "scripts" / "run_baseline_backtest.py"),
            "--forecast-local-time",
            str(args.forecast_local_time),
            "--min-train-days",
            str(args.min_train_days),
        ]
        if args.at_hour_utc is not None:
            baseline_backtest.extend(["--at-hour-utc", str(args.at_hour_utc)])
        if not args.strict_panel:
            baseline_backtest.append("--allow-incomplete-panel")
        commands.append(baseline_backtest)

    if not args.skip_publish:
        publish = [
            python,
            str(ROOT / "scripts" / "run_publish_forecast.py"),
            "--forecast-local-time",
            str(args.forecast_local_time),
            "--min-train-days",
            str(args.min_train_days),
            "--score-days",
            str(args.score_days),
            "--score-max-origins",
            str(args.score_max_origins),
            "--score-holdout-days",
            str(args.score_holdout_days),
        ]
        if args.at_hour_utc is not None:
            publish.extend(["--at-hour-utc", str(args.at_hour_utc)])
        if args.models:
            publish.extend(["--models", *args.models])
        publish.extend(["--weather-features-long-path", args.weather_features_long_path])
        if not args.strict_panel:
            publish.append("--allow-incomplete-panel")
        commands.append(publish)

    return commands


def run_command(command: list[str], *, dry_run: bool) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def _env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    return value if value else default


def _env_list(name: str) -> list[str] | None:
    value = os.environ.get(name)
    if not value:
        return None
    return shlex.split(value)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None = None) -> int | None:
    value = os.environ.get(name)
    return int(value) if value else default


def _tomorrow_copenhagen() -> str:
    return (datetime.now(ZoneInfo("Europe/Copenhagen")).date() + timedelta(days=1)).isoformat()


if __name__ == "__main__":
    main()
