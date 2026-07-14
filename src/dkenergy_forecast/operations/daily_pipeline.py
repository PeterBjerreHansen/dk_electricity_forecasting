from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dkenergy_forecast.layout import PROJECT_ROOT, RuntimeLayout, runtime_layout


DEFAULT_WEATHER_LOOKBACK_DAYS = 90
DEFAULT_PRICE_LOOKBACK_DAYS = 450


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    commands = build_commands(args)
    for command in commands:
        run_command(command, dry_run=args.dry_run)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the daily file-based data refresh and forecast publish pipeline.")
    parser.add_argument(
        "--eds-start",
        default=_env("EDS_START", _lookback_month_start_copenhagen(DEFAULT_PRICE_LOOKBACK_DAYS)),
    )
    parser.add_argument("--eds-end", default=_env("EDS_END"))
    parser.add_argument(
        "--open-meteo-start",
        default=_env(
            "OPEN_METEO_START",
            _lookback_month_start_copenhagen(DEFAULT_WEATHER_LOOKBACK_DAYS),
        ),
    )
    parser.add_argument("--open-meteo-end", default=_env("OPEN_METEO_END", _tomorrow_copenhagen()))
    parser.add_argument(
        "--runtime-root",
        default=_env("DKENERGY_RUNTIME_ROOT"),
        help="Optional root for generated data, results, and artifacts. Defaults to the repository root.",
    )
    parser.add_argument(
        "--at-hour-utc",
        type=int,
        default=_env_int("FORECAST_AT_HOUR_UTC"),
        help="Legacy fixed UTC forecast hour. Omit to use --forecast-local-time.",
    )
    parser.add_argument("--forecast-local-time", default=_env("FORECAST_LOCAL_TIME", "10:00"))
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
        help="Optional models for background diagnostics; live production ignores this list.",
    )
    parser.add_argument(
        "--weather-features-long-path",
        default=_env("WEATHER_FEATURES_LONG_PATH"),
        help="Open-Meteo long weather feature parquet passed to weather-aware publish models.",
    )
    parser.add_argument(
        "--chronos-model-artifact-path",
        default=_env("DKENERGY_CHRONOS_MODEL_ARTIFACT_PATH"),
        help="Local trained Chronos LoRA artifact path passed to production publishing.",
    )
    parser.add_argument(
        "--production-config",
        default=_env(
            "DKENERGY_PRODUCTION_CONFIG",
            str(PROJECT_ROOT / "config" / "production.json"),
        ),
        help="Source-controlled live primary/fallback configuration.",
    )
    parser.add_argument(
        "--run-kind",
        choices=["live", "replay"],
        default=_env("DKENERGY_RUN_KIND", "live"),
        help="Publish a live forecast or an explicitly historical replay.",
    )
    parser.add_argument(
        "--information-cutoff-utc",
        default=_env("DKENERGY_INFORMATION_CUTOFF_UTC"),
        help="Historical information cutoff. Required by the publish command for replay runs.",
    )
    parser.add_argument("--skip-price-ingest", action="store_true")
    parser.add_argument(
        "--skip-weather",
        action="store_true",
        help="Disable weather even when WITH_WEATHER is set.",
    )
    parser.add_argument("--skip-backtest", action="store_true")
    parser.add_argument("--skip-publish", action="store_true")
    parser.add_argument(
        "--with-diagnostics",
        action="store_true",
        default=_env_bool("WITH_DIAGNOSTICS", False),
        help="Run recent model diagnostics after live publication. Disabled by default.",
    )
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
    return parser.parse_args(argv)


def build_commands(args: argparse.Namespace) -> list[list[str]]:
    run_kind = getattr(args, "run_kind", "live")
    information_cutoff_utc = getattr(args, "information_cutoff_utc", None)
    if run_kind == "replay" and not information_cutoff_utc:
        raise ValueError("Replay daily runs require --information-cutoff-utc")
    if run_kind == "live" and information_cutoff_utc:
        raise ValueError("Live daily runs cannot set --information-cutoff-utc")

    python = sys.executable
    paths = _runtime_layout(args.runtime_root)
    weather_features_long_path = args.weather_features_long_path or str(paths.weather_features_long)
    chronos_model_artifact_path = args.chronos_model_artifact_path or str(paths.chronos_model_artifact)
    commands: list[list[str]] = []

    if not args.skip_price_ingest:
        fetch_prices = [
            python,
            str(PROJECT_ROOT / "scripts" / "fetch_eds_prices.py"),
            "--start",
            args.eds_start,
            "--raw-dir",
            str(paths.eds_raw),
        ]
        if args.eds_end:
            fetch_prices.extend(["--end", args.eds_end])
        commands.append(fetch_prices)

    build_prices = [
        python,
        str(PROJECT_ROOT / "scripts" / "build_price_panel.py"),
        "--raw-dir",
        str(paths.eds_raw),
        "--normalized-dir",
        str(paths.normalized),
        "--model-ready-dir",
        str(paths.model_ready),
        "--start",
        args.eds_start,
    ]
    if args.eds_end:
        build_prices.extend(["--end", args.eds_end])
    if not args.strict_panel:
        build_prices.append("--allow-incomplete-recent")
    commands.append(build_prices)

    refresh_weather = args.with_weather and not args.skip_weather

    if refresh_weather:
        commands.append(
            [
                python,
                str(PROJECT_ROOT / "scripts" / "fetch_open_meteo_previous_runs.py"),
                "--start",
                args.open_meteo_start,
                "--end",
                args.open_meteo_end,
                "--raw-dir",
                str(paths.open_meteo_raw),
            ]
        )
        commands.append(
            [
                python,
                str(PROJECT_ROOT / "scripts" / "build_open_meteo_weather_features.py"),
                "--start",
                args.open_meteo_start,
                "--end",
                args.open_meteo_end,
                "--raw-dir",
                str(paths.open_meteo_raw),
                "--normalized-dir",
                str(paths.normalized),
                "--features-dir",
                str(paths.features),
            ]
        )

    if not args.skip_backtest:
        baseline_backtest = [
            python,
            str(PROJECT_ROOT / "scripts" / "run_baseline_backtest.py"),
            "--panel-path",
            str(paths.price_panel),
            "--qa-path",
            str(paths.price_panel_qa),
            "--output-dir",
            str(paths.baseline_results),
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
            str(PROJECT_ROOT / "scripts" / "run_publish_forecast.py"),
            "--panel-path",
            str(paths.price_panel),
            "--qa-path",
            str(paths.price_panel_qa),
            "--artifact-root",
            str(paths.forecast_runs),
            "--latest-pointer-path",
            str(paths.latest_pointer),
            "--runtime-root",
            str(paths.root),
            "--production-config",
            str(args.production_config),
            "--min-train-days",
            str(args.min_train_days),
        ]
        publish.extend(["--weather-features-long-path", weather_features_long_path])
        publish.extend(["--run-kind", run_kind])
        if information_cutoff_utc:
            publish.extend(["--information-cutoff-utc", information_cutoff_utc])
        if args.chronos_model_artifact_path:
            publish.extend(["--chronos-model-artifact-path", chronos_model_artifact_path])
        if not args.strict_panel:
            publish.append("--allow-incomplete-panel")
        commands.append(publish)

    if args.with_diagnostics:
        diagnostics = [
            python,
            str(PROJECT_ROOT / "scripts" / "run_recent_diagnostics.py"),
            "--panel-path",
            str(paths.price_panel),
            "--qa-path",
            str(paths.price_panel_qa),
            "--output-dir",
            str(paths.recent_scores),
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
            "--weather-features-long-path",
            weather_features_long_path,
            "--chronos-model-artifact-path",
            chronos_model_artifact_path,
        ]
        if args.at_hour_utc is not None:
            diagnostics.extend(["--at-hour-utc", str(args.at_hour_utc)])
        if args.models:
            diagnostics.extend(["--models", *args.models])
        if not args.strict_panel:
            diagnostics.append("--allow-incomplete-panel")
        commands.append(diagnostics)

    return commands


def run_command(command: list[str], *, dry_run: bool) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _runtime_layout(runtime_root: str | Path | None) -> RuntimeLayout:
    return runtime_layout(Path(runtime_root) if runtime_root else PROJECT_ROOT)


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


def _lookback_month_start_copenhagen(days: int) -> str:
    lookback_date = datetime.now(ZoneInfo("Europe/Copenhagen")).date() - timedelta(days=days)
    return lookback_date.replace(day=1).isoformat()
