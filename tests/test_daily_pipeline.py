from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_daily_pipeline_does_not_refresh_weather_implicitly() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_daily_pipeline.py", "--dry-run"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "fetch_open_meteo_previous_runs.py" not in result.stdout
    assert "build_open_meteo_weather_features.py" not in result.stdout
    assert "build_weather_backtest_frame.py" not in result.stdout
    assert "run_publish_forecast.py" in result.stdout
    assert "--weather-features-long-path" in result.stdout
    assert "weather_open_meteo_area_hourly_long_open_meteo_previous_runs_v1.parquet" in result.stdout


def test_daily_pipeline_accepts_allow_incomplete_panel_alias() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_daily_pipeline.py", "--allow-incomplete-panel", "--dry-run"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "run_publish_forecast.py" in result.stdout
    assert "--allow-incomplete-panel" in result.stdout


def test_daily_pipeline_allows_explicit_weather_refresh_skip() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_daily_pipeline.py", "--dry-run", "--skip-weather"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "fetch_open_meteo_previous_runs.py" not in result.stdout
    assert "build_open_meteo_weather_features.py" not in result.stdout
    assert "--weather-features-long-path" in result.stdout
    assert "weather_open_meteo_area_hourly_long_open_meteo_previous_runs_v1.parquet" in result.stdout


def test_daily_pipeline_refreshes_weather_sources_when_requested() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_daily_pipeline.py", "--dry-run", "--with-weather"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "fetch_open_meteo_previous_runs.py" in result.stdout
    assert "build_open_meteo_weather_features.py" in result.stdout
    assert "build_weather_backtest_frame.py" not in result.stdout


def test_daily_pipeline_does_not_allow_diagnostics_to_select_the_live_model() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_daily_pipeline.py",
            "--dry-run",
            "--models",
            "same_hour_last_week",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "fetch_open_meteo_previous_runs.py" not in result.stdout
    assert "build_weather_backtest_frame.py" not in result.stdout
    publish_line = next(
        line for line in result.stdout.splitlines() if "run_publish_forecast.py" in line
    )
    assert "--models" not in publish_line


def test_daily_pipeline_runtime_root_rewrites_runtime_artifact_paths(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_daily_pipeline.py",
            "--dry-run",
            "--runtime-root",
            str(tmp_path),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert f"--raw-dir {tmp_path}/data/raw/energi_data_service" in result.stdout
    assert f"--panel-path {tmp_path}/data/model_ready/price_panel_hourly_v1.parquet" in result.stdout
    assert f"--artifact-root {tmp_path}/artifacts/forecast_runs" in result.stdout
    assert f"--latest-pointer-path {tmp_path}/artifacts/latest.json" in result.stdout
    assert f"--runtime-root {tmp_path}" in result.stdout


def test_daily_pipeline_defaults_to_local_forecast_time(monkeypatch) -> None:
    monkeypatch.delenv("FORECAST_AT_HOUR_UTC", raising=False)
    result = subprocess.run(
        [sys.executable, "scripts/run_daily_pipeline.py", "--dry-run"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--forecast-local-time 10:00" in result.stdout
    assert "--at-hour-utc" not in result.stdout


def test_daily_pipeline_runs_recent_diagnostics_only_when_explicit() -> None:
    default = subprocess.run(
        [sys.executable, "scripts/run_daily_pipeline.py", "--dry-run"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    requested = subprocess.run(
        [
            sys.executable,
            "scripts/run_daily_pipeline.py",
            "--dry-run",
            "--with-diagnostics",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "run_recent_diagnostics.py" not in default.stdout
    assert "run_recent_diagnostics.py" in requested.stdout
