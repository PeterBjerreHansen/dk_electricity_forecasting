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


def test_daily_pipeline_passes_model_selection_to_publish_without_weather_refresh() -> None:
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
    assert "--models same_hour_last_week" in result.stdout


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
    assert f"--published-history-dir {tmp_path}/results/published_forecast_history" in result.stdout
    assert f"--dashboard-path {tmp_path}/app_data/forecast_dashboard.json" in result.stdout
    assert (
        f"--chronos-model-artifact-path {tmp_path}/artifacts/models/"
        "chronos2_lora_calendar_weather_ctx1024_v1"
    ) in result.stdout


def test_daily_pipeline_defaults_to_local_forecast_time(monkeypatch) -> None:
    monkeypatch.delenv("FORECAST_AT_HOUR_UTC", raising=False)
    result = subprocess.run(
        [sys.executable, "scripts/run_daily_pipeline.py", "--dry-run"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "--forecast-local-time 12:00" in result.stdout
    assert "--at-hour-utc" not in result.stdout
