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
