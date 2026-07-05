from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_daily_pipeline_skips_weather_by_default() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_daily_pipeline.py", "--dry-run"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "fetch_open_meteo_previous_runs.py" not in result.stdout
    assert "build_weather_backtest_frame.py" not in result.stdout
    assert "run_publish_forecast.py" in result.stdout


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


def test_daily_pipeline_builds_weather_frame_only_when_requested() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_daily_pipeline.py",
            "--dry-run",
            "--with-weather-frame",
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "fetch_open_meteo_previous_runs.py" not in result.stdout
    assert "build_weather_backtest_frame.py --frame-kind recent" in result.stdout
