from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


def test_weather_feature_backtest_skips_missing_weather_groups() -> None:
    module = _load_script_module()
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "local_hour": [12],
            "weather_gfs_global_lead1d_temperature_2m": [4.0],
            "weather_gfs_global_lead1d_temperature_2m_coverage_ratio": [1.0],
        }
    )

    assert module.feature_columns_for_set(frame, "price_only") == ["area", "local_hour"]
    assert module.feature_columns_for_set(frame, "gfs_global") == [
        "area",
        "local_hour",
        "weather_gfs_global_lead1d_temperature_2m",
    ]
    assert module.feature_columns_for_set(frame, "icon_eu") == []
    assert module.feature_columns_for_set(frame, "ensemble") == []


def _load_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "run_weather_feature_backtest.py"
    spec = importlib.util.spec_from_file_location("run_weather_feature_backtest", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
