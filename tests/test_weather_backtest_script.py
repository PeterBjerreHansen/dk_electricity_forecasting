from __future__ import annotations

import pandas as pd
import pytest

from dkenergy_forecast.features import tabular_feature_columns_for_set
from dkenergy_forecast.tuning.catboost_validation import target_values


def test_weather_feature_sets_skip_missing_weather_groups() -> None:
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "local_hour": [12],
            "weather_gfs_global_lead1d_temperature_2m": [4.0],
            "weather_gfs_global_lead1d_temperature_2m_coverage_ratio": [1.0],
        }
    )

    assert tabular_feature_columns_for_set(frame, "price_full_engineered") == ["area", "local_hour"]
    assert tabular_feature_columns_for_set(frame, "all_weather") == [
        "area",
        "local_hour",
        "weather_gfs_global_lead1d_temperature_2m",
    ]
    assert tabular_feature_columns_for_set(frame, "ensemble") == []
    with pytest.raises(ValueError, match="Unknown CatBoost feature set"):
        tabular_feature_columns_for_set(frame, "gfs_global")


def test_weather_catboost_residual_target_values() -> None:
    frame = pd.DataFrame(
        {
            "y": [10.0, 20.0],
            "baseline_wdwe_weighted_median_y_pred": [7.0, 15.0],
        }
    )

    target = target_values(
        frame,
        target_mode="residual_baseline",
        residual_baseline_column="baseline_wdwe_weighted_median_y_pred",
    )

    assert target.tolist() == [3.0, 5.0]
