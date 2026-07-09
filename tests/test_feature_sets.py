from __future__ import annotations

import pandas as pd

from dkenergy_forecast.features.feature_sets import (
    ROBUST_BASELINE_COLUMN,
    tabular_feature_columns_for_set,
)
from dkenergy_forecast.features.price_features import PriceFeatureConfig


def test_price_feature_sets_are_explicit_and_stable() -> None:
    frame = pd.DataFrame(
        {
            "area": ["DK1", "DK2"],
            "ds_utc": pd.date_range("2024-01-01T00:00:00Z", periods=2, freq="h"),
            "y": [10.0, 20.0],
            "local_hour": [0, 1],
            "local_day_of_week": [0, 0],
            "local_month": [1, 1],
            "is_weekend": [False, False],
            "is_dst": [False, False],
            "utc_offset_hours": [1.0, 1.0],
            "lag_24h": [8.0, 18.0],
            "lag_168h": [7.0, 17.0],
            "rolling_mean_24h_asof_origin": [9.0, 19.0],
            "rolling_median_24h_asof_origin": [9.0, 19.0],
            "rolling_mean_168h_asof_origin": [9.0, 19.0],
            "rolling_median_168h_asof_origin": [9.0, 19.0],
            "rolling_mean_672h_asof_origin": [9.0, 19.0],
            "rolling_median_672h_asof_origin": [9.0, 19.0],
            "seasonal_median_local_hour": [9.0, 19.0],
            "seasonal_median_hour_weekend": [9.0, 19.0],
            "dk1_minus_dk2_lag_24h": [-10.0, -10.0],
            "dk1_minus_dk2_lag_168h": [-10.0, -10.0],
            ROBUST_BASELINE_COLUMN: [9.5, 19.5],
            "numeric_diagnostic_that_must_not_be_auto_selected": [1.0, 2.0],
            "weather_gfs_global_lead1d_temperature_2m": [3.0, 4.0],
        }
    )

    full = tabular_feature_columns_for_set(frame, "price_full_engineered")
    baseline_calendar = tabular_feature_columns_for_set(frame, "price_baseline_calendar")
    lags_calendar = tabular_feature_columns_for_set(frame, "price_lags_calendar")

    assert full[0] == "area"
    assert ROBUST_BASELINE_COLUMN in full
    assert "weather_gfs_global_lead1d_temperature_2m" not in full
    assert "numeric_diagnostic_that_must_not_be_auto_selected" not in full
    assert baseline_calendar == [
        "area",
        ROBUST_BASELINE_COLUMN,
        "local_hour",
        "local_day_of_week",
        "local_month",
        "is_weekend",
        "is_dst",
        "utc_offset_hours",
    ]
    assert set(lags_calendar) == {
        "area",
        "local_hour",
        "local_day_of_week",
        "local_month",
        "is_weekend",
        "is_dst",
        "utc_offset_hours",
        "lag_24h",
        "lag_168h",
        "dk1_minus_dk2_lag_24h",
        "dk1_minus_dk2_lag_168h",
    }


def test_price_feature_sets_follow_price_feature_config() -> None:
    config = PriceFeatureConfig(
        lag_hours=(24,),
        rolling_windows_hours=(24,),
        spread_lag_hours=(24,),
        include_weighted_median_baseline=False,
    )
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "local_hour": [0],
            "local_day_of_week": [0],
            "local_month": [1],
            "is_weekend": [False],
            "is_dst": [False],
            "utc_offset_hours": [1.0],
            "lag_24h": [8.0],
            "lag_168h": [7.0],
            "rolling_mean_24h_asof_origin": [9.0],
            "rolling_median_24h_asof_origin": [9.0],
            "rolling_mean_168h_asof_origin": [9.0],
            "rolling_median_168h_asof_origin": [9.0],
            "seasonal_median_local_hour": [9.0],
            "seasonal_median_hour_weekend": [9.0],
            "dk1_minus_dk2_lag_24h": [-10.0],
            ROBUST_BASELINE_COLUMN: [9.5],
        }
    )

    full = tabular_feature_columns_for_set(
        frame,
        "price_full_engineered",
        price_feature_config=config,
    )
    lags_calendar = tabular_feature_columns_for_set(
        frame,
        "price_lags_calendar",
        price_feature_config=config,
    )

    assert ROBUST_BASELINE_COLUMN not in full
    assert "lag_168h" not in full
    assert "rolling_mean_168h_asof_origin" not in full
    assert "lag_24h" in full
    assert set(lags_calendar) == {
        "area",
        "local_hour",
        "local_day_of_week",
        "local_month",
        "is_weekend",
        "is_dst",
        "utc_offset_hours",
        "lag_24h",
        "dk1_minus_dk2_lag_24h",
    }
