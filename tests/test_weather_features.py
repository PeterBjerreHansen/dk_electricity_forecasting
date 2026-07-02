from __future__ import annotations

import pandas as pd
import pytest

from dkenergy_forecast.features.weather_features import (
    add_weather_ensemble_features,
    join_weather_features,
    weather_value_columns,
)


def test_join_weather_features_masks_values_unavailable_at_forecast_origin() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["day_ahead_price_DK1", "day_ahead_price_DK1"],
            "area": ["DK1", "DK1"],
            "ds_utc": [
                pd.Timestamp("2025-01-02T00:00:00Z"),
                pd.Timestamp("2025-01-02T12:00:00Z"),
            ],
            "forecast_origin_utc": [
                pd.Timestamp("2025-01-01T10:00:00Z"),
                pd.Timestamp("2025-01-01T10:00:00Z"),
            ],
        }
    )
    area_features_long = pd.DataFrame(
        {
            "area": ["DK1", "DK1"],
            "ds_utc": [
                pd.Timestamp("2025-01-02T00:00:00Z"),
                pd.Timestamp("2025-01-02T12:00:00Z"),
            ],
            "feature_name": [
                "weather_icon_eu_lead1d_temperature_2m",
                "weather_icon_eu_lead1d_temperature_2m",
            ],
            "value": [4.0, 8.0],
            "location_coverage_ratio": [1.0, 1.0],
            "location_coverage_pass": [True, True],
            "feature_group_pass": [True, True],
            "forecast_available_at_utc": [
                pd.Timestamp("2025-01-01T00:00:00Z"),
                pd.Timestamp("2025-01-01T12:00:00Z"),
            ],
        }
    )

    joined = join_weather_features(frame, area_features_long)

    feature = "weather_icon_eu_lead1d_temperature_2m"
    assert joined.loc[0, feature] == pytest.approx(4.0)
    assert pd.isna(joined.loc[1, feature])
    assert joined.loc[0, f"{feature}_available_at_utc"] == pd.Timestamp("2025-01-01T00:00:00Z")


def test_join_weather_features_filters_failing_coverage_groups_by_default() -> None:
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T10:00:00Z")],
        }
    )
    area_features_long = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "feature_name": ["weather_gfs_global_lead1d_cloud_cover"],
            "value": [77.0],
            "location_coverage_ratio": [0.8],
            "location_coverage_pass": [False],
            "feature_group_pass": [False],
            "forecast_available_at_utc": [pd.Timestamp("2025-01-01T00:00:00Z")],
        }
    )

    joined = join_weather_features(frame, area_features_long)
    joined_without_gate = join_weather_features(
        frame,
        area_features_long,
        require_feature_group_pass=False,
    )

    assert "weather_gfs_global_lead1d_cloud_cover" not in joined.columns
    assert joined_without_gate.loc[0, "weather_gfs_global_lead1d_cloud_cover"] == pytest.approx(77.0)


def test_join_weather_features_filters_individual_low_coverage_hours() -> None:
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T10:00:00Z")],
        }
    )
    area_features_long = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "feature_name": ["weather_icon_eu_lead1d_wind_speed_100m"],
            "value": [12.0],
            "location_coverage_ratio": [0.4],
            "location_coverage_pass": [False],
            "feature_group_pass": [True],
            "forecast_available_at_utc": [pd.Timestamp("2025-01-01T00:00:00Z")],
        }
    )

    joined = join_weather_features(frame, area_features_long)

    assert "weather_icon_eu_lead1d_wind_speed_100m" not in joined.columns


def test_weather_ensemble_features_use_model_values_only() -> None:
    frame = pd.DataFrame(
        {
            "weather_gfs_global_lead1d_temperature_2m": [2.0],
            "weather_icon_eu_lead1d_temperature_2m": [4.0],
            "weather_metno_nordic_lead1d_temperature_2m": [6.0],
            "weather_icon_eu_lead1d_temperature_2m_coverage_ratio": [1.0],
        }
    )

    output = add_weather_ensemble_features(frame)

    assert output["weather_ensemble_lead1d_temperature_2m_mean"].iloc[0] == pytest.approx(4.0)
    assert output["weather_ensemble_lead1d_temperature_2m_min"].iloc[0] == pytest.approx(2.0)
    assert output["weather_ensemble_lead1d_temperature_2m_max"].iloc[0] == pytest.approx(6.0)
    assert output["weather_ensemble_lead1d_temperature_2m_spread"].iloc[0] == pytest.approx(4.0)
    assert "weather_icon_eu_lead1d_temperature_2m_coverage_ratio" not in weather_value_columns(output)
    second_output = add_weather_ensemble_features(output)
    assert not any(column.startswith("weather_ensemble_lead1d_temperature_2m_mean_") for column in second_output)
