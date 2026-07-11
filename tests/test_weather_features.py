from __future__ import annotations

import pandas as pd
import pytest

from dkenergy_forecast.features.weather_features import (
    add_weather_derived_features,
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

    feature = "weather_icon_eu_temperature_2m"
    assert joined.loc[0, feature] == pytest.approx(4.0)
    assert pd.isna(joined.loc[1, feature])
    assert joined.loc[0, f"{feature}_available_at_utc"] == pd.Timestamp("2025-01-01T00:00:00Z")


def test_join_weather_features_uses_latest_eligible_forecast_vintage() -> None:
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T10:00:00Z")],
        }
    )
    source_feature = "weather_gfs_global_lead1d_temperature_2m"
    feature = "weather_gfs_global_temperature_2m"
    area_features_long = pd.DataFrame(
        {
            "area": ["DK1", "DK1", "DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")] * 3,
            "feature_name": [source_feature, source_feature, source_feature],
            "value": [9.0, 1.0, 99.0],
            "location_coverage_ratio": [1.0, 1.0, 1.0],
            "location_coverage_pass": [True, True, True],
            "feature_group_pass": [True, True, True],
            "forecast_available_at_utc": [
                pd.Timestamp("2025-01-01T09:00:00Z"),
                pd.Timestamp("2025-01-01T00:00:00Z"),
                pd.Timestamp("2025-01-01T12:00:00Z"),
            ],
        }
    )

    joined = join_weather_features(frame, area_features_long)

    assert joined.loc[0, feature] == pytest.approx(9.0)
    assert joined.loc[0, f"{feature}_available_at_utc"] == pd.Timestamp("2025-01-01T09:00:00Z")


def test_join_weather_features_selects_lead_at_information_cutoff() -> None:
    target = pd.Timestamp("2025-01-02T11:00:00Z")
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [target],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T12:00:00Z")],
            "information_cutoff_utc": [pd.Timestamp("2025-01-01T10:00:00Z")],
        }
    )
    area_features_long = pd.DataFrame(
        {
            "area": ["DK1", "DK1"],
            "ds_utc": [target, target],
            "feature_name": [
                "weather_gfs_global_lead1d_temperature_2m",
                "weather_gfs_global_lead2d_temperature_2m",
            ],
            "lead_time_days": [1, 2],
            "value": [11.0, 22.0],
            "location_coverage_ratio": [1.0, 1.0],
            "location_coverage_pass": [True, True],
            "feature_group_pass": [True, True],
            "forecast_available_at_utc": [
                pd.Timestamp("2025-01-01T11:00:00Z"),
                pd.Timestamp("2024-12-31T11:00:00Z"),
            ],
        }
    )

    joined = join_weather_features(frame, area_features_long)

    feature = "weather_gfs_global_temperature_2m"
    assert joined.loc[0, feature] == pytest.approx(22.0)
    assert joined.loc[0, f"{feature}_selected_lead_time_days"] == 2
    assert joined.loc[0, f"{feature}_source_feature_name"] == (
        "weather_gfs_global_lead2d_temperature_2m"
    )


def test_join_weather_features_excludes_circular_wind_direction() -> None:
    target = pd.Timestamp("2025-01-02T00:00:00Z")
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [target],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T10:00:00Z")],
        }
    )
    weather = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [target],
            "feature_name": ["weather_gfs_global_lead1d_wind_direction_10m"],
            "value": [180.0],
            "location_coverage_ratio": [1.0],
            "location_coverage_pass": [True],
            "feature_group_pass": [True],
            "forecast_available_at_utc": [pd.Timestamp("2025-01-01T00:00:00Z")],
        }
    )

    joined = join_weather_features(frame, weather)

    assert not any("wind_direction" in column for column in joined.columns)


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

    assert "weather_gfs_global_cloud_cover" not in joined.columns
    assert joined_without_gate.loc[0, "weather_gfs_global_cloud_cover"] == pytest.approx(77.0)


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

    assert "weather_icon_eu_wind_speed_100m" not in joined.columns


def test_weather_ensemble_features_use_model_values_only() -> None:
    frame = pd.DataFrame(
        {
            "weather_gfs_global_temperature_2m": [2.0],
            "weather_icon_eu_temperature_2m": [4.0],
            "weather_metno_nordic_temperature_2m": [6.0],
            "weather_icon_eu_temperature_2m_coverage_ratio": [1.0],
        }
    )

    output = add_weather_ensemble_features(frame)

    assert output["weather_ensemble_temperature_2m_mean"].iloc[0] == pytest.approx(4.0)
    assert output["weather_ensemble_temperature_2m_min"].iloc[0] == pytest.approx(2.0)
    assert output["weather_ensemble_temperature_2m_max"].iloc[0] == pytest.approx(6.0)
    assert output["weather_ensemble_temperature_2m_spread"].iloc[0] == pytest.approx(4.0)
    assert "weather_icon_eu_temperature_2m_coverage_ratio" not in weather_value_columns(output)
    second_output = add_weather_ensemble_features(output)
    assert not any(column.startswith("weather_ensemble_temperature_2m_mean_") for column in second_output)


def test_weather_derived_features_add_physics_and_area_spread() -> None:
    frame = pd.DataFrame(
        {
            "area": ["DK1", "DK2"],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T10:00:00Z")] * 2,
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")] * 2,
            "weather_gfs_global_temperature_2m": [10.0, 4.0],
            "weather_gfs_global_wind_speed_10m": [10.0, 8.0],
            "weather_gfs_global_wind_speed_100m": [15.0, 9.0],
            "weather_gfs_global_shortwave_radiation": [100.0, 60.0],
            "weather_gfs_global_cloud_cover": [50.0, 25.0],
        }
    )

    output = add_weather_derived_features(frame)

    assert output.loc[0, "weather_gfs_global_wind_shear_100m_minus_10m"] == pytest.approx(5.0)
    assert output.loc[0, "weather_gfs_global_shortwave_x_cloud_cover"] == pytest.approx(50.0)
    assert output.loc[0, "weather_gfs_global_shortwave_x_clear_sky_proxy"] == pytest.approx(50.0)
    spread = "weather_gfs_global_temperature_2m_dk1_minus_dk2"
    assert output[spread].tolist() == pytest.approx([6.0, 6.0])
    assert spread in weather_value_columns(output)
