from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dkenergy_data.build.open_meteo_weather_v1 import (
    OpenMeteoRawBatch,
    WeatherLocation,
    build_area_feature_long,
    build_area_feature_wide,
    normalize_batch,
)


def test_normalize_previous_runs_payload_to_long_forecast_rows() -> None:
    batch = _raw_batch(
        "dk1_a",
        payload={
            "hourly": {
                "time": ["2025-01-02T00:00", "2025-01-02T01:00"],
                "temperature_2m_previous_day1": [3.0, 4.0],
                "temperature_2m_previous_day2": [2.5, 3.5],
            },
            "hourly_units": {
                "temperature_2m_previous_day1": "degC",
                "temperature_2m_previous_day2": "degC",
            },
        },
    )

    normalized = normalize_batch(
        batch,
        locations=[WeatherLocation("dk1_a", "DK1", 56.0, 10.0)],
        base_variables=["temperature_2m"],
        lead_time_days=[1, 2],
    )

    assert len(normalized) == 4
    assert set(normalized["lead_time_days"]) == {1, 2}
    first = normalized[
        (normalized["valid_time_utc"] == pd.Timestamp("2025-01-02T00:00:00Z"))
        & (normalized["lead_time_days"] == 1)
    ].iloc[0]
    assert first["forecast_available_at_utc"] == pd.Timestamp("2025-01-01T00:00:00Z")
    assert first["parameter_id"] == "temperature_2m"
    assert first["unit"] == "degC"
    assert first["raw_batch_id"] == "batch-dk1_a"


def test_area_aggregation_uses_available_points_without_imputation() -> None:
    locations = [
        WeatherLocation("dk1_a", "DK1", 56.0, 10.0),
        WeatherLocation("dk1_b", "DK1", 57.0, 9.0),
    ]
    normalized = pd.concat(
        [
            normalize_batch(
                _raw_batch(
                    "dk1_a",
                    payload={
                        "hourly": {
                            "time": ["2025-01-02T00:00", "2025-01-02T01:00"],
                            "temperature_2m_previous_day1": [3.0, 4.0],
                        },
                        "hourly_units": {"temperature_2m_previous_day1": "degC"},
                    },
                ),
                locations=locations,
                base_variables=["temperature_2m"],
                lead_time_days=[1],
            ),
            normalize_batch(
                _raw_batch(
                    "dk1_b",
                    payload={
                        "hourly": {
                            "time": ["2025-01-02T00:00", "2025-01-02T01:00"],
                            "temperature_2m_previous_day1": [5.0, None],
                        },
                        "hourly_units": {"temperature_2m_previous_day1": "degC"},
                    },
                ),
                locations=locations,
                base_variables=["temperature_2m"],
                lead_time_days=[1],
            ),
        ],
        ignore_index=True,
    )

    area_long = build_area_feature_long(
        normalized,
        locations=locations,
        coverage_threshold=0.95,
    )

    first = area_long[area_long["ds_utc"] == pd.Timestamp("2025-01-02T00:00:00Z")].iloc[0]
    second = area_long[area_long["ds_utc"] == pd.Timestamp("2025-01-02T01:00:00Z")].iloc[0]
    assert first["value"] == pytest.approx(4.0)
    assert first["location_coverage_ratio"] == pytest.approx(1.0)
    assert first["location_coverage_pass"]
    assert first["feature_window_coverage_ratio"] == pytest.approx(0.5)
    assert not first["feature_group_pass"]
    assert second["value"] == pytest.approx(4.0)
    assert second["location_coverage_ratio"] == pytest.approx(0.5)
    assert not second["location_coverage_pass"]
    assert second["feature_window_coverage_ratio"] == pytest.approx(0.5)
    assert not second["feature_group_pass"]


def test_area_feature_wide_keeps_values_coverage_flags_and_availability() -> None:
    locations = [
        WeatherLocation("dk1_a", "DK1", 56.0, 10.0),
        WeatherLocation("dk1_b", "DK1", 57.0, 9.0),
    ]
    normalized = pd.concat(
        [
            normalize_batch(
                _raw_batch(
                    "dk1_a",
                    payload={
                        "hourly": {
                            "time": ["2025-01-02T00:00"],
                            "wind_speed_10m_previous_day1": [6.0],
                        },
                        "hourly_units": {"wind_speed_10m_previous_day1": "m/s"},
                    },
                ),
                locations=locations,
                base_variables=["wind_speed_10m"],
                lead_time_days=[1],
            ),
            normalize_batch(
                _raw_batch(
                    "dk1_b",
                    payload={
                        "hourly": {
                            "time": ["2025-01-02T00:00"],
                            "wind_speed_10m_previous_day1": [8.0],
                        },
                        "hourly_units": {"wind_speed_10m_previous_day1": "m/s"},
                    },
                ),
                locations=locations,
                base_variables=["wind_speed_10m"],
                lead_time_days=[1],
            ),
        ],
        ignore_index=True,
    )
    area_long = build_area_feature_long(normalized, locations=locations)

    wide = build_area_feature_wide(area_long)

    feature = "weather_gfs_global_lead1d_wind_speed_10m"
    row = wide.iloc[0]
    assert row[feature] == pytest.approx(7.0)
    assert row[f"{feature}_coverage_ratio"] == pytest.approx(1.0)
    assert row[f"{feature}_passes_coverage"]
    assert row[f"{feature}_passes_location_coverage"]
    assert row[f"{feature}_available_at_utc"] == pd.Timestamp("2025-01-01T00:00:00Z")


def _raw_batch(location_id: str, *, payload: dict) -> OpenMeteoRawBatch:
    return OpenMeteoRawBatch(
        batch_id=f"batch-{location_id}",
        weather_model="gfs_global",
        location_id=location_id,
        retrieved_at_utc="2026-01-01T00:00:00+00:00",
        raw_path=Path(f"/tmp/{location_id}.json"),
        payload=payload,
    )
