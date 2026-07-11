from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest

from dkenergy_data.build.eds_prices_v1 import (
    RawBatch,
    build_model_ready_panel,
    normalize_batches,
)
from dkenergy_data.build.open_meteo_weather_v1 import (
    FORECAST_AVAILABILITY_TIME_TYPE,
    FORECAST_REFERENCE_TIME_TYPE,
    OpenMeteoRawBatch,
    WeatherLocation,
    build_area_feature_long,
    normalize_batch,
)
from dkenergy_forecast.backtesting.horizons import make_next_utc_hours_horizon
from dkenergy_forecast.features.weather_features import (
    EXCLUDED_WEATHER_PARAMETERS,
    WEATHER_CUTOFF_COLUMNS,
    WEATHER_SELECTION_POLICY,
    join_weather_features,
)
from dkenergy_forecast.models.chronos_production import (
    CONTEXT_WEATHER_FILL_POLICY,
    FUTURE_WEATHER_FILL_POLICY,
    WEATHER_HORIZON_COVERAGE_UNIT,
    Chronos2LoRAWeatherConfig,
    fill_lora_covariates,
    validate_artifact_weather_policy,
    validate_weather_horizon_coverage,
)
from scripts.train_chronos_lora import make_manifest


def test_price_panel_and_future_expose_target_contract_across_market_regimes() -> None:
    elspot = normalize_batches(
        [
            RawBatch(
                source_dataset="Elspotprices",
                batch_id="hourly",
                retrieved_at_utc="2026-01-01T00:00:00Z",
                raw_path=Path("hourly.json"),
                records=[
                    {
                        "HourUTC": "2025-09-30T21:00:00Z",
                        "HourDK": "2025-09-30T23:00:00",
                        "PriceArea": "DK1",
                        "SpotPriceDKK": 100.0,
                        "SpotPriceEUR": 13.4,
                    }
                ],
            )
        ]
    )
    quarter_hours = [
        {
            "TimeUTC": f"2025-09-30T22:{minute:02d}:00Z",
            "TimeDK": f"2025-10-01T00:{minute:02d}:00",
            "PriceArea": "DK1",
            "DayAheadPriceDKK": float(200 + minute),
            "DayAheadPriceEUR": float(20 + minute),
        }
        for minute in (0, 15, 30, 45)
    ]
    day_ahead = normalize_batches(
        [
            RawBatch(
                source_dataset="DayAheadPrices",
                batch_id="quarter-hourly",
                retrieved_at_utc="2026-01-01T00:00:00Z",
                raw_path=Path("quarter-hourly.json"),
                records=quarter_hours,
            )
        ]
    )

    panel, qa = build_model_ready_panel(elspot, day_ahead, required_areas=["DK1"])

    assert panel["market_regime"].tolist() == ["native_hourly", "native_quarter_hour"]
    assert panel["native_resolution_minutes"].tolist() == [60, 15]
    assert panel["target_aggregation"].tolist() == [
        "identity",
        "arithmetic_mean_of_4_quarter_hours",
    ]
    assert panel["target_definition"].nunique() == 1
    assert qa["target_contract"]["native_resolution_minutes"] == [15, 60]

    future = make_next_utc_hours_horizon(
        panel,
        pd.Timestamp("2025-09-30T20:00:00Z"),
        hours=2,
    )
    assert future["market_regime"].tolist() == ["native_hourly", "native_quarter_hour"]
    assert future["native_resolution_minutes"].tolist() == [60, 15]


def test_open_meteo_marks_synthetic_reference_provenance() -> None:
    location = WeatherLocation("dk1_a", "DK1", 56.0, 10.0)
    normalized = normalize_batch(
        OpenMeteoRawBatch(
            batch_id="batch-a",
            weather_model="gfs_global",
            location_id="dk1_a",
            retrieved_at_utc="2026-01-01T00:00:00Z",
            raw_path=Path("weather.json"),
            payload={
                "hourly": {
                    "time": ["2025-01-03T00:00"],
                    "temperature_2m_previous_day2": [4.5],
                },
                "hourly_units": {"temperature_2m_previous_day2": "degC"},
            },
        ),
        locations=[location],
        base_variables=["temperature_2m"],
        lead_time_days=[2],
    )
    area_long = build_area_feature_long(normalized, locations=[location])

    for frame in (normalized, area_long):
        row = frame.iloc[0]
        assert row["forecast_reference_time_type"] == FORECAST_REFERENCE_TIME_TYPE
        assert not row["forecast_reference_time_is_observed"]
        assert row["forecast_availability_time_type"] == FORECAST_AVAILABILITY_TIME_TYPE
        assert row["weather_vintage_id"].endswith("synthetic:20250101T000000Z")


def test_weather_join_selects_latest_vintage_for_each_model_parameter() -> None:
    target = pd.Timestamp("2025-01-03T00:00:00Z")
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [target],
            "forecast_origin_utc": [pd.Timestamp("2025-01-02T10:00:00Z")],
        }
    )
    temperature_source = "weather_gfs_global_lead1d_temperature_2m"
    wind_source = "weather_gfs_global_lead1d_wind_speed_10m"
    weather = pd.DataFrame(
        {
            "area": ["DK1", "DK1", "DK1"],
            "ds_utc": [target] * 3,
            "feature_name": [temperature_source, wind_source, temperature_source],
            "value": [1.0, 2.0, 3.0],
            "location_coverage_ratio": [1.0] * 3,
            "location_coverage_pass": [True] * 3,
            "feature_group_pass": [True] * 3,
            "forecast_available_at_utc": pd.to_datetime(
                ["2025-01-02T00:00:00Z", "2025-01-02T00:00:00Z", "2025-01-02T06:00:00Z"],
                utc=True,
            ),
            "forecast_reference_time": pd.to_datetime(
                ["2025-01-02T00:00:00Z", "2025-01-02T00:00:00Z", "2025-01-02T06:00:00Z"],
                utc=True,
            ),
            "weather_vintage_id": ["run-00", "run-00", "run-06"],
            "forecast_reference_time_type": ["observed"] * 3,
            "forecast_reference_time_is_observed": [True] * 3,
            "forecast_availability_time_type": ["observed"] * 3,
        }
    )

    joined = join_weather_features(frame, weather)

    temperature = "weather_gfs_global_temperature_2m"
    wind = "weather_gfs_global_wind_speed_10m"
    assert joined.loc[0, temperature] == pytest.approx(3.0)
    assert joined.loc[0, f"{temperature}_vintage_id"] == "run-06"
    assert joined.loc[0, wind] == pytest.approx(2.0)
    assert joined.loc[0, f"{wind}_vintage_id"] == "run-00"


def test_future_weather_fill_never_borrows_across_valid_times() -> None:
    feature = "weather_gfs_global_temperature_2m"
    frame = pd.DataFrame(
        {
            "unique_id": ["DK1"] * 3,
            "ds_utc": pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC"),
            feature: [None, 2.0, None],
        }
    )

    training = fill_lora_covariates(frame, [feature], role="training")
    future = fill_lora_covariates(frame, [feature], role="future")

    assert training[feature].tolist() == [0.0, 2.0, 0.0]
    assert future[feature].tolist() == [0.0, 2.0, 0.0]


def test_weather_horizon_coverage_has_explicit_error_and_zero_fallbacks() -> None:
    temperature = "weather_gfs_global_temperature_2m"
    wind = "weather_gfs_global_wind_speed_10m"
    frame = pd.DataFrame(
        {
            "unique_id": ["DK1", "DK1"],
            "ds_utc": pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC"),
            temperature: [1.0, None],
            wind: [None, 2.0],
        }
    )

    with pytest.raises(ValueError, match="future weather coverage"):
        validate_weather_horizon_coverage(
            frame,
            [temperature, wind],
            min_coverage=1.0,
            fallback_policy="error",
        )

    coverage = validate_weather_horizon_coverage(
        frame,
        [temperature, wind],
        min_coverage=1.0,
        fallback_policy="zero",
    )
    assert coverage == {"DK1": 0.5}


def test_runtime_weather_policy_must_match_artifact_manifest() -> None:
    manifest = {
        "weather_covariate_mode": "raw",
        "weather_selection_policy": {
            "name": WEATHER_SELECTION_POLICY,
            "cutoff_priority": list(WEATHER_CUTOFF_COLUMNS),
            "stable_feature_names": True,
            "excluded_parameters": list(EXCLUDED_WEATHER_PARAMETERS),
            "historical_cutoff_local_time": "10:00",
        },
        "weather_horizon_coverage_policy": {
            "unit": WEATHER_HORIZON_COVERAGE_UNIT,
            "minimum": 1.0,
            "insufficient_coverage_fallback": "zero",
        }
    }

    with pytest.raises(ValueError, match="does not match"):
        validate_artifact_weather_policy(
            manifest,
            Chronos2LoRAWeatherConfig(
                weather_covariate_mode="raw",
                weather_future_fallback_policy="error",
            ),
        )

    validate_artifact_weather_policy(
        manifest,
        Chronos2LoRAWeatherConfig(
            weather_covariate_mode="raw",
            weather_future_fallback_policy="zero",
        ),
    )


def test_chronos_manifest_records_weather_and_target_policies() -> None:
    panel = pd.DataFrame(
        {
            "dataset_version": ["v1"],
            "market_regime": ["native_hourly"],
            "native_resolution_minutes": [60],
            "target_aggregation": ["identity"],
            "target_definition": ["hourly_day_ahead_area_price_dkk_per_mwh"],
        }
    )
    weather = pd.DataFrame(
        {
            "ds_utc": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "forecast_reference_time_type": [FORECAST_REFERENCE_TIME_TYPE],
            "forecast_reference_time_is_observed": [False],
            "forecast_availability_time_type": [FORECAST_AVAILABILITY_TIME_TYPE],
        }
    )
    args = Namespace(
        base_model_id="amazon/chronos-2",
        context_length=24,
        prediction_length=24,
        num_steps=1,
        batch_size=1,
        learning_rate=1e-4,
        weather_covariate_mode="raw",
        weather_horizon_min_coverage=1.0,
        weather_future_fallback_policy="zero",
        validation_scores_path=None,
        validation_score_model_label=None,
        panel_path="panel.parquet",
        weather_features_long_path="weather.parquet",
    )

    manifest = make_manifest(
        args=args,
        panel=panel,
        weather=weather,
        covariates=["local_hour", "weather_gfs_global_temperature_2m"],
        diagnostics={},
    )

    assert manifest["weather_horizon_coverage_policy"] == {
        "unit": WEATHER_HORIZON_COVERAGE_UNIT,
        "minimum": 1.0,
        "insufficient_coverage_fallback": "zero",
    }
    assert manifest["covariate_fill_policy"]["training"] == CONTEXT_WEATHER_FILL_POLICY
    assert manifest["covariate_fill_policy"]["serving_future"] == FUTURE_WEATHER_FILL_POLICY
    assert manifest["covariate_fill_policy"]["temporal_fill"] is False
    assert manifest["weather_selection_policy"]["name"] == WEATHER_SELECTION_POLICY
    assert manifest["target_contract"]["native_resolution_minutes"] == [60]
