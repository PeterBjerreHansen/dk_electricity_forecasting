from __future__ import annotations

from pathlib import Path

import pandas as pd

from dkenergy_data.build.open_meteo_weather_v1 import (
    OpenMeteoRawBatch,
    WeatherLocation,
    build_area_feature_long,
    normalize_batch,
)
from dkenergy_forecast.backtesting.horizons import make_next_utc_hours_horizon
from dkenergy_forecast.backtesting.rolling_origin import rolling_origin_backtest
from dkenergy_forecast.types import TARGET_LEAKAGE_COLUMNS, add_copenhagen_calendar


def test_open_meteo_outputs_canonical_forecast_timing_columns() -> None:
    location = WeatherLocation("dk1_a", "DK1", 56.0, 10.0)
    batch = OpenMeteoRawBatch(
        batch_id="batch-dk1-a",
        weather_model="gfs_global",
        location_id="dk1_a",
        retrieved_at_utc="2026-01-01T00:00:00+00:00",
        raw_path=Path("/tmp/dk1_a.json"),
        payload={
            "hourly": {
                "time": ["2025-01-03T00:00"],
                "temperature_2m_previous_day2": [4.5],
            },
            "hourly_units": {"temperature_2m_previous_day2": "degC"},
        },
    )

    normalized = normalize_batch(
        batch,
        locations=[location],
        base_variables=["temperature_2m"],
        lead_time_days=[2],
    )
    area_long = build_area_feature_long(normalized, locations=[location])

    normalized_row = normalized.iloc[0]
    area_row = area_long.iloc[0]
    expected_valid_time = pd.Timestamp("2025-01-03T00:00:00Z")
    expected_reference_time = pd.Timestamp("2025-01-01T00:00:00Z")

    for row in [normalized_row, area_row]:
        assert row["forecast_reference_time"] == expected_reference_time
        assert row["valid_time"] == expected_valid_time
        assert row["lead_time_hours"] == 48
        assert row["model"] == "gfs_global"
        assert row["variable"] == "temperature_2m"
        assert row["price_area"] == "DK1"


def test_rolling_origin_future_frame_drops_target_columns_before_model_prediction() -> None:
    class InspectingModel:
        model_name = "inspecting_model"
        model_version = "test"

        def fit(self, history: pd.DataFrame) -> "InspectingModel":
            self.history = history.copy()
            return self

        def predict(self, future: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
            overlap = set(TARGET_LEAKAGE_COLUMNS).intersection(future.columns)
            assert overlap == set()
            assert history is not None
            assert (
                pd.to_datetime(history["price_available_at_utc"], utc=True)
                < future["forecast_origin_utc"].min()
            ).all()

            output = future[["unique_id", "ds_utc", "forecast_origin_utc", "horizon"]].copy()
            output["model_name"] = self.model_name
            output["model_version"] = self.model_version
            output["y_pred"] = 0.0
            return output

    panel = _panel(periods=50)
    origin = pd.Timestamp("2024-01-02T00:00:00Z")
    origins = pd.DataFrame({"forecast_origin_utc": [origin]})

    predictions = rolling_origin_backtest(
        model_factory=InspectingModel,
        panel=panel,
        origins=origins,
        horizon_builder=_leaky_horizon_builder,
    )

    assert "y" in predictions.columns
    assert predictions["y"].notna().all()


def _leaky_horizon_builder(panel: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    future = make_next_utc_hours_horizon(panel, origin, hours=2)
    target_columns = ["unique_id", "ds_utc", *TARGET_LEAKAGE_COLUMNS]
    return future.merge(panel[target_columns], on=["unique_id", "ds_utc"], how="left")


def _panel(periods: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "unique_id": ["day_ahead_price_DK1"] * periods,
            "ds_utc": pd.date_range("2024-01-01T00:00:00Z", periods=periods, freq="h"),
            "area": ["DK1"] * periods,
            "y": [float(value) for value in range(periods)],
            "dataset_version": ["v1"] * periods,
        }
    )
    frame = add_copenhagen_calendar(frame)
    frame["price_dkk_per_mwh"] = frame["y"]
    frame["price_eur_per_mwh"] = frame["y"] / 7.45
    return frame
