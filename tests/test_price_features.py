from __future__ import annotations

import pandas as pd
import pytest

from dkenergy_forecast.backtesting.horizons import make_next_utc_hours_horizon
from dkenergy_forecast.features.price_features import (
    PriceFeatureConfig,
    WEIGHTED_MEDIAN_BASELINE_COLUMN,
    build_price_feature_frame,
    build_training_matrix,
)
from dkenergy_forecast.types import add_copenhagen_calendar


def test_price_feature_frame_uses_only_values_before_forecast_origin() -> None:
    panel = _two_area_panel(periods=96)
    origin = pd.Timestamp("2024-01-03T10:00:00Z")
    future = make_next_utc_hours_horizon(panel, origin, hours=2)
    config = PriceFeatureConfig(
        lag_hours=(1, 2),
        rolling_windows_hours=(3,),
        seasonal_lookback_days=2,
        spread_lag_hours=(1, 2),
    )

    features = build_price_feature_frame(
        panel,
        future,
        forecast_origin_utc=origin,
        include_target=True,
        config=config,
    )
    dk1_first = features[
        (features["area"] == "DK1")
        & (features["ds_utc"] == origin + pd.Timedelta(hours=1))
    ].iloc[0]

    assert pd.isna(dk1_first["lag_1h"])
    assert dk1_first["lag_2h"] == _value_at(panel, "DK1", origin - pd.Timedelta(hours=1))
    assert pd.isna(dk1_first["dk1_minus_dk2_lag_1h"])
    assert dk1_first["dk1_minus_dk2_lag_2h"] == pytest.approx(-1000.0)

    expected_rolling = [
        _value_at(panel, "DK1", origin - pd.Timedelta(hours=hour))
        for hour in [3, 2, 1]
    ]
    assert dk1_first["rolling_mean_3h_asof_origin"] == pytest.approx(
        sum(expected_rolling) / len(expected_rolling)
    )
    assert dk1_first["y"] == _value_at(panel, "DK1", origin + pd.Timedelta(hours=1))


def test_training_matrix_keeps_complete_training_horizons_before_origin() -> None:
    panel = _two_area_panel(periods=24 * 12)
    origin = pd.Timestamp("2024-01-10T10:00:00Z")

    training = build_training_matrix(
        panel,
        origin,
        training_origin_days=5,
        at_hour_utc=10,
        config=PriceFeatureConfig(
            lag_hours=(24,),
            rolling_windows_hours=(24,),
            seasonal_lookback_days=7,
            spread_lag_hours=(24,),
        ),
    )

    assert not training.empty
    assert training["ds_utc"].max() < origin
    assert training["y"].notna().all()
    assert {"lag_24h", "rolling_mean_24h_asof_origin", "dk1_minus_dk2_lag_24h"}.issubset(
        training.columns
    )


def test_weighted_median_baseline_feature_is_origin_safe() -> None:
    panel = _two_area_panel(periods=24 * 20)
    origin = pd.Timestamp("2024-01-16T10:00:00Z")
    future = make_next_utc_hours_horizon(panel, origin, hours=24)
    config = PriceFeatureConfig(
        lag_hours=(24,),
        rolling_windows_hours=(24,),
        seasonal_lookback_days=7,
        spread_lag_hours=(24,),
    )

    clean = build_price_feature_frame(
        panel,
        future,
        forecast_origin_utc=origin,
        include_target=False,
        config=config,
    )
    leaky_panel = panel.copy()
    leaky_panel.loc[leaky_panel["ds_utc"] >= origin, "y"] = 999999.0
    leaky = build_price_feature_frame(
        leaky_panel,
        future,
        forecast_origin_utc=origin,
        include_target=False,
        config=config,
    )
    disabled = build_price_feature_frame(
        panel,
        future,
        forecast_origin_utc=origin,
        include_target=False,
        config=PriceFeatureConfig(
            lag_hours=(24,),
            rolling_windows_hours=(24,),
            seasonal_lookback_days=7,
            spread_lag_hours=(24,),
            include_weighted_median_baseline=False,
        ),
    )

    assert WEIGHTED_MEDIAN_BASELINE_COLUMN in config.feature_columns
    assert clean[WEIGHTED_MEDIAN_BASELINE_COLUMN].notna().any()
    pd.testing.assert_series_equal(
        clean[WEIGHTED_MEDIAN_BASELINE_COLUMN],
        leaky[WEIGHTED_MEDIAN_BASELINE_COLUMN],
        check_names=False,
    )
    assert WEIGHTED_MEDIAN_BASELINE_COLUMN not in disabled.columns


def _two_area_panel(*, periods: int) -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01T00:00:00Z", periods=periods, freq="h")
    frames = []
    for area, offset in [("DK1", 0.0), ("DK2", 1000.0)]:
        frame = pd.DataFrame(
            {
                "unique_id": [f"day_ahead_price_{area}"] * periods,
                "ds_utc": timestamps,
                "area": [area] * periods,
                "y": [offset + float(i) for i in range(periods)],
                "dataset_version": ["test"] * periods,
            }
        )
        frames.append(frame)
    panel = pd.concat(frames, ignore_index=True)
    panel = add_copenhagen_calendar(panel)
    panel["price_dkk_per_mwh"] = panel["y"]
    panel["price_eur_per_mwh"] = panel["y"] / 7.45
    return panel.sort_values(["area", "ds_utc"]).reset_index(drop=True)


def _value_at(panel: pd.DataFrame, area: str, timestamp: pd.Timestamp) -> float:
    value = panel.loc[
        (panel["area"] == area) & (panel["ds_utc"] == timestamp),
        "y",
    ]
    assert len(value) == 1
    return float(value.iloc[0])
