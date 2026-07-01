from __future__ import annotations

import importlib.util

import pandas as pd
import pytest

from dkenergy_forecast.backtesting.horizons import make_next_utc_hours_horizon
from dkenergy_forecast.features.price_features import (
    PriceFeatureConfig,
    build_price_feature_frame,
    build_training_matrix,
)
from dkenergy_forecast.models.catboost_quantile import (
    CatBoostQuantileModel,
    _load_catboost,
    _repair_quantile_crossing,
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


def test_catboost_optional_dependency_error_is_clear_when_missing() -> None:
    if importlib.util.find_spec("catboost") is not None:
        pytest.skip("CatBoost is installed in this environment")

    with pytest.raises(ImportError, match="optional CatBoost dependency"):
        _load_catboost()


def test_catboost_quantile_columns_are_sorted_by_alpha_for_repair() -> None:
    model = CatBoostQuantileModel(
        quantiles={"q90": 0.9, "q10": 0.1, "q50": 0.5},
        iterations=1,
    )
    frame = pd.DataFrame({"q10": [50.0], "q50": [90.0], "q90": [10.0]})

    repaired = _repair_quantile_crossing(frame, list(model.quantiles))

    assert list(model.quantiles) == ["q10", "q50", "q90"]
    assert repaired.loc[0, ["q10", "q50", "q90"]].tolist() == [10.0, 50.0, 90.0]


def test_catboost_rejects_duplicate_quantile_values() -> None:
    with pytest.raises(ValueError, match="Duplicate quantile value"):
        CatBoostQuantileModel(quantiles={"q10": 0.1, "q_low": 0.1}, iterations=1)


def test_catboost_feature_importance_frame_uses_last_models() -> None:
    class FakeBackendModel:
        def get_feature_importance(self) -> list[float]:
            return [2.0, 1.0]

    model = CatBoostQuantileModel()
    model.last_feature_columns_ = ["lag_24h", "area"]
    model.last_models_ = {
        "2024-01-02T10:00:00+00:00": {
            "q50": FakeBackendModel(),
        }
    }

    result = model.feature_importance_frame()

    assert result.to_dict(orient="records") == [
        {
            "forecast_origin_utc": "2024-01-02T10:00:00+00:00",
            "quantile": "q50",
            "feature": "lag_24h",
            "importance": 2.0,
        },
        {
            "forecast_origin_utc": "2024-01-02T10:00:00+00:00",
            "quantile": "q50",
            "feature": "area",
            "importance": 1.0,
        },
    ]


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
