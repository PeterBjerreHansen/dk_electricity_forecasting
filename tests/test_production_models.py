from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest

from dkenergy_forecast.backtesting.horizons import make_danish_delivery_day_horizon
from dkenergy_forecast.features.price_features import (
    WEIGHTED_MEDIAN_BASELINE_COLUMN,
    build_price_feature_frame,
)
from dkenergy_forecast.models.catboost_production import (
    CatBoostProductionConfig,
    ProductionCatBoostDayAhead,
)
from dkenergy_forecast.models.chronos_production import (
    ChronosProductionConfig,
    ChronosZeroShotDayAhead,
)
from dkenergy_forecast.types import add_copenhagen_calendar


def test_production_catboost_residual_adapter_returns_publishable_predictions(monkeypatch) -> None:
    monkeypatch.setattr(
        "dkenergy_forecast.models.catboost_production.load_catboost",
        lambda: (FakeCatBoostRegressor, FakePool),
    )
    panel = _panel(periods=24 * 70)
    origin = pd.Timestamp("2024-03-01T00:00:00Z")
    history = panel[panel["ds_utc"] < origin].copy()
    future = make_danish_delivery_day_horizon(panel, origin)
    future["y"] = 999999.0
    config = CatBoostProductionConfig(
        feature_set="price_baseline_calendar",
        target_mode="residual_baseline",
        training_origin_days=30,
        at_hour_utc=0,
        recency_half_life_days=None,
        params={"iterations": 1},
    )

    predictions = ProductionCatBoostDayAhead(config=config).fit(history).predict(future)
    future_features = build_price_feature_frame(
        history,
        future,
        forecast_origin_utc=origin,
        include_target=False,
        config=config.price_feature_config,
    )

    assert {
        "unique_id",
        "ds_utc",
        "forecast_origin_utc",
        "horizon",
        "model_name",
        "model_version",
        "y_pred",
    }.issubset(predictions.columns)
    assert predictions["y_pred"].tolist() == pytest.approx(
        (future_features[WEIGHTED_MEDIAN_BASELINE_COLUMN] + 1.5).tolist()
    )
    assert not predictions["y_pred"].eq(999999.0).any()


def test_production_catboost_module_does_not_import_optuna() -> None:
    sys.modules.pop("optuna", None)

    __import__("dkenergy_forecast.models.catboost_production")

    assert "optuna" not in sys.modules


def test_chronos_zero_shot_adapter_emits_quantiles_and_uses_q50_as_point_forecast() -> None:
    panel = _panel(periods=24 * 10)
    origin = pd.Timestamp("2024-01-09T00:00:00Z")
    history = panel[panel["ds_utc"] < origin].copy()
    future = make_danish_delivery_day_horizon(panel, origin).head(4)
    config = ChronosProductionConfig(context_length=48)

    predictions = ChronosZeroShotDayAhead(
        config=config,
        pipeline=FakeChronosPipeline(),
    ).fit(history).predict(future)

    assert {"q10", "q50", "q90", "y_pred"}.issubset(predictions.columns)
    assert predictions["y_pred"].tolist() == predictions["q50"].tolist()
    assert predictions["q10"].tolist() == pytest.approx([9.0, 10.0, 11.0, 12.0])
    assert predictions["q50"].tolist() == pytest.approx([10.0, 11.0, 12.0, 13.0])
    assert predictions["q90"].tolist() == pytest.approx([11.0, 12.0, 13.0, 14.0])


class FakePool:
    def __init__(self, data, label=None, weight=None, cat_features=None):
        self.data = data
        self.label = label
        self.weight = weight
        self.cat_features = cat_features


class FakeCatBoostRegressor:
    def __init__(self, **params):
        self.params = params

    def fit(self, pool):
        self.fit_pool = pool
        return self

    def predict(self, pool):
        return np.full(len(pool.data), 1.5)


class FakeChronosPipeline:
    def predict_quantiles(self, *, context, prediction_length, quantile_levels):
        series_count = len(context)
        base = np.arange(prediction_length, dtype=float) + 10.0
        quantiles = np.stack([base - 1.0, base, base + 1.0], axis=-1)
        return np.repeat(quantiles[None, :, :], series_count, axis=0)


def _panel(*, periods: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "unique_id": ["day_ahead_price_DK1"] * periods,
            "ds_utc": pd.date_range("2024-01-01T00:00:00Z", periods=periods, freq="h"),
            "area": ["DK1"] * periods,
            "y": [float(value) for value in range(periods)],
            "dataset_version": ["v1"] * periods,
        }
    )
    return add_copenhagen_calendar(frame)
