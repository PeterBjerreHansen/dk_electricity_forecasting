from __future__ import annotations

import sys
import json
import types
from pathlib import Path

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
    CALENDAR_COVARIATES,
    CHRONOS_LORA_ARTIFACT_SCHEMA_VERSION,
    Chronos2LoRAWeatherConfig,
    Chronos2LoRAWeatherDayAhead,
    build_lora_weather_prediction_frames,
    load_lora_artifact_manifest,
    load_chronos_lora_pipeline,
)
from dkenergy_forecast.models.chronos_zero_shot import (
    ChronosProductionConfig,
    ChronosZeroShotDayAhead,
)
from dkenergy_forecast.types import add_copenhagen_calendar
from scripts.train_chronos_lora import make_lora_training_frame


def test_production_catboost_residual_adapter_returns_publishable_predictions(monkeypatch) -> None:
    monkeypatch.setattr(
        "dkenergy_forecast.models.catboost_production.load_catboost",
        lambda: (FakeCatBoostRegressor, FakePool),
    )
    panel = _panel(periods=24 * 70)
    origin = pd.Timestamp("2024-03-01T00:00:00Z")
    history = panel.copy()
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


def test_chronos_lora_weather_adapter_emits_delivery_quantiles_and_uses_published_context(tmp_path) -> None:
    panel = _panel(periods=24 * 14)
    origin = pd.Timestamp("2024-01-09T10:00:00Z")
    history = panel.copy()
    future = make_danish_delivery_day_horizon(panel, origin)
    artifact_path = _chronos_artifact(
        tmp_path,
        covariates=[*CALENDAR_COVARIATES, "weather_gfs_global_lead1d_temperature_2m"],
        weather_future_fallback_policy="zero",
    )
    weather_path = _weather_path(
        tmp_path,
        panel=panel,
        origin=origin,
        max_ds_utc=future["ds_utc"].max(),
        feature_names=["weather_gfs_global_lead1d_temperature_2m"],
    )
    pipeline = FakeChronosPredictDfPipeline()

    predictions = Chronos2LoRAWeatherDayAhead(
        config=Chronos2LoRAWeatherConfig(
            model_artifact_path=artifact_path,
            weather_features_long_path=weather_path,
            context_length=48,
            weather_future_fallback_policy="zero",
        ),
        pipeline=pipeline,
    ).fit(history).predict(future)

    assert {"q10", "q50", "q90", "y_pred"}.issubset(predictions.columns)
    assert predictions["y_pred"].tolist() == predictions["q50"].tolist()
    assert predictions[["unique_id", "forecast_origin_utc", "ds_utc"]].duplicated().sum() == 0
    assert predictions["ds_utc"].tolist() == future["ds_utc"].tolist()
    assert pipeline.prediction_length == len(future)
    assert pipeline.context_df["timestamp"].max() == future["ds_utc"].min().tz_localize(None) - pd.Timedelta(hours=1)
    assert pipeline.future_df["timestamp"].min() == future["ds_utc"].min().tz_localize(None)
    assert pipeline.future_df["timestamp"].max() == future["ds_utc"].max().tz_localize(None)


def test_chronos_training_and_serving_share_point_in_time_weather_covariates() -> None:
    panel = _panel(periods=24 * 12)
    origin = pd.Timestamp("2024-01-08T11:00:00Z")
    future = make_danish_delivery_day_horizon(panel, origin)
    weather = _weather_vintages(panel)
    covariates = [
        *CALENDAR_COVARIATES,
        "weather_gfs_global_lead1d_temperature_2m",
        "weather_gfs_global_lead2d_temperature_2m",
    ]
    config = Chronos2LoRAWeatherConfig(
        context_length=24,
        weather_covariate_mode="raw",
        weather_future_fallback_policy="zero",
    )

    training, training_covariates, _diagnostics = make_lora_training_frame(
        panel,
        weather,
        first_eval_origin=origin,
        context_length=24,
        prediction_length=24,
        weather_covariate_mode="raw",
    )
    serving = build_lora_weather_prediction_frames(
        panel,
        future,
        weather=weather,
        covariates=covariates,
        config=config,
    )

    feature = "weather_gfs_global_lead1d_temperature_2m"
    target = pd.Timestamp("2024-01-08T20:00:00")
    training_value = training.loc[training["timestamp"].eq(target), feature].iloc[0]
    serving_value = serving.context_df.loc[serving.context_df["timestamp"].eq(target), feature].iloc[0]
    raw_value = weather.loc[
        weather["ds_utc"].eq(target.tz_localize("UTC")) & weather["feature_name"].eq(feature),
        "value",
    ].iloc[0]

    assert set(training_covariates) == set(covariates)
    assert training_value == serving_value
    assert training_value != raw_value


def test_chronos_lora_loader_uses_chronos2_pipeline_for_adapter_artifact(tmp_path, monkeypatch) -> None:
    artifact_path = _chronos_artifact(tmp_path, covariates=[*CALENDAR_COVARIATES])
    calls = []

    class FakeBaseChronosPipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):  # pragma: no cover - failure branch
            raise AssertionError("LoRA loader should not use BaseChronosPipeline")

    class FakeChronos2Pipeline:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append((args, kwargs))
            return "loaded-lora"

    monkeypatch.setitem(
        sys.modules,
        "chronos",
        types.SimpleNamespace(
            BaseChronosPipeline=FakeBaseChronosPipeline,
            Chronos2Pipeline=FakeChronos2Pipeline,
        ),
    )

    pipeline = load_chronos_lora_pipeline(
        Chronos2LoRAWeatherConfig(
            model_artifact_path=artifact_path,
            device_map="cpu",
            torch_dtype="auto",
        )
    )

    assert pipeline == "loaded-lora"
    assert calls == [((str(artifact_path),), {"device_map": "cpu", "torch_dtype": "auto"})]


def test_chronos_lora_loader_rejects_stale_feature_contract(tmp_path) -> None:
    artifact_path = tmp_path / "stale_chronos_artifact"
    artifact_path.mkdir()
    (artifact_path / "manifest.json").write_text(
        json.dumps({"artifact_schema_version": 1, "covariates": [*CALENDAR_COVARIATES]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected 2"):
        load_lora_artifact_manifest(artifact_path)


@pytest.mark.parametrize(
    ("origin", "expected_prediction_length", "expected_delivery_rows"),
    [
        ("2026-03-28T11:00:00Z", 23, 23),
        ("2026-10-24T10:00:00Z", 25, 25),
    ],
)
def test_chronos_lora_weather_adapter_handles_dst_bridge_lengths(
    tmp_path,
    origin,
    expected_prediction_length,
    expected_delivery_rows,
) -> None:
    panel = _panel(start="2026-02-01T00:00:00Z", periods=24 * 310)
    origin_ts = pd.Timestamp(origin)
    history = panel.copy()
    future = make_danish_delivery_day_horizon(panel, origin_ts)
    artifact_path = _chronos_artifact(
        tmp_path,
        covariates=[*CALENDAR_COVARIATES, "weather_gfs_global_lead1d_temperature_2m"],
        weather_future_fallback_policy="zero",
    )
    weather_path = _weather_path(
        tmp_path,
        panel=panel,
        origin=origin_ts,
        max_ds_utc=future["ds_utc"].max(),
        feature_names=["weather_gfs_global_lead1d_temperature_2m"],
    )
    pipeline = FakeChronosPredictDfPipeline()

    predictions = Chronos2LoRAWeatherDayAhead(
        config=Chronos2LoRAWeatherConfig(
            model_artifact_path=artifact_path,
            weather_features_long_path=weather_path,
            context_length=48,
            weather_future_fallback_policy="zero",
        ),
        pipeline=pipeline,
    ).fit(history).predict(future)

    assert len(predictions) == expected_delivery_rows
    assert pipeline.prediction_length == expected_prediction_length


def test_chronos_lora_weather_adapter_allows_partial_weather_nans_when_other_signal_exists(tmp_path) -> None:
    panel = _panel(periods=24 * 14)
    origin = pd.Timestamp("2024-01-09T10:00:00Z")
    history = panel.copy()
    future = make_danish_delivery_day_horizon(panel, origin)
    covariates = [
        *CALENDAR_COVARIATES,
        "weather_gfs_global_lead1d_temperature_2m",
        "weather_gfs_global_lead2d_temperature_2m",
    ]
    artifact_path = _chronos_artifact(
        tmp_path,
        covariates=covariates,
        weather_future_fallback_policy="zero",
    )
    weather_path = _weather_path(
        tmp_path,
        panel=panel,
        origin=origin,
        max_ds_utc=future["ds_utc"].max(),
        feature_names=[
            "weather_gfs_global_lead1d_temperature_2m",
            "weather_gfs_global_lead2d_temperature_2m",
        ],
        unavailable_feature_names={"weather_gfs_global_lead1d_temperature_2m"},
    )

    predictions = Chronos2LoRAWeatherDayAhead(
        config=Chronos2LoRAWeatherConfig(
            model_artifact_path=artifact_path,
            weather_features_long_path=weather_path,
            context_length=48,
            weather_future_fallback_policy="zero",
        ),
        pipeline=FakeChronosPredictDfPipeline(),
    ).fit(history).predict(future)

    assert len(predictions) == len(future)


def test_chronos_lora_weather_adapter_fails_for_missing_weather_file(tmp_path) -> None:
    panel = _panel(periods=24 * 14)
    origin = pd.Timestamp("2024-01-09T10:00:00Z")
    history = panel.copy()
    future = make_danish_delivery_day_horizon(panel, origin)
    artifact_path = _chronos_artifact(
        tmp_path,
        covariates=[*CALENDAR_COVARIATES, "weather_gfs_global_lead1d_temperature_2m"],
    )

    with pytest.raises(FileNotFoundError, match="weather feature file"):
        Chronos2LoRAWeatherDayAhead(
            config=Chronos2LoRAWeatherConfig(
                model_artifact_path=artifact_path,
                weather_features_long_path=tmp_path / "missing.parquet",
                context_length=48,
            ),
            pipeline=FakeChronosPredictDfPipeline(),
        ).fit(history).predict(future)


def test_chronos_lora_weather_adapter_enforces_artifact_covariate_schema(tmp_path) -> None:
    panel = _panel(periods=24 * 14)
    origin = pd.Timestamp("2024-01-09T10:00:00Z")
    history = panel.copy()
    future = make_danish_delivery_day_horizon(panel, origin)
    artifact_path = _chronos_artifact(
        tmp_path,
        covariates=[*CALENDAR_COVARIATES, "weather_missing_model_lead1d_temperature_2m"],
    )
    weather_path = _weather_path(
        tmp_path,
        panel=panel,
        origin=origin,
        max_ds_utc=future["ds_utc"].max(),
        feature_names=["weather_gfs_global_lead1d_temperature_2m"],
    )

    with pytest.raises(ValueError, match="artifact covariates"):
        Chronos2LoRAWeatherDayAhead(
            config=Chronos2LoRAWeatherConfig(
                model_artifact_path=artifact_path,
                weather_features_long_path=weather_path,
                context_length=48,
            ),
            pipeline=FakeChronosPredictDfPipeline(),
        ).fit(history).predict(future)


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


class FakeChronosPredictDfPipeline:
    def __init__(self):
        self.context_df = None
        self.future_df = None
        self.prediction_length = None

    def predict_df(self, context_df, future_df, prediction_length, **kwargs):
        self.context_df = context_df.copy()
        self.future_df = future_df.copy()
        self.prediction_length = prediction_length
        frames = []
        for item_id, frame in future_df.groupby("item_id", sort=False):
            values = np.arange(len(frame), dtype=float) + 10.0
            frames.append(
                pd.DataFrame(
                    {
                        "item_id": item_id,
                        "timestamp": frame["timestamp"].to_numpy(),
                        "0.1": values - 1.0,
                        "0.5": values,
                        "0.9": values + 1.0,
                    }
                )
            )
        return pd.concat(frames, ignore_index=True)


def _panel(*, periods: int, start: str = "2024-01-01T00:00:00Z") -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "unique_id": ["day_ahead_price_DK1"] * periods,
            "ds_utc": pd.date_range(start, periods=periods, freq="h"),
            "area": ["DK1"] * periods,
            "y": [float(value) for value in range(periods)],
            "dataset_version": ["v1"] * periods,
        }
    )
    return add_copenhagen_calendar(frame)


def _weather_vintages(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in panel[["area", "ds_utc"]].itertuples(index=False):
        for lead_days in (1, 2):
            rows.append(
                {
                    "area": row.area,
                    "ds_utc": row.ds_utc,
                    "feature_name": f"weather_gfs_global_lead{lead_days}d_temperature_2m",
                    "value": float(lead_days * 10_000 + row.ds_utc.day * 100 + row.ds_utc.hour),
                    "location_coverage_ratio": 1.0,
                    "location_coverage_pass": True,
                    "feature_group_pass": True,
                    "forecast_available_at_utc": row.ds_utc - pd.Timedelta(days=lead_days),
                }
            )
    return pd.DataFrame(rows)


def _chronos_artifact(
    tmp_path,
    *,
    covariates: list[str],
    weather_future_fallback_policy: str = "error",
) -> Path:
    artifact_path = tmp_path / "chronos_artifact"
    artifact_path.mkdir()
    (artifact_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_schema_version": CHRONOS_LORA_ARTIFACT_SCHEMA_VERSION,
                "covariates": covariates,
                "weather_horizon_coverage_policy": {
                    "unit": "required_weather_covariate_cells",
                    "minimum": 1.0,
                    "insufficient_coverage_fallback": weather_future_fallback_policy,
                },
            }
        ),
        encoding="utf-8",
    )
    return artifact_path


def _weather_path(
    tmp_path,
    *,
    panel: pd.DataFrame,
    origin: pd.Timestamp,
    max_ds_utc: pd.Timestamp,
    feature_names: list[str],
    unavailable_feature_names: set[str] | None = None,
) -> Path:
    unavailable = unavailable_feature_names or set()
    timestamps = pd.date_range(
        panel["ds_utc"].min(),
        max_ds_utc,
        freq="h",
        tz="UTC",
    )
    rows = []
    for timestamp in timestamps:
        for feature_name in feature_names:
            lead_days = 2 if "_lead2d_" in feature_name else 1
            rows.append(
                {
                    "area": "DK1",
                    "ds_utc": timestamp,
                    "feature_name": feature_name,
                    "value": 1.0,
                    "location_coverage_ratio": 1.0,
                    "location_coverage_pass": True,
                    "feature_group_pass": True,
                    "forecast_available_at_utc": origin + pd.Timedelta(hours=1)
                    if feature_name in unavailable
                    else timestamp - pd.Timedelta(days=lead_days),
                }
            )
    path = tmp_path / "weather.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path
