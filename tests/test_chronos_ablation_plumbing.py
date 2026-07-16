from __future__ import annotations

import sys
from argparse import Namespace

import pandas as pd
import pytest

from dkenergy_forecast.models.chronos_production import (
    Chronos2LoRAWeatherConfig,
    Chronos2LoRAWeatherDayAhead,
    Chronos2ProductionFrames,
    selected_weather_columns,
    validate_artifact_weather_policy,
)
from dkenergy_forecast.models.chronos_artifact_identity import (
    validate_chronos_release_identity,
)
from dkenergy_forecast.types import add_copenhagen_calendar
from scripts.train_chronos_lora import (
    make_lora_training_frame,
    make_manifest,
    parse_args,
)


def test_weather_selection_supports_mean_spread_and_derived_opt_in() -> None:
    frame = pd.DataFrame(
        {
            "weather_gfs_global_temperature_2m": [1.0],
            "weather_ensemble_temperature_2m_mean": [2.0],
            "weather_ensemble_temperature_2m_min": [1.0],
            "weather_ensemble_temperature_2m_max": [3.0],
            "weather_ensemble_temperature_2m_spread": [2.0],
            "weather_gfs_global_wind_shear_100m_minus_10m": [4.0],
        }
    )

    assert selected_weather_columns(frame, "ensemble_mean_spread") == [
        "weather_ensemble_temperature_2m_mean",
        "weather_ensemble_temperature_2m_spread",
    ]
    assert selected_weather_columns(
        frame,
        "ensemble_mean",
        include_derived_features=True,
    ) == [
        "weather_ensemble_temperature_2m_mean",
        "weather_gfs_global_wind_shear_100m_minus_10m",
    ]


def test_training_frame_materializes_requested_ablation_covariates() -> None:
    panel = _panel(periods=24 * 7)
    weather = _weather(panel)

    training, covariates, diagnostics = make_lora_training_frame(
        panel,
        weather,
        first_eval_origin=pd.Timestamp("2024-01-06T10:00:00Z"),
        context_length=24,
        prediction_length=24,
        weather_covariate_mode="ensemble_mean_spread",
        cross_learning=True,
        add_weather_derived_features=True,
    )

    assert "weather_ensemble_temperature_2m_mean" in covariates
    assert "weather_ensemble_temperature_2m_spread" in covariates
    assert "weather_gfs_global_wind_shear_100m_minus_10m" in covariates
    assert training[covariates].notna().all().all()
    assert diagnostics["cross_learning"] is True
    assert diagnostics["add_weather_derived_features"] is True


def test_manifest_and_runtime_validate_the_same_ablation_policy() -> None:
    panel = pd.DataFrame(
        {
            "dataset_version": ["v1"],
            "market_regime": ["native_hourly"],
            "native_resolution_minutes": [60],
            "target_aggregation": ["identity"],
            "target_definition": ["hourly_day_ahead_area_price_dkk_per_mwh"],
        }
    )
    weather = pd.DataFrame({"ds_utc": [pd.Timestamp("2025-01-01T00:00:00Z")]})
    args = Namespace(
        base_model_id="amazon/chronos-2",
        context_length=24,
        prediction_length=24,
        num_steps=1,
        batch_size=1,
        learning_rate=1e-4,
        weather_covariate_mode="ensemble_mean_spread",
        cross_learning=True,
        add_weather_ensemble_features=True,
        add_weather_derived_features=True,
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
        covariates=[
            "local_hour",
            "weather_ensemble_temperature_2m_mean",
            "weather_ensemble_temperature_2m_spread",
        ],
        diagnostics={},
    )
    config = Chronos2LoRAWeatherConfig(
        weather_covariate_mode="ensemble_mean_spread",
        cross_learning=True,
        add_weather_derived_features=True,
        weather_future_fallback_policy="zero",
    )

    validate_artifact_weather_policy(manifest, config)
    assert manifest["cross_learning"] is True
    assert manifest["add_weather_ensemble_features"] is True
    assert manifest["add_weather_derived_features"] is True

    with pytest.raises(ValueError, match="ablation policy"):
        validate_artifact_weather_policy(
            manifest,
            Chronos2LoRAWeatherConfig(
                weather_covariate_mode="ensemble_mean_spread",
                cross_learning=False,
                add_weather_derived_features=True,
                weather_future_fallback_policy="zero",
            ),
        )


def test_release_identity_includes_serving_ablation_policy() -> None:
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
        {"ds_utc": [pd.Timestamp("2025-01-01T00:00:00Z")]}
    )
    common = {
        "base_model_id": "amazon/chronos-2",
        "base_model_revision": "immutable-revision",
        "context_length": 24,
        "prediction_length": 24,
        "num_steps": 1,
        "batch_size": 1,
        "learning_rate": 1e-4,
        "weather_covariate_mode": "ensemble_mean",
        "add_weather_ensemble_features": True,
        "add_weather_derived_features": False,
        "weather_horizon_min_coverage": 1.0,
        "weather_future_fallback_policy": "zero",
        "validation_scores_path": None,
        "validation_score_model_label": None,
        "panel_path": "panel.parquet",
        "weather_features_long_path": "weather.parquet",
    }
    independent = make_manifest(
        args=Namespace(**common, cross_learning=False),
        panel=panel,
        weather=weather,
        covariates=["local_hour", "weather_ensemble_temperature_2m_mean"],
        diagnostics={},
    )
    joint = make_manifest(
        args=Namespace(**common, cross_learning=True),
        panel=panel,
        weather=weather,
        covariates=["local_hour", "weather_ensemble_temperature_2m_mean"],
        diagnostics={},
    )

    assert (
        independent["artifact_content_sha256"]
        == joint["artifact_content_sha256"]
    )
    assert independent["serving_policy_sha256"] != joint["serving_policy_sha256"]
    assert independent["release_id"] != joint["release_id"]
    validate_chronos_release_identity(independent)
    tampered = dict(independent)
    tampered["cross_learning"] = True
    with pytest.raises(ValueError, match="release identity does not match"):
        validate_chronos_release_identity(tampered)


def test_release_identity_validation_keeps_legacy_release_ids_compatible() -> None:
    validate_chronos_release_identity({"release_id": "legacy-release"})

    with pytest.raises(ValueError, match="incomplete"):
        validate_chronos_release_identity(
            {
                "release_id": "new-release",
                "serving_policy_sha256": "partial",
            }
        )


def test_prediction_passes_cross_learning_to_chronos() -> None:
    pipeline = _CapturePipeline()
    model = Chronos2LoRAWeatherDayAhead(
        config=Chronos2LoRAWeatherConfig(cross_learning=True),
        pipeline=pipeline,
    )
    frames = Chronos2ProductionFrames(
        context_df=pd.DataFrame(
            {
                "item_id": ["DK1"],
                "timestamp": [pd.Timestamp("2025-01-01T00:00:00")],
                "target": [1.0],
            }
        ),
        future_df=pd.DataFrame(
            {
                "item_id": ["DK1"],
                "timestamp": [pd.Timestamp("2025-01-01T01:00:00")],
            }
        ),
        horizon_metadata=pd.DataFrame(),
        prediction_length=1,
        covariates=[],
        weather_coverage_by_series={},
    )

    model._predict_df(frames)

    assert pipeline.kwargs["cross_learning"] is True


def test_training_cli_exposes_ablation_flags(monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_chronos_lora.py",
            "--weather-covariate-mode",
            "ensemble_mean_spread",
            "--cross-learning",
            "--add-weather-derived-features",
        ],
    )

    args = parse_args()

    assert args.weather_covariate_mode == "ensemble_mean_spread"
    assert args.cross_learning is True
    assert args.add_weather_derived_features is True


class _CapturePipeline:
    def __init__(self) -> None:
        self.kwargs = {}

    def predict_df(self, context_df, future_df, prediction_length, **kwargs):
        self.kwargs = kwargs
        return pd.DataFrame(
            {
                "item_id": ["DK1"],
                "timestamp": future_df["timestamp"],
                "0.1": [0.0],
                "0.5": [1.0],
                "0.9": [2.0],
            }
        )


def _panel(*, periods: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "unique_id": ["day_ahead_price_DK1"] * periods,
            "area": ["DK1"] * periods,
            "ds_utc": pd.date_range("2024-01-01", periods=periods, freq="h", tz="UTC"),
            "y": range(periods),
        }
    )
    return add_copenhagen_calendar(frame)


def _weather(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    features = {
        "weather_gfs_global_lead2d_temperature_2m": 1.0,
        "weather_icon_eu_lead2d_temperature_2m": 3.0,
        "weather_gfs_global_lead2d_wind_speed_10m": 5.0,
        "weather_gfs_global_lead2d_wind_speed_100m": 8.0,
    }
    for row in panel[["area", "ds_utc"]].itertuples(index=False):
        for feature_name, value in features.items():
            rows.append(
                {
                    "area": row.area,
                    "ds_utc": row.ds_utc,
                    "feature_name": feature_name,
                    "value": value,
                    "location_coverage_ratio": 1.0,
                    "location_coverage_pass": True,
                    "feature_group_pass": True,
                    "forecast_available_at_utc": row.ds_utc - pd.Timedelta(days=2),
                }
            )
    return pd.DataFrame(rows)
