from __future__ import annotations

import json

import pandas as pd
import pytest

from dkenergy_forecast.backtesting.origins import choose_recent_complete_daily_origins
from dkenergy_forecast.backtesting.horizons import make_danish_delivery_day_horizon
from dkenergy_forecast.backtesting.rolling_origin import rolling_origin_backtest
from dkenergy_forecast.evaluation.summary import model_score_table, probabilistic_metric_table
from dkenergy_forecast.models.registry import (
    default_production_model_labels,
    latest_publish_model_factories,
    production_model_specs,
)
from dkenergy_forecast.publishing import (
    build_dashboard_payload,
    make_forecast_run_manifest,
    normalize_published_predictions,
    validate_model_scores_schema,
    validate_prediction_artifact_schema,
    write_forecast_run_artifacts,
)
from dkenergy_forecast.types import add_copenhagen_calendar


def test_choose_recent_complete_daily_origins_is_shared_and_bounded() -> None:
    panel = _panel(periods=24 * 20)

    origins = choose_recent_complete_daily_origins(
        panel,
        days=10,
        at_hour_utc=10,
        max_origins=3,
        min_history_days=2,
        holdout_days=2,
    )

    assert len(origins) == 3
    assert origins["forecast_origin_utc"].is_monotonic_increasing
    assert origins["forecast_origin_utc"].max() <= (panel["ds_utc"].max() - pd.Timedelta(days=2)).normalize() + pd.Timedelta(hours=10)


def test_model_score_table_matches_published_contract() -> None:
    predictions = _predictions()

    scores = model_score_table(predictions)
    probabilistic = probabilistic_metric_table(predictions)

    assert {"model_label", "area", "mae", "rmse", "bias", "coverage", "interval_width"}.issubset(scores.columns)
    assert set(scores["area"]) == {"ALL", "DK1"}
    assert scores.loc[scores["area"] == "ALL", "coverage"].iloc[0] == pytest.approx(1.0)
    assert scores.loc[scores["area"] == "ALL", "interval_width"].iloc[0] == pytest.approx(4.0)
    assert set(probabilistic["metric"]) == {
        "pinball_q10",
        "pinball_q50",
        "pinball_q90",
        "p10_p90_coverage",
        "p10_p90_avg_width",
    }


def test_published_artifacts_validate_and_write(tmp_path) -> None:
    predictions = normalize_published_predictions(_predictions())
    scores = model_score_table(predictions)
    artifact_paths = {
        "predictions": str(tmp_path / "run" / "predictions.parquet"),
        "model_scores": str(tmp_path / "run" / "model_scores.parquet"),
        "manifest": str(tmp_path / "run" / "manifest.json"),
    }
    manifest = make_forecast_run_manifest(
        run_id="test_run",
        forecast_origin_utc="2024-01-02T10:00:00Z",
        predictions=predictions,
        scores=scores,
        artifact_paths=artifact_paths,
        dataset_version="v1",
        git_commit_value="abc123",
    )
    dashboard = build_dashboard_payload(
        predictions=predictions,
        scores=scores,
        manifest=manifest,
    )

    validate_prediction_artifact_schema(predictions)
    validate_model_scores_schema(scores)
    written = write_forecast_run_artifacts(
        tmp_path / "run",
        predictions=predictions,
        scores=scores,
        manifest=manifest,
        dashboard=dashboard,
    )

    assert written["predictions"].exists()
    assert written["model_scores"].exists()
    assert written["manifest"].exists()
    assert written["dashboard"].exists()
    saved_manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
    assert saved_manifest["run_id"] == "test_run"
    assert saved_manifest["status"] == "success"


def test_published_prediction_validation_rejects_duplicate_keys() -> None:
    predictions = normalize_published_predictions(_predictions())
    duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate key rows"):
        validate_prediction_artifact_schema(duplicated)


def test_future_publish_predictions_preserve_horizon_metadata_without_actuals() -> None:
    panel = _panel(periods=24 * 14)
    origin = panel["ds_utc"].max().normalize() + pd.Timedelta(days=1, hours=10)
    origins = pd.DataFrame({"forecast_origin_utc": [origin]})
    factories = latest_publish_model_factories(["same_hour_last_week"])

    predictions = rolling_origin_backtest(
        model_factory=factories["same_hour_last_week"],
        panel=panel,
        origins=origins,
        horizon_builder=lambda panel_arg, origin_arg: make_danish_delivery_day_horizon(
            panel_arg,
            origin_arg,
            days_ahead=1,
        ),
        min_train_rows=24,
    )
    predictions["model_label"] = "same_hour_last_week"
    published = normalize_published_predictions(predictions)

    validate_prediction_artifact_schema(published)
    assert len(published) == 24
    assert published["area"].eq("DK1").all()
    assert published["ds_local"].notna().all()
    assert published["local_date"].eq("2024-01-16").all()
    assert published["actual_price"].isna().all()


def test_production_registry_defaults_to_baselines_and_registers_weather_catboost() -> None:
    specs = production_model_specs()

    assert default_production_model_labels() == [
        "same_hour_last_week",
        "rolling_median_local_hour_28d",
        "rolling_median_hour_weekend_56d",
    ]
    assert "catboost_quantile" in specs
    assert not specs["catboost_quantile"].default_enabled
    assert "weather_catboost_all_weather" in specs
    assert specs["weather_catboost_all_weather"].family == "weather_catboost"
    assert not specs["weather_catboost_all_weather"].default_enabled
    assert not specs["weather_catboost_all_weather"].supports_latest_publish
    assert set(latest_publish_model_factories()) == set(default_production_model_labels())
    with pytest.raises(ValueError, match="not yet wired"):
        latest_publish_model_factories(["weather_catboost_all_weather"])


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


def _predictions() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "unique_id": ["day_ahead_price_DK1", "day_ahead_price_DK1"],
            "forecast_origin_utc": [pd.Timestamp("2024-01-02T10:00:00Z")] * 2,
            "ds_utc": pd.date_range("2024-01-03T00:00:00Z", periods=2, freq="h"),
            "area": ["DK1", "DK1"],
            "model_label": ["catboost_quantile", "catboost_quantile"],
            "y": [10.0, 20.0],
            "y_pred": [11.0, 18.0],
            "q10": [8.0, 17.0],
            "q50": [11.0, 18.0],
            "q90": [12.0, 21.0],
            "horizon": [1, 2],
            "dataset_version": ["v1", "v1"],
        }
    )
    return add_copenhagen_calendar(frame)
