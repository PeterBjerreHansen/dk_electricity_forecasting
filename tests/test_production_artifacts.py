from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
from dkenergy_forecast.models.comparison_registry import (
    comparison_model_factories,
    comparison_model_specs,
)
from dkenergy_forecast.publishing import (
    build_published_forecast_history,
    build_published_forecast_scores,
    build_dashboard_payload,
    make_forecast_run_manifest,
    normalize_published_predictions,
    validate_evaluated_prediction_artifact_schema,
    validate_model_scores_schema,
    validate_prediction_artifact_schema,
    write_forecast_run_artifacts,
    write_published_forecast_history,
)
from dkenergy_forecast.types import add_copenhagen_calendar


ROOT = Path(__file__).resolve().parents[1]


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
        "score_predictions": str(tmp_path / "run" / "score_predictions.parquet"),
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
        score_predictions=predictions,
    )

    validate_prediction_artifact_schema(predictions)
    validate_evaluated_prediction_artifact_schema(predictions)
    validate_model_scores_schema(scores)
    written = write_forecast_run_artifacts(
        tmp_path / "run",
        predictions=predictions,
        scores=scores,
        manifest=manifest,
        score_predictions=predictions,
        dashboard=dashboard,
    )

    assert written["predictions"].exists()
    assert written["score_predictions"].exists()
    assert written["model_scores"].exists()
    assert written["manifest"].exists()
    assert written["dashboard"].exists()
    saved_manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
    assert saved_manifest["run_id"] == "test_run"
    assert saved_manifest["status"] == "success"
    saved_dashboard = json.loads(written["dashboard"].read_text(encoding="utf-8"))
    assert len(saved_dashboard["recent_predictions"]) == len(predictions)


def test_published_prediction_validation_rejects_duplicate_keys() -> None:
    predictions = normalize_published_predictions(_predictions())
    duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate key rows"):
        validate_prediction_artifact_schema(duplicated)


def test_evaluated_prediction_validation_requires_actuals() -> None:
    predictions = normalize_published_predictions(_predictions()).drop(columns=["y"])

    with pytest.raises(ValueError, match="evaluated predictions"):
        validate_evaluated_prediction_artifact_schema(predictions)


def test_mixed_family_predictions_with_optional_quantiles_validate() -> None:
    frame = _predictions().copy()
    mixed = pd.concat(
        [
            frame.assign(model_label="same_hour_last_week", q10=float("nan"), q50=float("nan"), q90=float("nan")),
            frame.assign(model_label="catboost_price_manual_v1", q10=float("nan"), q50=float("nan"), q90=float("nan")),
            frame.assign(model_label="chronos_zero_shot_v1"),
        ],
        ignore_index=True,
    )

    published = normalize_published_predictions(mixed)
    scores = model_score_table(published)

    validate_prediction_artifact_schema(published)
    validate_model_scores_schema(scores)
    assert set(published["model_label"]) == {
        "same_hour_last_week",
        "catboost_price_manual_v1",
        "chronos_zero_shot_v1",
    }


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


def test_future_dashboard_payload_serializes_missing_actuals_as_null(tmp_path) -> None:
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
    scores = model_score_table(_predictions())
    manifest = make_forecast_run_manifest(
        run_id="future_run",
        forecast_origin_utc=origin,
        predictions=predictions,
        scores=scores,
        artifact_paths={},
        dataset_version="v1",
        git_commit_value="abc123",
    )
    dashboard = build_dashboard_payload(
        predictions=predictions,
        scores=scores,
        manifest=manifest,
    )

    written = write_forecast_run_artifacts(
        tmp_path / "run",
        predictions=predictions,
        scores=scores,
        manifest=manifest,
        dashboard=dashboard,
    )

    saved_dashboard = json.loads(written["dashboard"].read_text(encoding="utf-8"))
    first_prediction = saved_dashboard["predictions"][0]
    assert first_prediction["price_available_at_utc"] is None
    assert first_prediction["actual_price"] is None


def test_published_forecast_history_scores_immutable_run_predictions(tmp_path) -> None:
    artifact_root = tmp_path / "forecast_runs"
    run_dir = artifact_root / "forecast_20240102T100000Z"
    run_dir.mkdir(parents=True)
    predictions = add_copenhagen_calendar(
        pd.DataFrame(
            {
                "unique_id": ["day_ahead_price_DK1", "day_ahead_price_DK1"],
                "forecast_origin_utc": [pd.Timestamp("2024-01-02T10:00:00Z")] * 2,
                "ds_utc": pd.date_range("2024-01-03T00:00:00Z", periods=2, freq="h"),
                "area": ["DK1", "DK1"],
                "model_label": ["same_hour_last_week", "same_hour_last_week"],
                "y_pred": [9.0, 22.0],
                "horizon": [1, 2],
                "dataset_version": ["v1", "v1"],
            }
        )
    )
    normalize_published_predictions(predictions).to_parquet(run_dir / "predictions.parquet", index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "forecast_20240102T100000Z",
                "created_at_utc": "2024-01-02T10:00:00Z",
                "forecast_origin_utc": "2024-01-02T10:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    panel = add_copenhagen_calendar(
        pd.DataFrame(
            {
                "unique_id": ["day_ahead_price_DK1", "day_ahead_price_DK1"],
                "ds_utc": pd.date_range("2024-01-03T00:00:00Z", periods=2, freq="h"),
                "area": ["DK1", "DK1"],
                "y": [10.0, 20.0],
                "dataset_version": ["v1", "v1"],
            }
        )
    )

    history = build_published_forecast_history(artifact_root, panel)
    scores = build_published_forecast_scores(history)
    written = write_published_forecast_history(
        tmp_path / "published_forecast_history",
        predictions=history,
        scores=scores,
    )

    assert history["run_id"].tolist() == ["forecast_20240102T100000Z", "forecast_20240102T100000Z"]
    assert history["y"].tolist() == pytest.approx([10.0, 20.0])
    assert scores.loc[scores["area"] == "ALL", "mae"].iloc[0] == pytest.approx(1.5)
    assert scores["score_source"].eq("published_forecast_history").all()
    assert written["published_history_predictions"].exists()
    assert written["published_history_scores"].exists()


def test_score_published_forecasts_script_scores_saved_runs(tmp_path) -> None:
    artifact_root = tmp_path / "forecast_runs"
    run_dir = artifact_root / "forecast_20240102T100000Z"
    run_dir.mkdir(parents=True)
    predictions = add_copenhagen_calendar(
        pd.DataFrame(
            {
                "unique_id": ["day_ahead_price_DK1", "day_ahead_price_DK1"],
                "forecast_origin_utc": [pd.Timestamp("2024-01-02T10:00:00Z")] * 2,
                "ds_utc": pd.date_range("2024-01-03T00:00:00Z", periods=2, freq="h"),
                "area": ["DK1", "DK1"],
                "model_label": ["same_hour_last_week", "same_hour_last_week"],
                "y_pred": [9.0, 22.0],
                "horizon": [1, 2],
                "dataset_version": ["v1", "v1"],
            }
        )
    )
    normalize_published_predictions(predictions).to_parquet(run_dir / "predictions.parquet", index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "forecast_20240102T100000Z",
                "created_at_utc": "2024-01-02T10:00:00Z",
                "forecast_origin_utc": "2024-01-02T10:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    panel = add_copenhagen_calendar(
        pd.DataFrame(
            {
                "unique_id": ["day_ahead_price_DK1", "day_ahead_price_DK1"],
                "ds_utc": pd.date_range("2024-01-03T00:00:00Z", periods=2, freq="h"),
                "area": ["DK1", "DK1"],
                "y": [10.0, 20.0],
                "dataset_version": ["v1", "v1"],
            }
        )
    )
    panel_path = tmp_path / "price_panel.parquet"
    qa_path = tmp_path / "price_panel.qa.json"
    output_dir = tmp_path / "published_forecast_history"
    panel.to_parquet(panel_path, index=False)
    qa_path.write_text(json.dumps({"artifact_status": "final_historical"}), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            "scripts/score_published_forecasts.py",
            "--artifact-root",
            str(artifact_root),
            "--panel-path",
            str(panel_path),
            "--qa-path",
            str(qa_path),
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    history = pd.read_parquet(output_dir / "predictions.parquet")
    scores = pd.read_parquet(output_dir / "model_scores.parquet")
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "Evaluated prediction rows: 2" in result.stdout
    assert history["y"].tolist() == pytest.approx([10.0, 20.0])
    assert scores.loc[scores["area"] == "ALL", "mae"].iloc[0] == pytest.approx(1.5)
    assert scores["score_source"].eq("published_forecast_history").all()
    assert manifest["prediction_row_count"] == 2
    assert manifest["evaluated_forecast_run_count"] == 1


def test_production_registry_contains_only_deployed_models(monkeypatch, tmp_path) -> None:
    specs = production_model_specs()

    assert default_production_model_labels() == [
        "same_hour_last_week",
        "median_weekday_exp_hl4_floor10_42d__median_weekend_exp_hl28_floor20_56d",
        "chronos2_lora_calendar_weather_ctx1024_v1",
    ]
    assert set(specs) == set(default_production_model_labels())
    assert specs["chronos2_lora_calendar_weather_ctx1024_v1"].family == "chronos"
    assert specs["chronos2_lora_calendar_weather_ctx1024_v1"].required_extra == "chronos"
    assert specs["chronos2_lora_calendar_weather_ctx1024_v1"].default_enabled
    assert specs["chronos2_lora_calendar_weather_ctx1024_v1"].emits_quantiles
    assert specs["chronos2_lora_calendar_weather_ctx1024_v1"].requires_weather
    monkeypatch.setattr("dkenergy_forecast.models.registry.ensure_chronos_available", lambda: None)
    assert set(latest_publish_model_factories()) == set(default_production_model_labels())
    chronos_factory = latest_publish_model_factories(
        ["chronos2_lora_calendar_weather_ctx1024_v1"],
        weather_features_long_path=tmp_path / "weather.parquet",
        chronos_model_artifact_path=tmp_path / "chronos_model",
    )["chronos2_lora_calendar_weather_ctx1024_v1"]
    chronos_model = chronos_factory()
    assert chronos_model.config.weather_features_long_path == tmp_path / "weather.parquet"
    assert chronos_model.config.model_artifact_path == tmp_path / "chronos_model"
    with pytest.raises(ValueError, match="Unknown production model"):
        latest_publish_model_factories(["weather_catboost_all_weather"])
    with pytest.raises(ValueError, match="Unknown production model"):
        latest_publish_model_factories(["chronos_zero_shot_v1"])


def test_comparison_registry_holds_notebook_and_smoke_models(monkeypatch) -> None:
    specs = comparison_model_specs()

    assert specs["rolling_median_local_hour_28d"].family == "baseline"
    assert specs["rolling_median_hour_weekend_56d"].family == "baseline"
    assert specs["catboost_price_manual_v1"].family == "catboost"
    assert specs["catboost_price_manual_v1"].required_extra == "catboost"
    assert not specs["catboost_price_manual_v1"].supports_latest_publish
    assert specs["chronos_zero_shot_v1"].family == "chronos"
    assert specs["chronos_zero_shot_v1"].required_extra == "chronos"
    assert specs["chronos_zero_shot_v1"].emits_quantiles
    monkeypatch.setattr("dkenergy_forecast.models.comparison_registry.ensure_chronos_zero_shot_available", lambda: None)
    factories = comparison_model_factories(["chronos_zero_shot_v1"])
    assert set(factories) == {"chronos_zero_shot_v1"}


def test_selecting_comparison_catboost_model_fails_fast_without_catboost(monkeypatch) -> None:
    monkeypatch.setattr(
        "dkenergy_forecast.models.comparison_registry.ensure_catboost_available",
        lambda: (_ for _ in ()).throw(ImportError("Install CatBoost with catboost extra")),
    )

    with pytest.raises(ImportError, match="CatBoost"):
        comparison_model_factories(["catboost_price_manual_v1"])


def test_publish_model_listing_includes_required_extra_and_quantile_metadata() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_publish_forecast.py", "--list-models"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert (
        "chronos2_lora_calendar_weather_ctx1024_v1: chronos, default, latest-publish, "
        "extra=chronos, quantiles, weather"
    ) in result.stdout
    assert "catboost_price_manual_v1" not in result.stdout
    assert "chronos_zero_shot_v1" not in result.stdout


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
            "model_label": ["example_quantile_model", "example_quantile_model"],
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
