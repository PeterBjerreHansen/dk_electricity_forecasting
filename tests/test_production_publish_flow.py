from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd

from dkenergy_forecast.backtesting.horizons import make_danish_delivery_day_horizon
from dkenergy_forecast.operations.publish_forecast import run_publish_forecast
from dkenergy_forecast.types import add_copenhagen_calendar


def test_chronos_failure_publishes_a_labeled_fixed_fallback(tmp_path, monkeypatch) -> None:
    panel = _complete_context_panel()
    panel_path = tmp_path / "panel.parquet"
    panel_path.write_bytes(b"panel fixture")
    config_path = tmp_path / "production.json"
    config_path.write_text(
        '{"schema_version":1,"primary":{"model":"chronos_weather",'
        '"artifact_path":"models/release-1"},'
        '"fallback":{"model":"weighted_median_v1"}}',
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_predict(model_label, *, request, panel, **kwargs):
        calls.append(model_label)
        if model_label == "chronos_weather":
            raise ValueError("weather contract failed")
        horizon = make_danish_delivery_day_horizon(
            panel,
            request.information_cutoff_utc,
            delivery_date_local=request.delivery_date_local,
        )
        return horizon.assign(
            model_name=model_label,
            model_version="weighted_median_v1",
            y_pred=42.0,
            y=float("nan"),
        )

    monkeypatch.setattr(
        "dkenergy_forecast.operations.publish_forecast.load_price_panel",
        lambda *args, **kwargs: panel,
    )
    monkeypatch.setattr(
        "dkenergy_forecast.operations.publish_forecast._predict_one_model",
        fake_predict,
    )

    result = run_publish_forecast(
        _args(tmp_path, panel_path=panel_path, production_config=config_path),
        project_root=tmp_path,
    )

    assert calls == ["chronos_weather", "weighted_median_v1"]
    assert result.forecast_status == "degraded"
    assert result.published_model == "weighted_median_v1"
    predictions = pd.read_parquet(result.paths["predictions"])
    assert predictions["requested_model"].eq("chronos_weather").all()
    assert predictions["forecast_status"].eq("degraded").all()
    assert predictions["model_release_id"].eq("weighted_median_v1").all()
    manifest = json.loads(result.paths["manifest"].read_text(encoding="utf-8"))
    pointer = json.loads(result.paths["latest_pointer"].read_text(encoding="utf-8"))
    assert manifest["primary_failure"] == {
        "type": "ValueError",
        "message": "weather contract failed",
    }
    assert pointer["forecast_status"] == "degraded"
    assert result.paths["completion"].name == "COMPLETED.json"


def _complete_context_panel() -> pd.DataFrame:
    timestamps = pd.date_range("2026-01-01T23:00:00Z", periods=24, freq="h")
    return add_copenhagen_calendar(
        pd.DataFrame(
            {
                "unique_id": ["day_ahead_price_DK1"] * len(timestamps),
                "area": ["DK1"] * len(timestamps),
                "ds_utc": timestamps,
                "y": range(len(timestamps)),
                "dataset_version": ["test"] * len(timestamps),
            }
        )
    )


def _args(tmp_path, *, panel_path, production_config):
    return SimpleNamespace(
        generated_at_utc="2026-01-02T09:00:00Z",
        information_cutoff_utc=None,
        forecast_origin_utc=None,
        run_kind="live",
        delivery_date_local="2026-01-03",
        decision_deadline_utc="2999-01-02T11:00:00Z",
        decision_cutoff_utc=None,
        decision_deadline_local_time="12:00",
        runtime_root=tmp_path,
        production_config=production_config,
        chronos_model_artifact_path=None,
        panel_path=panel_path,
        qa_path=None,
        allow_incomplete_panel=True,
        min_train_days=1,
        weather_features_long_path=tmp_path / "weather.parquet",
        run_id="live-test",
        artifact_root=tmp_path / "artifacts" / "forecast_runs",
        latest_pointer_path=tmp_path / "artifacts" / "latest.json",
    )
