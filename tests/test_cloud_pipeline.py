from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from dkenergy_forecast.cloud_pipeline import CloudPipelineConfig, run_cloud_pipeline


def test_cloud_pipeline_syncs_state_runs_daily_and_uploads_latest_last(tmp_path) -> None:
    store = tmp_path / "store"
    model = store / "models" / "chronos2_lora_calendar_weather_ctx1024_v1"
    model.mkdir(parents=True)
    (model / "manifest.json").write_text("{}", encoding="utf-8")
    (store / "state" / "data" / "model_ready").mkdir(parents=True)
    (store / "state" / "data" / "model_ready" / "seed.json").write_text("seed", encoding="utf-8")
    (store / "state" / "data" / "raw" / "energi_data_service" / "seed").mkdir(parents=True)
    (store / "state" / "data" / "raw" / "energi_data_service" / "seed" / "batch.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (store / "state" / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (store / "state" / "data" / "raw" / "old.json").write_text("old", encoding="utf-8")
    (store / "forecast_runs" / "previous_run").mkdir(parents=True)
    (store / "forecast_runs" / "previous_run" / "predictions.parquet").write_text("old-predictions", encoding="utf-8")
    workdir = tmp_path / "workdir"
    calls = []

    def fake_runner(command, *, cwd, env, check):
        calls.append((command, cwd, env, check))
        assert env["DKENERGY_RUNTIME_ROOT"] == str(workdir)
        assert env["WITH_WEATHER"] == "1"
        assert (
            env["DKENERGY_CHRONOS_MODEL_ARTIFACT_PATH"]
            == str(workdir / "artifacts" / "models" / "chronos2_lora_calendar_weather_ctx1024_v1")
        )
        assert "--with-weather" in command
        _write_pipeline_outputs(workdir)
        return subprocess.CompletedProcess(command, 0)

    uploaded = run_cloud_pipeline(
        CloudPipelineConfig(
            artifact_store_uri=f"file://{store}",
            workdir=workdir,
            model_artifact_uri=f"file://{model}",
            score_max_origins=1,
        ),
        command_runner=fake_runner,
    )

    assert calls
    assert (workdir / "data" / "model_ready" / "seed.json").exists()
    assert (workdir / "data" / "raw" / "energi_data_service" / "seed" / "batch.json").exists()
    assert (workdir / "artifacts" / "forecast_runs" / "previous_run" / "predictions.parquet").exists()
    assert not (workdir / "data" / "raw" / "old.json").exists()
    assert "forecast_runs/run_1/manifest.json" in uploaded
    assert "published_forecast_history/model_scores.parquet" in uploaded
    assert uploaded[-1] == "latest/forecast_dashboard.json"
    assert (store / "latest" / "forecast_dashboard.json").exists()
    assert (store / "latest" / "price_panel_hourly_v1.parquet").exists()


def test_cloud_pipeline_fails_when_model_manifest_is_missing(tmp_path) -> None:
    store = tmp_path / "store"
    model = store / "models" / "chronos2_lora_calendar_weather_ctx1024_v1"
    model.mkdir(parents=True)

    with pytest.raises(FileNotFoundError, match="manifest.json"):
        run_cloud_pipeline(
            CloudPipelineConfig(
                artifact_store_uri=f"file://{store}",
                workdir=tmp_path / "workdir",
                model_artifact_uri=f"file://{model}",
            ),
            command_runner=lambda *args, **kwargs: subprocess.CompletedProcess(args, 0),
        )


def test_cloud_pipeline_fails_when_weather_artifact_is_missing(tmp_path) -> None:
    store = tmp_path / "store"
    model = store / "models" / "chronos2_lora_calendar_weather_ctx1024_v1"
    model.mkdir(parents=True)
    (model / "manifest.json").write_text("{}", encoding="utf-8")
    workdir = tmp_path / "workdir"

    def fake_runner(command, *, cwd, env, check):
        _write_pipeline_outputs(workdir, include_weather=False)
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(FileNotFoundError, match="weather feature artifact is missing"):
        run_cloud_pipeline(
            CloudPipelineConfig(
                artifact_store_uri=f"file://{store}",
                workdir=workdir,
                model_artifact_uri=f"file://{model}",
            ),
            command_runner=fake_runner,
        )


def test_cloud_pipeline_fails_when_weather_artifact_is_stale(tmp_path, monkeypatch) -> None:
    store = tmp_path / "store"
    model = store / "models" / "chronos2_lora_calendar_weather_ctx1024_v1"
    model.mkdir(parents=True)
    (model / "manifest.json").write_text("{}", encoding="utf-8")
    workdir = tmp_path / "workdir"
    monkeypatch.setenv("DKENERGY_WEATHER_MAX_STALENESS_HOURS", "1")

    def fake_runner(command, *, cwd, env, check):
        _write_pipeline_outputs(
            workdir,
            weather_timestamp=pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=3),
        )
        return subprocess.CompletedProcess(command, 0)

    with pytest.raises(ValueError, match="weather feature artifact is stale"):
        run_cloud_pipeline(
            CloudPipelineConfig(
                artifact_store_uri=f"file://{store}",
                workdir=workdir,
                model_artifact_uri=f"file://{model}",
            ),
            command_runner=fake_runner,
        )


def test_cloud_pipeline_dry_run_does_not_require_model_artifact(tmp_path) -> None:
    calls = []

    def fake_runner(command, *, cwd, env, check):
        calls.append(command)
        assert "--dry-run" in command
        return subprocess.CompletedProcess(command, 0)

    uploaded = run_cloud_pipeline(
        CloudPipelineConfig(
            artifact_store_uri=f"file://{tmp_path / 'store'}",
            workdir=tmp_path / "workdir",
            model_artifact_uri=f"file://{tmp_path / 'missing_model'}",
        ),
        dry_run=True,
        command_runner=fake_runner,
    )

    assert uploaded == []
    assert calls


def _write_pipeline_outputs(
    workdir: Path,
    *,
    include_weather: bool = True,
    weather_timestamp: pd.Timestamp | None = None,
) -> None:
    for path in [
        workdir / "data" / "model_ready",
        workdir / "data" / "features",
        workdir / "results" / "latest_forecast",
        workdir / "results" / "recent_scores",
        workdir / "results" / "published_forecast_history",
        workdir / "artifacts" / "forecast_runs" / "run_1",
        workdir / "app_data",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    (workdir / "data" / "model_ready" / "price_panel_hourly_v1.parquet").write_text("panel", encoding="utf-8")
    (workdir / "data" / "model_ready" / "price_panel_hourly_v1.qa.json").write_text("{}", encoding="utf-8")
    (workdir / "results" / "latest_forecast" / "predictions.parquet").write_text("predictions", encoding="utf-8")
    (workdir / "results" / "latest_forecast" / "manifest.json").write_text("{}", encoding="utf-8")
    (workdir / "results" / "recent_scores" / "model_scores.parquet").write_text("scores", encoding="utf-8")
    (workdir / "results" / "published_forecast_history" / "predictions.parquet").write_text(
        "published predictions",
        encoding="utf-8",
    )
    (workdir / "results" / "published_forecast_history" / "model_scores.parquet").write_text(
        "published scores",
        encoding="utf-8",
    )
    (workdir / "artifacts" / "forecast_runs" / "run_1" / "manifest.json").write_text("{}", encoding="utf-8")
    (workdir / "app_data" / "forecast_dashboard.json").write_text("{}", encoding="utf-8")
    if include_weather:
        timestamp = weather_timestamp or pd.Timestamp.now(tz="UTC")
        pd.DataFrame(
            {
                "forecast_available_at_utc": [timestamp],
                "ds_utc": [timestamp],
                "area": ["DK1"],
            }
        ).to_parquet(
            workdir
            / "data"
            / "features"
            / "weather_open_meteo_area_hourly_long_open_meteo_previous_runs_v1.parquet",
            index=False,
        )
