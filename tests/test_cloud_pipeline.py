from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
import pytest

from dkenergy_forecast.cloud_pipeline import (
    CloudPipelineConfig,
    CloudScoringConfig,
    run_cloud_pipeline,
    run_cloud_scoring,
)


def test_cloud_pipeline_syncs_state_runs_daily_and_uploads_latest_last(tmp_path) -> None:
    store = tmp_path / "store"
    model = store / "models" / "chronos_weather"
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
            == str(workdir / "artifacts" / "models" / "chronos_weather")
        )
        assert "--with-weather" in command
        assert "--skip-backtest" in command
        assert "--with-diagnostics" not in command
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
    assert "forecast_runs/run_1/COMPLETED.json" in uploaded
    assert uploaded[-1] == "latest.json"
    assert (store / "latest.json").exists()
    assert (store / "forecast_runs" / "run_1" / "forecast_dashboard.json").exists()
    assert (store / "latest" / "price_panel_hourly_v1.parquet").exists()


def test_cloud_pipeline_fails_when_model_manifest_is_missing(tmp_path) -> None:
    store = tmp_path / "store"
    model = store / "models" / "chronos_weather"
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


def test_cloud_pipeline_passes_replay_contract_to_daily_pipeline(tmp_path) -> None:
    store = tmp_path / "store"
    model = store / "models" / "chronos_weather"
    model.mkdir(parents=True)
    (model / "manifest.json").write_text("{}", encoding="utf-8")
    workdir = tmp_path / "workdir"

    def fake_runner(command, *, cwd, env, check):
        assert "--run-kind" in command
        assert command[command.index("--run-kind") + 1] == "replay"
        assert command[command.index("--information-cutoff-utc") + 1] == "2026-07-01T08:00:00Z"
        _write_pipeline_outputs(workdir)
        return subprocess.CompletedProcess(command, 0)

    run_cloud_pipeline(
        CloudPipelineConfig(
            artifact_store_uri=f"file://{store}",
            workdir=workdir,
            model_artifact_uri=f"file://{model}",
            run_kind="replay",
            information_cutoff_utc="2026-07-01T08:00:00Z",
        ),
        command_runner=fake_runner,
    )


def test_cloud_pipeline_rejects_replay_without_cutoff(tmp_path) -> None:
    with pytest.raises(ValueError, match="require information_cutoff_utc"):
        run_cloud_pipeline(
            CloudPipelineConfig(
                artifact_store_uri=f"file://{tmp_path / 'store'}",
                workdir=tmp_path / "workdir",
                model_artifact_uri=f"file://{tmp_path / 'model'}",
                run_kind="replay",
            ),
            dry_run=True,
        )


def test_cloud_pipeline_fails_when_weather_artifact_is_missing(tmp_path) -> None:
    store = tmp_path / "store"
    model = store / "models" / "chronos_weather"
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
    model = store / "models" / "chronos_weather"
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


def test_cloud_scoring_is_independent_from_live_publication(tmp_path) -> None:
    store = tmp_path / "store"
    panel_dir = store / "state" / "data" / "model_ready"
    run_dir = store / "forecast_runs" / "run_1"
    panel_dir.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    (panel_dir / "price_panel_hourly_v1.parquet").write_text("panel", encoding="utf-8")
    (run_dir / "COMPLETED.json").write_text("{}", encoding="utf-8")
    workdir = tmp_path / "workdir"

    def fake_runner(command, *, cwd, env, check):
        assert "score_published_forecasts.py" in command[1]
        output = workdir / "results" / "published_forecast_history"
        output.mkdir(parents=True)
        (output / "model_scores.parquet").write_text("scores", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0)

    uploaded = run_cloud_scoring(
        CloudScoringConfig(
            artifact_store_uri=f"file://{store}",
            workdir=workdir,
        ),
        command_runner=fake_runner,
    )

    assert uploaded == ["published_forecast_history/model_scores.parquet"]


def _write_pipeline_outputs(
    workdir: Path,
    *,
    include_weather: bool = True,
    weather_timestamp: pd.Timestamp | None = None,
) -> None:
    for path in [
        workdir / "data" / "model_ready",
        workdir / "data" / "features",
        workdir / "artifacts" / "forecast_runs" / "run_1",
    ]:
        path.mkdir(parents=True, exist_ok=True)
    (workdir / "data" / "model_ready" / "price_panel_hourly_v1.parquet").write_text("panel", encoding="utf-8")
    (workdir / "data" / "model_ready" / "price_panel_hourly_v1.qa.json").write_text("{}", encoding="utf-8")
    run_dir = workdir / "artifacts" / "forecast_runs" / "run_1"
    (run_dir / "predictions.parquet").write_text("predictions", encoding="utf-8")
    (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (run_dir / "forecast_dashboard.json").write_text("{}", encoding="utf-8")
    (run_dir / "COMPLETED.json").write_text(
        '{"status":"completed","run_id":"run_1","committed_at_utc":"2026-01-01T10:00:00Z"}',
        encoding="utf-8",
    )
    (workdir / "artifacts" / "latest.json").write_text(
        "{\n"
        '  "schema_version": 1,\n'
        '  "run_id": "run_1",\n'
        '  "run_prefix": "forecast_runs/run_1",\n'
        '  "completion_key": "forecast_runs/run_1/COMPLETED.json",\n'
        '  "delivery_date_local": "2999-01-02",\n'
        '  "information_cutoff_utc": "2999-01-01T09:00:00Z",\n'
        '  "decision_deadline_utc": "2999-01-01T12:00:00Z",\n'
        '  "committed_at_utc": "2999-01-01T09:05:00Z",\n'
        '  "status": "completed"\n'
        "}\n",
        encoding="utf-8",
    )
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
