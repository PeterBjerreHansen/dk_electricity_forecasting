from __future__ import annotations

import os
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from dkenergy_forecast.layout import (
    CHRONOS_MODEL_ARTIFACT_RELATIVE_PATH,
    PROJECT_ROOT,
    WEATHER_FEATURES_LONG_PROJECT_RELATIVE_PATH,
    runtime_layout,
)
from dkenergy_forecast.storage import ArtifactStore, join_uri
from dkenergy_forecast.publishing import atomic_write_json
from dkenergy_forecast.types import to_utc_timestamp


MODEL_ARTIFACT_RELATIVE_PATH = CHRONOS_MODEL_ARTIFACT_RELATIVE_PATH
PRODUCTION_CONFIG_PATH = PROJECT_ROOT / "config" / "production.json"
WEATHER_FEATURES_RELATIVE_PATH = WEATHER_FEATURES_LONG_PROJECT_RELATIVE_PATH
STATE_DOWNLOAD_PREFIXES = (
    "state/data/raw/energi_data_service",
    "state/data/raw/open_meteo",
    "state/data/model_ready",
    "state/data/features",
)
STATE_UPLOAD_PREFIXES = (
    ("data/model_ready", "state/data/model_ready"),
    ("data/features", "state/data/features"),
    ("data/normalized", "state/data/normalized"),
    ("data/raw/energi_data_service", "state/data/raw/energi_data_service"),
    ("data/raw/open_meteo", "state/data/raw/open_meteo"),
)


@dataclass(frozen=True)
class CloudPipelineConfig:
    artifact_store_uri: str
    workdir: Path
    model_artifact_uri: str
    python: str = sys.executable
    with_weather: bool = True
    score_max_origins: int | None = None
    run_kind: str = "live"
    information_cutoff_utc: str | None = None


@dataclass(frozen=True)
class CloudScoringConfig:
    artifact_store_uri: str
    workdir: Path
    python: str = sys.executable


CommandRunner = Callable[..., subprocess.CompletedProcess]


def default_model_artifact_uri(artifact_store_uri: str) -> str:
    payload = json.loads(PRODUCTION_CONFIG_PATH.read_text(encoding="utf-8"))
    artifact_path = str(payload["primary"]["artifact_path"])
    relative_path = artifact_path.removeprefix("artifacts/").strip("/")
    if not relative_path:
        raise ValueError("production.json primary artifact_path must not be empty")
    return join_uri(artifact_store_uri, relative_path)


def run_cloud_pipeline(
    config: CloudPipelineConfig,
    *,
    dry_run: bool = False,
    command_runner: CommandRunner = subprocess.run,
) -> list[str]:
    if config.run_kind == "replay" and not config.information_cutoff_utc:
        raise ValueError("Replay cloud runs require information_cutoff_utc")
    if config.run_kind == "live" and config.information_cutoff_utc:
        raise ValueError("Live cloud runs cannot set information_cutoff_utc")
    if config.run_kind not in {"live", "replay"}:
        raise ValueError(f"Unsupported cloud run_kind: {config.run_kind!r}")

    store = ArtifactStore(config.artifact_store_uri)
    workdir = config.workdir
    workdir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(f"Artifact store: {config.artifact_store_uri}")
        print(f"Workdir: {workdir}")
        print(f"Model artifact: {config.model_artifact_uri}")
    else:
        _download_runtime_state(store, workdir)
        _download_model_artifact(config.model_artifact_uri, workdir)

    command = [
        config.python,
        str(PROJECT_ROOT / "scripts" / "run_daily_pipeline.py"),
        "--skip-backtest",
        "--run-kind",
        config.run_kind,
    ]
    if config.information_cutoff_utc:
        command.extend(["--information-cutoff-utc", config.information_cutoff_utc])
    if config.with_weather:
        command.append("--with-weather")
    if dry_run:
        command.append("--dry-run")

    env = _pipeline_env(config)
    print("+ " + " ".join(command), flush=True)
    command_runner(command, cwd=PROJECT_ROOT, env=env, check=True)

    if dry_run:
        return []
    if config.with_weather:
        _require_fresh_weather_features(workdir / WEATHER_FEATURES_RELATIVE_PATH)
    return _upload_runtime_outputs(store, workdir)


def _pipeline_env(config: CloudPipelineConfig) -> dict[str, str]:
    env = os.environ.copy()
    env["DKENERGY_RUNTIME_ROOT"] = str(config.workdir)
    env["WITH_WEATHER"] = "1" if config.with_weather else "0"
    env["WEATHER_FEATURES_LONG_PATH"] = str(config.workdir / WEATHER_FEATURES_RELATIVE_PATH)
    env["DKENERGY_CHRONOS_MODEL_ARTIFACT_PATH"] = str(_local_model_artifact_path(config.workdir))
    return env


def _download_runtime_state(store: ArtifactStore, workdir: Path) -> None:
    for prefix in STATE_DOWNLOAD_PREFIXES:
        relative = prefix.removeprefix("state/")
        store.download_prefix(prefix, workdir / relative, required=False)
    store.download_prefix("forecast_runs", workdir / "artifacts" / "forecast_runs", required=False)
    store.download_file("latest.json", workdir / "artifacts" / "latest.json", required=False)


def _download_model_artifact(model_artifact_uri: str, workdir: Path) -> None:
    destination = _local_model_artifact_path(workdir)
    ArtifactStore(model_artifact_uri).download_prefix("", destination, required=True)
    manifest = destination / "manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(
            "Chronos LoRA artifact was downloaded but manifest.json is missing: "
            f"{manifest}. Upload the trained artifact with `make aws-bootstrap-model`."
        )


def _local_model_artifact_path(workdir: Path) -> Path:
    return workdir / "artifacts" / MODEL_ARTIFACT_RELATIVE_PATH


def _upload_runtime_outputs(store: ArtifactStore, workdir: Path) -> list[str]:
    paths = runtime_layout(workdir)
    uploaded: list[str] = []
    for source_relative, destination_prefix in STATE_UPLOAD_PREFIXES:
        uploaded.extend(store.upload_prefix(workdir / source_relative, destination_prefix))

    for source, key in _latest_data_artifacts(workdir):
        _require_output(source)
        store.upload_file(source, key)
        uploaded.append(key)

    pointer = _read_latest_pointer(paths.latest_pointer)
    run_prefix = str(pointer["run_prefix"])
    run_dir = paths.latest_pointer.parent / run_prefix
    completion = run_dir / "COMPLETED.json"
    _require_output(completion)
    if not store.exists(f"{run_prefix}/COMPLETED.json"):
        for source in sorted(path for path in run_dir.iterdir() if path.name != "COMPLETED.json"):
            if not source.is_file():
                continue
            key = f"{run_prefix}/{source.name}"
            store.upload_file(source, key)
            uploaded.append(key)

        committed_at = to_utc_timestamp(datetime.now(timezone.utc))
        deadline = to_utc_timestamp(pointer["decision_deadline_utc"])
        if committed_at > deadline:
            raise RuntimeError(
                "Cloud publication missed its decision deadline; latest.json was not updated: "
                f"committed_at_utc={committed_at.isoformat()}, "
                f"decision_deadline_utc={deadline.isoformat()}"
            )
        completion_payload = json.loads(completion.read_text(encoding="utf-8"))
        completion_payload["committed_at_utc"] = committed_at
        atomic_write_json(completion, completion_payload)
        store.upload_file(completion, f"{run_prefix}/COMPLETED.json")
        uploaded.append(f"{run_prefix}/COMPLETED.json")
        pointer["committed_at_utc"] = committed_at
        atomic_write_json(paths.latest_pointer, pointer)

    store.upload_file(paths.latest_pointer, "latest.json")
    uploaded.append("latest.json")
    return uploaded


def _latest_data_artifacts(workdir: Path) -> list[tuple[Path, str]]:
    paths = runtime_layout(workdir)
    return [
        (
            paths.price_panel,
            f"latest/{paths.price_panel.name}",
        ),
        (
            paths.price_panel_qa,
            f"latest/{paths.price_panel_qa.name}",
        ),
    ]


def _read_latest_pointer(path: Path) -> dict[str, object]:
    _require_output(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "status",
        "run_id",
        "run_prefix",
        "completion_key",
        "delivery_date_local",
        "information_cutoff_utc",
        "decision_deadline_utc",
        "committed_at_utc",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"Latest forecast pointer is missing fields: {missing}")
    if payload["schema_version"] != 1 or payload["status"] != "completed":
        raise ValueError("Latest forecast pointer is not a completed schema-v1 pointer")
    run_prefix = Path(str(payload["run_prefix"]))
    if run_prefix.is_absolute() or ".." in run_prefix.parts:
        raise ValueError(f"Latest forecast run_prefix must be relative: {run_prefix}")
    return payload


def run_cloud_scoring(
    config: CloudScoringConfig,
    *,
    dry_run: bool = False,
    command_runner: CommandRunner = subprocess.run,
) -> list[str]:
    """Hydrate saved forecasts, score them, and upload diagnostics independently."""

    store = ArtifactStore(config.artifact_store_uri)
    workdir = config.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    if not dry_run:
        store.download_prefix(
            "state/data/model_ready",
            workdir / "data" / "model_ready",
            required=True,
        )
        store.download_prefix(
            "forecast_runs",
            workdir / "artifacts" / "forecast_runs",
            required=True,
        )
    paths = runtime_layout(workdir)
    command = [
        config.python,
        str(PROJECT_ROOT / "scripts" / "score_published_forecasts.py"),
        "--artifact-root",
        str(paths.forecast_runs),
        "--panel-path",
        str(paths.price_panel),
        "--qa-path",
        str(paths.price_panel_qa),
        "--output-dir",
        str(paths.published_history),
        "--allow-incomplete-panel",
    ]
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return []
    command_runner(command, cwd=PROJECT_ROOT, env=os.environ.copy(), check=True)
    return store.upload_prefix(paths.published_history, "published_forecast_history")


def _require_output(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Expected pipeline output is missing: {path}")


def _require_fresh_weather_features(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "Expected Open-Meteo weather feature artifact is missing: "
            f"{path}. Run the cloud pipeline with weather refresh enabled."
        )

    try:
        import pandas as pd

        weather = pd.read_parquet(path)
    except Exception as exc:
        raise ValueError(f"Could not read Open-Meteo weather feature artifact: {path}") from exc

    timestamp_column = next(
        (
            column
            for column in (
                "forecast_available_at_utc",
                "forecast_reference_time",
                "ds_utc",
            )
            if column in weather.columns
        ),
        None,
    )
    if timestamp_column is None:
        raise ValueError(
            "Open-Meteo weather feature artifact must contain one of "
            "`forecast_available_at_utc`, `forecast_reference_time`, or `ds_utc`."
        )

    timestamps = pd.to_datetime(weather[timestamp_column], utc=True, errors="coerce").dropna()
    if timestamps.empty:
        raise ValueError(f"Open-Meteo weather feature artifact has no valid {timestamp_column} values: {path}")

    max_age_hours = int(os.environ.get("DKENERGY_WEATHER_MAX_STALENESS_HOURS", "48"))
    latest = timestamps.max()
    age = pd.Timestamp.now(tz="UTC") - latest
    if age > pd.Timedelta(hours=max_age_hours):
        raise ValueError(
            "Open-Meteo weather feature artifact is stale: "
            f"latest {timestamp_column} is {latest.isoformat()}, older than {max_age_hours} hours. "
            "Check Open-Meteo ingestion before publishing latest artifacts."
        )
