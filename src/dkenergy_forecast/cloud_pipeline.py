from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dkenergy_forecast.storage import ArtifactStore, join_uri


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_ARTIFACT_RELATIVE_PATH = "models/chronos2_lora_calendar_weather_ctx1024_v1"
WEATHER_FEATURES_RELATIVE_PATH = (
    "data/features/weather_open_meteo_area_hourly_long_open_meteo_previous_runs_v1.parquet"
)
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


CommandRunner = Callable[..., subprocess.CompletedProcess]


def default_model_artifact_uri(artifact_store_uri: str) -> str:
    return join_uri(artifact_store_uri, MODEL_ARTIFACT_RELATIVE_PATH)


def run_cloud_pipeline(
    config: CloudPipelineConfig,
    *,
    dry_run: bool = False,
    command_runner: CommandRunner = subprocess.run,
) -> list[str]:
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
    ]
    if config.with_weather:
        command.append("--with-weather")
    if config.score_max_origins is not None:
        command.extend(["--score-max-origins", str(config.score_max_origins)])
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
    uploaded: list[str] = []
    for source_relative, destination_prefix in STATE_UPLOAD_PREFIXES:
        uploaded.extend(store.upload_prefix(workdir / source_relative, destination_prefix))
    uploaded.extend(store.upload_prefix(workdir / "artifacts" / "forecast_runs", "forecast_runs"))
    uploaded.extend(store.upload_prefix(workdir / "results" / "recent_scores", "recent_scores"))
    uploaded.extend(store.upload_prefix(workdir / "results" / "published_forecast_history", "published_forecast_history"))

    for source, key in _latest_artifacts(workdir):
        _require_output(source)
        store.upload_file(source, key)
        uploaded.append(key)
    return uploaded


def _latest_artifacts(workdir: Path) -> list[tuple[Path, str]]:
    return [
        (
            workdir / "data" / "model_ready" / "price_panel_hourly_v1.parquet",
            "latest/price_panel_hourly_v1.parquet",
        ),
        (
            workdir / "data" / "model_ready" / "price_panel_hourly_v1.qa.json",
            "latest/price_panel_hourly_v1.qa.json",
        ),
        (
            workdir / "results" / "latest_forecast" / "predictions.parquet",
            "latest/predictions.parquet",
        ),
        (
            workdir / "results" / "latest_forecast" / "manifest.json",
            "latest/manifest.json",
        ),
        (
            workdir / "app_data" / "forecast_dashboard.json",
            "latest/forecast_dashboard.json",
        ),
    ]


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
            for column in ("forecast_available_at_utc", "forecast_reference_time", "ds_utc")
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
