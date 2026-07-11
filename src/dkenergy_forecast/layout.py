from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHRONOS_LORA_WEATHER_MODEL_LABEL = "chronos_weather"
CHRONOS_LORA_ARTIFACT_ID = CHRONOS_LORA_WEATHER_MODEL_LABEL
WEATHER_FEATURES_LONG_FILENAME = "weather_open_meteo_area_hourly_long_open_meteo_previous_runs_v1.parquet"
PRICE_PANEL_FILENAME = "price_panel_hourly_v1.parquet"
PRICE_PANEL_QA_FILENAME = "price_panel_hourly_v1.qa.json"

CHRONOS_MODEL_ARTIFACT_RELATIVE_PATH = Path("models") / CHRONOS_LORA_ARTIFACT_ID
CHRONOS_MODEL_ARTIFACT_PROJECT_RELATIVE_PATH = Path("artifacts") / CHRONOS_MODEL_ARTIFACT_RELATIVE_PATH
WEATHER_FEATURES_LONG_PROJECT_RELATIVE_PATH = Path("data") / "features" / WEATHER_FEATURES_LONG_FILENAME


@dataclass(frozen=True)
class ProjectLayout:
    root: Path
    src: Path
    scripts: Path


@dataclass(frozen=True)
class RuntimeLayout:
    root: Path
    eds_raw: Path
    open_meteo_raw: Path
    normalized: Path
    features: Path
    model_ready: Path
    price_panel: Path
    price_panel_qa: Path
    weather_features_long: Path
    chronos_model_artifact: Path
    baseline_results: Path
    forecast_runs: Path
    latest_pointer: Path
    latest_forecast: Path
    recent_scores: Path
    published_history: Path
    dashboard_json: Path


def project_layout(root: str | Path = PROJECT_ROOT) -> ProjectLayout:
    project_root = Path(root)
    return ProjectLayout(
        root=project_root,
        src=project_root / "src",
        scripts=project_root / "scripts",
    )


def runtime_layout(root: str | Path = PROJECT_ROOT) -> RuntimeLayout:
    runtime_root = Path(root)
    data = runtime_root / "data"
    model_ready = data / "model_ready"
    results = runtime_root / "results"
    return RuntimeLayout(
        root=runtime_root,
        eds_raw=data / "raw" / "energi_data_service",
        open_meteo_raw=data / "raw" / "open_meteo",
        normalized=data / "normalized",
        features=data / "features",
        model_ready=model_ready,
        price_panel=model_ready / PRICE_PANEL_FILENAME,
        price_panel_qa=model_ready / PRICE_PANEL_QA_FILENAME,
        weather_features_long=runtime_root / WEATHER_FEATURES_LONG_PROJECT_RELATIVE_PATH,
        chronos_model_artifact=runtime_root / CHRONOS_MODEL_ARTIFACT_PROJECT_RELATIVE_PATH,
        baseline_results=results / "baseline_v1",
        forecast_runs=runtime_root / "artifacts" / "forecast_runs",
        latest_pointer=runtime_root / "artifacts" / "latest.json",
        latest_forecast=results / "latest_forecast",
        recent_scores=results / "recent_scores",
        published_history=results / "published_forecast_history",
        dashboard_json=runtime_root / "app_data" / "forecast_dashboard.json",
    )
