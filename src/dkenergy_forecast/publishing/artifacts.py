from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dkenergy_forecast.types import normalize_utc_column, require_columns, to_utc_timestamp


PUBLISHED_PREDICTION_REQUIRED_COLUMNS = [
    "forecast_origin_utc",
    "ds_utc",
    "ds_local",
    "area",
    "model_label",
    "y_pred",
]

MODEL_SCORE_REQUIRED_COLUMNS = [
    "model_label",
    "area",
    "mae",
    "rmse",
    "bias",
    "coverage",
    "interval_width",
]


def normalize_published_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Normalize prediction rows for durable artifact and API use."""

    output = predictions.copy()
    if "model_label" not in output.columns and "model_name" in output.columns:
        if "model_version" in output.columns:
            output["model_label"] = output["model_name"] + ":" + output["model_version"].astype(str)
        else:
            output["model_label"] = output["model_name"]
    if "actual_price" not in output.columns and "y" in output.columns:
        output["actual_price"] = output["y"]

    require_columns(output, PUBLISHED_PREDICTION_REQUIRED_COLUMNS, "published predictions")
    output = normalize_utc_column(output, "forecast_origin_utc")
    output = normalize_utc_column(output, "ds_utc")
    if "created_at_utc" in output.columns:
        output = normalize_utc_column(output, "created_at_utc")
    return output.reset_index(drop=True)


def validate_prediction_artifact_schema(predictions: pd.DataFrame) -> None:
    frame = normalize_published_predictions(predictions)
    key_cols = [
        column
        for column in ["unique_id", "forecast_origin_utc", "ds_utc", "area", "model_label"]
        if column in frame.columns
    ]
    duplicate_count = int(frame.duplicated(key_cols).sum())
    if duplicate_count:
        raise ValueError(
            "Published predictions contain duplicate key rows: "
            f"{duplicate_count} duplicate(s) over {key_cols}"
        )
    if frame["model_label"].isna().any():
        raise ValueError("Published predictions contain missing model_label values")
    if frame["area"].isna().any():
        raise ValueError("Published predictions contain missing area values")


def validate_model_scores_schema(scores: pd.DataFrame) -> None:
    require_columns(scores, MODEL_SCORE_REQUIRED_COLUMNS, "model scores")
    duplicate_count = int(scores.duplicated(["model_label", "area"]).sum())
    if duplicate_count:
        raise ValueError(
            "Model scores contain duplicate (model_label, area) rows: "
            f"{duplicate_count}"
        )
    for column in ["mae", "rmse", "bias", "coverage", "interval_width"]:
        if column in scores.columns:
            pd.to_numeric(scores[column], errors="raise")


def make_forecast_run_manifest(
    *,
    run_id: str,
    forecast_origin_utc: object,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    artifact_paths: dict[str, str],
    dataset_version: list[str] | str | None,
    git_commit_value: str | None,
    status: str = "success",
    created_at_utc: object | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    forecast_origin = to_utc_timestamp(forecast_origin_utc)
    created_at = to_utc_timestamp(created_at_utc or datetime.now(timezone.utc))
    model_labels = sorted(predictions["model_label"].dropna().astype(str).unique().tolist())
    manifest = {
        "run_id": run_id,
        "status": status,
        "created_at_utc": created_at,
        "forecast_origin_utc": forecast_origin,
        "model_labels": model_labels,
        "dataset_version": dataset_version,
        "prediction_row_count": int(len(predictions)),
        "model_score_row_count": int(len(scores)),
        "artifact_paths": artifact_paths,
        "git_commit": git_commit_value,
    }
    if extra:
        manifest.update(extra)
    return manifest


def write_forecast_run_artifacts(
    run_dir: str | Path,
    *,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    manifest: dict[str, Any],
    dashboard: dict[str, Any] | None = None,
) -> dict[str, Path]:
    run_path = Path(run_dir)
    run_path.mkdir(parents=True, exist_ok=True)
    predictions_out = normalize_published_predictions(predictions)
    validate_prediction_artifact_schema(predictions_out)
    validate_model_scores_schema(scores)

    paths = {
        "predictions": run_path / "predictions.parquet",
        "model_scores": run_path / "model_scores.parquet",
        "manifest": run_path / "manifest.json",
    }
    predictions_out.to_parquet(paths["predictions"], index=False)
    scores.to_parquet(paths["model_scores"], index=False)
    write_json(paths["manifest"], manifest)
    if dashboard is not None:
        paths["dashboard"] = run_path / "forecast_dashboard.json"
        write_json(paths["dashboard"], dashboard)
    return paths


def update_latest_exports(
    *,
    latest_forecast_dir: str | Path,
    recent_scores_dir: str | Path,
    dashboard_path: str | Path,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    manifest: dict[str, Any],
    dashboard: dict[str, Any],
) -> dict[str, Path]:
    latest_dir = Path(latest_forecast_dir)
    scores_dir = Path(recent_scores_dir)
    dashboard_out = Path(dashboard_path)
    latest_dir.mkdir(parents=True, exist_ok=True)
    scores_dir.mkdir(parents=True, exist_ok=True)
    dashboard_out.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "latest_predictions": latest_dir / "predictions.parquet",
        "latest_manifest": latest_dir / "manifest.json",
        "recent_scores": scores_dir / "model_scores.parquet",
        "dashboard": dashboard_out,
    }
    normalize_published_predictions(predictions).to_parquet(paths["latest_predictions"], index=False)
    scores.to_parquet(paths["recent_scores"], index=False)
    write_json(paths["latest_manifest"], manifest)
    write_json(paths["dashboard"], dashboard)
    return paths


def build_dashboard_payload(
    *,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    prediction_rows = normalize_published_predictions(predictions).to_dict(orient="records")
    score_rows = scores.to_dict(orient="records")
    return {
        "generated_at_utc": datetime.now(timezone.utc),
        "run": manifest,
        "predictions": prediction_rows,
        "model_scores": score_rows,
    }


def unique_run_id(prefix: str, *, created_at_utc: object | None = None) -> str:
    created_at = to_utc_timestamp(created_at_utc or datetime.now(timezone.utc))
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def git_commit(cwd: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(cwd),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def replace_directory(source: str | Path, destination: str | Path) -> None:
    destination_path = Path(destination)
    if destination_path.exists():
        shutil.rmtree(destination_path)
    shutil.copytree(source, destination_path)
