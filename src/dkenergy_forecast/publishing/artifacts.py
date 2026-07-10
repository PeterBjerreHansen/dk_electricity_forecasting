from __future__ import annotations

import json
import math
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dkenergy_forecast.types import (
    PRICE_AVAILABILITY_COLUMN,
    ensure_price_availability,
    normalize_utc_column,
    require_columns,
    to_utc_timestamp,
)


PUBLISHED_PREDICTION_REQUIRED_COLUMNS = [
    "forecast_origin_utc",
    "ds_utc",
    "ds_local",
    "area",
    "model_label",
    "y_pred",
]

EVALUATED_PREDICTION_REQUIRED_COLUMNS = [
    *PUBLISHED_PREDICTION_REQUIRED_COLUMNS,
    "y",
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

MODEL_SCORE_COLUMNS = [
    "model_label",
    "area",
    "rows",
    "evaluated_rows",
    "mae",
    "rmse",
    "bias",
    "coverage",
    "interval_width",
    "missing_rate",
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
    if frame["y_pred"].isna().any():
        raise ValueError("Published predictions contain missing point forecast values")

    quantile_columns = ["q10", "q50", "q90"]
    present_quantiles = [column for column in quantile_columns if column in frame.columns]
    if present_quantiles:
        missing_quantile_columns = sorted(set(quantile_columns) - set(present_quantiles))
        if missing_quantile_columns:
            raise ValueError(
                "Published predictions have an incomplete quantile schema; "
                f"missing columns: {missing_quantile_columns}"
            )
        has_any_quantile = frame[quantile_columns].notna().any(axis=1)
        has_all_quantiles = frame[quantile_columns].notna().all(axis=1)
        partial_count = int((has_any_quantile & ~has_all_quantiles).sum())
        if partial_count:
            raise ValueError(
                "Published predictions contain partially populated q10/q50/q90 rows: "
                f"{partial_count}"
            )
        crossed_count = int(
            (
                has_all_quantiles
                & ((frame["q10"] > frame["q50"]) | (frame["q50"] > frame["q90"]))
            ).sum()
        )
        if crossed_count:
            raise ValueError(f"Published predictions contain crossed quantiles: {crossed_count}")


def validate_evaluated_prediction_artifact_schema(predictions: pd.DataFrame) -> None:
    frame = normalize_published_predictions(predictions)
    require_columns(frame, EVALUATED_PREDICTION_REQUIRED_COLUMNS, "evaluated predictions")
    validate_prediction_artifact_schema(frame)
    if frame["y"].isna().any():
        raise ValueError("Evaluated predictions contain missing actual target values")
    if frame["y_pred"].isna().any():
        raise ValueError("Evaluated predictions contain missing point forecast values")


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


def build_published_forecast_history(
    artifact_root: str | Path,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Score only immutable forecast-run predictions that now have actuals."""

    published = _load_forecast_run_predictions(artifact_root)
    if published.empty:
        return _empty_published_history_frame()

    actuals = _panel_actuals(panel)
    merged = published.merge(actuals, on=["unique_id", "ds_utc"], how="left")
    merged["y"] = merged["_actual_y"]
    merged["actual_price"] = merged["_actual_y"]
    if PRICE_AVAILABILITY_COLUMN in merged:
        merged[PRICE_AVAILABILITY_COLUMN] = merged[f"{PRICE_AVAILABILITY_COLUMN}_panel"].combine_first(
            merged[PRICE_AVAILABILITY_COLUMN]
        )
    elif f"{PRICE_AVAILABILITY_COLUMN}_panel" in merged:
        merged[PRICE_AVAILABILITY_COLUMN] = merged[f"{PRICE_AVAILABILITY_COLUMN}_panel"]
    merged = merged.drop(
        columns=[column for column in ["_actual_y", f"{PRICE_AVAILABILITY_COLUMN}_panel"] if column in merged],
    )
    evaluated = merged.loc[merged["y"].notna() & merged["y_pred"].notna()].copy()
    if evaluated.empty:
        return _empty_published_history_frame()
    return evaluated.sort_values(["forecast_origin_utc", "run_id", "model_label", "unique_id", "ds_utc"]).reset_index(
        drop=True
    )


def build_published_forecast_scores(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return _empty_model_score_frame(score_source="published_forecast_history")

    from dkenergy_forecast.evaluation.summary import add_prediction_diagnostics, model_score_table

    scores = model_score_table(add_prediction_diagnostics(history))
    scores["score_source"] = "published_forecast_history"
    return scores


def write_published_forecast_history(
    history_dir: str | Path,
    *,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
) -> dict[str, Path]:
    history_path = Path(history_dir)
    history_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "published_history_predictions": history_path / "predictions.parquet",
        "published_history_scores": history_path / "model_scores.parquet",
    }
    predictions.to_parquet(paths["published_history_predictions"], index=False)
    scores.to_parquet(paths["published_history_scores"], index=False)
    return paths


def _load_forecast_run_predictions(artifact_root: str | Path) -> pd.DataFrame:
    root = Path(artifact_root)
    if not root.exists():
        return _empty_published_prediction_frame()

    frames: list[pd.DataFrame] = []
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        predictions_path = run_dir / "predictions.parquet"
        if not predictions_path.exists():
            continue
        frame = normalize_published_predictions(pd.read_parquet(predictions_path))
        require_columns(frame, ["unique_id"], f"{predictions_path}")
        validate_prediction_artifact_schema(frame)
        manifest = _read_optional_manifest(run_dir / "manifest.json")
        frame["run_id"] = frame.get("run_id", manifest.get("run_id") or run_dir.name)
        frame["run_id"] = frame["run_id"].fillna(manifest.get("run_id") or run_dir.name).astype(str)
        frame["published_at_utc"] = manifest.get("created_at_utc")
        frame["published_forecast_origin_utc"] = manifest.get("forecast_origin_utc")
        frames.append(frame)

    if not frames:
        return _empty_published_prediction_frame()
    return pd.concat(frames, ignore_index=True)


def _read_optional_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _panel_actuals(panel: pd.DataFrame) -> pd.DataFrame:
    require_columns(panel, ["unique_id", "ds_utc", "y"], "price panel")
    actuals = ensure_price_availability(normalize_utc_column(panel, "ds_utc"))
    columns = ["unique_id", "ds_utc", "y"]
    if PRICE_AVAILABILITY_COLUMN in actuals:
        columns.append(PRICE_AVAILABILITY_COLUMN)
    output = actuals[columns].drop_duplicates(["unique_id", "ds_utc"]).copy()
    rename = {"y": "_actual_y"}
    if PRICE_AVAILABILITY_COLUMN in output:
        rename[PRICE_AVAILABILITY_COLUMN] = f"{PRICE_AVAILABILITY_COLUMN}_panel"
    return output.rename(columns=rename)


def _empty_published_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[*PUBLISHED_PREDICTION_REQUIRED_COLUMNS, "unique_id", "run_id"])


def _empty_published_history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *PUBLISHED_PREDICTION_REQUIRED_COLUMNS,
            "unique_id",
            "run_id",
            "published_at_utc",
            "published_forecast_origin_utc",
            "y",
            "actual_price",
        ]
    )


def _empty_model_score_frame(*, score_source: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(columns=[*MODEL_SCORE_COLUMNS, "score_source"])
    if score_source is not None:
        frame["score_source"] = frame["score_source"].astype("object")
    return frame


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
    score_predictions: pd.DataFrame | None = None,
    dashboard: dict[str, Any] | None = None,
) -> dict[str, Path]:
    run_path = Path(run_dir)
    predictions_out = normalize_published_predictions(predictions)
    validate_prediction_artifact_schema(predictions_out)
    validate_model_scores_schema(scores)
    score_predictions_out = None
    if score_predictions is not None:
        score_predictions_out = normalize_published_predictions(score_predictions)
        validate_evaluated_prediction_artifact_schema(score_predictions_out)

    if run_path.exists():
        raise FileExistsError(f"Immutable forecast run already exists: {run_path}")
    run_path.mkdir(parents=True)

    paths = {
        "predictions": run_path / "predictions.parquet",
        "model_scores": run_path / "model_scores.parquet",
        "manifest": run_path / "manifest.json",
    }
    predictions_out.to_parquet(paths["predictions"], index=False)
    scores.to_parquet(paths["model_scores"], index=False)
    if score_predictions_out is not None:
        paths["score_predictions"] = run_path / "score_predictions.parquet"
        score_predictions_out.to_parquet(paths["score_predictions"], index=False)
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
    score_predictions: pd.DataFrame | None = None,
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
    if score_predictions is not None:
        score_predictions_out = normalize_published_predictions(score_predictions)
        validate_evaluated_prediction_artifact_schema(score_predictions_out)
        paths["recent_predictions"] = scores_dir / "predictions.parquet"
        score_predictions_out.to_parquet(paths["recent_predictions"], index=False)
    write_json(paths["latest_manifest"], manifest)
    write_json(paths["dashboard"], dashboard)
    return paths


def build_dashboard_payload(
    *,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    manifest: dict[str, Any],
    score_predictions: pd.DataFrame | None = None,
    published_history_predictions: pd.DataFrame | None = None,
    published_history_scores: pd.DataFrame | None = None,
) -> dict[str, Any]:
    prediction_rows = normalize_published_predictions(predictions).to_dict(orient="records")
    score_rows = scores.to_dict(orient="records")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc),
        "run": manifest,
        "predictions": prediction_rows,
        "model_scores": score_rows,
    }
    if score_predictions is not None:
        payload["recent_predictions"] = normalize_published_predictions(score_predictions).to_dict(
            orient="records"
        )
    if published_history_predictions is not None:
        payload["published_history_predictions"] = published_history_predictions.to_dict(orient="records")
    if published_history_scores is not None:
        payload["published_history_scores"] = published_history_scores.to_dict(orient="records")
    return payload


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
        json.dumps(json_safe(payload), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def replace_directory(source: str | Path, destination: str | Path) -> None:
    destination_path = Path(destination)
    if destination_path.exists():
        shutil.rmtree(destination_path)
    shutil.copytree(source, destination_path)
