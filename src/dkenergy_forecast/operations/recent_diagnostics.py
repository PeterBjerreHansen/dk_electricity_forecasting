from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from dkenergy_forecast.backtesting.origins import choose_recent_complete_daily_origins
from dkenergy_forecast.evaluation.summary import (
    add_prediction_diagnostics,
    model_score_table,
    probabilistic_metric_table,
)
from dkenergy_forecast.io import load_price_panel
from dkenergy_forecast.layout import PROJECT_ROOT
from dkenergy_forecast.models.registry import latest_publish_model_factories, production_model_specs
from dkenergy_forecast.operations.publish_forecast import publish_predictions_for_origins
from dkenergy_forecast.publishing import git_commit, json_safe, unique_run_id


@dataclass(frozen=True)
class RecentDiagnosticsResult:
    run_id: str
    paths: dict[str, Path]


def run_recent_diagnostics(
    args: Any,
    *,
    project_root: Path = PROJECT_ROOT,
) -> RecentDiagnosticsResult:
    """Evaluate production models without participating in live publication.

    Diagnostics deliberately have their own versioned run namespace and mutable
    convenience exports. A failure here cannot modify a live forecast run or its
    latest pointer.
    """

    diagnostic_labels = args.models or list(production_model_specs())
    factories = latest_publish_model_factories(
        diagnostic_labels,
        weather_features_long_path=args.weather_features_long_path,
        chronos_model_artifact_path=args.chronos_model_artifact_path,
    )
    panel_path = Path(args.panel_path)
    qa_path = Path(args.qa_path) if args.qa_path else None
    panel = load_price_panel(
        panel_path,
        qa_path if qa_path and qa_path.exists() else None,
        require_final_historical=not args.allow_incomplete_panel,
    )
    origins = choose_recent_complete_daily_origins(
        panel,
        days=args.score_days,
        at_hour_utc=args.at_hour_utc,
        forecast_local_time=args.forecast_local_time,
        max_origins=args.score_max_origins,
        min_history_days=args.min_train_days,
        holdout_days=args.score_holdout_days,
    )
    predictions = publish_predictions_for_origins(
        panel=panel,
        origins=origins,
        factories=factories,
        min_train_days=args.min_train_days,
    )
    predictions = add_prediction_diagnostics(predictions)
    scores = model_score_table(predictions)
    probabilistic = probabilistic_metric_table(predictions)
    run_id = args.run_id or unique_run_id("diagnostics")
    predictions["diagnostic_run_id"] = run_id
    manifest = {
        "run_id": run_id,
        "run_kind": "diagnostic",
        "status": "success",
        "created_at_utc": datetime.now(timezone.utc),
        "forecast_origin_min_utc": origins["forecast_origin_utc"].min(),
        "forecast_origin_max_utc": origins["forecast_origin_utc"].max(),
        "forecast_origin_count": int(len(origins)),
        "prediction_row_count": int(len(predictions)),
        "model_labels": sorted(predictions["model_label"].dropna().unique().tolist()),
        "panel_path": str(panel_path),
        "qa_path": str(qa_path) if qa_path else None,
        "git_commit": git_commit(project_root),
    }
    return _write_diagnostics(
        Path(args.output_dir),
        run_id=run_id,
        predictions=predictions,
        scores=scores,
        probabilistic=probabilistic,
        manifest=manifest,
    )


def _write_diagnostics(
    output_dir: Path,
    *,
    run_id: str,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    probabilistic: pd.DataFrame,
    manifest: dict[str, Any],
) -> RecentDiagnosticsResult:
    run_root = output_dir / "runs"
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"Immutable diagnostic run already exists: {run_dir}")

    temporary = run_root / f".{run_id}.{uuid4().hex}.tmp"
    temporary.mkdir()
    try:
        predictions.to_parquet(temporary / "predictions.parquet", index=False)
        scores.to_parquet(temporary / "model_scores.parquet", index=False)
        probabilistic.to_parquet(temporary / "probabilistic_metrics.parquet", index=False)
        (temporary / "manifest.json").write_text(
            json.dumps(json_safe(manifest), allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, run_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    output_dir.mkdir(parents=True, exist_ok=True)
    latest = {
        "recent_predictions": output_dir / "predictions.parquet",
        "recent_scores": output_dir / "model_scores.parquet",
        "recent_probabilistic_metrics": output_dir / "probabilistic_metrics.parquet",
        "recent_manifest": output_dir / "manifest.json",
    }
    _atomic_copy(run_dir / "predictions.parquet", latest["recent_predictions"])
    _atomic_copy(run_dir / "model_scores.parquet", latest["recent_scores"])
    _atomic_copy(
        run_dir / "probabilistic_metrics.parquet",
        latest["recent_probabilistic_metrics"],
    )
    _atomic_copy(run_dir / "manifest.json", latest["recent_manifest"])
    return RecentDiagnosticsResult(run_id=run_id, paths={"run_dir": run_dir, **latest})


def _atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
