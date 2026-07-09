#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.io import load_price_panel  # noqa: E402
from dkenergy_forecast.publishing import (  # noqa: E402
    build_published_forecast_history,
    build_published_forecast_scores,
    git_commit,
    write_json,
    write_published_forecast_history,
)


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel_path)
    qa_path = Path(args.qa_path) if args.qa_path else None
    panel = load_price_panel(
        panel_path,
        qa_path if qa_path and qa_path.exists() else None,
        require_final_historical=not args.allow_incomplete_panel,
    )

    history = build_published_forecast_history(args.artifact_root, panel)
    scores = build_published_forecast_scores(history)
    written = write_published_forecast_history(
        args.output_dir,
        predictions=history,
        scores=scores,
    )
    manifest_path = Path(args.output_dir) / "manifest.json"
    write_json(
        manifest_path,
        {
            "score_source": "published_forecast_history",
            "artifact_root": str(Path(args.artifact_root)),
            "panel_path": str(panel_path),
            "qa_path": str(qa_path) if qa_path else None,
            "output_dir": str(Path(args.output_dir)),
            "prediction_row_count": int(len(history)),
            "model_score_row_count": int(len(scores)),
            "evaluated_forecast_run_count": (
                int(history["run_id"].nunique()) if "run_id" in history else 0
            ),
            "target_min_utc": history["ds_utc"].min() if not history.empty else None,
            "target_max_utc": history["ds_utc"].max() if not history.empty else None,
            "git_commit": git_commit(ROOT),
            "generated_at_utc": pd.Timestamp.now(tz="UTC"),
        },
    )

    print("Scored published forecast history")
    print(f"Evaluated prediction rows: {len(history)}")
    evaluated_run_count = history["run_id"].nunique() if "run_id" in history else 0
    print(f"Evaluated forecast runs: {evaluated_run_count}")
    print(f"Score rows: {len(scores)}")
    for label, path in {**written, "manifest": manifest_path}.items():
        print(f"Wrote {label}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score immutable published forecast runs against a price panel without "
            "recomputing model predictions."
        )
    )
    parser.add_argument(
        "--artifact-root",
        default=str(ROOT / "artifacts" / "forecast_runs"),
        help="Directory containing immutable forecast run subdirectories.",
    )
    parser.add_argument(
        "--panel-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.parquet"),
        help="Model-ready hourly price panel containing actual prices.",
    )
    parser.add_argument(
        "--qa-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.qa.json"),
        help="Optional QA JSON for the price panel.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "published_forecast_history"),
        help="Directory for evaluated predictions and published forecast score artifacts.",
    )
    parser.add_argument(
        "--allow-incomplete-panel",
        action="store_true",
        help="Allow QA artifacts that are not marked final_historical.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
