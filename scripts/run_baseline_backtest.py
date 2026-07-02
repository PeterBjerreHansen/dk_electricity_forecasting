#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.backtesting.horizons import make_danish_delivery_day_horizon  # noqa: E402
from dkenergy_forecast.backtesting.origins import choose_recent_complete_daily_origins  # noqa: E402
from dkenergy_forecast.backtesting.rolling_origin import rolling_origin_backtest  # noqa: E402
from dkenergy_forecast.evaluation.summary import (  # noqa: E402
    add_prediction_diagnostics,
    cheapest_k_table,
    model_score_table,
)
from dkenergy_forecast.io import load_price_panel  # noqa: E402
from dkenergy_forecast.models.registry import baseline_model_factories  # noqa: E402
from dkenergy_forecast.publishing import git_commit, json_safe  # noqa: E402


BASELINE_FACTORIES = baseline_model_factories()


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel_path)
    qa_path = Path(args.qa_path) if args.qa_path else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_price_panel(
        panel_path,
        qa_path if qa_path and qa_path.exists() else None,
        require_final_historical=not args.allow_incomplete_panel,
    )
    origins = choose_recent_complete_daily_origins(
        panel,
        days=args.days,
        at_hour_utc=args.at_hour_utc,
    )

    prediction_frames = []
    for model_label, factory in BASELINE_FACTORIES.items():
        predictions = rolling_origin_backtest(
            model_factory=factory,
            panel=panel,
            origins=origins,
            horizon_builder=lambda panel_arg, origin_arg: make_danish_delivery_day_horizon(
                panel_arg,
                origin_arg,
                days_ahead=1,
            ),
            min_train_rows=args.min_train_days * 24 * panel["area"].nunique(),
        )
        predictions["model_label"] = model_label
        prediction_frames.append(predictions)

    predictions = add_prediction_diagnostics(pd.concat(prediction_frames, ignore_index=True))

    metrics = model_score_table(predictions)
    value_metrics = cheapest_k_table(predictions, k=args.cheapest_k)
    manifest = make_manifest(
        args=args,
        panel=panel,
        origins=origins,
        predictions=predictions,
        panel_path=panel_path,
        qa_path=qa_path,
    )

    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    metrics.to_parquet(output_dir / "model_scores.parquet", index=False)
    metrics.to_parquet(output_dir / "metrics.parquet", index=False)
    value_metrics.to_parquet(output_dir / "value_metrics.parquet", index=False)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote predictions: {output_dir / 'predictions.parquet'}")
    print(f"Wrote model scores: {output_dir / 'model_scores.parquet'}")
    print(f"Wrote metrics: {output_dir / 'metrics.parquet'}")
    print(f"Wrote value metrics: {output_dir / 'value_metrics.parquet'}")
    print(f"Wrote manifest: {output_dir / 'run_manifest.json'}")
    print(metrics.loc[metrics["area"] == "ALL"].sort_values("mae").to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official local baseline backtest.")
    parser.add_argument(
        "--panel-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.parquet"),
    )
    parser.add_argument(
        "--qa-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.qa.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "baseline_v1"),
    )
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--at-hour-utc", type=int, default=10)
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument("--cheapest-k", type=int, default=6)
    parser.add_argument(
        "--allow-incomplete-panel",
        action="store_true",
        help="Allow QA artifacts that are not marked final_historical.",
    )
    return parser.parse_args()


def make_manifest(
    *,
    args: argparse.Namespace,
    panel: pd.DataFrame,
    origins: pd.DataFrame,
    predictions: pd.DataFrame,
    panel_path: Path,
    qa_path: Path | None,
) -> dict[str, Any]:
    return {
        "run_id": "baseline_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_path": str(panel_path),
        "qa_path": str(qa_path) if qa_path else None,
        "dataset_version": sorted(panel["dataset_version"].dropna().unique().tolist()),
        "model_labels": sorted(BASELINE_FACTORIES),
        "forecast_origin_min_utc": origins["forecast_origin_utc"].min(),
        "forecast_origin_max_utc": origins["forecast_origin_utc"].max(),
        "forecast_origin_count": int(len(origins)),
        "prediction_row_count": int(len(predictions)),
        "days": int(args.days),
        "at_hour_utc": int(args.at_hour_utc),
        "min_train_days": int(args.min_train_days),
        "cheapest_k": int(args.cheapest_k),
        "git_commit": git_commit(ROOT),
    }


if __name__ == "__main__":
    main()
