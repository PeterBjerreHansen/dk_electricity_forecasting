#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.static_dashboard import build_static_dashboard  # noqa: E402
from dkenergy_forecast.dashboard import (  # noqa: E402
    canonical_forecast_history,
    combine_prediction_history,
    dashboard_records,
    recent_model_history,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a self-contained static forecast dashboard."
    )
    parser.add_argument(
        "--input",
        help="forecast_dashboard.json path; defaults to the run in artifacts/latest.json",
    )
    parser.add_argument("--output", required=True, help="HTML output path")
    parser.add_argument("--title", default="Danish Electricity Forecasts")
    parser.add_argument(
        "--history-output",
        help="Optional parquet output for seeding the private forecast-history archive",
    )
    parser.add_argument(
        "--history",
        action="append",
        default=[],
        help="Optional evaluated predictions parquet; may be supplied more than once",
    )
    args = parser.parse_args()

    source = Path(args.input) if args.input else _latest_dashboard_path()
    destination = Path(args.output)
    payload = json.loads(source.read_text(encoding="utf-8"))
    history = _load_history(payload, [Path(value) for value in args.history])
    if args.history_output:
        history_output = Path(args.history_output)
        history_output.parent.mkdir(parents=True, exist_ok=True)
        canonical_forecast_history(history).to_parquet(history_output, index=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        build_static_dashboard(
            payload,
            title=args.title,
            history_predictions=dashboard_records(history),
        ),
        encoding="utf-8",
    )
    print(destination)


def _load_history(payload: dict, paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    selected_paths = paths or _default_history_paths()
    for path in selected_paths:
        if path.exists():
            frames.append(pd.read_parquet(path))
    recent = payload.get("recent_predictions")
    if isinstance(recent, list) and recent:
        frames.append(pd.DataFrame(recent))
    history = combine_prediction_history(frames)
    if history.empty:
        return history
    return pd.concat(
        [
            recent_model_history(history, area=area, days=30)
            for area in sorted(history["area"].dropna().unique())
        ],
        ignore_index=True,
    )


def _default_history_paths() -> list[Path]:
    paths = [
        ROOT / "results" / "notebook_chronos2_experimental_v1" / "predictions.parquet",
        ROOT / "results" / "baseline_v1" / "predictions.parquet",
    ]
    paths.extend(
        sorted(
            (ROOT / "results" / "chronos_weather").glob(
                "*/recent_diagnostics/predictions.parquet"
            )
        )
    )
    return paths


def _latest_dashboard_path() -> Path:
    pointer_path = ROOT / "artifacts" / "latest.json"
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    run_prefix = Path(str(pointer["run_prefix"]))
    if run_prefix.is_absolute() or ".." in run_prefix.parts:
        raise ValueError(f"latest.json has an invalid run_prefix: {run_prefix}")
    return pointer_path.parent / run_prefix / "forecast_dashboard.json"
if __name__ == "__main__":
    main()
