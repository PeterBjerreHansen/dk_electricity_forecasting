#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.layout import PROJECT_ROOT, runtime_layout  # noqa: E402
from dkenergy_forecast.operations.publish_forecast import (  # noqa: E402
    print_model_registry,
    run_publish_forecast,
)


DEFAULT_LAYOUT = runtime_layout(PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    if args.list_models:
        print_model_registry()
        return

    try:
        result = run_publish_forecast(args, project_root=PROJECT_ROOT)
    except (ValueError, ImportError) as exc:
        raise SystemExit(str(exc)) from exc

    print(f"Published forecast run: {result.run_id}")
    print(f"Delivery date: {result.request.delivery_date_local}")
    print(f"Information cutoff UTC: {result.request.information_cutoff_utc.isoformat()}")
    print(f"Published model: {result.published_model} ({result.forecast_status})")
    for label, path in result.paths.items():
        print(f"Wrote {label}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a file-based forecast run for the dashboard API path.")
    parser.add_argument(
        "--panel-path",
        default=str(DEFAULT_LAYOUT.price_panel),
    )
    parser.add_argument(
        "--qa-path",
        default=str(DEFAULT_LAYOUT.price_panel_qa),
    )
    parser.add_argument(
        "--information-cutoff-utc",
        help="Latest timestamp whose information may be used. Required for replay; live defaults to start time.",
    )
    parser.add_argument("--delivery-date-local", help="Danish delivery date (YYYY-MM-DD). Defaults to cutoff date + 1.")
    parser.add_argument(
        "--run-kind",
        choices=["live", "replay"],
        help="Defaults to live, or replay when an explicit historical cutoff is supplied.",
    )
    parser.add_argument(
        "--decision-deadline-utc",
        help="Publication deadline. Defaults to 12:00 Europe/Copenhagen on the cutoff date.",
    )
    parser.add_argument("--decision-deadline-local-time", default="12:00")
    parser.add_argument(
        "--generated-at-utc",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument(
        "--weather-features-long-path",
        default=str(DEFAULT_LAYOUT.weather_features_long),
        help="Open-Meteo long weather feature parquet used by weather-aware production models.",
    )
    parser.add_argument(
        "--chronos-model-artifact-path",
        default=os.environ.get(
            "DKENERGY_CHRONOS_MODEL_ARTIFACT_PATH",
        ),
        help="Override the immutable Chronos artifact path declared in production.json.",
    )
    parser.add_argument(
        "--production-config",
        default=str(PROJECT_ROOT / "config" / "production.json"),
        help="Source-controlled primary/fallback production configuration.",
    )
    parser.add_argument("--runtime-root", help=argparse.SUPPRESS)
    parser.add_argument("--list-models", action="store_true", help="Print registered production models and exit.")
    parser.add_argument("--run-id", help="Optional explicit immutable forecast run id.")
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_LAYOUT.forecast_runs),
    )
    parser.add_argument("--latest-pointer-path", default=str(DEFAULT_LAYOUT.latest_pointer))
    parser.add_argument(
        "--allow-incomplete-panel",
        action="store_true",
        help="Allow QA artifacts that are not marked final_historical.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
