#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.layout import PROJECT_ROOT, runtime_layout  # noqa: E402
from dkenergy_forecast.models.chronos_production import (  # noqa: E402
    PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG,
)
from dkenergy_forecast.models.registry import default_production_model_labels  # noqa: E402
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
    print(f"Forecast origin UTC: {result.forecast_origin_utc.isoformat()}")
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
        "--forecast-origin-utc",
        help="Forecast origin timestamp. Defaults to the latest panel UTC date at --at-hour-utc.",
    )
    parser.add_argument(
        "--at-hour-utc",
        type=int,
        help="Legacy fixed UTC forecast hour. Omit to use --forecast-local-time.",
    )
    parser.add_argument("--forecast-local-time", default="12:00")
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument(
        "--score-days",
        type=int,
        default=14,
        help="Lookback window for recent completed-origin scoring.",
    )
    parser.add_argument(
        "--score-max-origins",
        type=int,
        default=7,
        help="Maximum recent completed origins used for model scores.",
    )
    parser.add_argument(
        "--score-holdout-days",
        type=int,
        default=2,
        help="Holdout days between latest panel timestamp and scoring origins.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help=(
            "Production model labels to publish. Defaults to registry defaults: "
            f"{default_production_model_labels()}."
        ),
    )
    parser.add_argument(
        "--weather-features-long-path",
        default=str(PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG.weather_features_long_path),
        help="Open-Meteo long weather feature parquet used by weather-aware production models.",
    )
    parser.add_argument(
        "--chronos-model-artifact-path",
        default=os.environ.get(
            "DKENERGY_CHRONOS_MODEL_ARTIFACT_PATH",
            str(PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG.model_artifact_path),
        ),
        help="Local trained Chronos LoRA artifact directory used by the production Chronos model.",
    )
    parser.add_argument("--list-models", action="store_true", help="Print registered production models and exit.")
    parser.add_argument("--run-id", help="Optional explicit immutable forecast run id.")
    parser.add_argument(
        "--artifact-root",
        default=str(DEFAULT_LAYOUT.forecast_runs),
    )
    parser.add_argument(
        "--latest-forecast-dir",
        default=str(DEFAULT_LAYOUT.latest_forecast),
    )
    parser.add_argument(
        "--recent-scores-dir",
        default=str(DEFAULT_LAYOUT.recent_scores),
    )
    parser.add_argument(
        "--published-history-dir",
        default=str(DEFAULT_LAYOUT.published_history),
    )
    parser.add_argument(
        "--dashboard-path",
        default=str(DEFAULT_LAYOUT.dashboard_json),
    )
    parser.add_argument(
        "--allow-incomplete-panel",
        action="store_true",
        help="Allow QA artifacts that are not marked final_historical.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
