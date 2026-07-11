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
from dkenergy_forecast.models.registry import production_model_specs  # noqa: E402
from dkenergy_forecast.operations.recent_diagnostics import run_recent_diagnostics  # noqa: E402


DEFAULT_LAYOUT = runtime_layout(PROJECT_ROOT)


def main() -> None:
    args = parse_args()
    result = run_recent_diagnostics(args)
    print(f"Completed diagnostic run: {result.run_id}")
    for label, path in result.paths.items():
        print(f"Wrote {label}: {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run recent production-model diagnostics outside the live publication path."
    )
    parser.add_argument("--panel-path", default=str(DEFAULT_LAYOUT.price_panel))
    parser.add_argument("--qa-path", default=str(DEFAULT_LAYOUT.price_panel_qa))
    parser.add_argument("--output-dir", default=str(DEFAULT_LAYOUT.recent_scores))
    parser.add_argument("--forecast-local-time", default="10:00")
    parser.add_argument("--at-hour-utc", type=int)
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument("--score-days", type=int, default=14)
    parser.add_argument("--score-max-origins", type=int, default=7)
    parser.add_argument("--score-holdout-days", type=int, default=2)
    parser.add_argument("--models", nargs="+", help=f"Defaults to {list(production_model_specs())}.")
    parser.add_argument(
        "--weather-features-long-path",
        default=str(PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG.weather_features_long_path),
    )
    parser.add_argument(
        "--chronos-model-artifact-path",
        default=os.environ.get(
            "DKENERGY_CHRONOS_MODEL_ARTIFACT_PATH",
            str(PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG.model_artifact_path),
        ),
    )
    parser.add_argument("--run-id")
    parser.add_argument("--allow-incomplete-panel", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
