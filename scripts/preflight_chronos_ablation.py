#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from dkenergy_forecast.io import load_price_panel  # noqa: E402
from dkenergy_forecast.models.chronos_production import (  # noqa: E402
    WEATHER_COVARIATE_MODES,
    Chronos2LoRAWeatherConfig,
    load_lora_artifact_manifest,
    validate_artifact_weather_policy,
)
from dkenergy_forecast.publishing import json_safe  # noqa: E402
from dkenergy_forecast.types import to_utc_timestamp  # noqa: E402
from scripts.train_chronos_lora import make_lora_training_frame  # noqa: E402


IMMUTABLE_REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")


def main() -> None:
    args = parse_args()
    if not IMMUTABLE_REVISION_PATTERN.fullmatch(args.base_model_revision):
        raise ValueError(
            "--base-model-revision must be a 40-character immutable commit hash"
        )
    panel = load_price_panel(args.panel_path, args.qa_path)
    weather = pd.read_parquet(args.weather_features_long_path)
    report = build_preflight_report(
        panel,
        weather,
        ablation_id=args.ablation_id,
        base_model_id=args.base_model_id,
        base_model_revision=args.base_model_revision,
        first_eval_origin=to_utc_timestamp(args.first_eval_origin_utc),
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        weather_covariate_mode=args.weather_covariate_mode,
        cross_learning=args.cross_learning,
        add_weather_derived_features=args.add_weather_derived_features,
        information_cutoff_local_time=args.information_cutoff_local_time,
        artifact_path=args.artifact_path,
    )
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            json_safe(report),
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Chronos ablation preflight: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize and validate an ablation training-frame contract "
            "without loading Chronos or starting a training run."
        )
    )
    parser.add_argument("--ablation-id", choices=["A0", "A1", "A2", "A3", "A4"], required=True)
    parser.add_argument("--panel-path", required=True)
    parser.add_argument("--qa-path", required=True)
    parser.add_argument("--weather-features-long-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--artifact-path")
    parser.add_argument("--base-model-id", default="amazon/chronos-2")
    parser.add_argument("--base-model-revision", required=True)
    parser.add_argument("--first-eval-origin-utc", default="2026-04-01T10:00:00Z")
    parser.add_argument("--information-cutoff-local-time", default="10:00")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--prediction-length", type=int, default=36)
    parser.add_argument(
        "--weather-covariate-mode",
        choices=WEATHER_COVARIATE_MODES,
        required=True,
    )
    parser.add_argument("--cross-learning", action="store_true")
    parser.add_argument("--add-weather-derived-features", action="store_true")
    return parser.parse_args()


def build_preflight_report(
    panel: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    ablation_id: str,
    base_model_id: str,
    base_model_revision: str,
    first_eval_origin: pd.Timestamp,
    context_length: int,
    prediction_length: int,
    weather_covariate_mode: str,
    cross_learning: bool,
    add_weather_derived_features: bool,
    information_cutoff_local_time: str,
    artifact_path: str | Path | None = None,
) -> dict[str, Any]:
    training, covariates, diagnostics = make_lora_training_frame(
        panel,
        weather,
        first_eval_origin=first_eval_origin,
        context_length=context_length,
        prediction_length=prediction_length,
        weather_covariate_mode=weather_covariate_mode,
        information_cutoff_local_time=information_cutoff_local_time,
        cross_learning=cross_learning,
        add_weather_derived_features=add_weather_derived_features,
    )
    artifact = None
    if artifact_path is not None:
        artifact = _validate_existing_artifact(
            artifact_path,
            base_model_id=base_model_id,
            base_model_revision=base_model_revision,
            context_length=context_length,
            prediction_length=prediction_length,
            weather_covariate_mode=weather_covariate_mode,
            cross_learning=cross_learning,
            add_weather_derived_features=add_weather_derived_features,
            information_cutoff_local_time=information_cutoff_local_time,
        )
        if list(artifact["covariates"]) != covariates:
            raise ValueError(
                "Chronos artifact covariates do not match the materialized "
                "ablation training-frame contract"
            )
    return {
        "schema_version": "chronos_ablation_preflight_v1",
        "ablation_id": ablation_id,
        "research_only": True,
        "promotes_model": False,
        "frozen_origin_evidence": False,
        "evidence_note": (
            "This preflight validates feature and artifact contracts only; "
            "recent diagnostics are not frozen-origin model-selection evidence."
        ),
        "base_model_id": base_model_id,
        "base_model_revision": base_model_revision,
        "config": {
            "context_length": context_length,
            "prediction_length": prediction_length,
            "weather_covariate_mode": weather_covariate_mode,
            "cross_learning": cross_learning,
            "add_weather_ensemble_features": True,
            "add_weather_derived_features": add_weather_derived_features,
            "information_cutoff_local_time": information_cutoff_local_time,
        },
        "training_rows": int(len(training)),
        "covariates": covariates,
        "diagnostics": diagnostics,
        "artifact": artifact,
    }


def _validate_existing_artifact(
    artifact_path: str | Path,
    *,
    base_model_id: str,
    base_model_revision: str,
    context_length: int,
    prediction_length: int,
    weather_covariate_mode: str,
    cross_learning: bool,
    add_weather_derived_features: bool,
    information_cutoff_local_time: str,
) -> dict[str, Any]:
    manifest = load_lora_artifact_manifest(artifact_path)
    expected_scalars = {
        "base_model_id": base_model_id,
        "base_model_revision": base_model_revision,
        "context_length": context_length,
        "prediction_length": prediction_length,
    }
    mismatches = {
        key: {"expected": expected, "observed": manifest.get(key)}
        for key, expected in expected_scalars.items()
        if manifest.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Chronos artifact scalar contract mismatch: {mismatches}")
    coverage = manifest["weather_horizon_coverage_policy"]
    config = Chronos2LoRAWeatherConfig(
        model_artifact_path=artifact_path,
        base_model_id=base_model_id,
        context_length=context_length,
        prediction_length=prediction_length,
        information_cutoff_local_time=information_cutoff_local_time,
        weather_covariate_mode=weather_covariate_mode,
        cross_learning=cross_learning,
        add_weather_derived_features=add_weather_derived_features,
        weather_horizon_min_coverage=float(coverage["minimum"]),
        weather_future_fallback_policy=str(
            coverage["insufficient_coverage_fallback"]
        ),
    )
    validate_artifact_weather_policy(manifest, config)
    return {
        "path": str(Path(artifact_path)),
        "release_id": manifest.get("release_id"),
        "artifact_content_sha256": manifest.get("artifact_content_sha256"),
        "covariates": list(manifest["covariates"]),
    }


if __name__ == "__main__":
    main()
