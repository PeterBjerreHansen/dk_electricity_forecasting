#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.cloud_pipeline import (  # noqa: E402
    CloudPipelineConfig,
    default_model_artifact_uri,
    run_cloud_pipeline,
)


def main() -> None:
    args = parse_args()
    artifact_store_uri = args.artifact_store_uri or os.environ.get("DKENERGY_ARTIFACT_STORE_URI")
    if not artifact_store_uri:
        raise SystemExit("Missing --artifact-store-uri or DKENERGY_ARTIFACT_STORE_URI")
    model_artifact_uri = (
        args.model_artifact_uri
        or os.environ.get("DKENERGY_MODEL_ARTIFACT_URI")
        or default_model_artifact_uri(artifact_store_uri)
    )
    workdir = Path(args.workdir or os.environ.get("DKENERGY_WORKDIR", "/var/lib/dkenergy"))
    uploaded = run_cloud_pipeline(
        CloudPipelineConfig(
            artifact_store_uri=artifact_store_uri,
            workdir=workdir,
            model_artifact_uri=model_artifact_uri,
            with_weather=not args.skip_weather,
            score_max_origins=args.score_max_origins,
        ),
        dry_run=args.dry_run,
    )
    if uploaded:
        print("Uploaded artifact keys:")
        for key in uploaded:
            print(f"  {key}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the production daily pipeline with S3/file artifact synchronization.",
    )
    parser.add_argument("--artifact-store-uri", help="Artifact store root, e.g. s3://bucket/prefix.")
    parser.add_argument("--model-artifact-uri", help="Trained Chronos LoRA artifact directory URI.")
    parser.add_argument("--workdir", help="Writable runtime directory. Defaults to DKENERGY_WORKDIR or /var/lib/dkenergy.")
    parser.add_argument("--score-max-origins", type=int, help="Override recent scoring origin count.")
    parser.add_argument("--skip-weather", action="store_true", help="Do not refresh Open-Meteo before publishing.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
