#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.cloud_pipeline import CloudScoringConfig, run_cloud_scoring  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score completed production forecasts in the remote artifact store.",
    )
    parser.add_argument("--artifact-store-uri")
    parser.add_argument("--workdir")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    artifact_store_uri = args.artifact_store_uri or os.environ.get(
        "DKENERGY_ARTIFACT_STORE_URI"
    )
    if not artifact_store_uri:
        raise SystemExit("Missing --artifact-store-uri or DKENERGY_ARTIFACT_STORE_URI")
    workdir = Path(args.workdir or os.environ.get("DKENERGY_WORKDIR", "/var/lib/dkenergy"))
    uploaded = run_cloud_scoring(
        CloudScoringConfig(artifact_store_uri=artifact_store_uri, workdir=workdir),
        dry_run=args.dry_run,
    )
    for key in uploaded:
        print(f"Uploaded {key}")


if __name__ == "__main__":
    main()
