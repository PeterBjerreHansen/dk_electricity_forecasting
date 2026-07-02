#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_data.build.open_meteo_weather_v1 import (  # noqa: E402
    COVERAGE_THRESHOLD,
    DATASET_VERSION,
    build_open_meteo_weather_from_raw,
)


def main() -> None:
    args = parse_args()
    result = build_open_meteo_weather_from_raw(
        raw_root=Path(args.raw_dir),
        normalized_dir=Path(args.normalized_dir),
        features_dir=Path(args.features_dir),
        dataset_version=args.dataset_version,
        coverage_threshold=args.coverage_threshold,
    )

    print(f"Wrote normalized Open-Meteo forecasts: {result.normalized_path}")
    print(f"Wrote long area weather features: {result.area_features_long_path}")
    print(f"Wrote wide area weather features: {result.area_features_wide_path}")
    print(f"Wrote QA report: {result.qa_path}")
    print(
        "Rows: normalized={normalized_row_count}; area_features={area_feature_row_count}; "
        "coverage threshold={coverage_threshold}".format(**result.qa)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Open-Meteo area-hourly weather features.")
    parser.add_argument(
        "--raw-dir",
        default=str(ROOT / "data" / "raw" / "open_meteo"),
    )
    parser.add_argument(
        "--normalized-dir",
        default=str(ROOT / "data" / "normalized"),
    )
    parser.add_argument(
        "--features-dir",
        default=str(ROOT / "data" / "features"),
    )
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument("--coverage-threshold", type=float, default=COVERAGE_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    main()
