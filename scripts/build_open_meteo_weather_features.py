#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_data.build.open_meteo_weather_v1 import (  # noqa: E402
    COVERAGE_THRESHOLD,
    DATASET_VERSION,
    build_open_meteo_weather_from_raw,
)


DEFAULT_LOOKBACK_DAYS = 90


def main() -> None:
    args = parse_args()
    result = build_open_meteo_weather_from_raw(
        raw_root=Path(args.raw_dir),
        normalized_dir=Path(args.normalized_dir),
        features_dir=Path(args.features_dir),
        dataset_version=args.dataset_version,
        coverage_threshold=args.coverage_threshold,
        write_wide=args.write_wide,
        min_valid_time=args.start,
        max_valid_time=args.end,
    )

    print(f"Wrote normalized Open-Meteo forecasts: {result.normalized_path}")
    print(f"Wrote long area weather features: {result.area_features_long_path}")
    if result.area_features_wide_path is not None:
        print(f"Wrote wide area weather features: {result.area_features_wide_path}")
    print(f"Wrote QA report: {result.qa_path}")
    print(
        "Rows: normalized={normalized_row_count}; area_features={area_feature_row_count}; "
        "coverage threshold={coverage_threshold}".format(**result.qa)
    )


def parse_args() -> argparse.Namespace:
    default_start = _lookback_month_start_copenhagen(DEFAULT_LOOKBACK_DAYS)
    default_end = _tomorrow_copenhagen()
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
    parser.add_argument(
        "--start",
        default=default_start,
        help="Inclusive minimum valid_time_utc to materialize. Defaults to a 90-day rolling window.",
    )
    parser.add_argument(
        "--end",
        default=default_end,
        help="Inclusive maximum valid_time_utc to materialize. Defaults to tomorrow in Europe/Copenhagen.",
    )
    parser.add_argument("--coverage-threshold", type=float, default=COVERAGE_THRESHOLD)
    parser.add_argument(
        "--write-wide",
        action="store_true",
        help="Also write the derived wide table for ad hoc inspection. The long table is canonical.",
    )
    return parser.parse_args()


def _tomorrow_copenhagen() -> str:
    return (datetime.now(ZoneInfo("Europe/Copenhagen")).date() + timedelta(days=1)).isoformat()


def _lookback_month_start_copenhagen(days: int) -> str:
    lookback_date = datetime.now(ZoneInfo("Europe/Copenhagen")).date() - timedelta(days=days)
    return lookback_date.replace(day=1).isoformat()


if __name__ == "__main__":
    main()
