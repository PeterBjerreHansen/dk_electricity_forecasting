#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_data.build.eds_prices_v1 import DATASET_VERSION, build_price_panel_from_raw


def main() -> None:
    args = parse_args()
    result = build_price_panel_from_raw(
        raw_root=Path(args.raw_dir),
        normalized_dir=Path(args.normalized_dir),
        model_ready_dir=Path(args.model_ready_dir),
        dataset_version=args.dataset_version,
        allow_incomplete_recent=args.allow_incomplete_recent,
        required_areas=args.required_areas,
        start_local=date.fromisoformat(args.start) if args.start else None,
        end_local=date.fromisoformat(args.end) if args.end else None,
    )

    print(f"Wrote model-ready panel: {result.panel_path}")
    print(f"Wrote QA report: {result.qa_path}")
    for dataset, path in result.normalized_paths.items():
        print(f"Wrote normalized {dataset}: {path}")
    print(
        "Rows: {row_count}; UTC range: {min_ds_utc} -> {max_ds_utc}".format(
            **result.qa
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the EDS hourly price panel.")
    parser.add_argument(
        "--raw-dir",
        default=str(ROOT / "data" / "raw" / "energi_data_service"),
        help="Raw EDS input directory.",
    )
    parser.add_argument(
        "--normalized-dir",
        default=str(ROOT / "data" / "normalized"),
        help="Normalized parquet output directory.",
    )
    parser.add_argument(
        "--model-ready-dir",
        default=str(ROOT / "data" / "model_ready"),
        help="Model-ready parquet output directory.",
    )
    parser.add_argument("--dataset-version", default=DATASET_VERSION)
    parser.add_argument(
        "--required-areas",
        nargs="+",
        default=["DK1", "DK2"],
        help="Areas required in the model-ready panel; pass one area only for explicit experiments.",
    )
    parser.add_argument("--start", help="Optional inclusive local delivery start date.")
    parser.add_argument("--end", help="Optional exclusive local delivery end date.")
    parser.add_argument(
        "--allow-incomplete-recent",
        action="store_true",
        help="Drop only the latest incomplete DayAheadPrices hour per area.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
