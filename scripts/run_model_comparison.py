#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.evaluation import (  # noqa: E402
    EvaluationInterval,
    build_model_comparison,
    explicit_evaluation_interval,
    load_frozen_date_splits,
    sha256_file,
    write_model_comparison,
)


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    predictions = pd.read_parquet(predictions_path)
    interval, split_provenance = resolve_interval(args)
    report = build_model_comparison(
        predictions,
        reference_model=args.reference_model,
        comparison_model=args.comparison_model,
        reference_release=args.reference_release,
        comparison_release=args.comparison_release,
        interval=interval,
        confidence=args.confidence,
        block_length=args.block_length,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
        extreme_threshold=args.extreme_threshold,
        extreme_quantile=args.extreme_quantile,
        split_provenance=split_provenance,
        source_sha256=sha256_file(predictions_path),
    )
    written = write_model_comparison(report, args.output_dir)

    difference = report["overall"]["differences"]["mae"]
    print(
        f"Compared {args.comparison_model!r} with reference "
        f"{args.reference_model!r}"
    )
    print(f"Paired rows: {report['pairing']['paired_rows']}")
    print(f"Forecast origins: {report['pairing']['origin_count']}")
    print(f"Overall MAE difference (comparison - reference): {difference:.6f}")
    for label, path in written.items():
        print(f"Wrote {label}: {path}")


def resolve_interval(
    args: argparse.Namespace,
) -> tuple[EvaluationInterval, dict[str, object]]:
    uses_frozen_split = args.splits_json is not None or args.split is not None
    uses_explicit_interval = args.start_utc is not None or args.end_utc is not None
    if uses_frozen_split and uses_explicit_interval:
        raise ValueError(
            "Choose either --splits-json/--split or --start-utc/--end-utc, not both"
        )
    if uses_frozen_split:
        if not args.splits_json or not args.split:
            raise ValueError("--splits-json and --split must be provided together")
        frozen = load_frozen_date_splits(args.splits_json)
        return frozen.select(args.split), {
            "kind": "frozen_date_split",
            "split_name": args.split,
            "split_file": frozen.source_path.name,
            "split_file_sha256": frozen.sha256,
        }
    if args.start_utc is None or args.end_utc is None:
        raise ValueError("Provide a frozen split or both --start-utc and --end-utc")
    return explicit_evaluation_interval(
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        timestamp_column=args.timestamp_column,
    ), {"kind": "explicit_evaluation_interval"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two models on exactly paired prediction rows and write "
            "descriptive JSON and Markdown reports."
        )
    )
    parser.add_argument("--predictions", required=True, help="Prediction parquet file.")
    parser.add_argument(
        "--reference-model",
        required=True,
        help="model_label used as the comparison reference.",
    )
    parser.add_argument(
        "--comparison-model",
        required=True,
        help="model_label whose differences from the reference are reported.",
    )
    parser.add_argument("--reference-release", help="Optional model_release_id for the reference.")
    parser.add_argument("--comparison-release", help="Optional model_release_id for the comparison.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "model_comparison"),
        help="Directory for model_comparison.json and model_comparison.md.",
    )

    interval = parser.add_argument_group("evaluation interval")
    interval.add_argument("--start-utc", help="Explicit inclusive interval start.")
    interval.add_argument("--end-utc", help="Explicit exclusive interval end.")
    interval.add_argument(
        "--timestamp-column",
        default="forecast_origin_utc",
        help="Timestamp column used by an explicit interval.",
    )
    interval.add_argument("--splits-json", help="Frozen date-split JSON declaration.")
    interval.add_argument("--split", help="Named split from --splits-json.")

    statistics = parser.add_argument_group("statistics")
    statistics.add_argument("--confidence", type=float, default=0.95)
    statistics.add_argument("--block-length", type=int, default=7)
    statistics.add_argument("--bootstrap-resamples", type=int, default=2_000)
    statistics.add_argument("--seed", type=int, default=2026)
    statistics.add_argument("--extreme-quantile", type=float, default=0.95)
    statistics.add_argument("--extreme-threshold", type=float)
    return parser.parse_args()


if __name__ == "__main__":
    main()
