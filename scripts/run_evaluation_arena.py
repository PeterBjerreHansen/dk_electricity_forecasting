#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.evaluation import (  # noqa: E402
    PromotionPolicy,
    EvaluationInterval,
    build_evaluation_report,
    explicit_evaluation_interval,
    load_frozen_date_splits,
    sha256_file,
    write_evaluation_report,
)


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    predictions = pd.read_parquet(predictions_path)
    interval, split_provenance = resolve_interval(args)
    policy = PromotionPolicy(
        min_mae_relative_improvement=args.min_mae_relative_improvement,
        max_wis_relative_degradation=args.max_wis_relative_degradation,
        max_calibration_error_increase=args.max_calibration_error_increase,
        max_calibration_error=args.max_calibration_error,
        max_subgroup_mae_relative_degradation=(
            args.max_subgroup_mae_relative_degradation
        ),
        min_subgroup_rows=args.min_subgroup_rows,
        require_probabilistic_comparison=(
            not args.allow_missing_probabilistic_comparison
        ),
        require_mae_ci_improvement=not args.do_not_require_mae_ci_improvement,
    )
    report = build_evaluation_report(
        predictions,
        candidate_label=args.candidate,
        champion_label=args.champion,
        interval=interval,
        policy=policy,
        confidence=args.confidence,
        block_length=args.block_length,
        n_resamples=args.bootstrap_resamples,
        seed=args.seed,
        extreme_threshold=args.extreme_threshold,
        extreme_quantile=args.extreme_quantile,
        split_provenance=split_provenance,
        source_sha256=sha256_file(predictions_path),
    )
    written = write_evaluation_report(report, args.output_dir)

    print(f"Evaluation decision: {report['promotion']['decision']}")
    print(f"Paired rows: {report['pairing']['paired_rows']}")
    print(f"Forecast origins: {report['pairing']['origin_count']}")
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
        raise ValueError(
            "Provide a frozen split or both --start-utc and --end-utc"
        )
    return explicit_evaluation_interval(
        start_utc=args.start_utc,
        end_utc=args.end_utc,
        timestamp_column=args.timestamp_column,
    ), {
        "kind": "explicit_evaluation_interval",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a candidate and champion on exactly paired prediction rows and "
            "write deterministic JSON and Markdown evaluation reports."
        )
    )
    parser.add_argument("--predictions", required=True, help="Prediction parquet file.")
    parser.add_argument("--candidate", required=True, help="Candidate model_label.")
    parser.add_argument("--champion", required=True, help="Champion model_label.")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "results" / "evaluation_arena"),
        help="Directory for evaluation_report.json and evaluation_report.md.",
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

    policy = parser.add_argument_group("promotion policy")
    policy.add_argument("--min-mae-relative-improvement", type=float, default=0.01)
    policy.add_argument("--max-wis-relative-degradation", type=float, default=0.0)
    policy.add_argument("--max-calibration-error-increase", type=float, default=0.02)
    policy.add_argument("--max-calibration-error", type=float, default=0.10)
    policy.add_argument(
        "--max-subgroup-mae-relative-degradation",
        type=float,
        default=0.10,
    )
    policy.add_argument("--min-subgroup-rows", type=int, default=24)
    policy.add_argument(
        "--allow-missing-probabilistic-comparison",
        action="store_true",
        help=(
            "Allow a point-only champion; the candidate must still provide q10/q50/q90 "
            "and pass the absolute calibration check."
        ),
    )
    policy.add_argument(
        "--do-not-require-mae-ci-improvement",
        action="store_true",
        help="Skip the requirement that the paired MAE CI upper bound is non-positive.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
