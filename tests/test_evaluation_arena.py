from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from dkenergy_forecast.evaluation import (
    PromotionPolicy,
    block_bootstrap_mean_ci,
    build_evaluation_report,
    explicit_evaluation_interval,
    interval_score,
    load_frozen_date_splits,
    mean_absolute_calibration_error,
    model_score_table,
    paired_model_predictions,
    paired_origin_comparison,
    prepare_evaluation_strata,
    weighted_interval_score,
    write_evaluation_report,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "candidate_v2"
CHAMPION = "champion_v1"


def test_probabilistic_metrics_and_production_score_columns() -> None:
    predictions = pd.DataFrame(
        {
            "model_label": [CANDIDATE],
            "area": ["DK1"],
            "y": [0.0],
            "y_pred": [0.0],
            "q10": [-1.0],
            "q50": [0.0],
            "q90": [1.0],
        }
    )

    assert interval_score(predictions) == pytest.approx(2.0)
    assert weighted_interval_score(predictions) == pytest.approx(2.0 / 15.0)
    scores = model_score_table(predictions)
    all_area = scores[scores["area"].eq("ALL")].iloc[0]
    assert all_area["pinball_q10"] == pytest.approx(0.1)
    assert all_area["pinball_q50"] == pytest.approx(0.0)
    assert all_area["pinball_q90"] == pytest.approx(0.1)
    assert all_area["interval_score_80"] == pytest.approx(2.0)
    assert all_area["weighted_interval_score"] == pytest.approx(2.0 / 15.0)


def test_pairing_is_exact_and_comparisons_are_per_origin() -> None:
    predictions = _arena_predictions()

    paired = paired_model_predictions(
        predictions,
        candidate_label=CANDIDATE,
        champion_label=CHAMPION,
    )
    comparisons = paired_origin_comparison(
        predictions,
        candidate_label=CANDIDATE,
        champion_label=CHAMPION,
    )

    assert len(paired) == len(predictions) / 2
    assert comparisons["forecast_origin_utc"].is_monotonic_increasing
    assert comparisons["rows"].eq(10).all()
    assert comparisons["mae_difference"].eq(-5.0).all()
    assert comparisons["mae_winner"].eq(CANDIDATE).all()

    missing_champion_row = predictions.drop(
        predictions[predictions["model_label"].eq(CHAMPION)].index[0]
    )
    with pytest.raises(ValueError, match="prediction keys differ"):
        paired_model_predictions(
            missing_champion_row,
            candidate_label=CANDIDATE,
            champion_label=CHAMPION,
        )


def test_block_bootstrap_is_deterministic_over_origins() -> None:
    first = block_bootstrap_mean_ci(
        [-3.0, -2.0, -1.0, 0.0, 1.0],
        block_length=2,
        n_resamples=100,
        seed=7,
    )
    second = block_bootstrap_mean_ci(
        [-3.0, -2.0, -1.0, 0.0, 1.0],
        block_length=2,
        n_resamples=100,
        seed=7,
    )

    assert first == second
    assert first["mean"] == pytest.approx(-1.0)
    assert first["block_length"] == 2
    assert first["method"] == "circular_moving_block_bootstrap"


def test_frozen_splits_are_declared_hashed_and_non_overlapping(tmp_path: Path) -> None:
    split_path = tmp_path / "splits.json"
    split_path.write_text(
        json.dumps(
            {
                "frozen": True,
                "timestamp_column": "forecast_origin_utc",
                "splits": {
                    "validation": {
                        "start_utc": "2025-09-27T00:00:00Z",
                        "end_utc": "2025-09-30T00:00:00Z",
                    },
                    "test": {
                        "start_utc": "2025-09-30T00:00:00Z",
                        "end_utc": "2025-10-04T00:00:00Z",
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    frozen = load_frozen_date_splits(split_path)

    assert len(frozen.sha256) == 64
    assert frozen.select("test").start_utc == pd.Timestamp("2025-09-30T00:00:00Z")
    with pytest.raises(ValueError, match="Unknown frozen split"):
        frozen.select("holdout")

    split_path.write_text(
        json.dumps(
            {
                "frozen": True,
                "splits": {
                    "a": {"start_utc": "2025-01-01", "end_utc": "2025-02-01"},
                    "b": {"start_utc": "2025-01-31", "end_utc": "2025-03-01"},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlap"):
        load_frozen_date_splits(split_path)


def test_committed_frozen_split_example_is_valid() -> None:
    frozen = load_frozen_date_splits(
        ROOT / "config" / "evaluation_splits.example.json"
    )

    assert set(frozen.intervals) == {"development", "validation", "test"}


def test_stratification_covers_requested_operational_slices() -> None:
    predictions = _arena_predictions()

    prepared, threshold = prepare_evaluation_strata(predictions)

    assert threshold >= 0
    assert {"2025-09", "2025-10"}.issubset(set(prepared["evaluation_month"]))
    assert set(prepared["area"]) == {"DK1", "DK2"}
    assert prepared["evaluation_hour"].nunique() > 1
    assert set(prepared["evaluation_dst"]) == {"dst"}
    assert set(prepared["evaluation_price_sign"]) == {"negative", "non_negative"}
    assert set(prepared["evaluation_extreme_price"]) == {"extreme", "typical"}
    assert set(prepared["market_regime"]) == {
        "native_hourly",
        "quarter_hour_aggregated_to_hourly",
    }


def test_report_applies_promotion_policy_and_writes_strict_json(tmp_path: Path) -> None:
    predictions = _arena_predictions()
    interval = explicit_evaluation_interval(
        start_utc="2025-09-27T00:00:00Z",
        end_utc="2025-10-04T00:00:00Z",
    )

    report = build_evaluation_report(
        predictions,
        candidate_label=CANDIDATE,
        champion_label=CHAMPION,
        interval=interval,
        policy=PromotionPolicy(min_subgroup_rows=1),
        block_length=2,
        n_resamples=50,
        seed=11,
        source_sha256="abc123",
    )
    written = write_evaluation_report(report, tmp_path)

    assert report["promotion"]["decision"] == "promote_candidate"
    assert report["pairing"]["origin_count"] == 6
    assert len(report["paired_origin_comparisons"]) == 6
    assert all(
        row["passed"] is not False
        for row in report["stratification"]["guardrails"]
    )
    saved = json.loads(written["json"].read_text(encoding="utf-8"))
    assert saved["source_sha256"] == "abc123"
    assert saved["promotion"]["passed"] is True
    assert "NaN" not in written["json"].read_text(encoding="utf-8")
    assert "Per-origin paired comparison" in written["markdown"].read_text(
        encoding="utf-8"
    )


def test_point_champion_skips_only_unavailable_relative_probabilistic_checks() -> None:
    predictions = _arena_predictions()
    point_champion = predictions["model_label"].eq(CHAMPION)
    predictions.loc[point_champion, ["q10", "q50", "q90"]] = float("nan")
    interval = explicit_evaluation_interval(
        start_utc="2025-09-27T00:00:00Z",
        end_utc="2025-10-04T00:00:00Z",
    )

    report = build_evaluation_report(
        predictions,
        candidate_label=CANDIDATE,
        champion_label=CHAMPION,
        interval=interval,
        policy=PromotionPolicy(
            min_subgroup_rows=1,
            require_probabilistic_comparison=False,
        ),
        n_resamples=20,
    )
    checks = {check["name"]: check for check in report["promotion"]["checks"]}

    assert checks["weighted_interval_score"]["passed"] is None
    assert checks["calibration_vs_champion"]["passed"] is None
    assert checks["absolute_calibration"]["passed"] is True
    assert report["promotion"]["decision"] == "promote_candidate"

    predictions.loc[
        predictions["model_label"].eq(CANDIDATE), ["q10", "q50", "q90"]
    ] = float("nan")
    missing_candidate_report = build_evaluation_report(
        predictions,
        candidate_label=CANDIDATE,
        champion_label=CHAMPION,
        interval=interval,
        policy=PromotionPolicy(
            min_subgroup_rows=1,
            require_probabilistic_comparison=False,
        ),
        n_resamples=20,
    )
    missing_checks = {
        check["name"]: check
        for check in missing_candidate_report["promotion"]["checks"]
    }
    assert missing_checks["absolute_calibration"]["passed"] is False
    assert missing_candidate_report["promotion"]["decision"] == "retain_champion"


def test_cli_generates_json_and_markdown_from_prediction_parquet(tmp_path: Path) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    output_dir = tmp_path / "report"
    _arena_predictions().to_parquet(prediction_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_evaluation_arena.py",
            "--predictions",
            str(prediction_path),
            "--candidate",
            CANDIDATE,
            "--champion",
            CHAMPION,
            "--start-utc",
            "2025-09-27T00:00:00Z",
            "--end-utc",
            "2025-10-04T00:00:00Z",
            "--bootstrap-resamples",
            "20",
            "--block-length",
            "2",
            "--min-subgroup-rows",
            "1",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Evaluation decision: promote_candidate" in result.stdout
    assert (output_dir / "evaluation_report.json").exists()
    assert (output_dir / "evaluation_report.md").exists()


def test_calibration_metric_uses_q10_q50_q90_empirical_frequencies() -> None:
    candidate = _arena_predictions().query("model_label == @CANDIDATE")

    assert mean_absolute_calibration_error(candidate) == pytest.approx(0.0)


def _arena_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    origins = pd.date_range("2025-09-27T10:00:00Z", periods=6, freq="D")
    for origin_number, origin in enumerate(origins):
        targets = pd.date_range(origin.normalize() + pd.Timedelta(days=1), periods=10, freq="h")
        for position, target in enumerate(targets):
            area = "DK1" if position % 2 == 0 else "DK2"
            actual = -10.0 - origin_number if position == 0 else 100.0 + position + origin_number
            common = {
                "unique_id": f"day_ahead_price_{area}",
                "forecast_origin_utc": origin,
                "ds_utc": target,
                "area": area,
                "y": actual,
            }
            rows.append(
                {
                    **common,
                    "model_label": CANDIDATE,
                    "y_pred": actual,
                    "q10": actual + 0.1 if position == 0 else actual - 1.0,
                    "q50": actual + 0.1 if position < 5 else actual - 0.5,
                    "q90": actual + 0.2 if position < 9 else actual - 0.25,
                }
            )
            rows.append(
                {
                    **common,
                    "model_label": CHAMPION,
                    "y_pred": actual + 5.0,
                    "q10": actual + 1.0 if position == 0 else actual - 10.0,
                    "q50": actual + 5.0 if position < 5 else actual - 5.0,
                    "q90": actual + 10.0 if position < 9 else actual - 1.0,
                }
            )
    return pd.DataFrame(rows)
