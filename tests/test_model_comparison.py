from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from dkenergy_forecast.evaluation import (
    build_model_comparison,
    explicit_evaluation_interval,
    interval_score,
    load_frozen_date_splits,
    mean_absolute_calibration_error,
    model_score_table,
    moving_block_bootstrap_mean_ci,
    origin_metric_differences,
    pair_model_predictions,
    prepare_evaluation_strata,
    weighted_interval_score,
    write_model_comparison,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = "weighted_median_v1"
COMPARISON = "chronos_weather_release_2026_07"


def test_probabilistic_metrics_and_model_score_columns() -> None:
    predictions = pd.DataFrame(
        {
            "model_label": [COMPARISON],
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


def test_pairing_is_exact_and_differences_are_per_origin() -> None:
    predictions = _comparison_predictions()

    paired = pair_model_predictions(
        predictions,
        reference_model=REFERENCE,
        comparison_model=COMPARISON,
    )
    differences = origin_metric_differences(
        predictions,
        reference_model=REFERENCE,
        comparison_model=COMPARISON,
    )

    assert len(paired) == len(predictions) / 2
    assert differences["forecast_origin_utc"].is_monotonic_increasing
    assert differences["rows"].eq(10).all()
    assert differences["mae_difference"].eq(-5.0).all()
    assert differences["reference_mae"].eq(5.0).all()
    assert differences["comparison_mae"].eq(0.0).all()

    missing_reference_row = predictions.drop(
        predictions[predictions["model_label"].eq(REFERENCE)].index[0]
    )
    with pytest.raises(ValueError, match="prediction keys differ"):
        pair_model_predictions(
            missing_reference_row,
            reference_model=REFERENCE,
            comparison_model=COMPARISON,
        )


def test_pairing_rejects_duplicate_and_disagreeing_actual_rows() -> None:
    predictions = _comparison_predictions()
    duplicate = pd.concat(
        [
            predictions,
            predictions[predictions["model_label"].eq(COMPARISON)].iloc[[0]],
        ],
        ignore_index=True,
    )
    with pytest.raises(ValueError, match="duplicate prediction key"):
        pair_model_predictions(
            duplicate,
            reference_model=REFERENCE,
            comparison_model=COMPARISON,
        )

    disagreeing = predictions.copy()
    reference_row = disagreeing["model_label"].eq(REFERENCE).idxmax()
    disagreeing.loc[reference_row, "y"] += 1.0
    with pytest.raises(ValueError, match="disagree on actual"):
        pair_model_predictions(
            disagreeing,
            reference_model=REFERENCE,
            comparison_model=COMPARISON,
        )


def test_comparison_requires_an_explicit_release_when_a_label_has_multiple() -> None:
    predictions = _comparison_predictions().assign(model_release_id="release-1")
    extra = predictions[predictions["model_label"].eq(COMPARISON)].copy()
    extra["model_release_id"] = "release-2"
    predictions = pd.concat([predictions, extra], ignore_index=True)

    with pytest.raises(ValueError, match="multiple releases"):
        pair_model_predictions(
            predictions,
            reference_model=REFERENCE,
            comparison_model=COMPARISON,
        )

    paired = pair_model_predictions(
        predictions,
        reference_model=REFERENCE,
        comparison_model=COMPARISON,
        reference_release="release-1",
        comparison_release="release-2",
    )
    assert len(paired) == len(_comparison_predictions()) / 2

def test_moving_block_bootstrap_is_deterministic_over_origins() -> None:
    first = moving_block_bootstrap_mean_ci(
        [-3.0, -2.0, -1.0, 0.0, 1.0],
        block_length=2,
        n_resamples=100,
        seed=7,
    )
    second = moving_block_bootstrap_mean_ci(
        [-3.0, -2.0, -1.0, 0.0, 1.0],
        block_length=2,
        n_resamples=100,
        seed=7,
    )

    assert first == second
    assert first["mean"] == pytest.approx(-1.0)
    assert first["block_length"] == 2
    assert first["method"] == "circular_moving_block_bootstrap"


def test_frozen_splits_are_hashed_and_non_overlapping(tmp_path: Path) -> None:
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


def test_stratification_covers_useful_operational_slices() -> None:
    prepared, threshold = prepare_evaluation_strata(_comparison_predictions())

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


def test_report_is_descriptive_paired_and_strict_json(tmp_path: Path) -> None:
    report = build_model_comparison(
        _comparison_predictions(),
        reference_model=REFERENCE,
        comparison_model=COMPARISON,
        interval=_full_interval(),
        block_length=2,
        n_resamples=50,
        seed=11,
        source_sha256="abc123",
    )
    written = write_model_comparison(report, tmp_path)

    assert report["schema_version"] == "model_comparison_v1"
    assert report["difference_definition"] == "comparison_minus_reference"
    assert report["pairing"]["origin_count"] == 6
    assert report["overall"]["differences"]["mae"] == pytest.approx(-5.0)
    assert len(report["per_origin_differences"]) == 6
    assert report["bootstrap_confidence_intervals"]["mae"]["upper"] < 0
    saved_text = written["json"].read_text(encoding="utf-8")
    saved = json.loads(saved_text)
    assert saved["source_sha256"] == "abc123"
    assert "NaN" not in saved_text
    assert "decision" not in saved_text.lower()
    markdown = written["markdown"].read_text(encoding="utf-8")
    assert "Per-origin differences" in markdown
    assert "does not select or deploy a model" in markdown


def test_point_only_reference_keeps_available_descriptive_metrics() -> None:
    predictions = _comparison_predictions()
    predictions.loc[
        predictions["model_label"].eq(REFERENCE), ["q10", "q50", "q90"]
    ] = float("nan")

    report = build_model_comparison(
        predictions,
        reference_model=REFERENCE,
        comparison_model=COMPARISON,
        interval=_full_interval(),
        n_resamples=20,
    )

    assert report["overall"]["reference"]["metrics"]["mae"] == pytest.approx(5.0)
    assert report["overall"]["reference"]["metrics"]["weighted_interval_score"] != (
        report["overall"]["reference"]["metrics"]["weighted_interval_score"]
    )
    assert report["bootstrap_confidence_intervals"]["mae"] is not None
    assert report["bootstrap_confidence_intervals"][
        "weighted_interval_score"
    ] is None


def test_cli_generates_json_and_markdown_from_prediction_parquet(
    tmp_path: Path,
) -> None:
    prediction_path = tmp_path / "predictions.parquet"
    output_dir = tmp_path / "report"
    _comparison_predictions().to_parquet(prediction_path, index=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/run_model_comparison.py",
            "--predictions",
            str(prediction_path),
            "--reference-model",
            REFERENCE,
            "--comparison-model",
            COMPARISON,
            "--start-utc",
            "2025-09-27T00:00:00Z",
            "--end-utc",
            "2025-10-04T00:00:00Z",
            "--bootstrap-resamples",
            "20",
            "--block-length",
            "2",
            "--output-dir",
            str(output_dir),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Overall MAE difference (comparison - reference): -5.000000" in result.stdout
    assert (output_dir / "model_comparison.json").exists()
    assert (output_dir / "model_comparison.md").exists()


def test_calibration_metric_uses_q10_q50_q90_empirical_frequencies() -> None:
    comparison = _comparison_predictions().query("model_label == @COMPARISON")

    assert mean_absolute_calibration_error(comparison) == pytest.approx(0.0)


def _full_interval():
    return explicit_evaluation_interval(
        start_utc="2025-09-27T00:00:00Z",
        end_utc="2025-10-04T00:00:00Z",
    )


def _comparison_predictions() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    origins = pd.date_range("2025-09-27T10:00:00Z", periods=6, freq="D")
    for origin_number, origin in enumerate(origins):
        targets = pd.date_range(
            origin.normalize() + pd.Timedelta(days=1), periods=10, freq="h"
        )
        for position, target in enumerate(targets):
            area = "DK1" if position % 2 == 0 else "DK2"
            actual = (
                -10.0 - origin_number
                if position == 0
                else 100.0 + position + origin_number
            )
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
                    "model_label": COMPARISON,
                    "y_pred": actual,
                    "q10": actual + 0.1 if position == 0 else actual - 1.0,
                    "q50": actual + 0.1 if position < 5 else actual - 0.5,
                    "q90": actual + 0.2 if position < 9 else actual - 0.25,
                }
            )
            rows.append(
                {
                    **common,
                    "model_label": REFERENCE,
                    "y_pred": actual + 5.0,
                    "q10": actual + 1.0 if position == 0 else actual - 10.0,
                    "q50": actual + 5.0 if position < 5 else actual - 5.0,
                    "q90": actual + 10.0 if position < 9 else actual - 1.0,
                }
            )
    return pd.DataFrame(rows)
