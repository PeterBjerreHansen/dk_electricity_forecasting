from __future__ import annotations

import pandas as pd
import pytest

from dkenergy_forecast.operations.publish_forecast import (
    default_run_id,
    resolve_forecast_origin,
    resolve_run_kind,
    validate_live_deadline,
    validate_live_price_context,
)
from dkenergy_forecast.operations.recent_diagnostics import _write_diagnostics
from dkenergy_forecast.publishing import update_latest_exports
from dkenergy_forecast.types import add_copenhagen_calendar


def test_live_origin_uses_execution_date_instead_of_latest_panel_date() -> None:
    panel = pd.DataFrame({"ds_utc": [pd.Timestamp("2030-01-10T22:00:00Z")]})

    origin = resolve_forecast_origin(
        panel,
        None,
        None,
        "12:00",
        reference_time_utc="2026-01-02T08:00:00Z",
    )

    assert origin == pd.Timestamp("2026-01-02T11:00:00Z")


def test_explicit_origin_defaults_to_replay_and_live_id_is_stable() -> None:
    assert resolve_run_kind(None, supplied_origin="2026-01-02T11:00:00Z") == "replay"
    assert resolve_run_kind(None, supplied_origin=None) == "live"
    assert (
        default_run_id("live", "2026-01-02T11:00:00Z")
        == "live_20260102T110000Z"
    )


def test_late_live_run_is_rejected_but_replay_is_allowed() -> None:
    with pytest.raises(ValueError, match="after its decision cutoff"):
        validate_live_deadline(
            run_kind="live",
            generated_at="2026-01-02T11:00:01Z",
            decision_cutoff="2026-01-02T11:00:00Z",
        )

    validate_live_deadline(
        run_kind="replay",
        generated_at="2026-02-01T00:00:00Z",
        decision_cutoff="2026-01-02T11:00:00Z",
    )


def test_live_context_requires_complete_current_delivery_day() -> None:
    timestamps = pd.date_range("2026-01-01T23:00:00Z", periods=24, freq="h")
    panel = add_copenhagen_calendar(
        pd.DataFrame(
            {
                "unique_id": ["day_ahead_price_DK1"] * len(timestamps),
                "area": ["DK1"] * len(timestamps),
                "ds_utc": timestamps,
                "y": range(len(timestamps)),
            }
        )
    )
    origin = pd.Timestamp("2026-01-02T11:00:00Z")

    validate_live_price_context(panel, origin)
    with pytest.raises(ValueError, match="complete current Danish delivery day"):
        validate_live_price_context(panel.iloc[:-1], origin)


def test_recent_diagnostics_use_a_separate_versioned_namespace(tmp_path) -> None:
    predictions = pd.DataFrame({"value": [1.0]})
    scores = pd.DataFrame({"metric": [2.0]})
    probabilistic = pd.DataFrame({"metric": [3.0]})
    result = _write_diagnostics(
        tmp_path / "recent_scores",
        run_id="diagnostics_1",
        predictions=predictions,
        scores=scores,
        probabilistic=probabilistic,
        manifest={"run_id": "diagnostics_1", "status": "success"},
    )

    assert result.paths["run_dir"].is_dir()
    assert pd.read_parquet(result.paths["recent_predictions"]).equals(predictions)
    assert pd.read_parquet(result.paths["recent_scores"]).equals(scores)


def test_live_latest_promotion_does_not_touch_diagnostic_exports(tmp_path) -> None:
    predictions = add_copenhagen_calendar(
        pd.DataFrame(
            {
                "unique_id": ["day_ahead_price_DK1"],
                "forecast_origin_utc": [pd.Timestamp("2026-01-02T11:00:00Z")],
                "ds_utc": [pd.Timestamp("2026-01-02T23:00:00Z")],
                "area": ["DK1"],
                "model_label": ["baseline"],
                "y_pred": [10.0],
            }
        )
    )
    scores = pd.DataFrame(
        columns=[
            "model_label",
            "area",
            "mae",
            "rmse",
            "bias",
            "coverage",
            "interval_width",
        ]
    )
    recent_dir = tmp_path / "recent_scores"

    paths = update_latest_exports(
        latest_forecast_dir=tmp_path / "latest",
        recent_scores_dir=recent_dir,
        dashboard_path=tmp_path / "dashboard.json",
        predictions=predictions,
        scores=scores,
        manifest={"forecast_origin_utc": "2026-01-02T11:00:00Z"},
        dashboard={"predictions": []},
        write_recent_scores=False,
    )

    assert "recent_scores" not in paths
    assert not recent_dir.exists()
