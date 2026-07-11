from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from dkenergy_forecast.operations.contracts import ForecastRequest, load_production_config
from dkenergy_forecast.operations.publish_forecast import (
    default_run_id,
    resolve_forecast_request,
    resolve_run_kind,
    validate_live_price_context,
)
from dkenergy_forecast.operations.recent_diagnostics import _write_diagnostics
from dkenergy_forecast.types import add_copenhagen_calendar


def test_live_request_uses_actual_start_as_information_cutoff() -> None:
    request = resolve_forecast_request(
        _args(),
        generated_at="2026-01-02T09:03:00Z",
    )

    assert request.information_cutoff_utc == pd.Timestamp("2026-01-02T09:03:00Z")
    assert request.decision_deadline_utc == pd.Timestamp("2026-01-02T11:00:00Z")
    assert request.delivery_date_local.isoformat() == "2026-01-03"
    assert request.forecast_origin_utc == request.information_cutoff_utc


def test_explicit_cutoff_defaults_to_replay() -> None:
    args = _args(information_cutoff_utc="2026-01-02T09:00:00Z")
    request = resolve_forecast_request(args, generated_at="2026-02-01T00:00:00Z")

    assert request.run_kind == "replay"
    assert resolve_run_kind(None, supplied_origin=None) == "live"
    with pytest.raises(ValueError, match="Unsupported run_kind"):
        resolve_run_kind("shadow", supplied_origin=None)


def test_forecast_request_rejects_invalid_time_contracts() -> None:
    with pytest.raises(ValueError, match="after its decision deadline"):
        ForecastRequest(
            delivery_date_local=pd.Timestamp("2026-01-03").date(),
            information_cutoff_utc=pd.Timestamp("2026-01-02T09:00:00Z"),
            decision_deadline_utc=pd.Timestamp("2026-01-02T11:00:00Z"),
            generated_at_utc=pd.Timestamp("2026-01-02T11:00:01Z"),
        )
    with pytest.raises(ValueError, match="after the information-cutoff date"):
        ForecastRequest(
            delivery_date_local=pd.Timestamp("2026-01-02").date(),
            information_cutoff_utc=pd.Timestamp("2026-01-02T09:00:00Z"),
            decision_deadline_utc=pd.Timestamp("2026-01-02T11:00:00Z"),
            generated_at_utc=pd.Timestamp("2026-01-02T09:00:00Z"),
        )


def test_live_run_id_records_delivery_date_and_cutoff() -> None:
    assert default_run_id(
        "live",
        "2026-01-02T09:03:00Z",
        delivery_date_local="2026-01-03",
    ) == "live_20260103_20260102T090300Z"


def test_live_context_requires_complete_previous_delivery_day() -> None:
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

    validate_live_price_context(
        panel,
        "2026-01-02T09:00:00Z",
        delivery_date_local="2026-01-03",
    )
    with pytest.raises(ValueError, match="complete current Danish delivery day"):
        validate_live_price_context(
            panel.iloc[:-1],
            "2026-01-02T09:00:00Z",
            delivery_date_local="2026-01-03",
        )


def test_production_config_has_one_primary_and_one_fallback(tmp_path) -> None:
    config_path = tmp_path / "production.json"
    config_path.write_text(
        '{"schema_version":1,"primary":{"model":"chronos_weather",'
        '"artifact_path":"models/release-1"},'
        '"fallback":{"model":"weighted_median_v1"}}',
        encoding="utf-8",
    )

    config = load_production_config(config_path, runtime_root=tmp_path)

    assert config.primary_model == "chronos_weather"
    assert config.primary_artifact_path == tmp_path / "models" / "release-1"
    assert config.fallback_model == "weighted_median_v1"


def test_recent_diagnostics_use_a_separate_versioned_namespace(tmp_path) -> None:
    result = _write_diagnostics(
        tmp_path / "recent_scores",
        run_id="diagnostics_1",
        predictions=pd.DataFrame({"value": [1.0]}),
        scores=pd.DataFrame({"metric": [2.0]}),
        probabilistic=pd.DataFrame({"metric": [3.0]}),
        manifest={"run_id": "diagnostics_1", "status": "success"},
    )

    assert result.paths["run_dir"].is_dir()
    assert pd.read_parquet(result.paths["recent_predictions"])["value"].tolist() == [1.0]


def _args(**overrides):
    values = {
        "information_cutoff_utc": None,
        "forecast_origin_utc": None,
        "run_kind": None,
        "delivery_date_local": None,
        "decision_deadline_utc": None,
        "decision_cutoff_utc": None,
        "decision_deadline_local_time": "12:00",
    }
    values.update(overrides)
    return SimpleNamespace(**values)
