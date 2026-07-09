from __future__ import annotations

import sys
import types

import pandas as pd
import pytest


if "streamlit" not in sys.modules:
    sys.modules["streamlit"] = types.ModuleType("streamlit")

from app import streamlit_app


def test_backtest_visible_days_filters_recomputed_scores() -> None:
    predictions = pd.DataFrame(
        [
            _prediction_row("run-a", "2026-07-01T00:00:00Z", "DK1", "model_a", 10.0, 0.0),
            _prediction_row("run-a", "2026-07-08T00:00:00Z", "DK1", "model_a", 10.0, 12.0),
            _prediction_row("run-a", "2026-07-10T00:00:00Z", "DK1", "model_a", 10.0, 16.0),
            _prediction_row("run-a", "2026-07-08T00:00:00Z", "DK1", "model_b", 10.0, 14.0),
            _prediction_row("run-a", "2026-07-10T00:00:00Z", "DK2", "model_b", 20.0, 10.0),
            _prediction_row("run-b", "2026-07-20T00:00:00Z", "DK1", "model_a", 100.0, 100.0),
        ]
    )

    prepared = streamlit_app._prepare_backtest_predictions(predictions)
    run_frame = prepared[prepared["run_id"] == "run-a"]
    visible = streamlit_app._filter_recent_backtest_rows(run_frame, visible_days=3)
    scores = streamlit_app._score_table_for_predictions(visible)

    model_a_all = scores[(scores["model_label"] == "model_a") & (scores["area"] == "ALL")].iloc[0]
    assert model_a_all["rows"] == 2
    assert model_a_all["mae"] == pytest.approx(4.0)


def test_backtest_window_caption_shows_filtered_rows_and_dates() -> None:
    predictions = pd.DataFrame(
        [
            _prediction_row("run-a", "2026-07-01T22:00:00Z", "DK1", "model_a", 10.0, 11.0),
            _prediction_row("run-a", "2026-07-05T21:00:00Z", "DK1", "model_a", 10.0, 12.0),
        ]
    )
    prepared = streamlit_app._prepare_backtest_predictions(predictions)

    caption = streamlit_app._backtest_window_caption(prepared.iloc[[1]], total_rows=len(prepared))

    assert caption == "Showing 1 of 2 rows from 2026-07-05 21:00 UTC to 2026-07-05 21:00 UTC."


def test_backtest_predictions_get_default_run_id() -> None:
    predictions = pd.DataFrame(
        [_prediction_row("ignored", "2026-07-01T00:00:00Z", "DK1", "model_a", 10.0, 11.0)]
    ).drop(columns=["run_id"])

    prepared = streamlit_app._prepare_backtest_predictions(predictions)

    assert prepared["run_id"].tolist() == ["backtest"]
    assert str(prepared["ds_utc"].dtype) == "datetime64[ns, UTC]"


def test_backtest_tab_loader_prefers_backtest_artifacts(tmp_path) -> None:
    backtest_dir = tmp_path / "baseline_v1"
    backtest_dir.mkdir()
    pd.DataFrame(
        [_prediction_row("ignored", "2026-07-01T00:00:00Z", "DK1", "model_a", 10.0, 11.0)]
    ).drop(columns=["run_id"]).to_parquet(backtest_dir / "predictions.parquet", index=False)
    recent_payload = {
        "recent_predictions": [
            _prediction_row("recent-run", "2026-07-20T00:00:00Z", "DK1", "model_b", 10.0, 20.0)
        ]
    }

    loaded = streamlit_app._load_backtest_tab_predictions(
        recent_payload,
        tmp_path / "missing_recent.parquet",
        [backtest_dir],
    )

    assert loaded["run_id"].tolist() == ["baseline_v1"]
    assert loaded["model_label"].tolist() == ["model_a"]


def test_streamlit_parquet_loader_accepts_file_uri(tmp_path) -> None:
    parquet_path = tmp_path / "frame.parquet"
    pd.DataFrame([{"x": 1}]).to_parquet(parquet_path, index=False)

    loaded = streamlit_app._load_parquet(f"file://{parquet_path}")

    assert loaded.to_dict(orient="records") == [{"x": 1}]


def test_cloud_mode_disables_legacy_backtest_dirs_by_default(monkeypatch) -> None:
    monkeypatch.setenv("DKENERGY_ARTIFACT_STORE_URI", "s3://bucket/prefix")

    assert streamlit_app._paths_from_env("DKENERGY_BACKTEST_DIRS", []) == []


def test_legacy_backtests_are_opt_in_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DKENERGY_ENABLE_LEGACY_BACKTESTS", raising=False)

    assert not streamlit_app._env_bool("DKENERGY_ENABLE_LEGACY_BACKTESTS", default=False)


def test_backtest_loader_accepts_string_directories(tmp_path) -> None:
    backtest_dir = tmp_path / "baseline_v1"
    backtest_dir.mkdir()
    pd.DataFrame(
        [_prediction_row("ignored", "2026-07-01T00:00:00Z", "DK1", "model_a", 10.0, 11.0)]
    ).drop(columns=["run_id"]).to_parquet(backtest_dir / "predictions.parquet", index=False)

    loaded = streamlit_app._load_backtest_predictions([str(backtest_dir)])

    assert loaded["run_id"].tolist() == ["baseline_v1"]


def _prediction_row(
    run_id: str,
    ds_utc: str,
    area: str,
    model_label: str,
    y: float,
    y_pred: float,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "forecast_origin_utc": "2026-06-30T10:00:00Z",
        "ds_utc": ds_utc,
        "area": area,
        "model_label": model_label,
        "y": y,
        "y_pred": y_pred,
    }
