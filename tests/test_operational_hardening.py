from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from dkenergy_forecast.publishing import (
    atomic_write_parquet,
    build_published_forecast_history,
    file_sha256,
    make_forecast_run_manifest,
    write_forecast_run_artifacts,
    write_latest_pointer,
)
from dkenergy_forecast.types import add_copenhagen_calendar


def test_forecast_run_is_transactional_and_has_a_completion_receipt(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    written = _write_run(run_dir)

    manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
    completion = json.loads(written["completion"].read_text(encoding="utf-8"))
    assert manifest["artifact_sha256"]["predictions"] == file_sha256(
        written["predictions"]
    )
    assert completion["status"] == "completed"
    assert completion["run_id"] == manifest["run_id"]
    assert completion["manifest_sha256"] == file_sha256(written["manifest"])
    assert not list(run_dir.parent.glob(".run-1.tmp-*"))


def test_failed_write_leaves_no_visible_run(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "runs" / "run-1"

    def fail_after_partial_write(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated parquet failure"):
        _write_run(run_dir)

    assert not run_dir.exists()
    assert not list(run_dir.parent.glob(".run-1.tmp-*"))


def test_identical_concurrent_retries_share_one_run(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: _write_run(run_dir), range(2)))

    assert results[0] == results[1]
    assert (run_dir / "COMPLETED.json").exists()
    assert pd.read_parquet(run_dir / "predictions.parquet")["y_pred"].tolist() == [9.0]


def test_idempotency_key_rejects_different_forecasts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    _write_run(run_dir)

    with pytest.raises(ValueError, match="reused for different artifacts"):
        _write_run(run_dir, y_pred=999.0)


def test_history_uses_only_timely_completed_live_runs(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    target = "2024-01-03T00:00:00Z"
    _manual_run(root, "live", target=target, y_pred=9.0, run_kind="live")
    _manual_run(
        root,
        "late",
        target="2024-01-04T00:00:00Z",
        y_pred=19.0,
        run_kind="live",
        committed_at="2024-01-03T12:01:00Z",
        deadline="2024-01-03T12:00:00Z",
    )
    _manual_run(
        root,
        "replay",
        target="2024-01-05T00:00:00Z",
        y_pred=29.0,
        run_kind="replay",
    )
    _manual_run(
        root,
        "no-receipt",
        target="2024-01-06T00:00:00Z",
        y_pred=39.0,
        run_kind="live",
        write_completion=False,
    )
    panel = add_copenhagen_calendar(
        pd.DataFrame(
            {
                "unique_id": ["day_ahead_price_DK1"] * 4,
                "ds_utc": pd.to_datetime(
                    [target, "2024-01-04T00:00:00Z", "2024-01-05T00:00:00Z", "2024-01-06T00:00:00Z"],
                    utc=True,
                ),
                "area": ["DK1"] * 4,
                "y": [10.0, 20.0, 30.0, 40.0],
            }
        )
    )

    history = build_published_forecast_history(root, panel)

    assert history["run_id"].tolist() == ["live"]
    assert history["y_pred"].tolist() == [9.0]


def test_history_rejects_checksum_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "run-1"
    _write_run(run_dir)
    _predictions(y_pred=999.0).to_parquet(run_dir / "predictions.parquet", index=False)

    with pytest.raises(ValueError, match="checksum mismatch"):
        build_published_forecast_history(
            tmp_path / "runs",
            _predictions()[["unique_id", "ds_utc", "area", "y"]],
        )


def test_atomic_parquet_failure_preserves_previous_file(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "latest.parquet"
    atomic_write_parquet(path, pd.DataFrame({"value": [1]}))

    def fail_after_partial_write(self, temporary_path, *args, **kwargs):
        Path(temporary_path).write_bytes(b"partial")
        raise RuntimeError("simulated parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated parquet failure"):
        atomic_write_parquet(path, pd.DataFrame({"value": [2]}))

    assert pd.read_parquet(path)["value"].tolist() == [1]


def test_latest_pointer_references_a_completed_run(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    run_dir = artifacts / "forecast_runs" / "run-1"
    _write_run(run_dir)

    pointer_path = write_latest_pointer(artifacts / "latest.json", run_dir=run_dir)
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))

    assert pointer["status"] == "completed"
    assert pointer["run_prefix"] == "forecast_runs/run-1"
    assert pointer["completion_key"] == "forecast_runs/run-1/COMPLETED.json"


def test_latest_pointer_cannot_regress(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    current = artifacts / "forecast_runs" / "current"
    stale = artifacts / "forecast_runs" / "stale"
    _write_run(current, run_id="current", information_cutoff="2024-01-02T10:00:00Z")
    _write_run(stale, run_id="stale", information_cutoff="2024-01-02T09:00:00Z")
    pointer = artifacts / "latest.json"
    write_latest_pointer(pointer, run_dir=current)

    with pytest.raises(ValueError, match="older information cutoff"):
        write_latest_pointer(pointer, run_dir=stale)

    assert json.loads(pointer.read_text(encoding="utf-8"))["run_id"] == "current"


def _write_run(
    run_dir: Path,
    *,
    run_id: str = "run-1",
    y_pred: float = 9.0,
    information_cutoff: str = "2024-01-02T10:00:00Z",
) -> dict[str, Path]:
    predictions = _predictions(y_pred=y_pred, origin=information_cutoff)
    scores = _scores()
    manifest = make_forecast_run_manifest(
        run_id=run_id,
        forecast_origin_utc=information_cutoff,
        predictions=predictions,
        scores=scores,
        artifact_paths={},
        dataset_version="test",
        git_commit_value="abc123",
        created_at_utc=information_cutoff,
        run_kind="live",
        idempotency_key=f"delivery-20240103:{run_id}",
        extra={
            "delivery_date_local": "2024-01-03",
            "information_cutoff_utc": information_cutoff,
            "decision_deadline_utc": "2999-01-02T12:00:00Z",
            "forecast_status": "primary",
        },
    )
    return write_forecast_run_artifacts(
        run_dir,
        predictions=predictions,
        scores=scores,
        manifest=manifest,
    )


def _predictions(
    *,
    y_pred: float = 9.0,
    origin: str = "2024-01-02T10:00:00Z",
) -> pd.DataFrame:
    return add_copenhagen_calendar(
        pd.DataFrame(
            {
                "unique_id": ["day_ahead_price_DK1"],
                "forecast_origin_utc": [pd.Timestamp(origin)],
                "ds_utc": [pd.Timestamp("2024-01-03T00:00:00Z")],
                "area": ["DK1"],
                "model_label": ["chronos_weather"],
                "model_release_id": ["release-1"],
                "y_pred": [y_pred],
                "y": [10.0],
            }
        )
    )


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["model_label", "area", "mae", "rmse", "bias", "coverage", "interval_width"]
    )


def _manual_run(
    root: Path,
    run_id: str,
    *,
    target: str,
    y_pred: float,
    run_kind: str,
    committed_at: str = "2024-01-02T10:05:00Z",
    deadline: str = "2024-01-02T12:00:00Z",
    write_completion: bool = True,
) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    frame = _predictions(y_pred=y_pred).assign(ds_utc=pd.Timestamp(target))
    frame.to_parquet(run_dir / "predictions.parquet", index=False)
    manifest = {
        "run_id": run_id,
        "forecast_origin_utc": "2024-01-02T10:00:00Z",
        "run_kind": run_kind,
        "decision_deadline_utc": deadline,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    if write_completion:
        (run_dir / "COMPLETED.json").write_text(
            json.dumps(
                {
                    "status": "completed",
                    "run_id": run_id,
                    "committed_at_utc": committed_at,
                }
            ),
            encoding="utf-8",
        )
