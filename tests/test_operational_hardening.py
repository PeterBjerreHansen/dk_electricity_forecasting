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
    update_latest_exports,
    write_forecast_run_artifacts,
)
from dkenergy_forecast.types import add_copenhagen_calendar


def test_forecast_run_publish_is_transactional_and_records_checksums(
    tmp_path: Path,
) -> None:
    predictions = _predictions()
    scores = _scores()
    run_dir = tmp_path / "runs" / "run-1"
    manifest = _manifest(predictions, scores, idempotency_key="delivery-20240103")

    written = write_forecast_run_artifacts(
        run_dir,
        predictions=predictions,
        scores=scores,
        manifest=manifest,
    )

    saved_manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
    assert saved_manifest["artifact_sha256"]["predictions"] == file_sha256(
        written["predictions"]
    )
    assert saved_manifest["artifact_sha256"]["model_scores"] == file_sha256(
        written["model_scores"]
    )
    assert saved_manifest["artifact_identity_sha256"]
    assert not list(run_dir.parent.glob(".run-1.tmp-*"))


def test_failed_forecast_run_publish_leaves_no_visible_or_temporary_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = _predictions()
    scores = _scores()
    run_dir = tmp_path / "runs" / "run-1"

    def fail_after_partial_write(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="simulated parquet failure"):
        write_forecast_run_artifacts(
            run_dir,
            predictions=predictions,
            scores=scores,
            manifest=_manifest(predictions, scores),
        )

    assert not run_dir.exists()
    assert not list(run_dir.parent.glob(".run-1.tmp-*"))


def test_idempotent_concurrent_run_retries_share_one_immutable_result(
    tmp_path: Path,
) -> None:
    predictions = _predictions()
    scores = _scores()
    run_dir = tmp_path / "runs" / "run-1"
    manifest = _manifest(predictions, scores, idempotency_key="delivery-20240103")

    def publish():
        return write_forecast_run_artifacts(
            run_dir,
            predictions=predictions,
            scores=scores,
            manifest=manifest,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: publish(), range(2)))

    assert results[0] == results[1]
    assert pd.read_parquet(run_dir / "predictions.parquet")["y_pred"].tolist() == [9.0]
    assert not list(run_dir.parent.glob(".run-1.tmp-*"))


def test_idempotency_key_cannot_be_reused_for_different_forecasts(tmp_path: Path) -> None:
    predictions = _predictions()
    scores = _scores()
    run_dir = tmp_path / "runs" / "run-1"
    manifest = _manifest(predictions, scores, idempotency_key="delivery-20240103")
    write_forecast_run_artifacts(
        run_dir,
        predictions=predictions,
        scores=scores,
        manifest=manifest,
    )

    with pytest.raises(ValueError, match="reused for different artifacts"):
        write_forecast_run_artifacts(
            run_dir,
            predictions=predictions.assign(y_pred=999.0),
            scores=scores,
            manifest=manifest,
        )


def test_history_filters_lifecycle_and_deduplicates_publications(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    origin = "2024-01-02T10:00:00Z"
    target = "2024-01-03T00:00:00Z"
    _manual_run(
        root,
        "live-first",
        origin=origin,
        target=target,
        y_pred=9.0,
        created_at="2024-01-02T10:05:00Z",
        status="completed",
        run_kind="live",
        score_eligible=True,
    )
    _manual_run(
        root,
        "live-later",
        origin=origin,
        target=target,
        y_pred=999.0,
        created_at="2024-01-02T11:00:00Z",
        status="completed",
        run_kind="live",
        score_eligible=True,
    )
    _manual_run(
        root,
        "replay",
        origin=origin,
        target=target,
        y_pred=1000.0,
        created_at="2024-01-02T09:00:00Z",
        status="completed",
        run_kind="replay",
        score_eligible=True,
    )
    _manual_run(
        root,
        "running",
        origin=origin,
        target=target,
        y_pred=1001.0,
        created_at="2024-01-02T09:00:00Z",
        status="running",
        run_kind="live",
        score_eligible=True,
    )
    _manual_run(
        root,
        "shadow",
        origin="2024-01-03T10:00:00Z",
        target="2024-01-04T00:00:00Z",
        y_pred=19.0,
        created_at="2024-01-03T10:05:00Z",
        status="success",
        run_kind="shadow",
        score_eligible=True,
    )
    _manual_run(
        root,
        "legacy",
        origin="2024-01-04T10:00:00Z",
        target="2024-01-05T00:00:00Z",
        y_pred=29.0,
        created_at="2024-01-04T10:05:00Z",
        status="success",
    )
    partial = root / "partial"
    partial.mkdir()
    _prediction_row(origin=origin, target=target, y_pred=2000.0).to_parquet(
        partial / "predictions.parquet", index=False
    )
    panel = add_copenhagen_calendar(
        pd.DataFrame(
            {
                "unique_id": ["day_ahead_price_DK1"] * 3,
                "ds_utc": pd.to_datetime(
                    [target, "2024-01-04T00:00:00Z", "2024-01-05T00:00:00Z"], utc=True
                ),
                "area": ["DK1"] * 3,
                "y": [10.0, 20.0, 30.0],
            }
        )
    )

    history = build_published_forecast_history(root, panel)

    assert history["run_id"].tolist() == ["live-first", "shadow", "legacy"]
    assert history["y_pred"].tolist() == [9.0, 19.0, 29.0]
    assert history["run_kind"].tolist() == ["live", "shadow", "legacy"]


def test_history_rejects_a_checksum_mismatch(tmp_path: Path) -> None:
    predictions = _predictions()
    scores = _scores()
    run_dir = tmp_path / "runs" / "run-1"
    write_forecast_run_artifacts(
        run_dir,
        predictions=predictions,
        scores=scores,
        manifest=_manifest(predictions, scores),
    )
    predictions.assign(y_pred=999.0).to_parquet(run_dir / "predictions.parquet", index=False)
    panel = predictions[["unique_id", "ds_utc", "area", "y"]].copy()

    with pytest.raises(ValueError, match="checksum mismatch"):
        build_published_forecast_history(tmp_path / "runs", panel)


def test_atomic_parquet_failure_preserves_previous_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "latest.parquet"
    atomic_write_parquet(path, pd.DataFrame({"value": [1]}))

    def fail_after_partial_write(self, temporary_path, *args, **kwargs):
        Path(temporary_path).write_bytes(b"partial")
        raise RuntimeError("simulated parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_after_partial_write)
    with pytest.raises(RuntimeError, match="simulated parquet failure"):
        atomic_write_parquet(path, pd.DataFrame({"value": [2]}))

    assert pd.read_parquet(path)["value"].tolist() == [1]
    assert not list(tmp_path.glob(".latest.parquet.tmp-*"))


def test_latest_manifest_is_the_last_atomic_commit_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import dkenergy_forecast.publishing.artifacts as artifacts

    writes: list[Path] = []
    original = artifacts.atomic_write_json

    def record_write(path, payload):
        writes.append(Path(path))
        original(path, payload)

    monkeypatch.setattr(artifacts, "atomic_write_json", record_write)
    predictions = _predictions()
    scores = _scores()
    paths = update_latest_exports(
        latest_forecast_dir=tmp_path / "latest",
        recent_scores_dir=tmp_path / "scores",
        dashboard_path=tmp_path / "dashboard.json",
        predictions=predictions,
        scores=scores,
        manifest=_manifest(predictions, scores),
        dashboard={"status": "ok"},
    )

    assert writes[-1] == paths["latest_manifest"]
    saved_manifest = json.loads(paths["latest_manifest"].read_text(encoding="utf-8"))
    assert saved_manifest["latest_artifact_sha256"]["latest_predictions"] == file_sha256(
        paths["latest_predictions"]
    )


def test_latest_promotion_cannot_regress_to_an_older_origin(tmp_path: Path) -> None:
    predictions = _predictions()
    scores = _scores()
    current_manifest = _manifest(predictions, scores)
    update_latest_exports(
        latest_forecast_dir=tmp_path / "latest",
        recent_scores_dir=tmp_path / "scores",
        dashboard_path=tmp_path / "dashboard.json",
        predictions=predictions,
        scores=scores,
        manifest=current_manifest,
        dashboard={"version": "current"},
    )
    stale_manifest = dict(current_manifest)
    stale_manifest["forecast_origin_utc"] = "2024-01-01T10:00:00Z"

    with pytest.raises(ValueError, match="older origin"):
        update_latest_exports(
            latest_forecast_dir=tmp_path / "latest",
            recent_scores_dir=tmp_path / "scores",
            dashboard_path=tmp_path / "dashboard.json",
            predictions=predictions.assign(y_pred=999.0),
            scores=scores,
            manifest=stale_manifest,
            dashboard={"version": "stale"},
        )

    latest = pd.read_parquet(tmp_path / "latest" / "predictions.parquet")
    assert latest["y_pred"].tolist() == [9.0]
    assert json.loads((tmp_path / "dashboard.json").read_text(encoding="utf-8")) == {
        "version": "current"
    }


def _predictions() -> pd.DataFrame:
    return _prediction_row(
        origin="2024-01-02T10:00:00Z",
        target="2024-01-03T00:00:00Z",
        y_pred=9.0,
    ).assign(y=10.0)


def _prediction_row(*, origin: str, target: str, y_pred: float) -> pd.DataFrame:
    return add_copenhagen_calendar(
        pd.DataFrame(
            {
                "unique_id": ["day_ahead_price_DK1"],
                "forecast_origin_utc": [pd.Timestamp(origin)],
                "ds_utc": [pd.Timestamp(target)],
                "area": ["DK1"],
                "model_label": ["baseline"],
                "y_pred": [y_pred],
            }
        )
    )


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model_label": ["baseline"],
            "area": ["DK1"],
            "mae": [1.0],
            "rmse": [1.0],
            "bias": [-1.0],
            "coverage": [float("nan")],
            "interval_width": [float("nan")],
        }
    )


def _manifest(
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    idempotency_key: str | None = None,
) -> dict[str, object]:
    return make_forecast_run_manifest(
        run_id="run-1",
        forecast_origin_utc="2024-01-02T10:00:00Z",
        predictions=predictions,
        scores=scores,
        artifact_paths={},
        dataset_version="test",
        git_commit_value="abc123",
        created_at_utc="2024-01-02T10:05:00Z",
        run_kind="live",
        score_eligible=True,
        idempotency_key=idempotency_key,
    )


def _manual_run(
    root: Path,
    run_id: str,
    *,
    origin: str,
    target: str,
    y_pred: float,
    created_at: str,
    status: str,
    run_kind: str | None = None,
    score_eligible: bool | None = None,
) -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    _prediction_row(origin=origin, target=target, y_pred=y_pred).to_parquet(
        run_dir / "predictions.parquet", index=False
    )
    manifest: dict[str, object] = {
        "run_id": run_id,
        "created_at_utc": created_at,
        "forecast_origin_utc": origin,
        "status": status,
    }
    if run_kind is not None:
        manifest["run_kind"] = run_kind
    if score_eligible is not None:
        manifest["score_eligible"] = score_eligible
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
