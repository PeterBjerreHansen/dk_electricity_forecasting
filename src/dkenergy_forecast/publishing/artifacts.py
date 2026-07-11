from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dkenergy_forecast.types import (
    PRICE_AVAILABILITY_COLUMN,
    ensure_price_availability,
    normalize_utc_column,
    require_columns,
    to_utc_timestamp,
)


PUBLISHED_PREDICTION_REQUIRED_COLUMNS = [
    "forecast_origin_utc",
    "ds_utc",
    "ds_local",
    "area",
    "model_label",
    "y_pred",
]

EVALUATED_PREDICTION_REQUIRED_COLUMNS = [
    *PUBLISHED_PREDICTION_REQUIRED_COLUMNS,
    "y",
]

MODEL_SCORE_REQUIRED_COLUMNS = [
    "model_label",
    "area",
    "mae",
    "rmse",
    "bias",
    "coverage",
    "interval_width",
]

MODEL_SCORE_COLUMNS = [
    "model_label",
    "area",
    "rows",
    "evaluated_rows",
    "mae",
    "rmse",
    "bias",
    "coverage",
    "interval_width",
    "pinball_q10",
    "pinball_q50",
    "pinball_q90",
    "interval_score_80",
    "weighted_interval_score",
    "calibration_error",
    "missing_rate",
]

COMPLETED_RUN_STATUSES = frozenset({"success", "complete", "completed"})
SCOREABLE_RUN_KINDS = frozenset({"live", "shadow"})


def normalize_published_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    """Normalize prediction rows for durable artifact and API use."""

    output = predictions.copy()
    if "model_label" not in output.columns and "model_name" in output.columns:
        if "model_version" in output.columns:
            output["model_label"] = output["model_name"] + ":" + output["model_version"].astype(str)
        else:
            output["model_label"] = output["model_name"]
    if "actual_price" not in output.columns and "y" in output.columns:
        output["actual_price"] = output["y"]

    require_columns(output, PUBLISHED_PREDICTION_REQUIRED_COLUMNS, "published predictions")
    output = normalize_utc_column(output, "forecast_origin_utc")
    output = normalize_utc_column(output, "ds_utc")
    if "created_at_utc" in output.columns:
        output = normalize_utc_column(output, "created_at_utc")
    return output.reset_index(drop=True)


def validate_prediction_artifact_schema(predictions: pd.DataFrame) -> None:
    frame = normalize_published_predictions(predictions)
    key_cols = [
        column
        for column in ["unique_id", "forecast_origin_utc", "ds_utc", "area", "model_label"]
        if column in frame.columns
    ]
    duplicate_count = int(frame.duplicated(key_cols).sum())
    if duplicate_count:
        raise ValueError(
            "Published predictions contain duplicate key rows: "
            f"{duplicate_count} duplicate(s) over {key_cols}"
        )
    if frame["model_label"].isna().any():
        raise ValueError("Published predictions contain missing model_label values")
    if frame["area"].isna().any():
        raise ValueError("Published predictions contain missing area values")
    if frame["y_pred"].isna().any():
        raise ValueError("Published predictions contain missing point forecast values")

    quantile_columns = ["q10", "q50", "q90"]
    present_quantiles = [column for column in quantile_columns if column in frame.columns]
    if present_quantiles:
        missing_quantile_columns = sorted(set(quantile_columns) - set(present_quantiles))
        if missing_quantile_columns:
            raise ValueError(
                "Published predictions have an incomplete quantile schema; "
                f"missing columns: {missing_quantile_columns}"
            )
        has_any_quantile = frame[quantile_columns].notna().any(axis=1)
        has_all_quantiles = frame[quantile_columns].notna().all(axis=1)
        partial_count = int((has_any_quantile & ~has_all_quantiles).sum())
        if partial_count:
            raise ValueError(
                "Published predictions contain partially populated q10/q50/q90 rows: "
                f"{partial_count}"
            )
        crossed_count = int(
            (
                has_all_quantiles
                & ((frame["q10"] > frame["q50"]) | (frame["q50"] > frame["q90"]))
            ).sum()
        )
        if crossed_count:
            raise ValueError(f"Published predictions contain crossed quantiles: {crossed_count}")


def validate_evaluated_prediction_artifact_schema(predictions: pd.DataFrame) -> None:
    frame = normalize_published_predictions(predictions)
    require_columns(frame, EVALUATED_PREDICTION_REQUIRED_COLUMNS, "evaluated predictions")
    validate_prediction_artifact_schema(frame)
    if frame["y"].isna().any():
        raise ValueError("Evaluated predictions contain missing actual target values")
    if frame["y_pred"].isna().any():
        raise ValueError("Evaluated predictions contain missing point forecast values")


def validate_model_scores_schema(scores: pd.DataFrame) -> None:
    require_columns(scores, MODEL_SCORE_REQUIRED_COLUMNS, "model scores")
    duplicate_count = int(scores.duplicated(["model_label", "area"]).sum())
    if duplicate_count:
        raise ValueError(
            "Model scores contain duplicate (model_label, area) rows: "
            f"{duplicate_count}"
        )
    for column in ["mae", "rmse", "bias", "coverage", "interval_width"]:
        if column in scores.columns:
            pd.to_numeric(scores[column], errors="raise")


def build_published_forecast_history(
    artifact_root: str | Path,
    panel: pd.DataFrame,
) -> pd.DataFrame:
    """Score eligible immutable forecast runs that now have actuals.

    Modern manifests must explicitly describe a completed ``live`` or ``shadow``
    run with ``score_eligible=true``. Manifests written before those lifecycle
    fields existed remain scoreable when their status is absent or completed.
    A directory without a manifest is never considered complete and is ignored.
    """

    published = _load_forecast_run_predictions(artifact_root)
    if published.empty:
        return _empty_published_history_frame()

    actuals = _panel_actuals(panel)
    merged = published.merge(actuals, on=["unique_id", "ds_utc"], how="left")
    merged["y"] = merged["_actual_y"]
    merged["actual_price"] = merged["_actual_y"]
    if PRICE_AVAILABILITY_COLUMN in merged:
        merged[PRICE_AVAILABILITY_COLUMN] = merged[f"{PRICE_AVAILABILITY_COLUMN}_panel"].combine_first(
            merged[PRICE_AVAILABILITY_COLUMN]
        )
    elif f"{PRICE_AVAILABILITY_COLUMN}_panel" in merged:
        merged[PRICE_AVAILABILITY_COLUMN] = merged[f"{PRICE_AVAILABILITY_COLUMN}_panel"]
    merged = merged.drop(
        columns=[column for column in ["_actual_y", f"{PRICE_AVAILABILITY_COLUMN}_panel"] if column in merged],
    )
    evaluated = merged.loc[merged["y"].notna() & merged["y_pred"].notna()].copy()
    if evaluated.empty:
        return _empty_published_history_frame()
    return evaluated.sort_values(
        ["forecast_origin_utc", "run_id", "model_label", "unique_id", "ds_utc"]
    ).reset_index(drop=True)


def build_published_forecast_scores(history: pd.DataFrame) -> pd.DataFrame:
    if history.empty:
        return _empty_model_score_frame(score_source="published_forecast_history")

    from dkenergy_forecast.evaluation.summary import add_prediction_diagnostics, model_score_table

    scores = model_score_table(add_prediction_diagnostics(history))
    scores["score_source"] = "published_forecast_history"
    return scores


def write_published_forecast_history(
    history_dir: str | Path,
    *,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
) -> dict[str, Path]:
    history_path = Path(history_dir)
    history_path.mkdir(parents=True, exist_ok=True)
    paths = {
        "published_history_predictions": history_path / "predictions.parquet",
        "published_history_scores": history_path / "model_scores.parquet",
    }
    atomic_write_parquet(paths["published_history_predictions"], predictions)
    atomic_write_parquet(paths["published_history_scores"], scores)
    return paths


def _load_forecast_run_predictions(artifact_root: str | Path) -> pd.DataFrame:
    root = Path(artifact_root)
    if not root.exists():
        return _empty_published_prediction_frame()

    frames: list[pd.DataFrame] = []
    seen_run_ids: dict[str, Path] = {}
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest_path = run_dir / "manifest.json"
        manifest = _read_optional_manifest(manifest_path)
        if not manifest or not _manifest_is_score_eligible(manifest):
            continue
        predictions_path = run_dir / "predictions.parquet"
        if not predictions_path.exists():
            continue
        _verify_manifest_artifact_checksum(manifest, "predictions", predictions_path)
        frame = normalize_published_predictions(pd.read_parquet(predictions_path))
        require_columns(frame, ["unique_id"], f"{predictions_path}")
        validate_prediction_artifact_schema(frame)
        run_id = str(manifest.get("run_id") or run_dir.name)
        if run_id in seen_run_ids:
            raise ValueError(
                f"Published run_id {run_id!r} is used by both "
                f"{seen_run_ids[run_id]} and {run_dir}"
            )
        seen_run_ids[run_id] = run_dir
        if "run_id" in frame.columns:
            conflicting_run_ids = frame["run_id"].dropna().astype(str).ne(run_id)
            if conflicting_run_ids.any():
                raise ValueError(
                    f"Published prediction run_id does not match its manifest: {predictions_path}"
                )
        manifest_origin = manifest.get("forecast_origin_utc")
        if manifest_origin is not None:
            expected_origin = to_utc_timestamp(manifest_origin)
            if frame["forecast_origin_utc"].ne(expected_origin).any():
                raise ValueError(
                    "Published prediction forecast_origin_utc does not match its manifest: "
                    f"{predictions_path}"
                )
        frame["run_id"] = run_id
        frame["published_at_utc"] = manifest.get(
            "published_at_utc", manifest.get("created_at_utc")
        )
        frame["published_forecast_origin_utc"] = manifest_origin
        frame["run_kind"] = manifest.get("run_kind", "legacy")
        frame["score_eligible"] = manifest.get("score_eligible", True)
        frames.append(frame)

    if not frames:
        return _empty_published_prediction_frame()
    published = pd.concat(frames, ignore_index=True)
    published["published_at_utc"] = pd.to_datetime(
        published["published_at_utc"], utc=True, errors="coerce"
    )
    publication_key = [
        "forecast_origin_utc",
        "unique_id",
        "ds_utc",
        "area",
        "model_label",
    ]
    # A retry or corrected publication must not give one forecast target extra
    # weight. Keep the earliest durable publication; later rows could have seen
    # information unavailable to the original forecast.
    return (
        published.sort_values(
            [*publication_key, "published_at_utc", "run_id"],
            na_position="last",
        )
        .drop_duplicates(publication_key, keep="first")
        .reset_index(drop=True)
    )


def _read_optional_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_is_score_eligible(manifest: dict[str, Any]) -> bool:
    status = str(manifest.get("status", "success")).lower()
    if status not in COMPLETED_RUN_STATUSES:
        return False

    has_explicit_lifecycle = "run_kind" in manifest or "score_eligible" in manifest
    if not has_explicit_lifecycle:
        # Backward-compatible rule for runs published before lifecycle metadata
        # existed. The manifest itself remains the completion marker.
        return True
    return (
        manifest.get("run_kind") in SCOREABLE_RUN_KINDS
        and manifest.get("score_eligible") is True
    )


def _verify_manifest_artifact_checksum(
    manifest: dict[str, Any],
    artifact_name: str,
    path: Path,
) -> None:
    artifact_hashes = manifest.get("artifact_sha256", {})
    if not isinstance(artifact_hashes, dict):
        raise ValueError("Forecast run manifest artifact_sha256 must be an object")
    expected = artifact_hashes.get(artifact_name)
    if expected is None:
        return
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"Published artifact checksum mismatch for {path}: expected {expected}, got {actual}"
        )


def _panel_actuals(panel: pd.DataFrame) -> pd.DataFrame:
    require_columns(panel, ["unique_id", "ds_utc", "y"], "price panel")
    actuals = ensure_price_availability(normalize_utc_column(panel, "ds_utc"))
    columns = ["unique_id", "ds_utc", "y"]
    if PRICE_AVAILABILITY_COLUMN in actuals:
        columns.append(PRICE_AVAILABILITY_COLUMN)
    output = actuals[columns].drop_duplicates(["unique_id", "ds_utc"]).copy()
    rename = {"y": "_actual_y"}
    if PRICE_AVAILABILITY_COLUMN in output:
        rename[PRICE_AVAILABILITY_COLUMN] = f"{PRICE_AVAILABILITY_COLUMN}_panel"
    return output.rename(columns=rename)


def _empty_published_prediction_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=[*PUBLISHED_PREDICTION_REQUIRED_COLUMNS, "unique_id", "run_id"])


def _empty_published_history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *PUBLISHED_PREDICTION_REQUIRED_COLUMNS,
            "unique_id",
            "run_id",
            "published_at_utc",
            "published_forecast_origin_utc",
            "y",
            "actual_price",
        ]
    )


def _empty_model_score_frame(*, score_source: str | None = None) -> pd.DataFrame:
    frame = pd.DataFrame(columns=[*MODEL_SCORE_COLUMNS, "score_source"])
    if score_source is not None:
        frame["score_source"] = frame["score_source"].astype("object")
    return frame


def make_forecast_run_manifest(
    *,
    run_id: str,
    forecast_origin_utc: object,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    artifact_paths: dict[str, str],
    dataset_version: list[str] | str | None,
    git_commit_value: str | None,
    status: str = "success",
    created_at_utc: object | None = None,
    extra: dict[str, Any] | None = None,
    run_kind: str | None = None,
    score_eligible: bool | None = None,
    idempotency_key: str | None = None,
) -> dict[str, Any]:
    forecast_origin = to_utc_timestamp(forecast_origin_utc)
    created_at = to_utc_timestamp(created_at_utc or datetime.now(timezone.utc))
    model_labels = sorted(predictions["model_label"].dropna().astype(str).unique().tolist())
    manifest = {
        "run_id": run_id,
        "status": status,
        "created_at_utc": created_at,
        "forecast_origin_utc": forecast_origin,
        "model_labels": model_labels,
        "dataset_version": dataset_version,
        "prediction_row_count": int(len(predictions)),
        "model_score_row_count": int(len(scores)),
        "artifact_paths": artifact_paths,
        "git_commit": git_commit_value,
    }
    if extra:
        manifest.update(extra)
    if run_kind is not None:
        manifest["run_kind"] = run_kind
    if score_eligible is not None:
        manifest["score_eligible"] = score_eligible
    if idempotency_key is not None:
        manifest["idempotency_key"] = idempotency_key
    return manifest


def write_forecast_run_artifacts(
    run_dir: str | Path,
    *,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    manifest: dict[str, Any],
    score_predictions: pd.DataFrame | None = None,
    dashboard: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Transactionally publish one immutable run directory.

    Files are prepared in a hidden sibling directory. The completed directory
    becomes visible with one atomic rename, so readers never observe a partial
    run. When a manifest supplies ``idempotency_key``, an exact retry returns
    the existing paths; reusing the key for different core artifacts fails.
    """

    run_path = Path(run_dir)
    predictions_out = normalize_published_predictions(predictions)
    validate_prediction_artifact_schema(predictions_out)
    validate_model_scores_schema(scores)
    score_predictions_out = None
    if score_predictions is not None:
        score_predictions_out = normalize_published_predictions(score_predictions)
        validate_evaluated_prediction_artifact_schema(score_predictions_out)

    idempotency_key = manifest.get("idempotency_key")
    if idempotency_key is not None and (
        not isinstance(idempotency_key, str) or not idempotency_key.strip()
    ):
        raise ValueError("Forecast run idempotency_key must be a non-empty string")
    if run_path.exists() and idempotency_key is None:
        raise FileExistsError(f"Immutable forecast run already exists: {run_path}")

    run_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = Path(
        tempfile.mkdtemp(prefix=f".{run_path.name}.tmp-", dir=run_path.parent)
    )
    temp_paths = _forecast_run_paths(
        temp_path,
        include_score_predictions=score_predictions_out is not None,
        include_dashboard=dashboard is not None,
    )
    try:
        predictions_out.to_parquet(temp_paths["predictions"], index=False)
        scores.to_parquet(temp_paths["model_scores"], index=False)
        if score_predictions_out is not None:
            score_predictions_out.to_parquet(temp_paths["score_predictions"], index=False)
        if dashboard is not None:
            _write_json_direct(temp_paths["dashboard"], dashboard)

        artifact_sha256 = {
            name: file_sha256(path)
            for name, path in temp_paths.items()
            if name != "manifest"
        }
        manifest_out = dict(manifest)
        manifest_out["artifact_sha256"] = artifact_sha256
        manifest_out["artifact_identity_sha256"] = _artifact_identity_sha256(
            artifact_sha256
        )
        _write_json_direct(temp_paths["manifest"], manifest_out)

        if run_path.exists():
            return _resolve_existing_idempotent_run(
                run_path,
                candidate_manifest=manifest_out,
                expected_paths=temp_paths,
            )
        try:
            os.rename(temp_path, run_path)
        except OSError:
            if run_path.exists():
                return _resolve_existing_idempotent_run(
                    run_path,
                    candidate_manifest=manifest_out,
                    expected_paths=temp_paths,
                )
            raise
        return _forecast_run_paths(
            run_path,
            include_score_predictions=score_predictions_out is not None,
            include_dashboard=dashboard is not None,
        )
    finally:
        if temp_path.exists():
            shutil.rmtree(temp_path)


def update_latest_exports(
    *,
    latest_forecast_dir: str | Path,
    recent_scores_dir: str | Path,
    dashboard_path: str | Path,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    manifest: dict[str, Any],
    dashboard: dict[str, Any],
    score_predictions: pd.DataFrame | None = None,
    write_recent_scores: bool = True,
) -> dict[str, Path]:
    predictions_out = normalize_published_predictions(predictions)
    validate_prediction_artifact_schema(predictions_out)
    validate_model_scores_schema(scores)
    score_predictions_out = None
    if score_predictions is not None:
        score_predictions_out = normalize_published_predictions(score_predictions)
        validate_evaluated_prediction_artifact_schema(score_predictions_out)

    latest_dir = Path(latest_forecast_dir)
    scores_dir = Path(recent_scores_dir)
    dashboard_out = Path(dashboard_path)
    latest_dir.mkdir(parents=True, exist_ok=True)
    if write_recent_scores:
        scores_dir.mkdir(parents=True, exist_ok=True)
    dashboard_out.parent.mkdir(parents=True, exist_ok=True)

    paths = {
        "latest_predictions": latest_dir / "predictions.parquet",
        "latest_manifest": latest_dir / "manifest.json",
        "dashboard": dashboard_out,
    }
    if write_recent_scores:
        paths["recent_scores"] = scores_dir / "model_scores.parquet"
    if write_recent_scores and score_predictions_out is not None:
        paths["recent_predictions"] = scores_dir / "predictions.parquet"

    # The lock serializes this multi-file promotion. Each individual replace is
    # atomic, and the manifest is replaced last as the reader-visible commit
    # marker for the new set.
    lock_path = latest_dir.parent / f".{latest_dir.name}.update.lock"
    with _exclusive_file_lock(lock_path):
        _reject_stale_latest_promotion(paths["latest_manifest"], manifest)
        atomic_write_parquet(paths["latest_predictions"], predictions_out)
        if write_recent_scores:
            atomic_write_parquet(paths["recent_scores"], scores)
        if write_recent_scores and score_predictions_out is not None:
            atomic_write_parquet(paths["recent_predictions"], score_predictions_out)
        atomic_write_json(paths["dashboard"], dashboard)
        latest_hashes = {
            name: file_sha256(path)
            for name, path in paths.items()
            if name != "latest_manifest"
        }
        latest_manifest = dict(manifest)
        latest_manifest["latest_artifact_sha256"] = latest_hashes
        atomic_write_json(paths["latest_manifest"], latest_manifest)
    return paths


def build_dashboard_payload(
    *,
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    manifest: dict[str, Any],
    score_predictions: pd.DataFrame | None = None,
    published_history_predictions: pd.DataFrame | None = None,
    published_history_scores: pd.DataFrame | None = None,
) -> dict[str, Any]:
    prediction_rows = normalize_published_predictions(predictions).to_dict(orient="records")
    score_rows = scores.to_dict(orient="records")
    payload = {
        "generated_at_utc": datetime.now(timezone.utc),
        "run": manifest,
        "predictions": prediction_rows,
        "model_scores": score_rows,
    }
    if score_predictions is not None:
        payload["recent_predictions"] = normalize_published_predictions(score_predictions).to_dict(
            orient="records"
        )
    if published_history_predictions is not None:
        payload["published_history_predictions"] = published_history_predictions.to_dict(orient="records")
    if published_history_scores is not None:
        payload["published_history_scores"] = published_history_scores.to_dict(orient="records")
    return payload


def unique_run_id(prefix: str, *, created_at_utc: object | None = None) -> str:
    created_at = to_utc_timestamp(created_at_utc or datetime.now(timezone.utc))
    stamp = created_at.strftime("%Y%m%dT%H%M%SZ")
    return f"{prefix}_{stamp}"


def git_commit(cwd: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(cwd),
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_json(path: str | Path, payload: Any) -> None:
    """Backward-compatible alias for an atomic JSON replacement."""

    atomic_write_json(path, payload)


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Write JSON to a sibling temporary file and atomically replace ``path``."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _new_atomic_temp_path(out)
    try:
        _write_json_direct(temp_path, payload)
        os.replace(temp_path, out)
    finally:
        temp_path.unlink(missing_ok=True)


def atomic_write_parquet(path: str | Path, frame: pd.DataFrame) -> None:
    """Write Parquet to a sibling temporary file and atomically replace ``path``."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp_path = _new_atomic_temp_path(out)
    try:
        frame.to_parquet(temp_path, index=False)
        os.replace(temp_path, out)
    finally:
        temp_path.unlink(missing_ok=True)


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_direct(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_safe(payload), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _new_atomic_temp_path(destination: Path) -> Path:
    descriptor, value = tempfile.mkstemp(
        prefix=f".{destination.name}.tmp-",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(value)


@contextmanager
def _exclusive_file_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another artifact promotion is already in progress: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _reject_stale_latest_promotion(
    current_manifest_path: Path,
    candidate_manifest: dict[str, Any],
) -> None:
    if not current_manifest_path.exists():
        return
    current_manifest = _read_optional_manifest(current_manifest_path)
    current_origin = current_manifest.get("forecast_origin_utc")
    candidate_origin = candidate_manifest.get("forecast_origin_utc")
    if current_origin is None or candidate_origin is None:
        return
    if to_utc_timestamp(candidate_origin) < to_utc_timestamp(current_origin):
        raise ValueError(
            "Refusing to replace latest forecast with an older origin: "
            f"candidate={to_utc_timestamp(candidate_origin).isoformat()}, "
            f"current={to_utc_timestamp(current_origin).isoformat()}"
        )


def _forecast_run_paths(
    run_path: Path,
    *,
    include_score_predictions: bool,
    include_dashboard: bool,
) -> dict[str, Path]:
    paths = {
        "predictions": run_path / "predictions.parquet",
        "model_scores": run_path / "model_scores.parquet",
        "manifest": run_path / "manifest.json",
    }
    if include_score_predictions:
        paths["score_predictions"] = run_path / "score_predictions.parquet"
    if include_dashboard:
        paths["dashboard"] = run_path / "forecast_dashboard.json"
    return paths


def _artifact_identity_sha256(artifact_hashes: dict[str, str]) -> str:
    identity_hashes = {
        name: digest
        for name, digest in artifact_hashes.items()
        if name in {"predictions", "model_scores", "score_predictions"}
    }
    canonical = json.dumps(identity_hashes, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _resolve_existing_idempotent_run(
    run_path: Path,
    *,
    candidate_manifest: dict[str, Any],
    expected_paths: dict[str, Path],
) -> dict[str, Path]:
    idempotency_key = candidate_manifest.get("idempotency_key")
    if idempotency_key is None:
        raise FileExistsError(f"Immutable forecast run already exists: {run_path}")

    existing_manifest = _read_optional_manifest(run_path / "manifest.json")
    if existing_manifest.get("idempotency_key") != idempotency_key:
        raise FileExistsError(
            f"Immutable forecast run already exists with a different idempotency key: {run_path}"
        )
    if (
        existing_manifest.get("artifact_identity_sha256")
        != candidate_manifest.get("artifact_identity_sha256")
    ):
        raise ValueError(
            f"Forecast run idempotency key was reused for different artifacts: {idempotency_key}"
        )

    final_paths = _forecast_run_paths(
        run_path,
        include_score_predictions="score_predictions" in expected_paths,
        include_dashboard="dashboard" in expected_paths,
    )
    missing = [str(path) for path in final_paths.values() if not path.exists()]
    if missing:
        raise ValueError(
            "Existing idempotent forecast run is incomplete; missing: " + ", ".join(missing)
        )
    return final_paths


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def replace_directory(source: str | Path, destination: str | Path) -> None:
    destination_path = Path(destination)
    if destination_path.exists():
        shutil.rmtree(destination_path)
    shutil.copytree(source, destination_path)
