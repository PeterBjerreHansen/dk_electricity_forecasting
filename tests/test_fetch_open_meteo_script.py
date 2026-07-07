from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import date
from pathlib import Path


def _load_fetch_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_open_meteo_previous_runs.py"
    spec = importlib.util.spec_from_file_location("fetch_open_meteo_previous_runs_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


partition_existing_raw_batch_paths = _load_fetch_script().partition_existing_raw_batch_paths


def test_partition_existing_open_meteo_raw_batches_rejects_hash_mismatch(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    raw_path = (
        raw_root
        / "previous_runs"
        / "gfs_global"
        / "dk1_aarhus"
        / "fetched_at=20260101T000000Z"
        / "start=2025-01-01_end=2025-01-02.json"
    )
    raw_path.parent.mkdir(parents=True)
    content = b'{"hourly":{"time":["2025-01-01T00:00"]}}'
    raw_path.write_bytes(content)
    manifest = {
        "source_provider": "open_meteo",
        "source_product": "previous_runs",
        "raw_path": str(raw_path.relative_to(raw_root)),
        "saved_json_sha256": "0" * 64,
    }
    (raw_root / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    valid, invalid = partition_existing_raw_batch_paths(
        raw_root,
        "gfs_global",
        "dk1_aarhus",
        date(2025, 1, 1),
        date(2025, 1, 2),
    )

    assert valid == []
    assert invalid[0][0] == raw_path
    assert "manifest hash mismatch" in invalid[0][1]


def test_partition_existing_open_meteo_raw_batches_accepts_valid_json_and_hash(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    raw_path = (
        raw_root
        / "previous_runs"
        / "icon_eu"
        / "dk2_copenhagen"
        / "fetched_at=20260101T000000Z"
        / "start=2025-01-01_end=2025-01-02.json"
    )
    raw_path.parent.mkdir(parents=True)
    content = b'{"hourly":{"time":["2025-01-01T00:00"]}}'
    raw_path.write_bytes(content)
    manifest = {
        "source_provider": "open_meteo",
        "source_product": "previous_runs",
        "raw_path": str(raw_path.relative_to(raw_root)),
        "saved_json_sha256": hashlib.sha256(content).hexdigest(),
    }
    (raw_root / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    valid, invalid = partition_existing_raw_batch_paths(
        raw_root,
        "icon_eu",
        "dk2_copenhagen",
        date(2025, 1, 1),
        date(2025, 1, 2),
    )

    assert valid == [raw_path]
    assert invalid == []
