from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import date
from pathlib import Path


def _load_fetch_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "fetch_eds_prices.py"
    spec = importlib.util.spec_from_file_location("fetch_eds_prices_script", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


partition_existing_raw_batch_paths = _load_fetch_script().partition_existing_raw_batch_paths


def test_partition_existing_raw_batch_paths_rejects_hash_mismatch(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    raw_path = (
        raw_root
        / "DayAheadPrices"
        / "fetched_at=20260101T000000Z"
        / "start=2025-10-01_end=2025-10-02.json"
    )
    raw_path.parent.mkdir(parents=True)
    content = b'{"records":[]}'
    raw_path.write_bytes(content)
    manifest = {
        "source_dataset": "DayAheadPrices",
        "raw_path": str(raw_path.relative_to(raw_root)),
        "saved_json_sha256": "0" * 64,
    }
    (raw_root / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    valid, invalid = partition_existing_raw_batch_paths(
        raw_root,
        "DayAheadPrices",
        date(2025, 10, 1),
        date(2025, 10, 2),
    )

    assert valid == []
    assert invalid[0][0] == raw_path
    assert "manifest hash mismatch" in invalid[0][1]


def test_partition_existing_raw_batch_paths_accepts_valid_json_and_hash(tmp_path) -> None:
    raw_root = tmp_path / "raw"
    raw_path = (
        raw_root
        / "DayAheadPrices"
        / "fetched_at=20260101T000000Z"
        / "start=2025-10-01_end=2025-10-02.json"
    )
    raw_path.parent.mkdir(parents=True)
    content = b'{"records":[]}'
    raw_path.write_bytes(content)
    manifest = {
        "source_dataset": "DayAheadPrices",
        "raw_path": str(raw_path.relative_to(raw_root)),
        "saved_json_sha256": hashlib.sha256(content).hexdigest(),
    }
    (raw_root / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    valid, invalid = partition_existing_raw_batch_paths(
        raw_root,
        "DayAheadPrices",
        date(2025, 10, 1),
        date(2025, 10, 2),
    )

    assert valid == [raw_path]
    assert invalid == []
