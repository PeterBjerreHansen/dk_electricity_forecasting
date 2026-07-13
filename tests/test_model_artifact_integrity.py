from __future__ import annotations

import hashlib
import json

import pytest

from dkenergy_forecast.models.chronos_production import load_lora_artifact_manifest


def test_chronos_manifest_verifies_declared_model_file_hashes(tmp_path) -> None:
    model_file = tmp_path / "adapter_model.safetensors"
    model_file.write_bytes(b"model-v1")
    expected_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
    adapter_config = tmp_path / "adapter_config.json"
    adapter_config.write_text('{"revision":"base-revision"}', encoding="utf-8")
    manifest = {
        "artifact_schema_version": 3,
        "base_model_revision": "base-revision",
        "artifact_files_sha256": {
            model_file.name: expected_hash,
            adapter_config.name: hashlib.sha256(adapter_config.read_bytes()).hexdigest(),
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert load_lora_artifact_manifest(tmp_path)["artifact_files_sha256"] == {
        model_file.name: expected_hash,
        adapter_config.name: hashlib.sha256(adapter_config.read_bytes()).hexdigest(),
    }

    model_file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_lora_artifact_manifest(tmp_path)


def test_chronos_manifest_rejects_unpinned_adapter_base_revision(tmp_path) -> None:
    model_file = tmp_path / "adapter_model.safetensors"
    model_file.write_bytes(b"weights")
    adapter_config = tmp_path / "adapter_config.json"
    adapter_config.write_text('{"revision":null}', encoding="utf-8")
    manifest = {
        "artifact_schema_version": 3,
        "base_model_revision": "base-revision",
        "artifact_files_sha256": {
            model_file.name: hashlib.sha256(model_file.read_bytes()).hexdigest(),
            adapter_config.name: hashlib.sha256(adapter_config.read_bytes()).hexdigest(),
        },
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="adapter base revision"):
        load_lora_artifact_manifest(tmp_path)
