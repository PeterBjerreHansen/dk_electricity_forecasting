from __future__ import annotations

import hashlib
import json

import pytest

from dkenergy_forecast.models.chronos_production import load_lora_artifact_manifest


def test_chronos_manifest_verifies_declared_model_file_hashes(tmp_path) -> None:
    model_file = tmp_path / "adapter_model.safetensors"
    model_file.write_bytes(b"model-v1")
    expected_hash = hashlib.sha256(model_file.read_bytes()).hexdigest()
    manifest = {
        "artifact_schema_version": 3,
        "artifact_files_sha256": {model_file.name: expected_hash},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    assert load_lora_artifact_manifest(tmp_path)["artifact_files_sha256"] == {
        model_file.name: expected_hash
    }

    model_file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_lora_artifact_manifest(tmp_path)
