from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MODEL_EVIDENCE_SCHEMA_VERSION = "model_evidence_v1"
MODEL_COMPARISON_SCHEMA_VERSION = "model_comparison_v1"
LEGACY_EVIDENCE_TIER = "reconstructed_proxy"
_COMPARISON_BINDING_FIELDS = {
    "path",
    "schema_version",
    "sha256",
}


def validate_model_evidence_sidecar(
    sidecar_path: str | Path,
    comparison_report_path: str | Path,
) -> dict[str, Any]:
    """Validate a legacy evidence declaration and its exact comparison report."""

    sidecar_file = Path(sidecar_path)
    comparison_file = Path(comparison_report_path)
    sidecar = _load_json_object(sidecar_file, label="Model evidence sidecar")

    if sidecar.get("schema_version") != MODEL_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            "Model evidence sidecar schema_version must be "
            f"{MODEL_EVIDENCE_SCHEMA_VERSION!r}"
        )
    if sidecar.get("tier") != LEGACY_EVIDENCE_TIER:
        raise ValueError(
            "Legacy model_comparison_v1 evidence sidecars support only "
            f"{LEGACY_EVIDENCE_TIER!r}; stronger tiers require full audit or "
            "saved-live receipt validation"
        )

    binding = sidecar.get("comparison_report")
    if not isinstance(binding, dict):
        raise ValueError(
            "Model evidence sidecar requires a comparison_report binding"
        )
    if set(binding) != _COMPARISON_BINDING_FIELDS:
        raise ValueError(
            "comparison_report binding must contain exactly "
            f"{sorted(_COMPARISON_BINDING_FIELDS)}"
        )

    declared_path = binding.get("path")
    if not isinstance(declared_path, str) or not declared_path.strip():
        raise ValueError("comparison_report.path must be non-empty text")
    relative_path = Path(declared_path)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise ValueError(
            "comparison_report.path must be a safe relative path"
        )
    expected_report_path = (sidecar_file.parent / relative_path).resolve()
    if comparison_file.resolve() != expected_report_path:
        raise ValueError(
            "Comparison report path does not match the evidence sidecar binding"
        )

    expected_hash = binding.get("sha256")
    if not _is_sha256(expected_hash):
        raise ValueError(
            "comparison_report.sha256 must be a lowercase SHA-256 digest"
        )
    report_bytes = comparison_file.read_bytes()
    actual_hash = hashlib.sha256(report_bytes).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError(
            "Comparison report SHA-256 does not match the evidence sidecar"
        )

    report = _load_json_object(
        comparison_file,
        label="Bound model comparison report",
        raw=report_bytes,
    )
    declared_schema = binding.get("schema_version")
    if declared_schema != MODEL_COMPARISON_SCHEMA_VERSION:
        raise ValueError(
            "comparison_report.schema_version must be "
            f"{MODEL_COMPARISON_SCHEMA_VERSION!r}"
        )
    if report.get("schema_version") != declared_schema:
        raise ValueError(
            "Bound comparison report schema_version does not match the "
            "evidence sidecar"
        )

    availability_audit = sidecar.get("availability_audit")
    availability_audit_sha256 = sidecar.get("availability_audit_sha256")
    if availability_audit is None:
        if availability_audit_sha256 is not None:
            raise ValueError(
                "availability_audit_sha256 must be null when no audit is bound"
            )
    else:
        if not isinstance(availability_audit, dict):
            raise ValueError("availability_audit must be an object or null")
        if not _is_sha256(availability_audit_sha256):
            raise ValueError(
                "A bound availability audit requires a lowercase SHA-256 digest"
            )

    return sidecar


def _load_json_object(
    path: Path,
    *,
    label: str,
    raw: bytes | None = None,
) -> dict[str, Any]:
    payload_bytes = path.read_bytes() if raw is None else raw
    try:
        payload = json.loads(payload_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return payload


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
