from __future__ import annotations

import json
from pathlib import Path

import pytest

from dkenergy_forecast.evaluation.evidence_sidecar import (
    validate_model_evidence_sidecar,
)


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_DIR = (
    ROOT
    / "docs"
    / "model-evidence"
    / "chronos-weather-sha256-55814a7fd0d36973"
)


def test_committed_legacy_evidence_sidecar_is_bound_to_report() -> None:
    sidecar = validate_model_evidence_sidecar(
        EVIDENCE_DIR / "evidence.json",
        EVIDENCE_DIR / "model_comparison.json",
    )

    assert sidecar["tier"] == "reconstructed_proxy"
    assert sidecar["comparison_report"] == {
        "path": "model_comparison.json",
        "schema_version": "model_comparison_v1",
        "sha256": (
            "8922bbf3e459a66e776cf44a03c6b8dd70955c946922f35661b23e81ff811136"
        ),
    }


def test_legacy_evidence_sidecar_rejects_tampered_comparison(
    tmp_path: Path,
) -> None:
    sidecar_path = tmp_path / "evidence.json"
    report_path = tmp_path / "model_comparison.json"
    sidecar_path.write_bytes((EVIDENCE_DIR / "evidence.json").read_bytes())
    report_path.write_bytes(
        (EVIDENCE_DIR / "model_comparison.json").read_bytes() + b"\n"
    )

    with pytest.raises(ValueError, match="SHA-256"):
        validate_model_evidence_sidecar(sidecar_path, report_path)


def test_legacy_evidence_sidecar_rejects_unbound_report_path(
    tmp_path: Path,
) -> None:
    sidecar = json.loads((EVIDENCE_DIR / "evidence.json").read_text())
    sidecar["comparison_report"]["path"] = "different.json"
    sidecar_path = tmp_path / "evidence.json"
    sidecar_path.write_text(json.dumps(sidecar))
    report_path = tmp_path / "model_comparison.json"
    report_path.write_bytes(
        (EVIDENCE_DIR / "model_comparison.json").read_bytes()
    )

    with pytest.raises(ValueError, match="path does not match"):
        validate_model_evidence_sidecar(sidecar_path, report_path)


@pytest.mark.parametrize(
    "tier",
    [
        "strict_point_in_time",
        "saved_live_forecast",
    ],
)
def test_legacy_evidence_sidecar_rejects_stronger_tiers(
    tmp_path: Path,
    tier: str,
) -> None:
    sidecar = json.loads((EVIDENCE_DIR / "evidence.json").read_text())
    sidecar["tier"] = tier
    sidecar_path = tmp_path / "evidence.json"
    sidecar_path.write_text(json.dumps(sidecar))
    report_path = tmp_path / "model_comparison.json"
    report_path.write_bytes(
        (EVIDENCE_DIR / "model_comparison.json").read_bytes()
    )

    with pytest.raises(
        ValueError,
        match="support only 'reconstructed_proxy'",
    ):
        validate_model_evidence_sidecar(sidecar_path, report_path)
