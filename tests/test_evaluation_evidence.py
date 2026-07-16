from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from dkenergy_forecast.evaluation import (
    RECONSTRUCTED_PROXY,
    SAVED_LIVE_FORECAST,
    STRICT_POINT_IN_TIME,
    build_availability_audit,
    classify_evidence_tier,
    make_evidence_declaration,
    validate_evidence_coverage,
)


def _audit_frame(*, availability_type: str = "observed") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "forecast_origin_utc": [
                "2026-01-10T09:00:00Z",
                "2026-01-10T09:00:00Z",
                "2026-01-11T09:00:00Z",
            ],
            "information_cutoff_utc": [
                "2026-01-10T09:00:00Z",
                "2026-01-10T09:00:00Z",
                "2026-01-11T09:00:00Z",
            ],
            "available_at_utc": [
                "2026-01-10T08:00:00Z",
                "2026-01-10T08:30:00Z",
                "2026-01-11T08:00:00Z",
            ],
            "feature_name": ["load_forecast", "wind_forecast", "load_forecast"],
            "model_label": ["candidate"] * 3,
            "model_release_id": ["release-1"] * 3,
            "area": ["DK1", "DK1", "DK1"],
            "feature_role": ["external_forecast"] * 3,
            "availability_time_type": [availability_type] * 3,
            "source_revision": ["rev-1", "rev-2", "rev-3"],
            "revision_policy": ["provider_revision_as_of_cutoff"] * 3,
            "revision_as_of_verified": [True] * 3,
            "lineage_complete": [True] * 3,
        }
    )


def test_observed_vintages_produce_strict_point_in_time_evidence() -> None:
    audit = build_availability_audit(_audit_frame())

    assert audit["status"] == "passed"
    assert audit["origin_count"] == 2
    assert audit["eligible_row_count"] == 3
    assert audit["proxy_availability_count"] == 0
    assert classify_evidence_tier(audit) == STRICT_POINT_IN_TIME
    assert make_evidence_declaration(audit)["tier"] == STRICT_POINT_IN_TIME


@pytest.mark.parametrize(
    "availability_type",
    ["observed_source_publication", "observed_retrieval"],
)
def test_fundamentals_observed_availability_types_are_strict(
    availability_type: str,
) -> None:
    audit = build_availability_audit(
        _audit_frame(availability_type=availability_type)
    )

    assert audit["status"] == "passed"
    assert classify_evidence_tier(audit) == STRICT_POINT_IN_TIME


def test_proxy_availability_is_explicitly_downgraded() -> None:
    audit = build_availability_audit(
        _audit_frame(availability_type="synthetic_reference_time_proxy")
    )

    assert audit["status"] == "passed"
    assert audit["proxy_availability_count"] == 3
    assert classify_evidence_tier(audit) == RECONSTRUCTED_PROXY
    with pytest.raises(ValueError, match="strict_point_in_time"):
        make_evidence_declaration(audit, requested_tier=STRICT_POINT_IN_TIME)


def test_rows_after_the_information_cutoff_fail_the_evidence_contract() -> None:
    frame = _audit_frame()
    frame.loc[0, "available_at_utc"] = "2026-01-10T09:00:01Z"
    audit = build_availability_audit(frame)

    assert audit["status"] == "failed"
    assert audit["after_cutoff_row_count"] == 1
    with pytest.raises(ValueError, match="Availability audit failed"):
        classify_evidence_tier(audit)


def test_information_cutoff_after_origin_fails_the_evidence_contract() -> None:
    frame = _audit_frame()
    frame.loc[0, "information_cutoff_utc"] = "2026-01-10T09:00:01Z"
    audit = build_availability_audit(frame)

    assert audit["status"] == "failed"
    assert audit["cutoff_after_origin_count"] == 1


@pytest.mark.parametrize(
    "column, count_field",
    [
        ("feature_name", "missing_feature_count"),
        ("model_label", "missing_model_count"),
        ("model_release_id", "missing_model_release_count"),
        ("area", "missing_area_count"),
    ],
)
def test_missing_audit_identity_fails_closed(
    column: str,
    count_field: str,
) -> None:
    frame = _audit_frame()
    frame.loc[0, column] = None
    audit = build_availability_audit(frame)

    assert audit["status"] == "failed"
    assert audit[count_field] == 1


def test_price_style_cutoff_can_require_strictly_before() -> None:
    frame = _audit_frame()
    frame.loc[0, "available_at_utc"] = frame.loc[0, "information_cutoff_utc"]

    inclusive = build_availability_audit(frame)
    strict_before = build_availability_audit(frame, cutoff_inclusive=False)

    assert inclusive["status"] == "passed"
    assert strict_before["status"] == "failed"
    assert strict_before["after_cutoff_row_count"] == 1


def test_missing_source_revision_fails_closed() -> None:
    frame = _audit_frame()
    frame.loc[0, "source_revision"] = None
    audit = build_availability_audit(frame)

    assert audit["missing_source_revision_count"] == 1
    assert audit["status"] == "failed"
    with pytest.raises(ValueError, match="Availability audit failed"):
        classify_evidence_tier(audit)


def test_latest_snapshot_revision_is_explicit_proxy_evidence() -> None:
    frame = _audit_frame()
    frame["revision_policy"] = "latest_retrieved_snapshot"
    frame["revision_as_of_verified"] = False
    audit = build_availability_audit(frame)

    assert audit["status"] == "passed"
    assert audit["unverified_revision_count"] == 3
    assert audit["proxy_reasons"] == [
        "source_revision_not_verified_as_of_cutoff"
    ]
    assert classify_evidence_tier(audit) == RECONSTRUCTED_PROXY


def test_missing_revision_verification_is_invalid_not_proxy_evidence() -> None:
    frame = _audit_frame()
    frame["revision_as_of_verified"] = frame["revision_as_of_verified"].astype(
        "boolean"
    )
    frame.loc[0, "revision_as_of_verified"] = pd.NA
    audit = build_availability_audit(frame)

    assert audit["status"] == "failed"
    assert audit["missing_revision_verification_count"] == 1
    assert audit["unverified_revision_count"] == 0
    with pytest.raises(ValueError, match="Availability audit failed"):
        classify_evidence_tier(audit)


def test_missing_provenance_column_and_tampered_audit_fail_closed() -> None:
    with pytest.raises(ValueError, match="missing required columns"):
        build_availability_audit(_audit_frame().drop(columns=["revision_policy"]))

    audit = build_availability_audit(_audit_frame())
    audit["eligible_row_count"] = 999
    with pytest.raises(ValueError, match="SHA-256"):
        classify_evidence_tier(audit)


def test_rehashed_but_semantically_inconsistent_audit_fails_closed() -> None:
    audit = build_availability_audit(_audit_frame())
    audit["after_cutoff_row_count"] = 1
    audit["status"] = "passed"
    audit.pop("audit_sha256")
    audit["audit_sha256"] = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="status is inconsistent"):
        classify_evidence_tier(audit)


def test_rehashed_audit_with_inconsistent_origin_totals_fails_closed() -> None:
    audit = build_availability_audit(_audit_frame())
    audit["by_origin"][0]["eligible_row_count"] -= 1
    audit.pop("audit_sha256")
    audit["audit_sha256"] = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="must all be eligible"):
        classify_evidence_tier(audit)


def test_rehashed_audit_cannot_claim_strict_without_strict_rows() -> None:
    audit = build_availability_audit(_audit_frame())
    audit["strict_availability_count"] = 0
    audit.pop("audit_sha256")
    audit["audit_sha256"] = hashlib.sha256(
        json.dumps(audit, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="must cover every row"):
        classify_evidence_tier(audit)


def test_saved_live_forecast_is_a_distinct_evidence_tier() -> None:
    declaration = make_evidence_declaration(saved_live_receipt_verified=True)

    assert declaration["tier"] == SAVED_LIVE_FORECAST
    with pytest.raises(ValueError, match="verified live completion receipt"):
        make_evidence_declaration(requested_tier=SAVED_LIVE_FORECAST)


def test_audit_coverage_is_bound_to_evaluated_origin_and_cutoff() -> None:
    audit = build_availability_audit(_audit_frame())
    predictions = pd.DataFrame(
        {
            "forecast_origin_utc": [
                "2026-01-10T09:00:00Z",
                "2026-01-12T09:00:00Z",
            ],
            "information_cutoff_utc": [
                "2026-01-10T09:00:00Z",
                "2026-01-12T09:00:00Z",
            ],
            "model_label": ["candidate", "candidate"],
            "model_release_id": ["release-1", "release-1"],
            "area": ["DK1", "DK1"],
            "evidence_id": [audit["audit_sha256"]] * 2,
        }
    )

    with pytest.raises(ValueError, match="does not cover every evaluated"):
        validate_evidence_coverage(predictions, audit)

    covered = predictions.iloc[[0]].copy()
    covered["evidence_id"] = "0" * 64
    with pytest.raises(ValueError, match="not bound"):
        validate_evidence_coverage(covered, audit)
