from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Any

import pandas as pd

from dkenergy_forecast.types import require_columns


STRICT_POINT_IN_TIME = "strict_point_in_time"
RECONSTRUCTED_PROXY = "reconstructed_proxy"
SAVED_LIVE_FORECAST = "saved_live_forecast"
EVIDENCE_TIERS = {
    STRICT_POINT_IN_TIME,
    RECONSTRUCTED_PROXY,
    SAVED_LIVE_FORECAST,
}

DEFAULT_STRICT_AVAILABILITY_TYPES = {
    "observed",
    "observed_publication_time",
    "observed_retrieval",
    "observed_source_publication",
    "provider_event_timestamp",
    "provider_published_at",
    "provider_publication_time",
    "recorded_live_availability",
    "recorded_publication_time",
    "source_publication_time",
}
DEFAULT_PROXY_MARKERS = {
    "assumed",
    "convention",
    "inferred",
    "legacy",
    "proxy",
    "reconstructed",
    "synthetic",
}
AUDIT_INVALID_COUNT_FIELDS = (
    "missing_origin_count",
    "missing_cutoff_count",
    "cutoff_after_origin_count",
    "missing_availability_count",
    "after_cutoff_row_count",
    "missing_feature_count",
    "missing_model_count",
    "missing_model_release_count",
    "missing_area_count",
    "unknown_availability_type_count",
    "missing_source_revision_count",
    "missing_revision_policy_count",
    "missing_revision_verification_count",
    "incomplete_lineage_count",
    "unknown_feature_role_count",
)
AUDIT_PROXY_COUNT_FIELDS = (
    "proxy_availability_count",
    "unverified_revision_count",
)
AUDIT_REQUIRED_COUNT_FIELDS = (
    "row_count",
    "origin_count",
    "feature_count",
    "model_count",
    "eligible_row_count",
    "strict_availability_count",
    *AUDIT_INVALID_COUNT_FIELDS,
    *AUDIT_PROXY_COUNT_FIELDS,
)


def build_availability_audit(
    frame: pd.DataFrame,
    *,
    origin_column: str = "forecast_origin_utc",
    cutoff_column: str = "information_cutoff_utc",
    availability_column: str = "available_at_utc",
    feature_column: str = "feature_name",
    model_column: str = "model_label",
    model_release_column: str = "model_release_id",
    area_column: str = "area",
    feature_role_column: str = "feature_role",
    availability_type_column: str = "availability_time_type",
    source_revision_column: str = "source_revision",
    revision_policy_column: str = "revision_policy",
    revision_as_of_verified_column: str = "revision_as_of_verified",
    lineage_complete_column: str = "lineage_complete",
    cutoff_inclusive: bool = True,
    strict_availability_types: Iterable[str] = DEFAULT_STRICT_AVAILABILITY_TYPES,
    proxy_markers: Iterable[str] = DEFAULT_PROXY_MARKERS,
) -> dict[str, Any]:
    """Audit selected feature cells against the information cutoff at each origin.

    The input is expected to contain the feature rows actually selected for an
    evaluation, not every candidate vintage. This keeps the audit aligned with
    the precise information set supplied to the model.
    """

    required = [
        origin_column,
        cutoff_column,
        availability_column,
        feature_column,
        model_column,
        model_release_column,
        area_column,
        feature_role_column,
        availability_type_column,
        source_revision_column,
        revision_policy_column,
        revision_as_of_verified_column,
        lineage_complete_column,
    ]
    require_columns(frame, required, "availability audit frame")
    if frame.empty:
        raise ValueError("availability audit frame must not be empty")

    audited = frame.copy()
    for column in (origin_column, cutoff_column, availability_column):
        audited[column] = pd.to_datetime(audited[column], utc=True, errors="coerce")

    missing_origin = audited[origin_column].isna()
    missing_cutoff = audited[cutoff_column].isna()
    cutoff_after_origin = (
        ~missing_origin
        & ~missing_cutoff
        & audited[cutoff_column].gt(audited[origin_column])
    )
    normalized_features = _normalized_text_column(audited, feature_column)
    normalized_models = _normalized_text_column(audited, model_column)
    normalized_releases = _normalized_text_column(
        audited,
        model_release_column,
    )
    normalized_areas = _normalized_text_column(audited, area_column)
    missing_feature = normalized_features.eq("")
    missing_model = normalized_models.eq("")
    missing_model_release = normalized_releases.eq("")
    missing_area = normalized_areas.eq("")
    feature_roles = _normalized_text_column(audited, feature_role_column)
    known_future = feature_roles.eq("known_future")
    unknown_feature_role = ~feature_roles.isin(
        {"external_forecast", "historical_observation", "known_future"}
    )
    missing_availability = audited[availability_column].isna() & ~known_future
    comparable = ~(missing_cutoff | missing_availability | known_future)
    if cutoff_inclusive:
        eligible = comparable & (
            audited[availability_column] <= audited[cutoff_column]
        )
        cutoff_operator = "<="
    else:
        eligible = comparable & (
            audited[availability_column] < audited[cutoff_column]
        )
        cutoff_operator = "<"
    eligible = eligible | (known_future & ~missing_cutoff)
    after_cutoff = comparable & ~eligible

    normalized_types = _normalized_text_column(
        audited,
        availability_type_column,
    )
    strict_types = {str(value).strip().lower() for value in strict_availability_types}
    marker_values = {str(value).strip().lower() for value in proxy_markers}
    proxy_type = normalized_types.map(
        lambda value: any(marker in value for marker in marker_values)
        if value
        else False
    )
    strict_type = normalized_types.isin(strict_types) | (
        known_future & normalized_types.eq("not_applicable_known_future")
    )
    missing_type = normalized_types.eq("")
    unknown_type = ~(proxy_type | strict_type | missing_type)

    normalized_revisions = _normalized_text_column(
        audited,
        source_revision_column,
    )
    missing_revision = normalized_revisions.eq("")
    normalized_revision_policies = _normalized_text_column(
        audited,
        revision_policy_column,
    )
    missing_revision_policy = normalized_revision_policies.eq("")
    revision_as_of_verified = _boolean_column(
        audited,
        revision_as_of_verified_column,
    )
    missing_revision_verification = revision_as_of_verified.isna()
    lineage_complete = _boolean_column(audited, lineage_complete_column)

    invalid_count = int(
        (
            missing_origin
            | missing_cutoff
            | cutoff_after_origin
            | missing_availability
            | after_cutoff
            | missing_feature
            | missing_model
            | missing_model_release
            | missing_area
            | missing_type
            | unknown_type
            | missing_revision
            | missing_revision_policy
            | missing_revision_verification
            | lineage_complete.isna()
            | lineage_complete.eq(False)
            | unknown_feature_role
        ).sum()
    )
    status = "failed" if invalid_count else "passed"

    by_origin: list[dict[str, Any]] = []
    valid_origins = audited.loc[
        ~missing_origin
        & ~missing_cutoff
        & ~cutoff_after_origin
        & ~missing_model
        & ~missing_model_release
        & ~missing_area
    ]
    for (origin, cutoff, model, release, area), group in valid_origins.groupby(
        [
            origin_column,
            cutoff_column,
            model_column,
            model_release_column,
            area_column,
        ],
        sort=True,
    ):
        indexes = group.index
        by_origin.append(
            {
                "forecast_origin_utc": origin.isoformat(),
                "information_cutoff_utc": cutoff.isoformat(),
                "model_label": str(model),
                "model_release_id": str(release),
                "area": str(area),
                "row_count": int(len(group)),
                "feature_count": int(group[feature_column].nunique(dropna=True)),
                "eligible_row_count": int(eligible.loc[indexes].sum()),
                "after_cutoff_row_count": int(after_cutoff.loc[indexes].sum()),
                "missing_availability_count": int(
                    missing_availability.loc[indexes].sum()
                ),
                "proxy_availability_count": int(proxy_type.loc[indexes].sum()),
                "unknown_availability_type_count": int(
                    (unknown_type | missing_type).loc[indexes].sum()
                ),
                "missing_source_revision_count": int(
                    missing_revision.loc[indexes].sum()
                ),
                "unverified_revision_count": int(
                    revision_as_of_verified.loc[indexes].eq(False).sum()
                ),
                "missing_revision_verification_count": int(
                    missing_revision_verification.loc[indexes].sum()
                ),
                "incomplete_lineage_count": int(
                    (
                        lineage_complete.loc[indexes].isna()
                        | lineage_complete.loc[indexes].eq(False)
                    ).sum()
                ),
            }
        )

    payload = {
        "schema_version": "availability_audit_v1",
        "status": status,
        "cutoff_operator": cutoff_operator,
        "columns": {
            "forecast_origin": origin_column,
            "information_cutoff": cutoff_column,
            "available_at": availability_column,
            "feature": feature_column,
            "model_label": model_column,
            "model_release_id": model_release_column,
            "area": area_column,
            "feature_role": feature_role_column,
            "availability_time_type": availability_type_column,
            "source_revision": source_revision_column,
            "revision_policy": revision_policy_column,
            "revision_as_of_verified": revision_as_of_verified_column,
            "lineage_complete": lineage_complete_column,
        },
        "row_count": int(len(audited)),
        "origin_count": int(audited[origin_column].nunique(dropna=True)),
        "feature_count": int(audited[feature_column].nunique(dropna=True)),
        "model_count": int(
            audited[[model_column, model_release_column]]
            .drop_duplicates()
            .shape[0]
        ),
        "eligible_row_count": int(eligible.sum()),
        "after_cutoff_row_count": int(after_cutoff.sum()),
        "missing_origin_count": int(missing_origin.sum()),
        "missing_cutoff_count": int(missing_cutoff.sum()),
        "cutoff_after_origin_count": int(cutoff_after_origin.sum()),
        "missing_availability_count": int(missing_availability.sum()),
        "missing_feature_count": int(missing_feature.sum()),
        "missing_model_count": int(missing_model.sum()),
        "missing_model_release_count": int(missing_model_release.sum()),
        "missing_area_count": int(missing_area.sum()),
        "strict_availability_count": int(strict_type.sum()),
        "proxy_availability_count": int(proxy_type.sum()),
        "unknown_availability_type_count": int((unknown_type | missing_type).sum()),
        "missing_source_revision_count": int(missing_revision.sum()),
        "missing_revision_policy_count": int(missing_revision_policy.sum()),
        "missing_revision_verification_count": int(
            missing_revision_verification.sum()
        ),
        "unverified_revision_count": int(revision_as_of_verified.eq(False).sum()),
        "incomplete_lineage_count": int(
            (lineage_complete.isna() | lineage_complete.eq(False)).sum()
        ),
        "unknown_feature_role_count": int(unknown_feature_role.sum()),
        "availability_types": sorted(
            value for value in normalized_types.unique().tolist() if value
        ),
        "revision_policies": sorted(
            value for value in normalized_revision_policies.unique().tolist() if value
        ),
        "proxy_reasons": _proxy_reasons(
            proxy_count=int(proxy_type.sum()),
            unverified_revision_count=int(
                revision_as_of_verified.eq(False).sum()
            ),
        ),
        "by_origin": by_origin,
    }
    payload["audit_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return payload


def classify_evidence_tier(
    availability_audit: dict[str, Any] | None = None,
    *,
    saved_live_receipt_verified: bool = False,
) -> str:
    """Return the strongest defensible evidence tier for an evaluation."""

    if saved_live_receipt_verified:
        return SAVED_LIVE_FORECAST
    if availability_audit is None:
        return RECONSTRUCTED_PROXY
    _validate_availability_audit(availability_audit)
    if availability_audit["status"] != "passed":
        raise ValueError(
            "Availability audit failed; the evaluation contains unavailable or "
            "unverifiable feature rows"
        )
    proxy_or_unverifiable = any(
        int(availability_audit.get(field, 0)) > 0
        for field in (
            "proxy_availability_count",
            "unverified_revision_count",
        )
    )
    if proxy_or_unverifiable:
        return RECONSTRUCTED_PROXY
    return STRICT_POINT_IN_TIME


def make_evidence_declaration(
    availability_audit: dict[str, Any] | None = None,
    *,
    requested_tier: str | None = None,
    saved_live_receipt_verified: bool = False,
    note: str | None = None,
) -> dict[str, Any]:
    """Build a validated evidence declaration for a comparison artifact."""

    inferred_tier = classify_evidence_tier(
        availability_audit,
        saved_live_receipt_verified=saved_live_receipt_verified,
    )
    if requested_tier is not None and requested_tier not in EVIDENCE_TIERS:
        raise ValueError(
            f"Unknown evidence tier {requested_tier!r}; expected one of "
            f"{sorted(EVIDENCE_TIERS)}"
        )
    if requested_tier == SAVED_LIVE_FORECAST and not saved_live_receipt_verified:
        raise ValueError(
            "saved_live_forecast evidence requires a verified live completion receipt"
        )
    if requested_tier == STRICT_POINT_IN_TIME and inferred_tier != STRICT_POINT_IN_TIME:
        raise ValueError(
            "strict_point_in_time evidence requires a passing availability audit "
            "with observed availability timestamps and source revisions"
        )
    if (
        requested_tier is not None
        and inferred_tier == SAVED_LIVE_FORECAST
        and requested_tier != SAVED_LIVE_FORECAST
    ):
        raise ValueError(
            "A saved live forecast must retain the saved_live_forecast evidence tier"
        )

    tier = requested_tier or inferred_tier
    basis = {
        SAVED_LIVE_FORECAST: (
            "Predictions were persisted from a live run before outcomes were known."
        ),
        STRICT_POINT_IN_TIME: (
            "Every audited feature row was available by its origin cutoff using "
            "observed availability timestamps and recorded source revisions."
        ),
        RECONSTRUCTED_PROXY: (
            "At least part of the historical information set was reconstructed, "
            "proxied, or lacks complete source-vintage evidence."
        ),
    }[tier]
    return {
        "schema_version": "model_evidence_v1",
        "tier": tier,
        "basis": basis,
        "integrity_scope": (
            "Receipt, manifest, and artifact hashes are internally consistent; "
            "saved-live evidence additionally assumes the source artifact store "
            "enforces append-only or externally controlled run records."
            if tier == SAVED_LIVE_FORECAST
            else None
        ),
        "note": note,
        "saved_live_receipt_verified": saved_live_receipt_verified,
        "availability_audit_sha256": (
            availability_audit.get("audit_sha256")
            if availability_audit is not None
            else None
        ),
        "availability_audit": availability_audit,
    }


def validate_evidence_coverage(
    predictions: pd.DataFrame,
    availability_audit: dict[str, Any],
) -> None:
    """Require the audit to cover every evaluated origin/cutoff pair."""

    _validate_availability_audit(availability_audit)
    require_columns(
        predictions,
        [
            "forecast_origin_utc",
            "information_cutoff_utc",
            "model_label",
            "model_release_id",
            "area",
            "evidence_id",
        ],
        "evaluated predictions",
    )
    audit_sha256 = availability_audit["audit_sha256"]
    evidence_ids = set(predictions["evidence_id"].dropna().astype(str))
    if predictions["evidence_id"].isna().any() or evidence_ids != {audit_sha256}:
        raise ValueError(
            "Evaluated predictions are not bound to the supplied availability "
            "audit SHA-256"
        )
    evaluated = predictions[
        [
            "forecast_origin_utc",
            "information_cutoff_utc",
            "model_label",
            "model_release_id",
            "area",
        ]
    ].copy()
    for column in ("forecast_origin_utc", "information_cutoff_utc"):
        evaluated[column] = pd.to_datetime(evaluated[column], utc=True, errors="coerce")
    for column in ("model_label", "model_release_id", "area"):
        missing_text = (
            evaluated[column].isna()
            | evaluated[column].astype("string").str.strip().eq("").fillna(True)
        )
        if missing_text.any():
            raise ValueError(
                f"Evaluated predictions contain missing {column} values"
            )
        evaluated[column] = evaluated[column].astype(str)
    if evaluated[
        ["forecast_origin_utc", "information_cutoff_utc"]
    ].isna().any().any():
        raise ValueError(
            "Evaluated predictions contain missing origin or information cutoff"
        )
    expected = set(evaluated.itertuples(index=False, name=None))

    audited: set[tuple[pd.Timestamp, pd.Timestamp, str, str, str]] = set()
    for row in availability_audit.get("by_origin", []):
        if not isinstance(row, dict):
            raise ValueError("Availability audit by_origin entries must be objects")
        origin = pd.to_datetime(
            row.get("forecast_origin_utc"),
            utc=True,
            errors="coerce",
        )
        cutoff = pd.to_datetime(
            row.get("information_cutoff_utc"),
            utc=True,
            errors="coerce",
        )
        if pd.isna(origin) or pd.isna(cutoff):
            raise ValueError(
                "Availability audit by_origin entries require origin and cutoff"
            )
        model = row.get("model_label")
        release = row.get("model_release_id")
        area = row.get("area")
        if any(not isinstance(value, str) or not value for value in (model, release, area)):
            raise ValueError(
                "Availability audit by_origin entries require model, release, and area"
            )
        audited.add((origin, cutoff, model, release, area))

    missing = sorted(expected - audited)
    if missing:
        sample = [
            {
                "forecast_origin_utc": origin.isoformat(),
                "information_cutoff_utc": cutoff.isoformat(),
            }
            | {
                "model_label": model,
                "model_release_id": release,
                "area": area,
            }
            for origin, cutoff, model, release, area in missing[:5]
        ]
        raise ValueError(
            "Availability audit does not cover every evaluated origin/cutoff; "
            f"missing_sample={sample}"
        )


def _normalized_text_column(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].fillna("").astype(str).str.strip().str.lower()


def _boolean_column(frame: pd.DataFrame, column: str) -> pd.Series:
    values = frame[column]
    invalid = values.notna() & ~values.isin([True, False])
    if invalid.any():
        raise ValueError(f"{column} must contain only booleans")
    return values.astype("boolean")


def _proxy_reasons(
    *,
    proxy_count: int,
    unverified_revision_count: int,
) -> list[str]:
    reasons: list[str] = []
    if proxy_count:
        reasons.append("synthetic_or_conventional_availability")
    if unverified_revision_count:
        reasons.append("source_revision_not_verified_as_of_cutoff")
    return reasons


def _validate_availability_audit(audit: dict[str, Any]) -> None:
    if audit.get("schema_version") != "availability_audit_v1":
        raise ValueError("Unsupported availability audit schema")
    if audit.get("status") not in {"passed", "failed"}:
        raise ValueError("Availability audit status must be 'passed' or 'failed'")
    recorded_hash = audit.get("audit_sha256")
    if not isinstance(recorded_hash, str) or len(recorded_hash) != 64:
        raise ValueError("Availability audit is missing a valid audit_sha256")
    unhashed = {key: value for key, value in audit.items() if key != "audit_sha256"}
    calculated_hash = hashlib.sha256(
        json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if calculated_hash != recorded_hash:
        raise ValueError("Availability audit SHA-256 does not match its contents")

    counts = {
        field: _nonnegative_audit_count(audit, field)
        for field in AUDIT_REQUIRED_COUNT_FIELDS
    }
    if counts["row_count"] < 1:
        raise ValueError("Availability audit row_count must be positive")
    for field in ("origin_count", "feature_count", "model_count"):
        if counts[field] < 1:
            raise ValueError(f"Availability audit {field} must be positive")
    invalid_count = sum(counts[field] for field in AUDIT_INVALID_COUNT_FIELDS)
    expected_status = "failed" if invalid_count else "passed"
    if audit["status"] != expected_status:
        raise ValueError(
            "Availability audit status is inconsistent with its invalid-row "
            f"counts: status={audit['status']!r}, expected={expected_status!r}"
        )

    by_origin = audit.get("by_origin")
    if not isinstance(by_origin, list):
        raise ValueError("Availability audit by_origin must be a list")
    if audit["status"] == "passed":
        if counts["eligible_row_count"] != counts["row_count"]:
            raise ValueError(
                "Passing availability audit eligible_row_count must equal row_count"
            )
        if (
            counts["strict_availability_count"]
            + counts["proxy_availability_count"]
            != counts["row_count"]
        ):
            raise ValueError(
                "Passing availability audit strict and proxy availability counts "
                "must cover every row"
            )
        _validate_passing_audit_origins(
            by_origin,
            top_level_counts=counts,
        )


def _nonnegative_audit_count(audit: dict[str, Any], field: str) -> int:
    value = audit.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"Availability audit {field} must be a non-negative integer"
        )
    return value


def _validate_passing_audit_origins(
    by_origin: list[Any],
    *,
    top_level_counts: dict[str, int],
) -> None:
    if not by_origin:
        raise ValueError("A passing availability audit requires by_origin coverage")

    required_identity = (
        "forecast_origin_utc",
        "information_cutoff_utc",
        "model_label",
        "model_release_id",
        "area",
    )
    required_counts = (
        "row_count",
        "eligible_row_count",
        "after_cutoff_row_count",
        "missing_availability_count",
        "proxy_availability_count",
        "unknown_availability_type_count",
        "missing_source_revision_count",
        "unverified_revision_count",
        "missing_revision_verification_count",
        "incomplete_lineage_count",
    )
    seen: set[tuple[str, str, str, str, str]] = set()
    totals = {field: 0 for field in required_counts}
    for row in by_origin:
        if not isinstance(row, dict):
            raise ValueError("Availability audit by_origin entries must be objects")
        identity_values: list[str] = []
        for field in required_identity:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "Availability audit by_origin entries require non-empty "
                    f"{field}"
                )
            identity_values.append(value.strip())
        for field in ("forecast_origin_utc", "information_cutoff_utc"):
            timestamp = pd.to_datetime(row[field], utc=True, errors="coerce")
            if pd.isna(timestamp):
                raise ValueError(
                    f"Availability audit by_origin entry has invalid {field}"
                )
        identity = tuple(identity_values)
        if identity in seen:
            raise ValueError(
                "Availability audit by_origin contains duplicate origin/cutoff/"
                "model/release/area coverage"
            )
        seen.add(identity)

        for field in required_counts:
            value = row.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    "Availability audit by_origin "
                    f"{field} must be a non-negative integer"
                )
            totals[field] += value
        if row["row_count"] < 1:
            raise ValueError(
                "Availability audit by_origin row_count must be positive"
            )
        if row["eligible_row_count"] != row["row_count"]:
            raise ValueError(
                "Passing availability audit by_origin rows must all be eligible"
            )
        if any(
            row[field] != 0
            for field in (
                "after_cutoff_row_count",
                "missing_availability_count",
                "unknown_availability_type_count",
                "missing_source_revision_count",
                "missing_revision_verification_count",
                "incomplete_lineage_count",
            )
        ):
            raise ValueError(
                "Passing availability audit by_origin entries contain invalid "
                "feature rows"
            )

    top_level_mapping = {
        "row_count": "row_count",
        "eligible_row_count": "eligible_row_count",
        "after_cutoff_row_count": "after_cutoff_row_count",
        "missing_availability_count": "missing_availability_count",
        "proxy_availability_count": "proxy_availability_count",
        "unknown_availability_type_count": "unknown_availability_type_count",
        "missing_source_revision_count": "missing_source_revision_count",
        "unverified_revision_count": "unverified_revision_count",
        "missing_revision_verification_count": (
            "missing_revision_verification_count"
        ),
        "incomplete_lineage_count": "incomplete_lineage_count",
    }
    mismatches = {
        top_level_field: {
            "top_level": top_level_counts[top_level_field],
            "by_origin": totals[origin_field],
        }
        for top_level_field, origin_field in top_level_mapping.items()
        if top_level_counts[top_level_field] != totals[origin_field]
    }
    if mismatches:
        raise ValueError(
            "Availability audit top-level counts disagree with by_origin: "
            f"{mismatches}"
        )
