from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any


CHRONOS_SERVING_POLICY_IDENTITY_FIELDS = (
    "artifact_schema_version",
    "model_name",
    "base_model_id",
    "base_model_revision",
    "model_family",
    "context_length",
    "prediction_length",
    "weather_covariate_mode",
    "cross_learning",
    "add_weather_ensemble_features",
    "add_weather_derived_features",
    "covariates",
    "price_availability_policy",
    "forecast_origin_policy",
    "weather_availability_policy",
    "weather_selection_policy",
    "weather_horizon_coverage_policy",
    "covariate_fill_policy",
)


def attach_chronos_release_identity(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind trained content and serving semantics into one release identity."""

    output = dict(manifest)
    artifact_content_sha256 = output.get("artifact_content_sha256")
    if not isinstance(artifact_content_sha256, str) or not artifact_content_sha256:
        raise ValueError(
            "Chronos manifest requires artifact_content_sha256 before release "
            "identity can be attached"
        )
    missing = [
        field
        for field in CHRONOS_SERVING_POLICY_IDENTITY_FIELDS
        if field not in output
    ]
    if missing:
        raise ValueError(
            "Chronos manifest is missing serving-policy identity fields: "
            f"{missing}"
        )

    serving_policy = {
        field: output[field]
        for field in CHRONOS_SERVING_POLICY_IDENTITY_FIELDS
    }
    serving_policy_sha256 = _canonical_sha256(serving_policy)
    release_identity_sha256 = _canonical_sha256(
        {
            "artifact_content_sha256": artifact_content_sha256,
            "serving_policy_sha256": serving_policy_sha256,
        }
    )
    output["serving_policy_sha256"] = serving_policy_sha256
    output["release_identity_sha256"] = release_identity_sha256
    output["release_id"] = f"sha256-{release_identity_sha256[:16]}"
    return output


def validate_chronos_release_identity(manifest: Mapping[str, Any]) -> None:
    """Verify policy-aware identity fields when a manifest provides them."""

    policy_identity_fields = {
        "serving_policy_sha256",
        "release_identity_sha256",
    }
    present_policy_fields = policy_identity_fields.intersection(manifest)
    # Legacy manifests already have a release_id, but predate policy-aware
    # identity hashes. Keep those readable while requiring every newly
    # introduced policy identity field to arrive as one complete bundle.
    if not present_policy_fields:
        return
    identity_fields = {*policy_identity_fields, "release_id"}
    if not identity_fields.issubset(manifest):
        raise ValueError(
            "Chronos manifest has an incomplete policy-aware release identity"
        )
    expected = attach_chronos_release_identity(manifest)
    mismatched = [
        field
        for field in sorted(identity_fields)
        if manifest.get(field) != expected[field]
    ]
    if mismatched:
        raise ValueError(
            "Chronos manifest policy-aware release identity does not match its "
            f"content: {mismatched}"
        )


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
