#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.models.chronos_artifact_identity import (  # noqa: E402
    attach_chronos_release_identity,
)
from dkenergy_forecast.models.chronos_production import (  # noqa: E402
    Chronos2LoRAWeatherConfig,
    load_lora_artifact_manifest,
    validate_artifact_weather_policy,
)


def main() -> None:
    args = parse_args()
    manifest = clone_chronos_policy_artifact(
        args.source_artifact,
        args.output_dir,
        cross_learning=args.cross_learning,
        expected_base_model_revision=args.expected_base_model_revision,
    )
    print(
        "Cloned Chronos policy artifact "
        f"{manifest['policy_clone']['source_release_id']} -> {manifest['release_id']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Clone immutable Chronos weights while changing only a serving-time "
            "policy that does not require retraining."
        )
    )
    parser.add_argument("--source-artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--cross-learning",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="Cross-series inference policy for the cloned artifact.",
    )
    parser.add_argument(
        "--expected-base-model-revision",
        required=True,
        help="Immutable base-model revision that the source adapter must pin.",
    )
    return parser.parse_args()


def clone_chronos_policy_artifact(
    source_artifact: str | Path,
    output_dir: str | Path,
    *,
    cross_learning: bool,
    expected_base_model_revision: str,
) -> dict[str, object]:
    source = Path(source_artifact).resolve()
    destination = Path(output_dir).resolve()
    if source == destination:
        raise ValueError("Chronos source and destination artifacts must differ")
    if destination.exists():
        raise FileExistsError(f"Chronos clone destination already exists: {destination}")

    source_manifest = load_lora_artifact_manifest(source)
    observed_revision = source_manifest.get("base_model_revision")
    if observed_revision != expected_base_model_revision:
        raise ValueError(
            "Chronos source base-model revision does not match the pinned "
            f"experiment revision: source={observed_revision!r}, "
            f"expected={expected_base_model_revision!r}"
        )
    source_cross_learning = _manifest_boolean(
        source_manifest,
        "cross_learning",
        default=False,
    )
    if source_cross_learning == cross_learning:
        raise ValueError(
            "Chronos policy clone must change cross_learning; "
            f"source and requested value are both {cross_learning}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.tmp")
    try:
        shutil.copytree(source, temporary)
        cloned_manifest = dict(source_manifest)
        cloned_manifest["cross_learning"] = cross_learning
        cloned_manifest["add_weather_ensemble_features"] = _manifest_boolean(
            source_manifest,
            "add_weather_ensemble_features",
            default=True,
        )
        cloned_manifest["add_weather_derived_features"] = _manifest_boolean(
            source_manifest,
            "add_weather_derived_features",
            default=False,
        )
        cloned_manifest["policy_clone"] = {
            "schema_version": "chronos_policy_clone_v1",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_release_id": source_manifest.get("release_id"),
            "source_artifact_content_sha256": source_manifest.get(
                "artifact_content_sha256"
            ),
            "changed_fields": {
                "cross_learning": {
                    "from": source_cross_learning,
                    "to": cross_learning,
                }
            },
            "weights_retrained": False,
        }
        cloned_manifest = attach_chronos_release_identity(cloned_manifest)
        (temporary / "manifest.json").write_text(
            json.dumps(
                cloned_manifest,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _validate_clone(
            temporary,
            cloned_manifest,
            cross_learning=cross_learning,
        )
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return load_lora_artifact_manifest(destination)


def _validate_clone(
    artifact_path: Path,
    manifest: dict[str, object],
    *,
    cross_learning: bool,
) -> None:
    config = Chronos2LoRAWeatherConfig(
        model_artifact_path=artifact_path,
        base_model_id=str(manifest["base_model_id"]),
        context_length=int(manifest["context_length"]),
        prediction_length=int(manifest["prediction_length"]),
        information_cutoff_local_time=str(
            manifest["weather_selection_policy"]["historical_cutoff_local_time"]
        ),
        weather_covariate_mode=str(manifest["weather_covariate_mode"]),
        cross_learning=cross_learning,
        add_weather_ensemble_features=bool(
            manifest["add_weather_ensemble_features"]
        ),
        add_weather_derived_features=bool(
            manifest["add_weather_derived_features"]
        ),
        weather_horizon_min_coverage=float(
            manifest["weather_horizon_coverage_policy"]["minimum"]
        ),
        weather_future_fallback_policy=str(
            manifest["weather_horizon_coverage_policy"][
                "insufficient_coverage_fallback"
            ]
        ),
    )
    loaded = load_lora_artifact_manifest(artifact_path)
    validate_artifact_weather_policy(loaded, config)


def _manifest_boolean(
    manifest: dict[str, object],
    name: str,
    *,
    default: bool,
) -> bool:
    value = manifest.get(name, default)
    if not isinstance(value, bool):
        raise ValueError(f"Chronos artifact policy {name!r} must be boolean")
    return value


if __name__ == "__main__":
    main()
