from __future__ import annotations

import hashlib
import json
from pathlib import Path

from dkenergy_forecast.experiments import load_experiment_spec
from dkenergy_forecast.features.weather_features import (
    EXCLUDED_WEATHER_PARAMETERS,
    WEATHER_CUTOFF_COLUMNS,
    WEATHER_SELECTION_POLICY,
)
from dkenergy_forecast.models.chronos_artifact_identity import (
    attach_chronos_release_identity,
)
from dkenergy_forecast.models.chronos_production import (
    CHRONOS_LORA_ARTIFACT_SCHEMA_VERSION,
    WEATHER_HORIZON_COVERAGE_UNIT,
)
from scripts.clone_chronos_policy_artifact import clone_chronos_policy_artifact


ROOT = Path(__file__).resolve().parents[1]
REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
SPEC_EXPECTATIONS = {
    "chronos_a0_current.json": {
        "ablation_id": "A0",
        "mode": "ensemble_mean",
        "cross_learning": False,
        "derived": False,
        "trains": False,
        "base_artifact": True,
    },
    "chronos_a1_cross_learning.json": {
        "ablation_id": "A1",
        "mode": "ensemble_mean",
        "cross_learning": True,
        "derived": False,
        "trains": False,
        "base_artifact": True,
    },
    "chronos_a2_derived_weather.json": {
        "ablation_id": "A2",
        "mode": "ensemble_mean",
        "cross_learning": False,
        "derived": True,
        "trains": True,
        "base_artifact": False,
    },
    "chronos_a3_ensemble_spread.json": {
        "ablation_id": "A3",
        "mode": "ensemble_mean_spread",
        "cross_learning": False,
        "derived": False,
        "trains": True,
        "base_artifact": False,
    },
    "chronos_a4_combined.json": {
        "ablation_id": "A4",
        "mode": "ensemble_mean_spread",
        "cross_learning": True,
        "derived": True,
        "trains": True,
        "base_artifact": False,
    },
}


def test_a0_to_a4_specs_are_research_only_reproducible_pipelines() -> None:
    for filename, expected in SPEC_EXPECTATIONS.items():
        path = ROOT / "config" / "experiments" / filename
        for profile in ("smoke", "full"):
            resolved = load_experiment_spec(path, profile=profile).effective
            metadata = resolved["metadata"]
            input_names = {item["name"] for item in resolved["inputs"]}
            argv = [
                value
                for stage in resolved["stages"]
                for value in stage["argv"]
            ]

            assert metadata["ablation_id"] == expected["ablation_id"]
            assert metadata["research_only"] is True
            assert metadata["promotes_model"] is False
            assert metadata["frozen_origin_evidence"] is False
            assert "not frozen-origin" in metadata["evidence_note"]
            assert metadata["base_model_revision"] == REVISION
            assert {"price-panel", "price-panel-qa", "weather-features"} <= input_names
            assert ("base-artifact" in input_names) is expected["base_artifact"]
            assert all(
                output.startswith("{run_dir}/")
                for stage in resolved["stages"]
                for output in stage.get("outputs", [])
            )
            assert all(
                not value.startswith("{project_root}/results")
                and not value.startswith("{project_root}/artifacts")
                for value in argv
                if value.endswith(("diagnostics", "artifact", "preflight.json"))
            )
            if profile == "smoke":
                assert "scripts/train_chronos_lora.py" not in argv

        full = load_experiment_spec(path, profile="full").effective
        full_argv = [value for stage in full["stages"] for value in stage["argv"]]
        assert (
            "scripts/train_chronos_lora.py" in full_argv
        ) is expected["trains"]
        if expected["trains"]:
            assert _option_value(full_argv, "--base-model-revision") == REVISION
        assert _option_value(
            full_argv,
            "--chronos-weather-covariate-mode",
        ) == expected["mode"]
        assert (
            "--chronos-cross-learning" in full_argv
        ) is expected["cross_learning"]
        assert (
            "--chronos-add-weather-derived-features" in full_argv
        ) is expected["derived"]


def test_a1_policy_clone_reuses_weights_but_gets_a_distinct_release(tmp_path: Path) -> None:
    source = _artifact(tmp_path)
    source_manifest = json.loads((source / "manifest.json").read_text())
    destination = tmp_path / "a1"

    cloned = clone_chronos_policy_artifact(
        source,
        destination,
        cross_learning=True,
        expected_base_model_revision=REVISION,
    )
    repeated = clone_chronos_policy_artifact(
        source,
        tmp_path / "a1-repeated",
        cross_learning=True,
        expected_base_model_revision=REVISION,
    )

    assert cloned["cross_learning"] is True
    assert cloned["artifact_content_sha256"] == source_manifest["artifact_content_sha256"]
    assert cloned["artifact_files_sha256"] == source_manifest["artifact_files_sha256"]
    assert cloned["serving_policy_sha256"] != source_manifest["serving_policy_sha256"]
    assert cloned["release_id"] != source_manifest["release_id"]
    assert repeated["release_id"] == cloned["release_id"]
    assert cloned["policy_clone"]["weights_retrained"] is False
    assert cloned["policy_clone"]["changed_fields"] == {
        "cross_learning": {"from": False, "to": True}
    }
    assert (destination / "adapter_model.safetensors").read_bytes() == (
        source / "adapter_model.safetensors"
    ).read_bytes()


def test_policy_clone_can_change_missing_weather_fallback_without_retraining(
    tmp_path: Path,
) -> None:
    source = _artifact(tmp_path)
    source_manifest = json.loads((source / "manifest.json").read_text())

    cloned = clone_chronos_policy_artifact(
        source,
        tmp_path / "zero-fallback",
        cross_learning=None,
        weather_future_fallback_policy="zero",
        expected_base_model_revision=REVISION,
    )

    assert cloned["cross_learning"] is False
    assert cloned["weather_horizon_coverage_policy"]["insufficient_coverage_fallback"] == "zero"
    assert cloned["artifact_content_sha256"] == source_manifest["artifact_content_sha256"]
    assert cloned["policy_clone"]["changed_fields"] == {
        "weather_horizon_coverage_policy.insufficient_coverage_fallback": {
            "from": "error",
            "to": "zero",
        }
    }


def _option_value(argv: list[str], option: str) -> str | None:
    if option not in argv:
        return None
    return argv[argv.index(option) + 1]


def _artifact(tmp_path: Path) -> Path:
    artifact = tmp_path / "source"
    artifact.mkdir()
    files = {
        "README.md": b"test artifact\n",
        "adapter_config.json": json.dumps({"revision": REVISION}).encode(),
        "adapter_model.safetensors": b"immutable weights",
    }
    for name, content in files.items():
        (artifact / name).write_bytes(content)
    file_hashes = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in files.items()
    }
    content_hash = hashlib.sha256(
        json.dumps(
            file_hashes,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest = attach_chronos_release_identity(
        {
            "artifact_schema_version": CHRONOS_LORA_ARTIFACT_SCHEMA_VERSION,
            "artifact_content_sha256": content_hash,
            "artifact_files_sha256": file_hashes,
            "model_name": "chronos_weather",
            "base_model_id": "amazon/chronos-2",
            "base_model_revision": REVISION,
            "model_family": "chronos2_lora_calendar_weather",
            "context_length": 24,
            "prediction_length": 24,
            "weather_covariate_mode": "ensemble_mean",
            "cross_learning": False,
            "add_weather_ensemble_features": True,
            "add_weather_derived_features": False,
            "covariates": ["local_hour", "weather_ensemble_temperature_2m_mean"],
            "price_availability_policy": {
                "column": "price_available_at_utc",
                "publication_local_time": "12:00",
                "timezone": "Europe/Copenhagen",
                "eligibility_operator": "< forecast_origin_utc",
            },
            "forecast_origin_policy": {
                "timezone": "Europe/Copenhagen",
                "local_time": "12:00",
            },
            "weather_availability_policy": {
                "training_cutoff": "delivery-day-minus-one at 10:00 Europe/Copenhagen",
                "serving_context_cutoff": "delivery-day-minus-one at 10:00 Europe/Copenhagen",
                "serving_future_cutoff": "information_cutoff_utc",
                "eligibility_operator": (
                    "forecast_available_at_utc <= information_cutoff_utc"
                ),
            },
            "weather_selection_policy": {
                "name": WEATHER_SELECTION_POLICY,
                "cutoff_priority": list(WEATHER_CUTOFF_COLUMNS),
                "stable_feature_names": True,
                "excluded_parameters": list(EXCLUDED_WEATHER_PARAMETERS),
                "historical_cutoff_local_time": "10:00",
            },
            "weather_horizon_coverage_policy": {
                "unit": WEATHER_HORIZON_COVERAGE_UNIT,
                "minimum": 1.0,
                "insufficient_coverage_fallback": "error",
            },
            "covariate_fill_policy": {
                "training": "no_temporal_fill_then_zero",
                "serving_context": "no_temporal_fill_then_zero",
                "serving_future": "no_temporal_fill_then_zero",
                "temporal_fill": False,
                "missing_value": 0.0,
            },
        }
    )
    (artifact / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return artifact
