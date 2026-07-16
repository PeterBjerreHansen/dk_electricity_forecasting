from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


SPEC_SCHEMA_VERSION = "experiment_spec_v1"
SUPPORTED_EXPERIMENT_KIND = "command_pipeline"
_NAME_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
_SECRET_ENV_KEY_PATTERN = re.compile(
    r"(?:^|_)(?:API_KEY|CREDENTIALS?|PASSWORD|PASSWD|SECRET|TOKEN)(?:$|_)",
    re.IGNORECASE,
)
_ROOT_FIELDS = {
    "schema_version",
    "name",
    "kind",
    "description",
    "metadata",
    "inputs",
    "env",
    "stages",
    "profiles",
}
_PROFILE_FIELDS = {"description", "metadata", "inputs", "env", "stages"}
_STAGE_FIELDS = {
    "name",
    "argv",
    "cwd",
    "env",
    "outputs",
    "timeout_seconds",
}


@dataclass(frozen=True)
class ResolvedExperimentSpec:
    source_path: Path
    source_sha256: str
    profile: str
    effective: dict[str, Any]

    @property
    def name(self) -> str:
        return str(self.effective["name"])


def load_experiment_spec(
    path: str | Path,
    *,
    profile: str,
) -> ResolvedExperimentSpec:
    source_path = Path(path)
    raw_bytes = source_path.read_bytes()
    try:
        raw = json.loads(raw_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Experiment spec is not valid JSON: {source_path}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Experiment spec root must be a JSON object")

    effective = resolve_experiment_profile(raw, profile=profile)
    return ResolvedExperimentSpec(
        source_path=source_path,
        source_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        profile=profile,
        effective=effective,
    )


def resolve_experiment_profile(
    raw: Mapping[str, Any],
    *,
    profile: str,
) -> dict[str, Any]:
    unknown = set(raw) - _ROOT_FIELDS
    if unknown:
        raise ValueError(f"Unknown experiment spec fields: {sorted(unknown)}")
    if raw.get("schema_version") != SPEC_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version must be {SPEC_SCHEMA_VERSION!r}"
        )
    if raw.get("kind") != SUPPORTED_EXPERIMENT_KIND:
        raise ValueError(
            "Unsupported experiment kind "
            f"{raw.get('kind')!r}; expected {SUPPORTED_EXPERIMENT_KIND!r}"
        )
    name = raw.get("name")
    if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
        raise ValueError(
            "Experiment name must contain only lowercase letters, digits, "
            "hyphens, and underscores"
        )
    if not isinstance(profile, str) or not _NAME_PATTERN.fullmatch(profile):
        raise ValueError("Profile name is invalid")

    profiles = raw.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError("profiles must be a JSON object")
    missing_standard_profiles = {"smoke", "full"} - set(profiles)
    if missing_standard_profiles:
        raise ValueError(
            "Experiment specs must define smoke and full profiles; missing "
            f"{sorted(missing_standard_profiles)}"
        )
    for profile_name, profile_value in profiles.items():
        if (
            not isinstance(profile_name, str)
            or not _NAME_PATTERN.fullmatch(profile_name)
            or not isinstance(profile_value, dict)
        ):
            raise ValueError("Every experiment profile must be a named JSON object")
        unknown_profile_fields = set(profile_value) - _PROFILE_FIELDS
        if unknown_profile_fields:
            raise ValueError(
                f"Unknown fields in profile {profile_name!r}: "
                f"{sorted(unknown_profile_fields)}"
            )
    if profile not in profiles:
        available = sorted(profiles) if isinstance(profiles, dict) else []
        raise ValueError(
            f"Unknown experiment profile {profile!r}; available: {available}"
        )
    override = profiles[profile]

    base = {key: copy.deepcopy(value) for key, value in raw.items() if key != "profiles"}
    effective = _deep_merge(base, override)
    effective["profile"] = profile
    _validate_effective_spec(effective)
    return effective


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _validate_effective_spec(spec: Mapping[str, Any]) -> None:
    description = spec.get("description")
    if description is not None and not isinstance(description, str):
        raise ValueError("description must be a string")
    metadata = spec.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a JSON object")
    _validate_environment_mapping(spec.get("env", {}), field="env")
    _validate_inputs(spec.get("inputs", []))

    stages = spec.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("command_pipeline specs must contain at least one stage")
    names: set[str] = set()
    declared_outputs: set[str] = set()
    for index, stage in enumerate(stages, start=1):
        if not isinstance(stage, dict):
            raise ValueError(f"Stage {index} must be a JSON object")
        unknown = set(stage) - _STAGE_FIELDS
        if unknown:
            raise ValueError(
                f"Unknown fields in stage {index}: {sorted(unknown)}"
            )
        name = stage.get("name")
        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Stage {index} has an invalid name")
        if name in names:
            raise ValueError(f"Duplicate stage name: {name}")
        names.add(name)
        argv = stage.get("argv")
        if (
            not isinstance(argv, list)
            or not argv
            or any(not isinstance(value, str) or not value for value in argv)
        ):
            raise ValueError(f"Stage {name!r} argv must be a non-empty string list")
        cwd = stage.get("cwd", "{project_root}")
        if not isinstance(cwd, str) or not cwd:
            raise ValueError(f"Stage {name!r} cwd must be a non-empty string")
        _validate_environment_mapping(
            stage.get("env", {}),
            field=f"stage {name!r} env",
        )
        outputs = stage.get("outputs", [])
        if (
            not isinstance(outputs, list)
            or any(not isinstance(value, str) or not value for value in outputs)
        ):
            raise ValueError(f"Stage {name!r} outputs must be a string list")
        if len(outputs) != len(set(outputs)):
            raise ValueError(f"Stage {name!r} contains duplicate outputs")
        reused_outputs = sorted(set(outputs) & declared_outputs)
        if reused_outputs:
            raise ValueError(
                "Experiment stage outputs must be owned by exactly one stage; "
                f"reused={reused_outputs}"
            )
        declared_outputs.update(outputs)
        timeout = stage.get("timeout_seconds")
        if timeout is not None and (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or timeout <= 0
        ):
            raise ValueError(f"Stage {name!r} timeout_seconds must be positive")


def _validate_inputs(inputs: Any) -> None:
    if not isinstance(inputs, list):
        raise ValueError("inputs must be a list")
    names: set[str] = set()
    for index, entry in enumerate(inputs, start=1):
        if not isinstance(entry, dict):
            raise ValueError(f"Input {index} must be a JSON object")
        unknown = set(entry) - {"name", "path", "required"}
        if unknown:
            raise ValueError(f"Unknown fields in input {index}: {sorted(unknown)}")
        name = entry.get("name")
        path = entry.get("path")
        required = entry.get("required", True)
        if not isinstance(name, str) or not _NAME_PATTERN.fullmatch(name):
            raise ValueError(f"Input {index} has an invalid name")
        if name in names:
            raise ValueError(f"Duplicate input name: {name}")
        names.add(name)
        if not isinstance(path, str) or not path:
            raise ValueError(f"Input {name!r} path must be a non-empty string")
        if not isinstance(required, bool):
            raise ValueError(f"Input {name!r} required must be boolean")


def _validate_string_mapping(value: Any, *, field: str) -> None:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in value.items()
    ):
        raise ValueError(f"{field} must map strings to strings")


def _validate_environment_mapping(value: Any, *, field: str) -> None:
    _validate_string_mapping(value, field=field)
    secret_keys = sorted(
        key for key in value if _SECRET_ENV_KEY_PATTERN.search(key)
    )
    if secret_keys:
        raise ValueError(
            f"{field} must not persist secret-like environment keys: {secret_keys}"
        )
