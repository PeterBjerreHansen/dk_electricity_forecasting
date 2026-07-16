from __future__ import annotations

import json
from pathlib import Path

import pytest

from dkenergy_forecast.experiments import (
    prepare_experiment,
    resolve_experiment_profile,
)


def test_smoke_and_full_profiles_are_resolved_before_identity_hashing(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname='test'\n")
    spec_path = project_root / "experiment.json"
    spec_path.write_text(
        json.dumps(
            _spec(
                smoke_stages=[_stage("smoke-stage")],
                full_stages=[_stage("full-stage")],
            )
        )
    )

    smoke = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )
    full = prepare_experiment(
        spec_path,
        profile="full",
        project_root=project_root,
    )

    assert smoke.experiment_id != full.experiment_id
    smoke_effective = json.loads(
        (smoke.run_dir / "effective_spec.json").read_text()
    )
    full_effective = json.loads(
        (full.run_dir / "effective_spec.json").read_text()
    )
    assert "profiles" not in smoke_effective
    assert smoke_effective["profile"] == "smoke"
    assert smoke_effective["stages"][0]["name"] == "smoke-stage"
    assert full_effective["profile"] == "full"
    assert full_effective["stages"][0]["name"] == "full-stage"


def test_unknown_kind_and_missing_standard_profile_are_rejected() -> None:
    raw = _spec()
    raw["kind"] = "python_callable"
    with pytest.raises(ValueError, match="Unsupported experiment kind"):
        resolve_experiment_profile(raw, profile="smoke")

    raw = _spec()
    del raw["profiles"]["full"]
    with pytest.raises(ValueError, match="must define smoke and full"):
        resolve_experiment_profile(raw, profile="smoke")


@pytest.mark.parametrize(
    "environment_key",
    [
        "API_KEY",
        "HF_TOKEN",
        "DATABASE_PASSWORD",
        "AWS_SECRET_ACCESS_KEY",
        "SERVICE_CREDENTIAL",
    ],
)
def test_secret_like_declared_environment_keys_are_rejected(
    environment_key: str,
) -> None:
    raw = _spec()
    raw["env"] = {environment_key: "must-not-be-persisted"}

    with pytest.raises(ValueError, match="secret-like environment keys"):
        resolve_experiment_profile(raw, profile="smoke")


def test_input_content_changes_the_deterministic_experiment_id(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    input_path = project_root / "input.txt"
    input_path.write_text("first")
    spec = _spec()
    spec["inputs"] = [{"name": "fixture", "path": "input.txt"}]
    spec_path = project_root / "experiment.json"
    spec_path.write_text(json.dumps(spec))

    first = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )
    input_path.write_text("second")
    second = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )

    assert first.experiment_id != second.experiment_id


def test_dirty_code_and_environment_are_part_of_the_experiment_id(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    source_dir = project_root / "src"
    source_dir.mkdir(parents=True)
    source_path = source_dir / "pipeline.py"
    source_path.write_text("VERSION = 1\n")
    spec_path = project_root / "experiment.json"
    spec_path.write_text(json.dumps(_spec()))

    first = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )
    source_path.write_text("VERSION = 2\n")
    changed_code = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )
    environment_a = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
        code={"fingerprint": "fixed"},
        environment={"environment": "a"},
    )
    environment_b = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
        code={"fingerprint": "fixed"},
        environment={"environment": "b"},
    )

    assert first.experiment_id != changed_code.experiment_id
    assert environment_a.experiment_id != environment_b.experiment_id


def test_full_profile_rejects_an_explicitly_dirty_source_identity(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    spec_path = project_root / "experiment.json"
    spec_path.write_text(json.dumps(_spec()))

    smoke = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
        code={"fingerprint": "dirty", "dirty": True},
        environment={"environment": "fixed"},
    )
    assert smoke.state == "prepared"
    with pytest.raises(ValueError, match="clean Git worktree"):
        prepare_experiment(
            spec_path,
            profile="full",
            project_root=project_root,
            code={"fingerprint": "dirty", "dirty": True},
            environment={"environment": "fixed"},
        )


def _spec(
    *,
    smoke_stages: list[dict[str, object]] | None = None,
    full_stages: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "schema_version": "experiment_spec_v1",
        "name": "test-pipeline",
        "kind": "command_pipeline",
        "profiles": {
            "smoke": {"stages": smoke_stages or [_stage("smoke")]},
            "full": {"stages": full_stages or [_stage("full")]},
        },
    }


def _stage(name: str) -> dict[str, object]:
    return {
        "name": name,
        "argv": ["python", "-c", "print('ok')"],
    }
