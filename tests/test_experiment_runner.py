from __future__ import annotations

import fcntl
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

import dkenergy_forecast.experiments.runner as runner_module
from dkenergy_forecast.experiments import (
    AlreadyRunningError,
    IntegrityError,
    launch_detached,
    prepare_experiment,
    run_experiment_worker,
    verify_completed_run,
)


ROOT = Path(__file__).resolve().parents[1]


def test_completed_retry_is_a_verified_no_op_and_completion_is_written_last(
    tmp_path: Path,
) -> None:
    project_root, spec_path = _make_project(
        tmp_path,
        stages=[
            _write_stage(
                "write-artifact",
                "artifact.txt",
                content="complete",
            )
        ],
    )
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )

    completed = run_experiment_worker(prepared.run_dir)
    manifest = verify_completed_run(prepared.run_dir)
    retried = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )
    worker_retry = run_experiment_worker(prepared.run_dir)

    assert completed.state == "completed"
    assert retried.no_op is True
    assert worker_retry.no_op is True
    assert manifest["status"] == "completed"
    assert (prepared.run_dir / "artifact.txt").read_text() == "complete"
    assert (prepared.run_dir / "COMPLETED.json").stat().st_mtime_ns >= (
        prepared.run_dir / "manifest.json"
    ).stat().st_mtime_ns


def test_failed_pipeline_resumes_from_verified_stage_receipts(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname='test'\n")
    first_code = (
        "from pathlib import Path; import os; "
        "root=Path(os.environ['DKENERGY_EXPERIMENT_RUN_DIR']); "
        "counter=root/'counter.txt'; "
        "counter.write_text((counter.read_text() if counter.exists() else '')+'1'); "
        "(root/'first.txt').write_text('first')"
    )
    second_code = (
        "from pathlib import Path; import os, sys; "
        "project=Path('{project_root}'); "
        "sys.exit(7) if not (project/'allow-second-stage').exists() else None; "
        "Path(os.environ['DKENERGY_EXPERIMENT_RUN_DIR'], 'second.txt').write_text('second')"
    )
    spec_path = project_root / "experiment.json"
    _write_spec(
        spec_path,
        [
            {
                "name": "first",
                "argv": [sys.executable, "-c", first_code],
                "outputs": ["first.txt", "counter.txt"],
            },
            {
                "name": "second",
                "argv": [sys.executable, "-c", second_code],
                "outputs": ["second.txt"],
            },
        ],
    )
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )

    with pytest.raises(Exception):
        run_experiment_worker(prepared.run_dir)
    failed_status = json.loads((prepared.run_dir / "status.json").read_text())
    assert failed_status["state"] == "failed"
    assert (prepared.run_dir / "stages" / "001-first.json").exists()
    assert not (prepared.run_dir / "stages" / "002-second.json").exists()

    (project_root / "allow-second-stage").write_text("yes")
    resumed = run_experiment_worker(prepared.run_dir)

    assert resumed.state == "completed"
    assert (prepared.run_dir / "counter.txt").read_text() == "1"
    assert (prepared.run_dir / "second.txt").read_text() == "second"
    assert len(list((prepared.run_dir / "logs").glob("001-first.attempt-*.log"))) == 1
    assert len(list((prepared.run_dir / "logs").glob("002-second.attempt-*.log"))) == 2


def test_tampered_completed_output_fails_verification_and_no_op_retry(
    tmp_path: Path,
) -> None:
    project_root, spec_path = _make_project(
        tmp_path,
        stages=[_write_stage("write", "artifact.txt", content="original")],
    )
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )
    run_experiment_worker(prepared.run_dir)
    (prepared.run_dir / "artifact.txt").write_text("tampered")

    with pytest.raises(IntegrityError, match="output hash"):
        verify_completed_run(prepared.run_dir)
    with pytest.raises(IntegrityError, match="output hash"):
        prepare_experiment(
            spec_path,
            profile="smoke",
            project_root=project_root,
        )


def test_worker_lock_is_nonblocking(tmp_path: Path) -> None:
    project_root, spec_path = _make_project(
        tmp_path,
        stages=[_write_stage("write", "artifact.txt")],
    )
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )
    lock_path = prepared.run_dir / ".worker.lock"
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        with pytest.raises(AlreadyRunningError):
            run_experiment_worker(prepared.run_dir)
        duplicate_launch = launch_detached(
            prepared.run_dir,
            worker_script=ROOT / "scripts" / "run_experiment.py",
        )
        assert time.monotonic() - started < 0.5
        assert duplicate_launch.no_op is True
        assert duplicate_launch.state == "running"
        assert duplicate_launch.pid is None


def test_detached_launch_returns_before_short_worker_finishes(
    tmp_path: Path,
) -> None:
    sleeping_code = (
        "from pathlib import Path; import os, time; "
        "time.sleep(0.5); "
        "Path(os.environ['DKENERGY_EXPERIMENT_RUN_DIR'], 'artifact.txt').write_text('done')"
    )
    project_root, spec_path = _make_project(
        tmp_path,
        stages=[
            {
                "name": "short-background-work",
                "argv": [sys.executable, "-c", sleeping_code],
                "outputs": ["artifact.txt"],
            }
        ],
    )
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )

    started = time.monotonic()
    launched = launch_detached(
        prepared.run_dir,
        worker_script=ROOT / "scripts" / "run_experiment.py",
    )
    elapsed = time.monotonic() - started

    assert launched.pid is not None
    assert elapsed < 0.4
    deadline = time.monotonic() + 10
    while (
        not (prepared.run_dir / "COMPLETED.json").exists()
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    verify_completed_run(prepared.run_dir)


def test_declared_outputs_cannot_escape_run_directory(tmp_path: Path) -> None:
    project_root, spec_path = _make_project(
        tmp_path,
        stages=[
            {
                "name": "escape",
                "argv": [sys.executable, "-c", "print('never runs')"],
                "outputs": ["../outside.txt"],
            }
        ],
    )
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )

    with pytest.raises(ValueError, match="below the run directory"):
        run_experiment_worker(prepared.run_dir)
    assert not (prepared.run_dir.parent / "outside.txt").exists()
    status = json.loads((prepared.run_dir / "status.json").read_text())
    assert status["state"] == "failed"


def test_resume_does_not_certify_stale_output_from_failed_attempt(
    tmp_path: Path,
) -> None:
    code = (
        "from pathlib import Path; import os, sys; "
        "root=Path(os.environ['DKENERGY_EXPERIMENT_RUN_DIR']); "
        "marker=root/'allow-success'; output=root/'artifact.txt'; "
        "output.write_text('partial') if not marker.exists() else None; "
        "sys.exit(7) if not marker.exists() else None"
    )
    project_root, spec_path = _make_project(
        tmp_path,
        stages=[
            {
                "name": "stale-output",
                "argv": [sys.executable, "-c", code],
                "outputs": ["artifact.txt"],
            }
        ],
    )
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )

    with pytest.raises(Exception):
        run_experiment_worker(prepared.run_dir)
    assert (prepared.run_dir / "artifact.txt").read_text() == "partial"
    (prepared.run_dir / "allow-success").write_text("yes")

    with pytest.raises(IntegrityError, match="did not produce declared output"):
        run_experiment_worker(prepared.run_dir)
    assert not (prepared.run_dir / "artifact.txt").exists()


def test_stage_cannot_mutate_declared_input_and_still_complete(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    input_path = project_root / "input.txt"
    input_path.write_text("original")
    code = (
        "from pathlib import Path; import os; "
        f"Path({str(input_path)!r}).write_text('mutated'); "
        "Path(os.environ['DKENERGY_EXPERIMENT_RUN_DIR'], 'artifact.txt').write_text('done')"
    )
    spec_path = project_root / "experiment.json"
    spec = {
        "schema_version": "experiment_spec_v1",
        "name": "input-mutation",
        "kind": "command_pipeline",
        "inputs": [{"name": "input", "path": "input.txt"}],
        "profiles": {
            profile: {
                "stages": [
                    {
                        "name": "mutate",
                        "argv": [sys.executable, "-c", code],
                        "outputs": ["artifact.txt"],
                    }
                ]
            }
            for profile in ("smoke", "full")
        },
    }
    spec_path.write_text(json.dumps(spec))
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )

    with pytest.raises(IntegrityError, match="inputs changed"):
        run_experiment_worker(prepared.run_dir)
    assert not (prepared.run_dir / "stages" / "001-mutate.json").exists()


def test_later_stage_cannot_mutate_an_earlier_stage_output(
    tmp_path: Path,
) -> None:
    mutating_code = (
        "from pathlib import Path; import os; "
        "root=Path(os.environ['DKENERGY_EXPERIMENT_RUN_DIR']); "
        "(root/'first.txt').write_text('mutated'); "
        "(root/'second.txt').write_text('second')"
    )
    project_root, spec_path = _make_project(
        tmp_path,
        stages=[
            _write_stage("first", "first.txt", content="original"),
            {
                "name": "second",
                "argv": [sys.executable, "-c", mutating_code],
                "outputs": ["second.txt"],
            },
        ],
    )
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )

    with pytest.raises(IntegrityError, match="output hash"):
        run_experiment_worker(prepared.run_dir)

    status = json.loads((prepared.run_dir / "status.json").read_text())
    assert status["state"] == "failed"
    assert not (prepared.run_dir / "COMPLETED.json").exists()


def _declared_output_stage(name: str, output: str) -> dict[str, object]:
    return {
        "name": name,
        "argv": [sys.executable, "-c", "print('never runs')"],
        "outputs": [output],
    }


@pytest.mark.parametrize(
    "stages, message",
    [
        (
            [
                _declared_output_stage("first", "bundle"),
                _declared_output_stage("second", "bundle/nested.txt"),
            ],
            "cannot overlap or nest",
        ),
        (
            [_declared_output_stage("root", ".")],
            "cannot replace it",
        ),
        (
            [_declared_output_stage("reserved", "status.json")],
            "reserved run path",
        ),
    ],
)
def test_stage_outputs_cannot_overlap_or_replace_runner_artifacts(
    tmp_path: Path,
    stages: list[dict[str, object]],
    message: str,
) -> None:
    project_root, spec_path = _make_project(tmp_path, stages=stages)
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )

    with pytest.raises(ValueError, match=message):
        run_experiment_worker(prepared.run_dir)
    assert (prepared.run_dir / "identity.json").exists()
    assert not (prepared.run_dir / "COMPLETED.json").exists()


def test_declared_input_directories_cannot_contain_symlinks(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    input_dir = project_root / "input"
    input_dir.mkdir(parents=True)
    target = project_root / "target.txt"
    target.write_text("target")
    (input_dir / "linked.txt").symlink_to(target)
    spec_path = project_root / "experiment.json"
    _write_spec(
        spec_path,
        [_write_stage("write", "artifact.txt")],
        inputs=[{"name": "input", "path": "input"}],
    )

    with pytest.raises(ValueError, match="must not contain symlinks"):
        prepare_experiment(
            spec_path,
            profile="smoke",
            project_root=project_root,
        )


def test_declared_output_directories_cannot_contain_symlinks(
    tmp_path: Path,
) -> None:
    code = (
        "from pathlib import Path; import os; "
        "root=Path(os.environ['DKENERGY_EXPERIMENT_RUN_DIR']); "
        "(root/'bundle').mkdir(); "
        "(root/'target.txt').write_text('target'); "
        "(root/'bundle'/'linked.txt').symlink_to(root/'target.txt')"
    )
    project_root, spec_path = _make_project(
        tmp_path,
        stages=[
            {
                "name": "write-symlink",
                "argv": [sys.executable, "-c", code],
                "outputs": ["bundle"],
            }
        ],
    )
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )

    with pytest.raises(ValueError, match="must not contain symlinks"):
        run_experiment_worker(prepared.run_dir)


@pytest.mark.parametrize("mode", ["timeout", "failure", "success"])
def test_stage_process_groups_do_not_leave_descendants_running(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    ready = tmp_path / f"{mode}-child-ready"
    survivor = tmp_path / f"{mode}-child-survived"
    child_code = (
        "import signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(ready)!r}).write_text('ready')\n"
        "time.sleep(0.6)\n"
        f"Path({str(survivor)!r}).write_text('survived')\n"
    )
    parent_ending = {
        "timeout": "time.sleep(60)\n",
        "failure": "raise SystemExit(7)\n",
        "success": "",
    }[mode]
    parent_code = (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"ready = Path({str(ready)!r})\n"
        "deadline = time.monotonic() + 5\n"
        "while not ready.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "if not ready.exists():\n"
        "    raise SystemExit(99)\n"
        f"{parent_ending}"
    )
    stage: dict[str, object] = {
        "name": f"descendant-{mode}",
        "argv": [sys.executable, "-c", parent_code],
    }
    if mode == "timeout":
        stage["timeout_seconds"] = 0.2
    project_root, spec_path = _make_project(tmp_path, stages=[stage])
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )
    monkeypatch.setattr(
        runner_module,
        "PROCESS_GROUP_TERMINATION_GRACE_SECONDS",
        0.05,
    )

    if mode == "timeout":
        with pytest.raises(subprocess.TimeoutExpired):
            run_experiment_worker(prepared.run_dir)
    elif mode == "failure":
        with pytest.raises(subprocess.CalledProcessError):
            run_experiment_worker(prepared.run_dir)
    else:
        assert run_experiment_worker(prepared.run_dir).state == "completed"

    assert ready.exists()
    time.sleep(0.7)
    assert not survivor.exists()


def test_ambient_environment_is_hashed_without_persisting_secret_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_value = "top-secret-test-value"
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret_value)
    project_root, spec_path = _make_project(
        tmp_path,
        stages=[_write_stage("write", "artifact.txt")],
    )
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )

    identity_text = (prepared.run_dir / "identity.json").read_text()
    assert secret_value not in identity_text
    identity = json.loads(identity_text)
    ambient = identity["identity_payload"]["environment"]["ambient_environment"]
    assert ambient["AWS_SECRET_ACCESS_KEY"] != secret_value

    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "changed-secret-test-value")
    with pytest.raises(IntegrityError, match="AWS_SECRET_ACCESS_KEY"):
        run_experiment_worker(prepared.run_dir)


def test_stage_executable_drift_is_detected_before_execution(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    executable = project_root / "stage-tool"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    spec_path = project_root / "experiment.json"
    _write_spec(
        spec_path,
        [{"name": "tool", "argv": [str(executable)]}],
    )
    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )
    identity = json.loads((prepared.run_dir / "identity.json").read_text())
    executable_record = identity["identity_payload"]["stage_executables"][0]
    assert executable_record["invocation_path"] == str(executable.absolute())
    assert executable_record["resolved_path"] == str(executable.resolve())
    assert executable_record["sha256"]

    executable.write_text("#!/bin/sh\nexit 9\n")
    with pytest.raises(IntegrityError, match="stage executables changed"):
        run_experiment_worker(prepared.run_dir)


def test_stage_execution_preserves_a_symlinked_invocation_path(
    tmp_path: Path,
) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    python_link = project_root / "python-link"
    python_link.symlink_to(sys.executable)
    code = (
        "from pathlib import Path; import os; "
        "Path(os.environ['DKENERGY_EXPERIMENT_RUN_DIR'], 'artifact.txt')"
        ".write_text('done')"
    )
    spec_path = project_root / "experiment.json"
    _write_spec(
        spec_path,
        [
            {
                "name": "linked-python",
                "argv": [str(python_link), "-c", code],
                "outputs": ["artifact.txt"],
            }
        ],
    )

    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
    )
    run_experiment_worker(prepared.run_dir)

    receipt = json.loads(
        (prepared.run_dir / "stages" / "001-linked-python.json").read_text()
    )
    assert receipt["argv"][0] == str(python_link.absolute())
    assert receipt["executable"]["resolved_path"] == str(
        python_link.resolve()
    )


def test_failed_initialization_never_exposes_a_partial_final_run_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, spec_path = _make_project(
        tmp_path,
        stages=[_write_stage("write", "artifact.txt")],
    )
    original_write = runner_module._write_json_atomic

    def fail_during_initialization(
        path: Path,
        value: dict[str, object],
    ) -> None:
        if path.name == "context.json":
            raise OSError("simulated initialization failure")
        original_write(path, value)

    monkeypatch.setattr(
        runner_module,
        "_write_json_atomic",
        fail_during_initialization,
    )

    with pytest.raises(OSError, match="simulated initialization failure"):
        prepare_experiment(
            spec_path,
            profile="smoke",
            project_root=project_root,
        )

    experiment_parent = project_root / "results" / "experiments" / "test-pipeline"
    assert experiment_parent.exists()
    assert list(experiment_parent.iterdir()) == []


def test_custom_in_project_results_roots_must_use_results_tree(
    tmp_path: Path,
) -> None:
    project_root, spec_path = _make_project(
        tmp_path,
        stages=[_write_stage("write", "artifact.txt")],
    )

    with pytest.raises(ValueError, match="ignored results/ tree"):
        prepare_experiment(
            spec_path,
            profile="smoke",
            project_root=project_root,
            results_root=project_root / "custom-results",
        )

    prepared = prepare_experiment(
        spec_path,
        profile="smoke",
        project_root=project_root,
        results_root=project_root / "results" / "custom",
    )
    assert prepared.run_dir.is_relative_to(project_root / "results" / "custom")


def test_results_root_cannot_overlap_declared_inputs(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    shared = tmp_path / "shared"
    shared.mkdir()
    input_path = shared / "input.txt"
    input_path.write_text("input")
    spec_path = project_root / "experiment.json"
    _write_spec(
        spec_path,
        [_write_stage("write", "artifact.txt")],
        inputs=[{"name": "input", "path": str(input_path)}],
    )

    with pytest.raises(ValueError, match="must not overlap a declared input"):
        prepare_experiment(
            spec_path,
            profile="smoke",
            project_root=project_root,
            results_root=shared,
        )


def _make_project(
    tmp_path: Path,
    *,
    stages: list[dict[str, object]],
) -> tuple[Path, Path]:
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "pyproject.toml").write_text("[project]\nname='test'\n")
    spec_path = project_root / "experiment.json"
    _write_spec(spec_path, stages)
    return project_root, spec_path


def _write_spec(
    path: Path,
    stages: list[dict[str, object]],
    *,
    inputs: list[dict[str, object]] | None = None,
) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "experiment_spec_v1",
                "name": "test-pipeline",
                "kind": "command_pipeline",
                "inputs": inputs or [],
                "profiles": {
                    "smoke": {"stages": stages},
                    "full": {"stages": stages},
                },
            }
        )
    )


def _write_stage(
    name: str,
    output: str,
    *,
    content: str = "ok",
) -> dict[str, object]:
    code = (
        "from pathlib import Path; import os; "
        f"Path(os.environ['DKENERGY_EXPERIMENT_RUN_DIR'], {output!r}).write_text({content!r})"
    )
    return {
        "name": name,
        "argv": [sys.executable, "-c", code],
        "outputs": [output],
    }
