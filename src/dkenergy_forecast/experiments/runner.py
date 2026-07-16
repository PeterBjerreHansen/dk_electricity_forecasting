from __future__ import annotations

import fcntl
import json
import os
import signal
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from dkenergy_forecast.experiments.provenance import (
    canonical_sha256,
    code_fingerprint,
    environment_fingerprint,
    hash_declared_inputs,
    hash_path,
    reject_path_symlinks,
    resolve_stage_executables,
)
from dkenergy_forecast.experiments.spec import load_experiment_spec


IDENTITY_SCHEMA_VERSION = "experiment_identity_v1"
STATUS_SCHEMA_VERSION = "experiment_status_v1"
RECEIPT_SCHEMA_VERSION = "experiment_stage_receipt_v1"
MANIFEST_SCHEMA_VERSION = "experiment_manifest_v1"
COMPLETION_SCHEMA_VERSION = "experiment_completion_v1"
RESERVED_RUN_PATHS = {
    ".worker.lock",
    "COMPLETED.json",
    "context.json",
    "effective_spec.json",
    "identity.json",
    "logs",
    "manifest.json",
    "stages",
    "status.json",
}
PROCESS_GROUP_TERMINATION_GRACE_SECONDS = 1.0


class IntegrityError(RuntimeError):
    pass


class AlreadyRunningError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExperimentRun:
    experiment_id: str
    run_dir: Path
    state: str
    no_op: bool = False
    pid: int | None = None


def prepare_experiment(
    spec_path: str | Path,
    *,
    profile: str,
    project_root: str | Path,
    results_root: str | Path | None = None,
    code: Mapping[str, Any] | None = None,
    environment: Mapping[str, str] | None = None,
) -> ExperimentRun:
    root = Path(project_root).resolve()
    resolved = load_experiment_spec(spec_path, profile=profile)
    experiments_root = _resolve_experiments_root(
        results_root,
        project_root=root,
        declared_inputs=resolved.effective.get("inputs", []),
    )
    input_hashes = hash_declared_inputs(
        resolved.effective.get("inputs", []),
        project_root=root,
    )
    code_record = dict(code) if code is not None else code_fingerprint(root)
    if profile == "full" and code_record.get("dirty") is True:
        raise ValueError(
            "Full experiments require a clean Git worktree or immutable source "
            "identity; commit or stash source changes first"
        )
    environment_record = (
        dict(environment) if environment is not None else environment_fingerprint()
    )
    stage_executables = resolve_stage_executables(
        resolved.effective,
        project_root=root,
    )
    identity_payload = {
        "effective_spec": resolved.effective,
        "code": code_record,
        "inputs": input_hashes,
        "environment": environment_record,
        "stage_executables": stage_executables,
    }
    identity_sha256 = canonical_sha256(identity_payload)
    experiment_id = f"{resolved.name}-{identity_sha256[:20]}"
    identity = {
        "schema_version": IDENTITY_SCHEMA_VERSION,
        "experiment_id": experiment_id,
        "identity_sha256": identity_sha256,
        "identity_payload": identity_payload,
    }

    run_dir = experiments_root / resolved.name / experiment_id
    reject_path_symlinks(run_dir)
    context = {
        "schema_version": "experiment_context_v1",
        "project_root": str(root),
        "source_spec_path": str(resolved.source_path.resolve()),
        "source_spec_sha256": resolved.source_sha256,
        "profile": profile,
    }

    if _initialize_run_directory(
        run_dir,
        identity=identity,
        effective_spec=resolved.effective,
        context=context,
    ):
        return ExperimentRun(experiment_id, run_dir, "prepared")

    saved_identity = _read_json(run_dir / "identity.json")
    if saved_identity != identity:
        raise IntegrityError(
            f"Experiment ID collision or altered identity in {run_dir}"
        )
    _verify_effective_spec(run_dir, identity)
    if (run_dir / "COMPLETED.json").exists():
        verify_completed_run(run_dir)
        return ExperimentRun(experiment_id, run_dir, "completed", no_op=True)
    state = str(_read_json(run_dir / "status.json").get("state"))
    return ExperimentRun(experiment_id, run_dir, state)


def run_experiment_worker(run_dir: str | Path) -> ExperimentRun:
    path = _resolve_run_directory(run_dir)
    identity = _read_json(path / "identity.json")
    experiment_id = str(identity["experiment_id"])
    _verify_effective_spec(path, identity)

    with _worker_lock(path):
        if (path / "COMPLETED.json").exists():
            verify_completed_run(path)
            return ExperimentRun(experiment_id, path, "completed", no_op=True)

        context = _read_json(path / "context.json")
        project_root = Path(context["project_root"]).resolve()
        effective_spec = identity["identity_payload"]["effective_spec"]
        previous_status = _read_json(path / "status.json")
        attempt = int(previous_status.get("attempt", 0)) + 1
        started_at = _utc_now()
        _write_status(
            path,
            {
                "state": "running",
                "created_at_utc": previous_status.get("created_at_utc"),
                "started_at_utc": started_at,
                "attempt": attempt,
                "pid": os.getpid(),
            },
        )

        try:
            _verify_runtime_provenance(
                identity=identity,
                effective_spec=effective_spec,
                project_root=project_root,
            )
            receipts = _run_stages(
                run_dir=path,
                project_root=project_root,
                experiment_id=experiment_id,
                effective_spec=effective_spec,
                identity=identity,
            )
            _verify_runtime_provenance(
                identity=identity,
                effective_spec=effective_spec,
                project_root=project_root,
            )
            _verify_stage_receipt_snapshot(
                run_dir=path,
                experiment_id=experiment_id,
                stages=effective_spec["stages"],
                stage_executables=identity["identity_payload"][
                    "stage_executables"
                ],
                receipt_entries=receipts,
            )
        except BaseException as exc:
            _write_status(
                path,
                {
                    "state": "failed",
                    "created_at_utc": previous_status.get("created_at_utc"),
                    "started_at_utc": started_at,
                    "failed_at_utc": _utc_now(),
                    "attempt": attempt,
                    "pid": os.getpid(),
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    },
                },
            )
            raise

        completed_at = _utc_now()
        final_status = {
            "state": "completed",
            "created_at_utc": previous_status.get("created_at_utc"),
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "attempt": attempt,
            "pid": os.getpid(),
        }
        _write_status(path, final_status)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "status": "completed",
            "profile": effective_spec["profile"],
            "kind": effective_spec["kind"],
            "name": effective_spec["name"],
            "identity_sha256": identity["identity_sha256"],
            "effective_spec_sha256": canonical_sha256(effective_spec),
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "stage_receipts": [
                {
                    "stage_name": receipt["stage_name"],
                    "path": receipt["path"],
                    "sha256": receipt["sha256"],
                }
                for receipt in receipts
            ],
        }
        manifest_path = path / "manifest.json"
        _write_json_atomic(manifest_path, manifest)
        completion = {
            "schema_version": COMPLETION_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "status": "completed",
            "manifest_sha256": hash_path(manifest_path)["sha256"],
            "completed_at_utc": completed_at,
        }
        _write_json_atomic(path / "COMPLETED.json", completion)
        return ExperimentRun(experiment_id, path, "completed")


def launch_detached(
    run_dir: str | Path,
    *,
    worker_script: str | Path,
    python_executable: str | Path = sys.executable,
) -> ExperimentRun:
    path = _resolve_run_directory(run_dir)
    identity = _read_json(path / "identity.json")
    experiment_id = str(identity["experiment_id"])
    if (path / "COMPLETED.json").exists():
        verify_completed_run(path)
        return ExperimentRun(experiment_id, path, "completed", no_op=True)
    if _worker_lock_is_held(path):
        return ExperimentRun(
            experiment_id,
            path,
            "running",
            no_op=True,
        )

    context = _read_json(path / "context.json")
    log_path = path / "logs" / "worker.log"
    log_handle = log_path.open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            [
                str(python_executable),
                str(Path(worker_script).resolve()),
                "worker",
                "--run-dir",
                str(path),
            ],
            cwd=context["project_root"],
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    finally:
        log_handle.close()
    return ExperimentRun(
        experiment_id,
        path,
        "prepared",
        pid=process.pid,
    )


def verify_completed_run(run_dir: str | Path) -> dict[str, Any]:
    path = _resolve_run_directory(run_dir)
    identity = _read_json(path / "identity.json")
    _verify_effective_spec(path, identity)
    completion = _read_json(path / "COMPLETED.json")
    manifest_path = path / "manifest.json"
    manifest = _read_json(manifest_path)
    expected_manifest_sha = completion.get("manifest_sha256")
    actual_manifest_sha = hash_path(manifest_path)["sha256"]
    if expected_manifest_sha != actual_manifest_sha:
        raise IntegrityError("Completed experiment manifest hash does not match")
    experiment_id = identity.get("experiment_id")
    if (
        completion.get("schema_version") != COMPLETION_SCHEMA_VERSION
        or completion.get("status") != "completed"
        or completion.get("experiment_id") != experiment_id
    ):
        raise IntegrityError("Invalid experiment completion marker")
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("status") != "completed"
        or manifest.get("experiment_id") != experiment_id
        or manifest.get("identity_sha256") != identity.get("identity_sha256")
    ):
        raise IntegrityError("Invalid experiment manifest")
    status = _read_json(path / "status.json")
    if status.get("state") != "completed":
        raise IntegrityError("Completed experiment status is not completed")

    effective_spec = identity["identity_payload"]["effective_spec"]
    stages = effective_spec["stages"]
    stage_executables = identity["identity_payload"]["stage_executables"]
    receipt_entries = manifest.get("stage_receipts")
    if (
        not isinstance(receipt_entries, list)
        or not isinstance(stage_executables, list)
        or len(receipt_entries) != len(stages)
        or len(stage_executables) != len(stages)
    ):
        raise IntegrityError("Experiment manifest stage receipt count is invalid")
    for index, (stage, executable, entry) in enumerate(
        zip(stages, stage_executables, receipt_entries, strict=True),
        start=1,
    ):
        receipt_path = path / str(entry.get("path"))
        if hash_path(receipt_path)["sha256"] != entry.get("sha256"):
            raise IntegrityError(
                f"Stage receipt hash does not match for {stage['name']!r}"
            )
        _verify_stage_receipt(
            run_dir=path,
            experiment_id=str(experiment_id),
            stage_index=index,
            stage=stage,
            executable=executable,
            receipt_path=receipt_path,
        )
    return manifest


def _verify_stage_receipt_snapshot(
    *,
    run_dir: Path,
    experiment_id: str,
    stages: Sequence[Mapping[str, Any]],
    stage_executables: Sequence[Mapping[str, Any]],
    receipt_entries: Sequence[Mapping[str, str]],
) -> None:
    if (
        len(receipt_entries) != len(stages)
        or len(stage_executables) != len(stages)
    ):
        raise IntegrityError("Experiment stage receipt count is invalid")
    for index, (stage, executable, entry) in enumerate(
        zip(stages, stage_executables, receipt_entries, strict=True),
        start=1,
    ):
        expected_path = _stage_receipt_path(
            run_dir,
            index,
            str(stage["name"]),
        )
        recorded_path = entry.get("path")
        if (
            not isinstance(recorded_path, str)
            or (run_dir / recorded_path).resolve() != expected_path.resolve()
        ):
            raise IntegrityError(
                f"Stage receipt path is invalid for {stage['name']!r}"
            )
        actual_sha = hash_path(expected_path)["sha256"]
        if actual_sha != entry.get("sha256"):
            raise IntegrityError(
                f"Stage receipt hash does not match for {stage['name']!r}"
            )
        _verify_stage_receipt(
            run_dir=run_dir,
            experiment_id=experiment_id,
            stage_index=index,
            stage=stage,
            executable=executable,
            receipt_path=expected_path,
        )


def read_experiment_status(run_dir: str | Path) -> dict[str, Any]:
    path = _resolve_run_directory(run_dir)
    status = _read_json(path / "status.json")
    status["run_dir"] = str(path)
    return status


def _resolve_run_directory(run_dir: str | Path) -> Path:
    candidate = Path(run_dir)
    reject_path_symlinks(candidate)
    return candidate.resolve()


def _verify_runtime_provenance(
    *,
    identity: Mapping[str, Any],
    effective_spec: Mapping[str, Any],
    project_root: Path,
) -> None:
    payload = identity["identity_payload"]
    actual_inputs = hash_declared_inputs(
        effective_spec.get("inputs", []),
        project_root=project_root,
    )
    if actual_inputs != payload.get("inputs"):
        raise IntegrityError("Experiment inputs changed after preparation")
    if code_fingerprint(project_root) != payload.get("code"):
        raise IntegrityError("Project code changed after experiment preparation")
    actual_environment = environment_fingerprint()
    expected_environment = payload.get("environment")
    if actual_environment != expected_environment:
        changed_keys = _changed_environment_keys(
            expected_environment,
            actual_environment,
        )
        raise IntegrityError(
            "Execution environment changed after experiment preparation; "
            f"changed_keys={changed_keys}"
        )
    actual_executables = resolve_stage_executables(
        effective_spec,
        project_root=project_root,
    )
    if actual_executables != payload.get("stage_executables"):
        raise IntegrityError(
            "Experiment stage executables changed after preparation"
        )


def _run_stages(
    *,
    run_dir: Path,
    project_root: Path,
    experiment_id: str,
    effective_spec: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> list[dict[str, str]]:
    receipts = []
    template_context = {
        "run_dir": str(run_dir),
        "project_root": str(project_root),
        "experiment_id": experiment_id,
        "profile": str(effective_spec["profile"]),
    }
    _validate_resolved_stage_outputs(
        run_dir=run_dir,
        stages=effective_spec["stages"],
        template_context=template_context,
    )
    stage_executables = identity["identity_payload"]["stage_executables"]
    if len(stage_executables) != len(effective_spec["stages"]):
        raise IntegrityError("Experiment stage executable count is invalid")
    for index, stage in enumerate(effective_spec["stages"], start=1):
        executable = stage_executables[index - 1]
        if executable.get("stage_name") != stage.get("name"):
            raise IntegrityError("Experiment stage executable order is invalid")
        _verify_runtime_provenance(
            identity=identity,
            effective_spec=effective_spec,
            project_root=project_root,
        )
        receipt_path = _stage_receipt_path(run_dir, index, str(stage["name"]))
        relative_receipt = receipt_path.relative_to(run_dir).as_posix()
        if receipt_path.exists():
            _verify_stage_receipt(
                run_dir=run_dir,
                experiment_id=experiment_id,
                stage_index=index,
                stage=stage,
                executable=executable,
                receipt_path=receipt_path,
            )
            receipts.append(
                {
                    "stage_name": str(stage["name"]),
                    "path": relative_receipt,
                    "sha256": str(hash_path(receipt_path)["sha256"]),
                }
            )
            continue

        argv = [_expand(value, template_context) for value in stage["argv"]]
        # Preserve the invocation path because Python virtual environments and
        # other launchers can derive runtime semantics from it. The executable
        # record separately binds that path to its resolved target and hash.
        argv[0] = str(executable["invocation_path"])
        cwd_text = _expand(stage.get("cwd", "{project_root}"), template_context)
        cwd = Path(cwd_text)
        if not cwd.is_absolute():
            cwd = project_root / cwd
        cwd = cwd.resolve()
        if not cwd.is_dir():
            raise FileNotFoundError(f"Stage cwd does not exist: {cwd}")

        output_records = _resolve_outputs(
            run_dir=run_dir,
            outputs=stage.get("outputs", []),
            template_context=template_context,
        )
        for record in output_records:
            Path(record["absolute_path"]).parent.mkdir(parents=True, exist_ok=True)
        _clear_uncommitted_outputs(run_dir, output_records)

        attempt = _next_stage_attempt(run_dir, index, str(stage["name"]))
        log_path = (
            run_dir
            / "logs"
            / f"{index:03d}-{stage['name']}.attempt-{attempt:03d}.log"
        )
        env = os.environ.copy()
        env.update(
            {
                "DKENERGY_EXPERIMENT_RUN_DIR": str(run_dir),
                "DKENERGY_EXPERIMENT_ID": experiment_id,
                "DKENERGY_EXPERIMENT_PROFILE": str(effective_spec["profile"]),
                "PYTHONUNBUFFERED": "1",
            }
        )
        env.update(
            {
                key: _expand(value, template_context)
                for key, value in effective_spec.get("env", {}).items()
            }
        )
        env.update(
            {
                key: _expand(value, template_context)
                for key, value in stage.get("env", {}).items()
            }
        )
        started_at = _utc_now()
        started = time.monotonic()
        with log_path.open("wb") as log_handle:
            _run_stage_process(
                argv,
                cwd=cwd,
                env=env,
                log_handle=log_handle,
                timeout=stage.get("timeout_seconds"),
            )
        _verify_runtime_provenance(
            identity=identity,
            effective_spec=effective_spec,
            project_root=project_root,
        )
        completed_at = _utc_now()

        hashed_outputs = []
        for record in output_records:
            output_path = _safe_run_output(
                run_dir,
                Path(record["absolute_path"]),
            )
            output_hash = hash_path(output_path)
            if not output_hash["exists"]:
                raise IntegrityError(
                    f"Stage {stage['name']!r} did not produce declared output "
                    f"{record['declared_path']!r}"
                )
            hashed_outputs.append(
                {
                    "declared_path": record["declared_path"],
                    "path": record["path"],
                    **output_hash,
                }
            )
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "experiment_id": experiment_id,
            "stage_index": index,
            "stage_name": stage["name"],
            "stage_spec_sha256": canonical_sha256(stage),
            "executable": executable,
            "argv": argv,
            "cwd": str(cwd),
            "log_path": log_path.relative_to(run_dir).as_posix(),
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "elapsed_seconds": round(time.monotonic() - started, 6),
            "outputs": hashed_outputs,
        }
        _write_json_atomic(receipt_path, receipt)
        receipts.append(
            {
                "stage_name": str(stage["name"]),
                "path": relative_receipt,
                "sha256": str(hash_path(receipt_path)["sha256"]),
            }
        )
    return receipts


def _validate_resolved_stage_outputs(
    *,
    run_dir: Path,
    stages: Sequence[Mapping[str, Any]],
    template_context: Mapping[str, str],
) -> None:
    resolved: list[tuple[str, Path]] = []
    for stage in stages:
        for record in _resolve_outputs(
            run_dir=run_dir,
            outputs=stage.get("outputs", []),
            template_context=template_context,
        ):
            candidate = Path(record["absolute_path"])
            for previous_name, previous in resolved:
                if candidate == previous or candidate in previous.parents or previous in candidate.parents:
                    raise ValueError(
                        "Experiment stage outputs cannot overlap or nest: "
                        f"{previous_name!r}={previous}, "
                        f"{stage['name']!r}={candidate}"
                    )
            resolved.append((str(stage["name"]), candidate))


def _clear_uncommitted_outputs(
    run_dir: Path,
    output_records: Sequence[Mapping[str, str]],
) -> None:
    for record in output_records:
        output_path = _safe_run_output(run_dir, Path(record["absolute_path"]))
        if output_path.is_dir():
            shutil.rmtree(output_path)
        elif output_path.exists() or output_path.is_symlink():
            output_path.unlink()
        output_path.parent.mkdir(parents=True, exist_ok=True)


def _verify_stage_receipt(
    *,
    run_dir: Path,
    experiment_id: str,
    stage_index: int,
    stage: Mapping[str, Any],
    executable: Mapping[str, Any],
    receipt_path: Path,
) -> dict[str, Any]:
    receipt = _read_json(receipt_path)
    if (
        receipt.get("schema_version") != RECEIPT_SCHEMA_VERSION
        or receipt.get("experiment_id") != experiment_id
        or receipt.get("stage_index") != stage_index
        or receipt.get("stage_name") != stage.get("name")
        or receipt.get("stage_spec_sha256") != canonical_sha256(stage)
        or receipt.get("executable") != executable
    ):
        raise IntegrityError(f"Invalid stage receipt: {receipt_path}")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, list):
        raise IntegrityError(f"Stage receipt outputs are invalid: {receipt_path}")
    declared_outputs = stage.get("outputs", [])
    if [output.get("declared_path") for output in outputs] != declared_outputs:
        raise IntegrityError(
            f"Stage receipt declared outputs are invalid: {receipt_path}"
        )
    for output in outputs:
        relative = output.get("path")
        if not isinstance(relative, str):
            raise IntegrityError(f"Stage receipt output path is invalid: {receipt_path}")
        output_path = _safe_run_output(run_dir, run_dir / relative)
        actual = hash_path(output_path)
        expected = {
            key: value
            for key, value in output.items()
            if key not in {"declared_path", "path"}
        }
        if actual != expected:
            raise IntegrityError(
                f"Stage output hash does not match receipt: {output_path}"
            )
    return receipt


def _resolve_outputs(
    *,
    run_dir: Path,
    outputs: Sequence[str],
    template_context: Mapping[str, str],
) -> list[dict[str, str]]:
    records = []
    for declared in outputs:
        expanded = _expand(declared, template_context)
        candidate = Path(expanded)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        safe_path = _safe_run_output(run_dir, candidate)
        records.append(
            {
                "declared_path": declared,
                "absolute_path": str(safe_path),
                "path": safe_path.relative_to(run_dir.resolve()).as_posix(),
            }
        )
    return records


def _safe_run_output(run_dir: Path, candidate: Path) -> Path:
    root = run_dir.resolve()
    reject_path_symlinks(candidate)
    resolved = Path(os.path.abspath(candidate))
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(
            "Experiment outputs must remain below the run directory and cannot "
            f"replace it: {resolved}"
        )
    relative = resolved.relative_to(root)
    if relative.parts[0] in RESERVED_RUN_PATHS:
        raise ValueError(
            f"Experiment output uses reserved run path: {relative.as_posix()}"
        )
    return resolved


def _run_stage_process(
    argv: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    log_handle: Any,
    timeout: float | None,
) -> None:
    process = subprocess.Popen(
        argv,
        cwd=cwd,
        env=dict(env),
        stdin=subprocess.DEVNULL,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        close_fds=True,
    )
    try:
        return_code = process.wait(timeout=timeout)
    except BaseException:
        _terminate_process_group(process)
        raise
    if return_code:
        _terminate_process_group(process)
        raise subprocess.CalledProcessError(return_code, list(argv))
    if _process_group_exists(process.pid):
        _terminate_process_group(process)


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    process_group = process.pid
    _signal_process_group(process_group, signal.SIGTERM)
    deadline = time.monotonic() + PROCESS_GROUP_TERMINATION_GRACE_SECONDS
    while time.monotonic() < deadline:
        process.poll()
        if not _process_group_exists(process_group):
            break
        time.sleep(0.02)
    if _process_group_exists(process_group):
        _signal_process_group(process_group, signal.SIGKILL)
    try:
        process.wait(timeout=PROCESS_GROUP_TERMINATION_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        _signal_process_group(process_group, signal.SIGKILL)
        process.wait()
    deadline = time.monotonic() + PROCESS_GROUP_TERMINATION_GRACE_SECONDS
    while (
        _process_group_exists(process_group)
        and time.monotonic() < deadline
    ):
        time.sleep(0.02)


def _signal_process_group(process_group: int, signal_number: int) -> None:
    try:
        os.killpg(process_group, signal_number)
    except ProcessLookupError:
        return


def _process_group_exists(process_group: int) -> bool:
    try:
        os.killpg(process_group, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _resolve_experiments_root(
    results_root: str | Path | None,
    *,
    project_root: Path,
    declared_inputs: Sequence[Mapping[str, Any]],
) -> Path:
    if results_root is None:
        candidate = project_root / "results" / "experiments"
    else:
        candidate = Path(results_root)
    reject_path_symlinks(candidate, recursive=False)
    resolved = Path(os.path.abspath(candidate)).resolve()
    if resolved == project_root or resolved in project_root.parents:
        raise ValueError(
            "Experiment results root cannot be the project root or one of its "
            f"ancestors: {resolved}"
        )
    if results_root is not None and resolved.is_relative_to(project_root):
        conventional = (project_root / "results").resolve()
        if resolved != conventional and not resolved.is_relative_to(conventional):
            raise ValueError(
                "An in-project experiment results root must remain under the "
                f"ignored results/ tree: {resolved}"
            )
    for entry in declared_inputs:
        declared = Path(str(entry["path"]))
        if not declared.is_absolute():
            declared = project_root / declared
        reject_path_symlinks(declared, recursive=False)
        input_path = Path(os.path.abspath(declared)).resolve()
        if (
            input_path == resolved
            or input_path.is_relative_to(resolved)
            or resolved.is_relative_to(input_path)
        ):
            raise ValueError(
                "Experiment results root must not overlap a declared input: "
                f"results_root={resolved}, input={input_path}"
            )
    return resolved


def _changed_environment_keys(
    expected: Any,
    actual: Mapping[str, Any],
) -> list[str]:
    if not isinstance(expected, Mapping):
        return ["<environment-record>"]
    changed = [
        key
        for key in sorted(set(expected) | set(actual))
        if key != "ambient_environment" and expected.get(key) != actual.get(key)
    ]
    expected_ambient = expected.get("ambient_environment")
    actual_ambient = actual.get("ambient_environment")
    if not isinstance(expected_ambient, Mapping):
        changed.append("ambient_environment")
        return changed
    if not isinstance(actual_ambient, Mapping):
        changed.append("ambient_environment")
        return changed
    changed.extend(
        f"ambient_environment.{key}"
        for key in sorted(set(expected_ambient) | set(actual_ambient))
        if expected_ambient.get(key) != actual_ambient.get(key)
    )
    return changed


def _initialize_run_directory(
    run_dir: Path,
    *,
    identity: Mapping[str, Any],
    effective_spec: Mapping[str, Any],
    context: Mapping[str, Any],
) -> bool:
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    if run_dir.exists():
        return False
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{run_dir.name}.init-",
            dir=run_dir.parent,
        )
    )
    try:
        (temporary / "logs").mkdir()
        (temporary / "stages").mkdir()
        _write_json_atomic(temporary / "identity.json", identity)
        _write_json_atomic(temporary / "effective_spec.json", effective_spec)
        _write_json_atomic(temporary / "context.json", context)
        _write_status(
            temporary,
            {
                "state": "prepared",
                "created_at_utc": _utc_now(),
                "attempt": 0,
            },
        )
        try:
            os.rename(temporary, run_dir)
        except OSError:
            if run_dir.exists():
                return False
            raise
        return True
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _verify_effective_spec(
    run_dir: Path,
    identity: Mapping[str, Any],
) -> None:
    effective = _read_json(run_dir / "effective_spec.json")
    if effective != identity["identity_payload"]["effective_spec"]:
        raise IntegrityError("Saved effective spec does not match experiment identity")
    if canonical_sha256(identity["identity_payload"]) != identity.get("identity_sha256"):
        raise IntegrityError("Experiment identity hash does not match its payload")


@contextmanager
def _worker_lock(run_dir: Path) -> Iterator[None]:
    lock_path = run_dir / ".worker.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise AlreadyRunningError(
                f"Another worker already owns {lock_path}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _worker_lock_is_held(run_dir: Path) -> bool:
    lock_path = run_dir / ".worker.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    return False


def _write_status(run_dir: Path, fields: Mapping[str, Any]) -> None:
    _write_json_atomic(
        run_dir / "status.json",
        {
            "schema_version": STATUS_SCHEMA_VERSION,
            "experiment_id": _read_json(run_dir / "identity.json")["experiment_id"],
            **fields,
        },
    )


def _stage_receipt_path(run_dir: Path, index: int, name: str) -> Path:
    return run_dir / "stages" / f"{index:03d}-{name}.json"


def _next_stage_attempt(run_dir: Path, index: int, name: str) -> int:
    matches = list((run_dir / "logs").glob(f"{index:03d}-{name}.attempt-*.log"))
    return len(matches) + 1


def _expand(value: str, context: Mapping[str, str]) -> str:
    expanded = value
    for key, replacement in context.items():
        expanded = expanded.replace("{" + key + "}", replacement)
    return expanded


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise IntegrityError(f"Missing experiment artifact: {path}") from exc
    except json.JSONDecodeError as exc:
        raise IntegrityError(f"Invalid JSON experiment artifact: {path}") from exc
    if not isinstance(value, dict):
        raise IntegrityError(f"Experiment artifact must be an object: {path}")
    return value


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    temporary.write_text(payload, encoding="utf-8")
    os.replace(temporary, path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
