from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping


ENVIRONMENT_PACKAGES = (
    "dk-energy-forecasts",
    "pandas",
    "pyarrow",
    "chronos-forecasting",
    "torch",
    "transformers",
    "peft",
    "catboost",
)
RUNNER_CONTROLLED_ENVIRONMENT_KEYS = {
    "COLUMNS",
    "DKENERGY_EXPERIMENT_ID",
    "DKENERGY_EXPERIMENT_PROFILE",
    "DKENERGY_EXPERIMENT_RUN_DIR",
    "LINES",
    "PYTHONUNBUFFERED",
}


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def hash_path(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    reject_path_symlinks(candidate)
    if not candidate.exists():
        return {"exists": False, "kind": "missing", "sha256": None}
    if candidate.is_file():
        return {
            "exists": True,
            "kind": "file",
            "sha256": _file_sha256(candidate),
            "size_bytes": candidate.stat().st_size,
        }
    if candidate.is_dir():
        entries = []
        total_size = 0
        for file_path in sorted(
            (value for value in candidate.rglob("*") if value.is_file()),
            key=lambda value: value.relative_to(candidate).as_posix(),
        ):
            relative = file_path.relative_to(candidate).as_posix()
            size = file_path.stat().st_size
            total_size += size
            entries.append(
                {
                    "path": relative,
                    "sha256": _file_sha256(file_path),
                    "size_bytes": size,
                }
            )
        return {
            "exists": True,
            "kind": "directory",
            "sha256": canonical_sha256(entries),
            "file_count": len(entries),
            "size_bytes": total_size,
        }
    raise ValueError(f"Cannot hash unsupported path type: {candidate}")


def reject_path_symlinks(
    path: str | Path,
    *,
    recursive: bool = True,
) -> None:
    """Reject a symlink at the path, in an existing ancestor, or below a directory."""

    candidate = Path(os.path.abspath(Path(path)))
    for current in (candidate, *candidate.parents):
        if current.is_symlink():
            raise ValueError(f"Experiment paths must not use symlinks: {current}")
    if not recursive or not candidate.is_dir():
        return
    for directory, directory_names, file_names in os.walk(
        candidate,
        followlinks=False,
    ):
        directory_path = Path(directory)
        for name in [*directory_names, *file_names]:
            nested = directory_path / name
            if nested.is_symlink():
                raise ValueError(
                    f"Experiment paths must not contain symlinks: {nested}"
                )


def hash_declared_inputs(
    inputs: Iterable[Mapping[str, Any]],
    *,
    project_root: str | Path,
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    hashed = []
    for entry in inputs:
        declared_path = str(entry["path"])
        path = Path(declared_path)
        resolved = path if path.is_absolute() else root / path
        required = bool(entry.get("required", True))
        digest = hash_path(resolved)
        if required and not digest["exists"]:
            raise FileNotFoundError(
                f"Required experiment input {entry['name']!r} does not exist: {resolved}"
            )
        hashed.append(
            {
                "name": str(entry["name"]),
                "path": declared_path,
                "required": required,
                **digest,
            }
        )
    return hashed


def environment_fingerprint(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    ambient = os.environ if environ is None else environ
    return {
        "python_implementation": platform.python_implementation(),
        "python_version": platform.python_version(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "package_versions": {
            package: _package_version(package)
            for package in ENVIRONMENT_PACKAGES
        },
        "ambient_environment": {
            key: hashlib.sha256(str(value).encode("utf-8")).hexdigest()
            for key, value in sorted(ambient.items())
            if key not in RUNNER_CONTROLLED_ENVIRONMENT_KEYS
        },
    }


def resolve_stage_executables(
    effective_spec: Mapping[str, Any],
    *,
    project_root: str | Path,
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Resolve and fingerprint the executable used by every experiment stage."""

    root = Path(project_root).resolve()
    ambient = dict(os.environ if environ is None else environ)
    context = {
        "project_root": str(root),
        "profile": str(effective_spec["profile"]),
    }
    global_env = {
        key: _expand_known(value, context)
        for key, value in effective_spec.get("env", {}).items()
    }
    records = []
    for stage in effective_spec["stages"]:
        stage_env = dict(ambient)
        stage_env.update(global_env)
        stage_env.update(
            {
                key: _expand_known(value, context)
                for key, value in stage.get("env", {}).items()
            }
        )
        cwd_text = _expand_known(stage.get("cwd", "{project_root}"), context)
        cwd = _resolve_stage_cwd(cwd_text, project_root=root)
        requested = _expand_known(str(stage["argv"][0]), context)
        executable = _resolve_executable(
            requested,
            cwd=cwd,
            path_value=stage_env.get("PATH", os.defpath),
        )
        invocation = Path(os.path.abspath(executable))
        resolved = executable.resolve(strict=True)
        digest = hash_path(resolved)
        records.append(
            {
                "stage_name": str(stage["name"]),
                "requested": requested,
                "invocation_path": str(invocation),
                "resolved_path": str(resolved),
                "mode": stat.S_IMODE(resolved.stat().st_mode),
                **digest,
            }
        )
    return records


def code_fingerprint(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        head = _git(root, "rev-parse", "HEAD").decode("ascii").strip()
        status = _git(
            root,
            "status",
            "--porcelain=v1",
            "-z",
            "--untracked-files=all",
        )
        diff = _git(root, "diff", "--binary", "HEAD", "--", ".")
    except (FileNotFoundError, subprocess.CalledProcessError, UnicodeDecodeError):
        return _fallback_code_fingerprint(root)

    untracked = _untracked_paths(status)
    untracked_hashes = []
    for relative in untracked:
        candidate = root / relative
        if candidate.exists():
            untracked_hashes.append(
                {"path": relative, **hash_path(candidate)}
            )
    details = {
        "method": "git_worktree_v1",
        "git_head": head,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(),
        "untracked": untracked_hashes,
    }
    return {**details, "fingerprint": canonical_sha256(details)}


def _git(root: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    ).stdout


def _untracked_paths(status: bytes) -> list[str]:
    paths = []
    for record in status.split(b"\0"):
        if record.startswith(b"?? "):
            paths.append(os.fsdecode(record[3:]))
    return sorted(paths)


def _fallback_code_fingerprint(root: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    for relative in ("src", "scripts"):
        directory = root / relative
        if directory.exists():
            candidates.extend(
                value for value in directory.rglob("*.py") if value.is_file()
            )
    for relative in ("pyproject.toml", "constraints-production.txt"):
        candidate = root / relative
        if candidate.exists():
            candidates.append(candidate)
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": _file_sha256(path),
        }
        for path in sorted(set(candidates))
    ]
    details = {
        "method": "source_tree_v1",
        "dirty": None,
        "files": entries,
    }
    return {**details, "fingerprint": canonical_sha256(details)}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return version(package)
    except PackageNotFoundError:
        return None


def _expand_known(value: str, context: Mapping[str, str]) -> str:
    expanded = value
    for key, replacement in context.items():
        expanded = expanded.replace("{" + key + "}", replacement)
    return expanded


def _resolve_stage_cwd(value: str, *, project_root: Path) -> Path | None:
    if "{" in value or "}" in value:
        return None
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    return candidate.resolve()


def _resolve_executable(
    requested: str,
    *,
    cwd: Path | None,
    path_value: str,
) -> Path:
    if "{" in requested or "}" in requested:
        raise ValueError(
            "Stage executable cannot depend on unresolved experiment templates: "
            f"{requested!r}"
        )
    if os.sep in requested or (os.altsep and os.altsep in requested):
        candidate = Path(requested)
        if not candidate.is_absolute():
            if cwd is None:
                raise ValueError(
                    "Relative stage executable requires a statically resolvable cwd: "
                    f"{requested!r}"
                )
            candidate = cwd / candidate
        return _require_executable(candidate, requested=requested)

    if cwd is None and any(
        not Path(entry or ".").is_absolute()
        for entry in path_value.split(os.pathsep)
    ):
        raise ValueError(
            "Stage executable uses a relative PATH entry with a dynamic cwd: "
            f"{requested!r}"
        )
    for entry in path_value.split(os.pathsep):
        directory = Path(entry or ".")
        if not directory.is_absolute():
            assert cwd is not None
            directory = cwd / directory
        candidate = directory / requested
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    raise FileNotFoundError(
        f"Experiment stage executable is not available on PATH: {requested!r}"
    )


def _require_executable(candidate: Path, *, requested: str) -> Path:
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise FileNotFoundError(
            f"Experiment stage executable is missing or not executable: "
            f"{requested!r} -> {candidate}"
        )
    return candidate
