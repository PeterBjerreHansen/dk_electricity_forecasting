#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.experiments import (  # noqa: E402
    launch_detached,
    prepare_experiment,
    run_experiment_worker,
    verify_completed_run,
)
from dkenergy_forecast.experiments.runner import read_experiment_status  # noqa: E402


def main() -> None:
    args = parse_args()
    if args.command == "run" and args.profile == "full":
        raise SystemExit(
            "Full experiment profiles cannot run in the foreground; use "
            "`python scripts/run_experiment.py launch <spec> --profile full`."
        )
    if args.command in {"prepare", "run", "launch"}:
        prepared = prepare_experiment(
            args.spec,
            profile=args.profile,
            project_root=args.project_root,
            results_root=args.results_root,
        )
        if args.command == "prepare":
            _print_run(prepared)
            return
        if args.command == "run":
            result = (
                prepared
                if prepared.no_op
                else run_experiment_worker(prepared.run_dir)
            )
            _print_run(result)
            return
        result = (
            prepared
            if prepared.no_op
            else launch_detached(
                prepared.run_dir,
                worker_script=Path(__file__),
            )
        )
        _print_run(result)
        return

    if args.command == "worker":
        run_experiment_worker(args.run_dir)
        return
    if args.command == "status":
        print(
            json.dumps(
                read_experiment_status(args.run_dir),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "verify":
        manifest = verify_completed_run(args.run_dir)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return
    raise AssertionError(f"Unhandled command: {args.command}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare, resume, or detach reproducible command-pipeline experiments. "
            "This runner never promotes a model artifact."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("prepare", "Resolve and persist an experiment without executing it."),
        (
            "run",
            "Prepare and run/resume a non-full experiment in the foreground.",
        ),
        ("launch", "Prepare and launch a detached local worker immediately."),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("spec", help="Path to an experiment_spec_v1 JSON file.")
        child.add_argument(
            "--profile",
            required=True,
            help="Profile to resolve, normally smoke or full.",
        )
        child.add_argument(
            "--project-root",
            default=str(ROOT),
            help="Project root used for code and relative input provenance.",
        )
        child.add_argument(
            "--results-root",
            help="Override the default <project-root>/results/experiments root.",
        )

    worker = subparsers.add_parser(
        "worker",
        help="Internal worker process used by detached launches.",
    )
    worker.add_argument("--run-dir", required=True)
    status = subparsers.add_parser("status", help="Read a run's status artifact.")
    status.add_argument("--run-dir", required=True)
    verify = subparsers.add_parser(
        "verify",
        help="Verify a completed manifest, receipts, and declared outputs.",
    )
    verify.add_argument("--run-dir", required=True)
    return parser.parse_args()


def _print_run(run: object) -> None:
    print(
        json.dumps(
            {
                "experiment_id": run.experiment_id,
                "run_dir": str(run.run_dir),
                "state": run.state,
                "no_op": run.no_op,
                "pid": run.pid,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
