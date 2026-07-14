#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


def main() -> None:
    command = sys.argv[1:] or ["pipeline"]
    mode = command[0]
    if mode == "pipeline":
        _exec([sys.executable, "scripts/run_cloud_pipeline.py", *command[1:]])
    if mode == "score-published-cloud":
        _exec([sys.executable, "scripts/run_cloud_scoring.py", *command[1:]])
    raise SystemExit(f"Unknown container command: {mode}")


def _exec(command: list[str]) -> None:
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
