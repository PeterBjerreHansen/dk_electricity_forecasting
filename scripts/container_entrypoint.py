#!/usr/bin/env python3
from __future__ import annotations

import os
import sys


def main() -> None:
    command = sys.argv[1:] or ["web"]
    mode = command[0]
    if mode == "web":
        port = os.environ.get("PORT") or os.environ.get("STREAMLIT_PORT") or "8501"
        _exec(
            [
                "streamlit",
                "run",
                "app/streamlit_app.py",
                "--server.address=0.0.0.0",
                f"--server.port={port}",
                "--server.headless=true",
            ]
        )
    if mode == "pipeline":
        _exec([sys.executable, "scripts/run_cloud_pipeline.py", *command[1:]])
    if mode == "score-published-cloud":
        _exec([sys.executable, "scripts/run_cloud_scoring.py", *command[1:]])
    _exec(command)


def _exec(command: list[str]) -> None:
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
