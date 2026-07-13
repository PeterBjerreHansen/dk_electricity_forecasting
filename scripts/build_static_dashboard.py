#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.static_dashboard import build_static_dashboard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained static forecast dashboard.")
    parser.add_argument("--input", required=True, help="forecast_dashboard.json input path")
    parser.add_argument("--output", required=True, help="HTML output path")
    parser.add_argument("--title", default="Danish Electricity Forecasts")
    args = parser.parse_args()

    source = Path(args.input)
    destination = Path(args.output)
    payload = json.loads(source.read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(build_static_dashboard(payload, title=args.title), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
