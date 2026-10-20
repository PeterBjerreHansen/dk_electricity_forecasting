from __future__ import annotations

import argparse
from datetime import date, datetime
from zoneinfo import ZoneInfo

from dkenergy_databricks.lakehouse import Lakehouse


def parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--schema", required=True)
    parser.add_argument(
        "--run-date",
        default="",
        help="Copenhagen run date (YYYY-MM-DD); defaults to today. Use for backfills.",
    )
    return parser


def lakehouse(args: argparse.Namespace) -> Lakehouse:
    return Lakehouse(catalog=args.catalog, schema=args.schema)


def run_date(args: argparse.Namespace) -> date:
    if args.run_date:
        return date.fromisoformat(args.run_date)
    return datetime.now(ZoneInfo("Europe/Copenhagen")).date()
