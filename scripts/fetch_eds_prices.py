#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_data.build.eds_prices_v1 import SOURCE_SPECS, STITCH_BOUNDARY_DATE
from dkenergy_data.sources.energidataservice import (
    EnergiDataServiceClient,
    dataset_params,
    iter_date_chunks,
    write_dataset_response,
    write_metadata_response,
)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_dir)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if start >= end:
        raise SystemExit("--start must be before --end")

    client = EnergiDataServiceClient(
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        max_rate_limit_sleep_seconds=args.max_rate_limit_sleep_seconds,
    )
    windows = list(iter_price_source_windows(start, end, args.chunk_months))
    datasets = sorted({dataset for dataset, _, _ in windows})

    print(f"Fetching EDS metadata for: {', '.join(datasets)}")
    for dataset in datasets:
        response = client.fetch_metadata(dataset)
        path = write_metadata_response(raw_root, dataset, response)
        print(f"  metadata {dataset}: {path}")

    total_records = 0
    skipped_batches = 0
    for dataset, window_start, window_end in windows:
        valid_existing, invalid_existing = partition_existing_raw_batch_paths(
            raw_root,
            dataset,
            window_start,
            window_end,
        )
        for path, reason in invalid_existing:
            print(
                f"  ignore invalid existing {dataset} {window_start} -> {window_end}: "
                f"{path} ({reason})",
                flush=True,
            )
        if valid_existing and not args.force:
            skipped_batches += 1
            print(
                f"  skip {dataset} {window_start} -> {window_end}: "
                f"{len(valid_existing)} valid existing raw file(s)",
                flush=True,
            )
            continue

        spec = SOURCE_SPECS[dataset]
        params = dataset_params(
            start=window_start,
            end=window_end,
            areas=args.areas,
            columns=spec.columns,
            time_column=spec.time_column,
        )
        response = client.fetch_dataset(dataset, params)
        result = write_dataset_response(raw_root, dataset, window_start, window_end, response)
        record_count = result.manifest_entry["record_count"] or 0
        total_records += record_count
        print(
            f"  {dataset} {window_start} -> {window_end}: "
            f"{record_count} rows, {result.path}",
            flush=True,
        )
        if args.sleep_between_requests_seconds:
            time.sleep(args.sleep_between_requests_seconds)

    print(
        f"Fetched {total_records} records across "
        f"{len(windows) - skipped_batches} new data batches; skipped {skipped_batches}.",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    default_end = (
        datetime.now(ZoneInfo("Europe/Copenhagen")).date()
        + timedelta(days=1)
    ).isoformat()
    parser = argparse.ArgumentParser(description="Fetch raw DK1/DK2 EDS price data.")
    parser.add_argument("--start", default="1999-07-01", help="Inclusive local EDS start date.")
    parser.add_argument("--end", default=default_end, help="Exclusive local EDS end date.")
    parser.add_argument("--areas", nargs="+", default=["DK1", "DK2"], help="Price areas to fetch.")
    parser.add_argument("--chunk-months", type=int, default=3, help="Months per API request.")
    parser.add_argument(
        "--raw-dir",
        default=str(ROOT / "data" / "raw" / "energi_data_service"),
        help="Raw EDS output directory.",
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--max-rate-limit-sleep-seconds", type=int, default=300)
    parser.add_argument("--sleep-between-requests-seconds", type=float, default=0.25)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fetch and write a new raw batch even when the same dataset/start/end exists.",
    )
    return parser.parse_args()


def iter_price_source_windows(
    start: date,
    end: date,
    chunk_months: int,
) -> list[tuple[str, date, date]]:
    windows: list[tuple[str, date, date]] = []

    if start < STITCH_BOUNDARY_DATE:
        elspot_end = min(end, STITCH_BOUNDARY_DATE)
        for window_start, window_end in iter_date_chunks(start, elspot_end, chunk_months):
            windows.append(("Elspotprices", window_start, window_end))

    if end > STITCH_BOUNDARY_DATE:
        day_ahead_start = max(start, STITCH_BOUNDARY_DATE)
        for window_start, window_end in iter_date_chunks(day_ahead_start, end, chunk_months):
            windows.append(("DayAheadPrices", window_start, window_end))

    return windows


def existing_raw_batch_paths(
    raw_root: Path,
    dataset: str,
    start: date,
    end: date,
) -> list[Path]:
    pattern = f"fetched_at=*/start={start.isoformat()}_end={end.isoformat()}.json"
    return sorted((raw_root / dataset).glob(pattern))


def partition_existing_raw_batch_paths(
    raw_root: Path,
    dataset: str,
    start: date,
    end: date,
) -> tuple[list[Path], list[tuple[Path, str]]]:
    manifest_hashes = manifest_hashes_by_raw_path(raw_root)
    valid: list[Path] = []
    invalid: list[tuple[Path, str]] = []

    for path in existing_raw_batch_paths(raw_root, dataset, start, end):
        reason = invalid_raw_batch_reason(path, manifest_hashes.get(path))
        if reason:
            invalid.append((path, reason))
        else:
            valid.append(path)
    return valid, invalid


def invalid_raw_batch_reason(path: Path, expected_sha256: str | None) -> str | None:
    try:
        content = path.read_bytes()
    except OSError as exc:
        return f"cannot read file: {exc}"

    if expected_sha256:
        actual_sha256 = hashlib.sha256(content).hexdigest()
        if actual_sha256 != expected_sha256:
            return f"manifest hash mismatch: expected {expected_sha256}, got {actual_sha256}"

    try:
        payload = json.loads(content)
    except ValueError as exc:
        return f"invalid JSON: {exc}"

    records = payload.get("records") if isinstance(payload, dict) else None
    if not isinstance(records, list):
        return "payload does not contain a records list"
    return None


def manifest_hashes_by_raw_path(raw_root: Path) -> dict[Path, str]:
    manifest_path = raw_root / "manifest.jsonl"
    if not manifest_path.exists():
        return {}

    hashes: dict[Path, str] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            raw_path = Path(entry["raw_path"])
            if not raw_path.is_absolute():
                raw_path = raw_root / raw_path
            expected_sha256 = entry.get("saved_json_sha256") or entry.get("response_sha256")
            if expected_sha256:
                hashes[raw_path] = str(expected_sha256)
    return hashes


if __name__ == "__main__":
    main()
