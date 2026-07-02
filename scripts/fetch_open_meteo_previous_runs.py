#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_data.build.open_meteo_weather_v1 import (  # noqa: E402
    BASE_VARIABLES,
    LEAD_TIME_DAYS,
    OPEN_METEO_LOCATION_BASKET,
    OPEN_METEO_MODELS,
    WeatherLocation,
)
from dkenergy_data.sources.energidataservice import iter_date_chunks  # noqa: E402
from dkenergy_data.sources.open_meteo import (  # noqa: E402
    OpenMeteoClient,
    previous_runs_params,
    write_previous_runs_response,
)


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_dir)
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    if start > end:
        raise SystemExit("--start must be on or before --end")

    models = args.models
    locations = selected_locations(args.locations)
    lead_days = args.lead_time_days
    base_variables = args.variables
    client = OpenMeteoClient(
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
    )
    windows = [
        (window_start, window_end_exclusive - timedelta(days=1))
        for window_start, window_end_exclusive in iter_date_chunks(
            start,
            end + timedelta(days=1),
            args.chunk_months,
        )
    ]

    fetched = 0
    skipped = 0
    for model in models:
        for location in locations:
            for window_start, window_end in windows:
                if existing_raw_batch_paths(raw_root, model, location.location_id, window_start, window_end) and not args.force:
                    skipped += 1
                    print(
                        f"  skip {model} {location.location_id} {window_start} -> {window_end}",
                        flush=True,
                    )
                    continue
                params = previous_runs_params(
                    latitude=location.latitude,
                    longitude=location.longitude,
                    start=window_start,
                    end=window_end,
                    weather_model=model,
                    base_variables=base_variables,
                    lead_time_days=lead_days,
                )
                response = client.fetch_previous_runs(params)
                result = write_previous_runs_response(
                    raw_root,
                    weather_model=model,
                    location_id=location.location_id,
                    start=window_start,
                    end=window_end,
                    response=response,
                )
                fetched += 1
                row_count = result.manifest_entry.get("row_count")
                print(
                    f"  {model} {location.location_id} {window_start} -> {window_end}: "
                    f"{row_count} hourly rows, {result.path}",
                    flush=True,
                )
                if args.sleep_between_requests_seconds:
                    time.sleep(args.sleep_between_requests_seconds)

    print(f"Fetched {fetched} Open-Meteo batches; skipped {skipped}.", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch raw Open-Meteo Previous Runs forecasts.")
    parser.add_argument("--start", default="2024-07-01", help="Inclusive start date.")
    parser.add_argument("--end", required=True, help="Inclusive end date.")
    parser.add_argument("--models", nargs="+", default=list(OPEN_METEO_MODELS))
    parser.add_argument("--variables", nargs="+", default=list(BASE_VARIABLES))
    parser.add_argument("--lead-time-days", nargs="+", type=int, default=list(LEAD_TIME_DAYS))
    parser.add_argument(
        "--locations",
        nargs="+",
        help="Optional location ids from the v1 basket. Defaults to all DK1/DK2 basket points.",
    )
    parser.add_argument("--chunk-months", type=int, default=1)
    parser.add_argument(
        "--raw-dir",
        default=str(ROOT / "data" / "raw" / "open_meteo"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=1.0)
    parser.add_argument("--sleep-between-requests-seconds", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def selected_locations(location_ids: list[str] | None) -> list[WeatherLocation]:
    if not location_ids:
        return list(OPEN_METEO_LOCATION_BASKET)
    by_id = {location.location_id: location for location in OPEN_METEO_LOCATION_BASKET}
    missing = sorted(set(location_ids) - set(by_id))
    if missing:
        raise SystemExit(f"Unknown Open-Meteo location ids: {missing}")
    return [by_id[location_id] for location_id in location_ids]


def existing_raw_batch_paths(
    raw_root: Path,
    weather_model: str,
    location_id: str,
    start: date,
    end: date,
) -> list[Path]:
    pattern = (
        f"previous_runs/{weather_model}/{location_id}/fetched_at=*/"
        f"start={start.isoformat()}_end={end.isoformat()}.json"
    )
    candidates = sorted(raw_root.glob(pattern))
    return [path for path in candidates if _looks_like_open_meteo_payload(path)]


def _looks_like_open_meteo_payload(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    times = hourly.get("time") if isinstance(hourly, dict) else None
    return isinstance(times, list)


if __name__ == "__main__":
    main()
