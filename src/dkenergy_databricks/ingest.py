from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import pandas as pd
import requests

from dkenergy_data.build.eds_prices_v1 import SOURCE_SPECS, load_raw_batches
from dkenergy_data.price_windows import iter_price_source_windows
from dkenergy_data.sources.energidataservice import (
    EnergiDataServiceClient,
    EnergiDataServiceError,
    dataset_params,
    write_dataset_response,
    write_metadata_response,
)


LandingStatus = Literal["fetched", "up_to_date", "api_unreachable"]

BRONZE_COLUMNS = [
    "batch_id",
    "source_dataset",
    "retrieved_at_utc",
    "raw_path",
    "record_index",
    "record_json",
]


@dataclass(frozen=True)
class LandingResult:
    status: LandingStatus
    fetched_batches: int
    skipped_batches: int
    record_count: int
    error: str | None = None


def default_fetch_range(today_local: date, *, history_days: int = 450) -> tuple[date, date]:
    """Inclusive start and exclusive end of the price history kept in the landing zone.

    The end is tomorrow so today's delivery prices (published yesterday) are
    included; tomorrow's prices are not known before the forecast origin.
    """

    start = (today_local - timedelta(days=history_days)).replace(day=1)
    return start, today_local + timedelta(days=1)


def land_eds_prices(
    raw_root: Path,
    *,
    start: date,
    end: date,
    chunk_months: int = 1,
    areas: Iterable[str] = ("DK1", "DK2"),
    client: EnergiDataServiceClient | None = None,
) -> LandingResult:
    """Fetch raw EDS responses that are not yet in ``raw_root``.

    Files use the same layout and manifest as the local pipeline, so the
    existing panel builder reads them unchanged. Network failures are reported
    rather than raised: on Databricks Free Edition outbound access is limited,
    and the raw files can then be landed from outside (see the GitHub workflow).
    """

    client = client or EnergiDataServiceClient(max_retries=1)
    windows = iter_price_source_windows(start, end, chunk_months)
    pending = [
        window for window in windows
        if not _window_already_landed(raw_root, *window)
    ]
    skipped = len(windows) - len(pending)
    if not pending:
        return LandingResult("up_to_date", 0, skipped, 0)

    fetched = 0
    records = 0
    try:
        for dataset in sorted({dataset for dataset, _, _ in pending}):
            write_metadata_response(raw_root, dataset, client.fetch_metadata(dataset))
        for dataset, window_start, window_end in pending:
            spec = SOURCE_SPECS[dataset]
            response = client.fetch_dataset(
                dataset,
                dataset_params(
                    start=window_start,
                    end=window_end,
                    areas=list(areas),
                    columns=spec.columns,
                    time_column=spec.time_column,
                ),
            )
            result = write_dataset_response(raw_root, dataset, window_start, window_end, response)
            fetched += 1
            records += result.manifest_entry["record_count"] or 0
    except (EnergiDataServiceError, requests.RequestException) as exc:
        return LandingResult("api_unreachable", fetched, skipped, records, error=str(exc))
    return LandingResult("fetched", fetched, skipped, records)


def bronze_records_frame(
    raw_root: Path,
    *,
    known_batch_ids: Iterable[str] = (),
) -> pd.DataFrame:
    """One row per raw API record for batches not yet in the bronze table."""

    known = set(known_batch_ids)
    rows: list[dict[str, object]] = []
    for dataset in SOURCE_SPECS:
        for batch in load_raw_batches(raw_root, dataset):
            if batch.batch_id in known:
                continue
            known.add(batch.batch_id)
            for index, record in enumerate(batch.records):
                rows.append(
                    {
                        "batch_id": batch.batch_id,
                        "source_dataset": batch.source_dataset,
                        "retrieved_at_utc": batch.retrieved_at_utc,
                        "raw_path": str(batch.raw_path.relative_to(raw_root)),
                        "record_index": index,
                        "record_json": json.dumps(record, ensure_ascii=False, sort_keys=True),
                    }
                )
    return pd.DataFrame(rows, columns=BRONZE_COLUMNS)


def _window_already_landed(raw_root: Path, dataset: str, start: date, end: date) -> bool:
    pattern = f"fetched_at=*/start={start.isoformat()}_end={end.isoformat()}.json"
    return any((raw_root / dataset).glob(pattern))
