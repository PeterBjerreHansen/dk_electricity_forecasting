from __future__ import annotations

import calendar
import hashlib
import json
import re
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

import requests


BASE_URL = "https://api.energidataservice.dk"
RETRYABLE_STATUS_CODES = {500, 502, 503, 504}


class EnergiDataServiceError(RuntimeError):
    """Raised when Energi Data Service cannot satisfy a request."""


@dataclass(frozen=True)
class ApiResponse:
    url: str
    params: dict[str, Any]
    status_code: int
    content: bytes
    payload: dict[str, Any]
    retrieved_at_utc: datetime
    response_sha256: str


@dataclass(frozen=True)
class RawWriteResult:
    batch_id: str
    path: Path
    manifest_entry: dict[str, Any]


class EnergiDataServiceClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        max_rate_limit_sleep_seconds: int = 300,
        retry_backoff_seconds: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.max_rate_limit_sleep_seconds = max_rate_limit_sleep_seconds
        self.retry_backoff_seconds = retry_backoff_seconds
        self.session = session or requests.Session()

    def fetch_metadata(self, dataset: str) -> ApiResponse:
        return self._get(f"/meta/dataset/{dataset}", params={})

    def fetch_dataset(self, dataset: str, params: Mapping[str, Any]) -> ApiResponse:
        return self._get(f"/dataset/{dataset}", params=dict(params))

    def _get(self, path: str, params: dict[str, Any]) -> ApiResponse:
        url = f"{self.base_url}{path}"
        prepared = requests.Request("GET", url, params=params).prepare()
        request_url = prepared.url or url

        for attempt in range(self.max_retries + 1):
            retrieved_at = datetime.now(timezone.utc)
            try:
                response = self.session.get(
                    url,
                    params=params,
                    timeout=self.timeout_seconds,
                    headers={"Accept": "application/json"},
                )
            except requests.RequestException as exc:
                if attempt < self.max_retries:
                    time.sleep(_retry_wait_seconds(attempt, self.retry_backoff_seconds))
                    continue
                raise EnergiDataServiceError(
                    f"EDS request failed after {attempt + 1} attempt(s): {exc}"
                ) from exc

            if response.status_code == 429 and attempt < self.max_retries:
                wait_seconds = _rate_limit_wait_seconds(response)
                if wait_seconds > self.max_rate_limit_sleep_seconds:
                    raise EnergiDataServiceError(
                        "EDS rate limit wait exceeds configured cap: "
                        f"{wait_seconds}s > {self.max_rate_limit_sleep_seconds}s"
                    )
                time.sleep(wait_seconds)
                continue

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                time.sleep(_retry_wait_seconds(attempt, self.retry_backoff_seconds))
                continue

            if not response.ok:
                raise EnergiDataServiceError(
                    f"EDS request failed with HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise EnergiDataServiceError("EDS response was not valid JSON") from exc

            return ApiResponse(
                url=request_url,
                params=dict(params),
                status_code=response.status_code,
                content=response.content,
                payload=payload,
                retrieved_at_utc=retrieved_at,
                response_sha256=hashlib.sha256(response.content).hexdigest(),
            )

        raise EnergiDataServiceError("EDS request failed after retries")


def dataset_params(
    *,
    start: date,
    end: date,
    areas: list[str],
    columns: list[str],
    time_column: str,
) -> dict[str, Any]:
    return {
        "start": start.isoformat(),
        "end": end.isoformat(),
        "filter": json.dumps({"PriceArea": areas}, separators=(",", ":")),
        "columns": ",".join(columns),
        "sort": f"{time_column},PriceArea",
        "limit": 0,
    }


def iter_date_chunks(
    start: date,
    end: date,
    months_per_chunk: int = 1,
) -> Iterator[tuple[date, date]]:
    if months_per_chunk < 1:
        raise ValueError("months_per_chunk must be >= 1")
    if start >= end:
        return

    cursor = start
    while cursor < end:
        next_month = _add_months(date(cursor.year, cursor.month, 1), months_per_chunk)
        chunk_end = min(end, next_month)
        yield cursor, chunk_end
        cursor = chunk_end


def write_metadata_response(raw_root: Path, dataset: str, response: ApiResponse) -> Path:
    timestamp = _format_timestamp_for_path(response.retrieved_at_utc)
    path = raw_root / "metadata" / f"{dataset}_{timestamp}.json"
    _write_bytes(path, response.content)
    return path


def write_dataset_response(
    raw_root: Path,
    dataset: str,
    start: date,
    end: date,
    response: ApiResponse,
) -> RawWriteResult:
    timestamp = _format_timestamp_for_path(response.retrieved_at_utc)
    path = (
        raw_root
        / dataset
        / f"fetched_at={timestamp}"
        / f"start={start.isoformat()}_end={end.isoformat()}.json"
    )
    _write_bytes(path, response.content)

    saved_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    batch_id = f"{dataset}_{start.isoformat()}_{end.isoformat()}_{saved_sha256[:12]}"
    records = response.payload.get("records")
    record_count = len(records) if isinstance(records, list) else None
    manifest_entry = {
        "batch_id": batch_id,
        "source_dataset": dataset,
        "request_url": response.url,
        "request_params": response.params,
        "retrieved_at_utc": response.retrieved_at_utc.isoformat(),
        "http_status": response.status_code,
        "record_count": record_count,
        "response_sha256": response.response_sha256,
        "saved_json_sha256": saved_sha256,
        "raw_path": str(path.relative_to(raw_root)),
    }
    _append_jsonl(raw_root / "manifest.jsonl", manifest_entry)
    return RawWriteResult(batch_id=batch_id, path=path, manifest_entry=manifest_entry)


def _rate_limit_wait_seconds(response: requests.Response) -> int:
    retry_after = response.headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        return max(1, int(retry_after))

    match = re.search(r"Try again in (\d+) seconds", response.text)
    if match:
        return max(1, int(match.group(1)))

    return 60


def _retry_wait_seconds(attempt: int, backoff_seconds: float) -> float:
    return max(0.0, backoff_seconds * (2**attempt))


def _format_timestamp_for_path(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
