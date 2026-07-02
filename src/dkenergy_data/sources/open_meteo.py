from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import requests


BASE_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class OpenMeteoError(RuntimeError):
    """Raised when Open-Meteo cannot satisfy a request."""


@dataclass(frozen=True)
class OpenMeteoResponse:
    url: str
    params: dict[str, Any]
    status_code: int
    content: bytes
    payload: dict[str, Any]
    retrieved_at_utc: datetime
    response_sha256: str


@dataclass(frozen=True)
class OpenMeteoRawWriteResult:
    batch_id: str
    path: Path
    manifest_entry: dict[str, Any]


class OpenMeteoClient:
    def __init__(
        self,
        base_url: str = BASE_URL,
        timeout_seconds: float = 60.0,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.session = session or requests.Session()

    def fetch_previous_runs(self, params: Mapping[str, Any]) -> OpenMeteoResponse:
        return self._get(dict(params))

    def _get(self, params: dict[str, Any]) -> OpenMeteoResponse:
        prepared = requests.Request("GET", self.base_url, params=params).prepare()
        request_url = prepared.url or self.base_url

        for attempt in range(self.max_retries + 1):
            retrieved_at = datetime.now(timezone.utc)
            try:
                response = self.session.get(
                    self.base_url,
                    params=params,
                    timeout=self.timeout_seconds,
                    headers={"Accept": "application/json"},
                )
            except requests.RequestException as exc:
                if attempt < self.max_retries:
                    time.sleep(_retry_wait_seconds(attempt, self.retry_backoff_seconds))
                    continue
                raise OpenMeteoError(
                    f"Open-Meteo request failed after {attempt + 1} attempt(s): {exc}"
                ) from exc

            if response.status_code in RETRYABLE_STATUS_CODES and attempt < self.max_retries:
                time.sleep(_retry_wait_seconds(attempt, self.retry_backoff_seconds))
                continue

            if not response.ok:
                raise OpenMeteoError(
                    f"Open-Meteo request failed with HTTP {response.status_code}: "
                    f"{response.text[:500]}"
                )

            try:
                payload = response.json()
            except ValueError as exc:
                raise OpenMeteoError("Open-Meteo response was not valid JSON") from exc

            if payload.get("error"):
                raise OpenMeteoError(f"Open-Meteo returned an error payload: {payload}")

            return OpenMeteoResponse(
                url=request_url,
                params=dict(params),
                status_code=response.status_code,
                content=response.content,
                payload=payload,
                retrieved_at_utc=retrieved_at,
                response_sha256=hashlib.sha256(response.content).hexdigest(),
            )

        raise OpenMeteoError("Open-Meteo request failed after retries")


def previous_runs_params(
    *,
    latitude: float,
    longitude: float,
    start: date,
    end: date,
    weather_model: str,
    base_variables: list[str],
    lead_time_days: list[int],
) -> dict[str, Any]:
    hourly = [
        f"{variable}_previous_day{lead_day}"
        for variable in base_variables
        for lead_day in lead_time_days
    ]
    return {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "hourly": ",".join(hourly),
        "models": weather_model,
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }


def write_previous_runs_response(
    raw_root: Path,
    *,
    weather_model: str,
    location_id: str,
    start: date,
    end: date,
    response: OpenMeteoResponse,
) -> OpenMeteoRawWriteResult:
    timestamp = _format_timestamp_for_path(response.retrieved_at_utc)
    path = (
        raw_root
        / "previous_runs"
        / weather_model
        / location_id
        / f"fetched_at={timestamp}"
        / f"start={start.isoformat()}_end={end.isoformat()}.json"
    )
    _write_bytes(path, response.content)

    saved_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    batch_id = (
        f"open_meteo_previous_runs_{weather_model}_{location_id}_"
        f"{start.isoformat()}_{end.isoformat()}_{saved_sha256[:12]}"
    )
    hourly = response.payload.get("hourly")
    row_count = len(hourly.get("time", [])) if isinstance(hourly, dict) else None
    manifest_entry = {
        "batch_id": batch_id,
        "source_provider": "open_meteo",
        "source_product": "previous_runs",
        "weather_model": weather_model,
        "location_id": location_id,
        "request_url": response.url,
        "request_params": response.params,
        "retrieved_at_utc": response.retrieved_at_utc.isoformat(),
        "http_status": response.status_code,
        "row_count": row_count,
        "response_sha256": response.response_sha256,
        "saved_json_sha256": saved_sha256,
        "raw_path": str(path.relative_to(raw_root)),
    }
    _append_jsonl(raw_root / "manifest.jsonl", manifest_entry)
    return OpenMeteoRawWriteResult(
        batch_id=batch_id,
        path=path,
        manifest_entry=manifest_entry,
    )


def _retry_wait_seconds(attempt: int, backoff_seconds: float) -> float:
    return max(0.0, backoff_seconds * (2**attempt))


def _format_timestamp_for_path(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
