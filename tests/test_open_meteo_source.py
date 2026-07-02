from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone

import pytest
import requests

from dkenergy_data.sources.open_meteo import (
    OpenMeteoClient,
    OpenMeteoError,
    OpenMeteoResponse,
    previous_runs_params,
    write_previous_runs_response,
)


def test_previous_runs_params_use_units_and_explicit_lead_variables() -> None:
    params = previous_runs_params(
        latitude=55.6761,
        longitude=12.5683,
        start=date(2025, 1, 1),
        end=date(2025, 1, 2),
        weather_model="icon_eu",
        base_variables=["temperature_2m", "wind_speed_10m"],
        lead_time_days=[1, 2],
    )

    assert params == {
        "latitude": 55.6761,
        "longitude": 12.5683,
        "start_date": "2025-01-01",
        "end_date": "2025-01-02",
        "hourly": (
            "temperature_2m_previous_day1,"
            "temperature_2m_previous_day2,"
            "wind_speed_10m_previous_day1,"
            "wind_speed_10m_previous_day2"
        ),
        "models": "icon_eu",
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }


def test_write_previous_runs_response_preserves_raw_bytes_and_manifest_hash(tmp_path) -> None:
    content = b'{"hourly":{"time":["2025-01-01T00:00"]},"generationtime_ms":1.2}'
    response = OpenMeteoResponse(
        url="https://example.test",
        params={"models": "gfs_global"},
        status_code=200,
        content=content,
        payload=json.loads(content.decode("utf-8")),
        retrieved_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        response_sha256=hashlib.sha256(content).hexdigest(),
    )

    result = write_previous_runs_response(
        tmp_path,
        weather_model="gfs_global",
        location_id="dk1_aarhus",
        start=date(2025, 1, 1),
        end=date(2025, 1, 1),
        response=response,
    )

    assert result.path.read_bytes() == content
    assert result.manifest_entry["response_sha256"] == hashlib.sha256(content).hexdigest()
    assert result.manifest_entry["saved_json_sha256"] == hashlib.sha256(content).hexdigest()
    manifest_entry = json.loads((tmp_path / "manifest.jsonl").read_text(encoding="utf-8"))
    assert manifest_entry["raw_path"] == str(result.path.relative_to(tmp_path))


def test_open_meteo_client_retries_5xx_statuses() -> None:
    session = FakeSession(
        [
            FakeResponse(503, {"reason": "temporary"}),
            FakeResponse(200, {"hourly": {"time": []}}),
        ]
    )
    client = OpenMeteoClient(
        base_url="https://example.test",
        max_retries=1,
        retry_backoff_seconds=0,
        session=session,
    )

    response = client.fetch_previous_runs({"models": "gfs_global"})

    assert response.status_code == 200
    assert response.payload == {"hourly": {"time": []}}
    assert session.call_count == 2


def test_open_meteo_client_retries_request_exceptions() -> None:
    session = FakeSession(
        [
            requests.Timeout("temporary timeout"),
            FakeResponse(200, {"hourly": {"time": []}}),
        ]
    )
    client = OpenMeteoClient(
        base_url="https://example.test",
        max_retries=1,
        retry_backoff_seconds=0,
        session=session,
    )

    response = client.fetch_previous_runs({"models": "icon_eu"})

    assert response.payload == {"hourly": {"time": []}}
    assert session.call_count == 2


def test_open_meteo_client_raises_after_retries_are_exhausted() -> None:
    session = FakeSession([requests.ConnectionError("connection reset")])
    client = OpenMeteoClient(
        base_url="https://example.test",
        max_retries=0,
        retry_backoff_seconds=0,
        session=session,
    )

    with pytest.raises(OpenMeteoError, match="failed after 1 attempt"):
        client.fetch_previous_runs({"models": "metno_nordic"})


class FakeSession:
    def __init__(self, responses: list[FakeResponse | requests.RequestException]) -> None:
        self.responses = responses
        self.call_count = 0

    def get(self, *args, **kwargs):
        response = self.responses[self.call_count]
        self.call_count += 1
        if isinstance(response, requests.RequestException):
            raise response
        return response


class FakeResponse:
    def __init__(self, status_code: int, payload: dict) -> None:
        self.status_code = status_code
        self.payload = payload
        self.content = json.dumps(payload).encode("utf-8")
        self.text = self.content.decode("utf-8")

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict:
        return self.payload
