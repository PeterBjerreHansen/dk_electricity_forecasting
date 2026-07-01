from __future__ import annotations

import json

import pytest
import requests

from dkenergy_data.sources.energidataservice import (
    EnergiDataServiceClient,
    EnergiDataServiceError,
)


def test_eds_client_retries_selected_5xx_statuses() -> None:
    session = FakeSession(
        [
            FakeResponse(503, {"error": "temporary"}),
            FakeResponse(200, {"records": [{"ok": True}]}),
        ]
    )
    client = EnergiDataServiceClient(
        base_url="https://example.test",
        max_retries=1,
        retry_backoff_seconds=0,
        session=session,
    )

    response = client.fetch_dataset("DayAheadPrices", {"limit": 0})

    assert response.status_code == 200
    assert response.payload["records"] == [{"ok": True}]
    assert session.call_count == 2


def test_eds_client_retries_request_exceptions() -> None:
    session = FakeSession(
        [
            requests.Timeout("temporary timeout"),
            FakeResponse(200, {"records": []}),
        ]
    )
    client = EnergiDataServiceClient(
        base_url="https://example.test",
        max_retries=1,
        retry_backoff_seconds=0,
        session=session,
    )

    response = client.fetch_dataset("DayAheadPrices", {"limit": 0})

    assert response.payload == {"records": []}
    assert session.call_count == 2


def test_eds_client_raises_after_request_exception_retries_are_exhausted() -> None:
    session = FakeSession([requests.ConnectionError("connection reset")])
    client = EnergiDataServiceClient(
        base_url="https://example.test",
        max_retries=0,
        retry_backoff_seconds=0,
        session=session,
    )

    with pytest.raises(EnergiDataServiceError, match="request failed after 1 attempt"):
        client.fetch_dataset("DayAheadPrices", {"limit": 0})


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
        self.headers = {}

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self) -> dict:
        return self.payload
