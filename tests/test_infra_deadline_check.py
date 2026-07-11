from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "infra"
    / "aws"
    / "functions"
    / "check_forecast_deadline.py"
)


def _load_module(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "boto3",
        SimpleNamespace(client=lambda service: SimpleNamespace(service=service)),
    )
    spec = importlib.util.spec_from_file_location("check_forecast_deadline", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _environment(monkeypatch) -> None:
    monkeypatch.setenv("ARTIFACT_BUCKET", "artifacts")
    monkeypatch.setenv("ARTIFACT_PREFIX", "project")
    monkeypatch.setenv("SCHEDULE_TIMEZONE", "Europe/Copenhagen")
    monkeypatch.setenv("DELIVERY_DATE_OFFSET_DAYS", "1")
    monkeypatch.setenv("MARKER_MAX_AGE_MINUTES", "360")


def test_valid_completed_pointer_passes_deadline_check(monkeypatch) -> None:
    module = _load_module(monkeypatch)
    _environment(monkeypatch)
    now = datetime(2026, 7, 11, 10, 15, tzinfo=timezone.utc)
    pointer = {
        "run_id": "live-2026-07-12",
        "status": "completed",
        "delivery_date_local": "2026-07-12",
        "committed_at_utc": "2026-07-11T09:58:00Z",
        "decision_deadline_utc": "2026-07-11T10:00:00Z",
        "completion_key": "forecast_runs/live-2026-07-12/COMPLETED.json",
    }
    monkeypatch.setattr(
        module,
        "_read_json",
        lambda bucket, key: ({"run_id": pointer["run_id"], "status": "completed"}, now),
    )

    problems: list[str] = []
    module._validate_pointer(pointer, now, now, problems)

    assert problems == []


def test_invalid_or_late_pointer_reports_each_contract_failure(monkeypatch) -> None:
    module = _load_module(monkeypatch)
    _environment(monkeypatch)
    now = datetime(2026, 7, 11, 10, 15, tzinfo=timezone.utc)
    pointer = {
        "run_id": "live-old",
        "status": "uploading",
        "delivery_date_local": "2026-07-11",
        "committed_at_utc": "2026-07-11T10:05:00Z",
        "decision_deadline_utc": "2026-07-11T10:00:00Z",
        "completion_key": "forecast_runs/live-old/COMPLETED.json",
    }

    def missing_completion(bucket, key):
        raise FileNotFoundError(f"s3://{bucket}/{key}")

    monkeypatch.setattr(module, "_read_json", missing_completion)

    problems: list[str] = []
    module._validate_pointer(pointer, now - timedelta(hours=7), now, problems)

    assert any("not 'completed'" in problem for problem in problems)
    assert any("delivery_date_local" in problem for problem in problems)
    assert any("last modified" in problem for problem in problems)
    assert any("after its" in problem for problem in problems)
    assert any("completion receipt is unavailable" in problem for problem in problems)
