from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone

import pandas as pd
import pytest

from dkenergy_data.build.eds_prices_v1 import (
    RawBatch,
    build_price_panel_from_raw,
    build_model_ready_panel,
    normalize_batches,
)
from dkenergy_data.sources.energidataservice import ApiResponse, write_dataset_response


def test_build_model_ready_panel_stitches_and_aggregates_day_ahead() -> None:
    elspot = normalize_batches(
        [
            RawBatch(
                source_dataset="Elspotprices",
                batch_id="old",
                retrieved_at_utc="2026-01-01T00:00:00+00:00",
                raw_path=None,  # type: ignore[arg-type]
                records=[
                    {
                        "HourUTC": "2025-09-30T20:00:00",
                        "HourDK": "2025-09-30T22:00:00",
                        "PriceArea": "DK1",
                        "SpotPriceDKK": 100.0,
                        "SpotPriceEUR": 13.4,
                    },
                    {
                        "HourUTC": "2025-09-30T21:00:00",
                        "HourDK": "2025-09-30T23:00:00",
                        "PriceArea": "DK1",
                        "SpotPriceDKK": 120.0,
                        "SpotPriceEUR": 16.1,
                    },
                ],
            )
        ]
    )
    day_ahead = normalize_batches(
        [
            RawBatch(
                source_dataset="DayAheadPrices",
                batch_id="new",
                retrieved_at_utc="2026-01-01T00:00:00+00:00",
                raw_path=None,  # type: ignore[arg-type]
                records=[
                    {
                        "TimeUTC": "2025-09-30T22:00:00",
                        "TimeDK": "2025-10-01T00:00:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 200.0,
                        "DayAheadPriceEUR": 26.8,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:15:00",
                        "TimeDK": "2025-10-01T00:15:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 220.0,
                        "DayAheadPriceEUR": 29.5,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:30:00",
                        "TimeDK": "2025-10-01T00:30:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 240.0,
                        "DayAheadPriceEUR": 32.2,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:45:00",
                        "TimeDK": "2025-10-01T00:45:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 260.0,
                        "DayAheadPriceEUR": 34.9,
                    },
                ],
            )
        ]
    )

    panel, qa = build_model_ready_panel(elspot, day_ahead, required_areas=["DK1"])

    assert panel["ds_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S%z").tolist() == [
        "2025-09-30T20:00:00+0000",
        "2025-09-30T21:00:00+0000",
        "2025-09-30T22:00:00+0000",
    ]
    assert panel["y"].tolist() == [100.0, 120.0, 230.0]
    assert panel["source_dataset"].tolist() == [
        "Elspotprices",
        "Elspotprices",
        "DayAheadPrices",
    ]
    assert qa["transition_boundary_check"]["status"] == "pass"
    assert qa["missing_hour_count"] == 0
    assert qa["shared_utc_coverage_check"]["status"] == "pass"


def test_day_ahead_requires_four_quarter_hours_per_hour() -> None:
    day_ahead = normalize_batches(
        [
            RawBatch(
                source_dataset="DayAheadPrices",
                batch_id="new",
                retrieved_at_utc="2026-01-01T00:00:00+00:00",
                raw_path=None,  # type: ignore[arg-type]
                records=[
                    {
                        "TimeUTC": "2025-09-30T22:00:00",
                        "TimeDK": "2025-10-01T00:00:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 200.0,
                        "DayAheadPriceEUR": 26.8,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:15:00",
                        "TimeDK": "2025-10-01T00:15:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 220.0,
                        "DayAheadPriceEUR": 29.5,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:30:00",
                        "TimeDK": "2025-10-01T00:30:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 240.0,
                        "DayAheadPriceEUR": 32.2,
                    },
                ],
            )
        ]
    )
    elspot = normalize_batches([])

    with pytest.raises(ValueError, match="incomplete hourly quarter-hour groups"):
        build_model_ready_panel(elspot, day_ahead, required_areas=["DK1"])


def test_source_local_timestamp_mismatch_fails_normalization() -> None:
    batches = [
        RawBatch(
            source_dataset="Elspotprices",
            batch_id="old",
            retrieved_at_utc="2026-01-01T00:00:00+00:00",
            raw_path=None,  # type: ignore[arg-type]
            records=[
                {
                    "HourUTC": "2025-09-30T20:00:00",
                    "HourDK": "2025-09-30T23:00:00",
                    "PriceArea": "DK1",
                    "SpotPriceDKK": 100.0,
                    "SpotPriceEUR": 13.4,
                },
            ],
        )
    ]

    with pytest.raises(ValueError, match="source local timestamps"):
        normalize_batches(batches)


def test_conflicting_duplicate_normalized_rows_fail() -> None:
    batches = [
        RawBatch(
            source_dataset="Elspotprices",
            batch_id="old",
            retrieved_at_utc="2026-01-01T00:00:00+00:00",
            raw_path=None,  # type: ignore[arg-type]
            records=[
                {
                    "HourUTC": "2025-09-30T20:00:00",
                    "HourDK": "2025-09-30T22:00:00",
                    "PriceArea": "DK1",
                    "SpotPriceDKK": 100.0,
                    "SpotPriceEUR": 13.4,
                },
                {
                    "HourUTC": "2025-09-30T20:00:00",
                    "HourDK": "2025-09-30T22:00:00",
                    "PriceArea": "DK1",
                    "SpotPriceDKK": 101.0,
                    "SpotPriceEUR": 13.4,
                },
            ],
        )
    ]

    with pytest.raises(ValueError, match="Conflicting duplicate"):
        normalize_batches(batches)


def test_build_price_panel_from_raw_writes_parquet_and_qa(tmp_path) -> None:
    raw_root = tmp_path / "raw" / "energi_data_service"
    raw_path = (
        raw_root
        / "DayAheadPrices"
        / "fetched_at=20260101T000000Z"
        / "start=2025-10-01_end=2025-10-02.json"
    )
    raw_path.parent.mkdir(parents=True)
    payload = {
        "dataset": "DayAheadPrices",
        "records": [
            {
                "TimeUTC": "2025-09-30T22:00:00",
                "TimeDK": "2025-10-01T00:00:00",
                "PriceArea": "DK1",
                "DayAheadPriceDKK": 200.0,
                "DayAheadPriceEUR": 26.8,
            },
            {
                "TimeUTC": "2025-09-30T22:15:00",
                "TimeDK": "2025-10-01T00:15:00",
                "PriceArea": "DK1",
                "DayAheadPriceDKK": 220.0,
                "DayAheadPriceEUR": 29.5,
            },
            {
                "TimeUTC": "2025-09-30T22:30:00",
                "TimeDK": "2025-10-01T00:30:00",
                "PriceArea": "DK1",
                "DayAheadPriceDKK": 240.0,
                "DayAheadPriceEUR": 32.2,
            },
            {
                "TimeUTC": "2025-09-30T22:45:00",
                "TimeDK": "2025-10-01T00:45:00",
                "PriceArea": "DK1",
                "DayAheadPriceDKK": 260.0,
                "DayAheadPriceEUR": 34.9,
            },
            {
                "TimeUTC": "2025-09-30T22:00:00",
                "TimeDK": "2025-10-01T00:00:00",
                "PriceArea": "DK2",
                "DayAheadPriceDKK": 300.0,
                "DayAheadPriceEUR": 40.2,
            },
            {
                "TimeUTC": "2025-09-30T22:15:00",
                "TimeDK": "2025-10-01T00:15:00",
                "PriceArea": "DK2",
                "DayAheadPriceDKK": 320.0,
                "DayAheadPriceEUR": 42.9,
            },
            {
                "TimeUTC": "2025-09-30T22:30:00",
                "TimeDK": "2025-10-01T00:30:00",
                "PriceArea": "DK2",
                "DayAheadPriceDKK": 340.0,
                "DayAheadPriceEUR": 45.6,
            },
            {
                "TimeUTC": "2025-09-30T22:45:00",
                "TimeDK": "2025-10-01T00:45:00",
                "PriceArea": "DK2",
                "DayAheadPriceDKK": 360.0,
                "DayAheadPriceEUR": 48.3,
            },
        ],
    }
    raw_path.write_text(json.dumps(payload), encoding="utf-8")

    result = build_price_panel_from_raw(
        raw_root=raw_root,
        normalized_dir=tmp_path / "normalized",
        model_ready_dir=tmp_path / "model_ready",
    )

    assert result.panel_path.exists()
    assert result.qa_path.exists()
    assert result.qa["row_count"] == 2
    assert result.qa["min_price_dkk_per_mwh"] == 230.0
    assert result.qa["areas"] == ["DK1", "DK2"]
    assert result.qa["shared_utc_coverage_check"]["status"] == "pass"
    assert result.qa["build_scope"] == "available_raw_history"
    assert result.qa["requested_start_local"] is None
    assert result.qa["raw_source_audit"]["total_batch_count"] == 1
    assert result.qa["raw_source_audit"]["by_dataset"]["DayAheadPrices"]["record_count"] == 8


def test_bounded_raw_build_filters_old_null_price_prefix_before_validation(tmp_path) -> None:
    raw_root = tmp_path / "raw" / "energi_data_service"
    raw_path = (
        raw_root
        / "Elspotprices"
        / "fetched_at=20260101T000000Z"
        / "start=2000-09-30_end=2000-10-02.json"
    )
    raw_path.parent.mkdir(parents=True)
    payload = {
        "dataset": "Elspotprices",
        "records": [
            {
                "HourUTC": "2000-09-30T21:00:00",
                "HourDK": "2000-09-30T23:00:00",
                "PriceArea": "DK1",
                "SpotPriceDKK": 100.0,
                "SpotPriceEUR": 13.4,
            },
            {
                "HourUTC": "2000-09-30T21:00:00",
                "HourDK": "2000-09-30T23:00:00",
                "PriceArea": "DK2",
                "SpotPriceDKK": None,
                "SpotPriceEUR": None,
            },
            {
                "HourUTC": "2000-09-30T22:00:00",
                "HourDK": "2000-10-01T00:00:00",
                "PriceArea": "DK1",
                "SpotPriceDKK": 110.0,
                "SpotPriceEUR": 14.7,
            },
            {
                "HourUTC": "2000-09-30T22:00:00",
                "HourDK": "2000-10-01T00:00:00",
                "PriceArea": "DK2",
                "SpotPriceDKK": 120.0,
                "SpotPriceEUR": 16.1,
            },
        ],
    }
    raw_path.write_text(json.dumps(payload), encoding="utf-8")

    result = build_price_panel_from_raw(
        raw_root=raw_root,
        normalized_dir=tmp_path / "normalized",
        model_ready_dir=tmp_path / "model_ready",
        start_local=date(2000, 10, 1),
    )

    assert result.qa["row_count"] == 2
    assert result.qa["min_ds_utc"] == "2000-09-30T22:00:00+00:00"
    assert result.qa["requested_start_local"] == "2000-10-01"
    assert result.qa["observed_start_local"] == "2000-10-01"
    assert result.qa["null_price_count"] == 0


def test_default_model_ready_panel_requires_both_price_areas() -> None:
    day_ahead = normalize_batches(
        [
            RawBatch(
                source_dataset="DayAheadPrices",
                batch_id="new",
                retrieved_at_utc="2026-01-01T00:00:00+00:00",
                raw_path=None,  # type: ignore[arg-type]
                records=[
                    {
                        "TimeUTC": "2025-09-30T22:00:00",
                        "TimeDK": "2025-10-01T00:00:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 200.0,
                        "DayAheadPriceEUR": 26.8,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:15:00",
                        "TimeDK": "2025-10-01T00:15:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 220.0,
                        "DayAheadPriceEUR": 29.5,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:30:00",
                        "TimeDK": "2025-10-01T00:30:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 240.0,
                        "DayAheadPriceEUR": 32.2,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:45:00",
                        "TimeDK": "2025-10-01T00:45:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 260.0,
                        "DayAheadPriceEUR": 34.9,
                    },
                ],
            )
        ]
    )
    elspot = normalize_batches([])

    with pytest.raises(ValueError, match="area coverage mismatch"):
        build_model_ready_panel(elspot, day_ahead)


def test_build_price_panel_from_raw_filters_to_requested_experiment_area(tmp_path) -> None:
    raw_root = tmp_path / "raw" / "energi_data_service"
    raw_path = (
        raw_root
        / "Elspotprices"
        / "fetched_at=20260101T000000Z"
        / "start=2000-10-01_end=2000-10-02.json"
    )
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "HourUTC": "2000-09-30T22:00:00",
                        "HourDK": "2000-10-01T00:00:00",
                        "PriceArea": "DK1",
                        "SpotPriceDKK": 100.0,
                        "SpotPriceEUR": 13.4,
                    },
                    {
                        "HourUTC": "2000-09-30T22:00:00",
                        "HourDK": "2000-10-01T00:00:00",
                        "PriceArea": "DK2",
                        "SpotPriceDKK": 120.0,
                        "SpotPriceEUR": 16.1,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = build_price_panel_from_raw(
        raw_root=raw_root,
        normalized_dir=tmp_path / "normalized",
        model_ready_dir=tmp_path / "model_ready",
        required_areas=["DK1"],
        start_local=date(2000, 10, 1),
        end_local=date(2000, 10, 2),
    )
    panel = pd.read_parquet(result.panel_path)

    assert panel["area"].tolist() == ["DK1"]
    assert result.qa["shared_utc_coverage_check"]["required_areas"] == ["DK1"]


def test_raw_dataset_response_is_preserved_byte_for_byte(tmp_path) -> None:
    content = b'{"dataset":"DayAheadPrices","records":[{"TimeUTC":"x"}]}'
    response = ApiResponse(
        url="https://api.example.test/dataset/DayAheadPrices",
        params={"limit": 0},
        status_code=200,
        content=content,
        payload=json.loads(content),
        retrieved_at_utc=datetime(2026, 1, 1, tzinfo=timezone.utc),
        response_sha256=hashlib.sha256(content).hexdigest(),
    )

    result = write_dataset_response(
        tmp_path,
        "DayAheadPrices",
        date(2025, 10, 1),
        date(2025, 10, 2),
        response,
    )

    assert result.path.read_bytes() == content
    assert result.manifest_entry["response_sha256"] == hashlib.sha256(content).hexdigest()
    assert result.manifest_entry["saved_json_sha256"] == result.manifest_entry["response_sha256"]


def test_manifest_raw_hash_mismatch_fails_build(tmp_path) -> None:
    raw_root = tmp_path / "raw" / "energi_data_service"
    raw_path = (
        raw_root
        / "DayAheadPrices"
        / "fetched_at=20260101T000000Z"
        / "start=2025-10-01_end=2025-10-02.json"
    )
    raw_path.parent.mkdir(parents=True)
    raw_path.write_text(json.dumps({"records": []}), encoding="utf-8")
    manifest = {
        "batch_id": "bad_hash",
        "source_dataset": "DayAheadPrices",
        "retrieved_at_utc": "2026-01-01T00:00:00+00:00",
        "raw_path": str(raw_path.relative_to(raw_root)),
        "response_sha256": "0" * 64,
        "saved_json_sha256": "0" * 64,
    }
    (raw_root / "manifest.jsonl").write_text(json.dumps(manifest) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="hash does not match manifest"):
        build_price_panel_from_raw(
            raw_root=raw_root,
            normalized_dir=tmp_path / "normalized",
            model_ready_dir=tmp_path / "model_ready",
        )


def test_allow_incomplete_recent_records_dropped_quarter_hour_sample() -> None:
    day_ahead = normalize_batches(
        [
            RawBatch(
                source_dataset="DayAheadPrices",
                batch_id="new",
                retrieved_at_utc="2026-01-01T00:00:00+00:00",
                raw_path=None,  # type: ignore[arg-type]
                records=[
                    {
                        "TimeUTC": "2025-09-30T22:00:00",
                        "TimeDK": "2025-10-01T00:00:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 200.0,
                        "DayAheadPriceEUR": 26.8,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:15:00",
                        "TimeDK": "2025-10-01T00:15:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 220.0,
                        "DayAheadPriceEUR": 29.5,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:30:00",
                        "TimeDK": "2025-10-01T00:30:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 240.0,
                        "DayAheadPriceEUR": 32.2,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:45:00",
                        "TimeDK": "2025-10-01T00:45:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 260.0,
                        "DayAheadPriceEUR": 34.9,
                    },
                    {
                        "TimeUTC": "2025-09-30T23:00:00",
                        "TimeDK": "2025-10-01T01:00:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 300.0,
                        "DayAheadPriceEUR": 40.2,
                    },
                    {
                        "TimeUTC": "2025-09-30T23:15:00",
                        "TimeDK": "2025-10-01T01:15:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 320.0,
                        "DayAheadPriceEUR": 42.9,
                    },
                ],
            )
        ]
    )

    panel, qa = build_model_ready_panel(
        normalize_batches([]),
        day_ahead,
        required_areas=["DK1"],
        allow_incomplete_recent=True,
        start_local=date(2025, 10, 1),
        end_local=date(2025, 10, 2),
    )

    assert len(panel) == 1
    assert qa["artifact_status"] == "incomplete_live_refresh"
    assert qa["build_scope"] == "bounded_local_range"
    assert qa["requested_start_local"] == "2025-10-01"
    assert qa["requested_end_local"] == "2025-10-02"
    assert qa["invalid_quarter_hour_group_count"] == 1
    assert qa["invalid_quarter_hour_group_sample"] == [
        {
            "area": "DK1",
            "hour_utc": "2025-09-30T23:00:00+00:00",
            "quarter_hour_count": 2,
            "is_latest_for_area": True,
        }
    ]


def test_bounded_local_range_without_full_requested_coverage_is_not_final() -> None:
    day_ahead = normalize_batches(
        [
            RawBatch(
                source_dataset="DayAheadPrices",
                batch_id="new",
                retrieved_at_utc="2026-01-01T00:00:00+00:00",
                raw_path=None,  # type: ignore[arg-type]
                records=[
                    {
                        "TimeUTC": "2025-09-30T22:00:00",
                        "TimeDK": "2025-10-01T00:00:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 200.0,
                        "DayAheadPriceEUR": 26.8,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:15:00",
                        "TimeDK": "2025-10-01T00:15:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 220.0,
                        "DayAheadPriceEUR": 29.5,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:30:00",
                        "TimeDK": "2025-10-01T00:30:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 240.0,
                        "DayAheadPriceEUR": 32.2,
                    },
                    {
                        "TimeUTC": "2025-09-30T22:45:00",
                        "TimeDK": "2025-10-01T00:45:00",
                        "PriceArea": "DK1",
                        "DayAheadPriceDKK": 260.0,
                        "DayAheadPriceEUR": 34.9,
                    },
                ],
            )
        ]
    )

    panel, qa = build_model_ready_panel(
        normalize_batches([]),
        day_ahead,
        required_areas=["DK1"],
        start_local=date(2025, 10, 1),
        end_local=date(2025, 10, 2),
    )

    assert len(panel) == 1
    assert qa["artifact_status"] == "incomplete_bounded_range"
    assert qa["bounded_local_range_check"]["status"] == "fail"
    assert qa["bounded_local_range_check"]["issues"] == [
        "observed_end_before_requested_end"
    ]
