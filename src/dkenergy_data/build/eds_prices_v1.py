from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from dkenergy_forecast.types import (
    DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
    PRICE_AVAILABILITY_COLUMN,
    TARGET_CONTRACT_COLUMNS,
    TARGET_REGIME_BOUNDARY_LOCAL,
    add_target_contract,
    add_price_availability,
)


COPENHAGEN_TZ = "Europe/Copenhagen"
DATASET_VERSION = "v1"
STITCH_BOUNDARY_LOCAL = TARGET_REGIME_BOUNDARY_LOCAL
STITCH_BOUNDARY_DATE = date(2025, 10, 1)
ALLOWED_AREAS = ("DK1", "DK2")


@dataclass(frozen=True)
class SourceSpec:
    dataset: str
    time_column: str
    local_time_column: str
    area_column: str
    price_dkk_column: str
    price_eur_column: str
    source_resolution_minutes: int

    @property
    def columns(self) -> list[str]:
        return [
            self.time_column,
            self.local_time_column,
            self.area_column,
            self.price_dkk_column,
            self.price_eur_column,
        ]


@dataclass(frozen=True)
class RawBatch:
    source_dataset: str
    batch_id: str
    retrieved_at_utc: str
    raw_path: Path
    records: list[dict[str, Any]]


@dataclass(frozen=True)
class BuildResult:
    panel_path: Path
    qa_path: Path
    normalized_paths: dict[str, Path]
    qa: dict[str, Any]


SOURCE_SPECS: dict[str, SourceSpec] = {
    "Elspotprices": SourceSpec(
        dataset="Elspotprices",
        time_column="HourUTC",
        local_time_column="HourDK",
        area_column="PriceArea",
        price_dkk_column="SpotPriceDKK",
        price_eur_column="SpotPriceEUR",
        source_resolution_minutes=60,
    ),
    "DayAheadPrices": SourceSpec(
        dataset="DayAheadPrices",
        time_column="TimeUTC",
        local_time_column="TimeDK",
        area_column="PriceArea",
        price_dkk_column="DayAheadPriceDKK",
        price_eur_column="DayAheadPriceEUR",
        source_resolution_minutes=15,
    ),
}

NORMALIZED_COLUMNS = [
    "source_dataset",
    "source_time_utc",
    "source_time_local_text",
    "area",
    "price_dkk_per_mwh",
    "price_eur_per_mwh",
    "source_resolution_minutes",
    "raw_batch_id",
    "retrieved_at_utc",
]

PANEL_COLUMNS = [
    "unique_id",
    "ds_utc",
    "ds_local",
    "local_date",
    "local_hour",
    "local_day_of_week",
    "local_month",
    "is_weekend",
    "is_dst",
    "utc_offset_hours",
    "area",
    "y",
    *TARGET_CONTRACT_COLUMNS,
    "price_dkk_per_mwh",
    "price_eur_per_mwh",
    "source_dataset",
    "source_resolution_minutes",
    "dataset_version",
]


def load_raw_batches(raw_root: Path, dataset: str) -> list[RawBatch]:
    manifest_entries = _read_manifest_entries(raw_root, dataset)
    if manifest_entries:
        return [_batch_from_manifest_entry(raw_root, entry) for entry in manifest_entries]

    return _load_raw_batches_from_glob(raw_root, dataset)


def normalize_batches(
    batches: Iterable[RawBatch],
    allowed_areas: Iterable[str] = ALLOWED_AREAS,
    start_local: date | None = None,
    end_local: date | None = None,
) -> pd.DataFrame:
    frames = [
        normalize_records(
            batch,
            allowed_areas=allowed_areas,
            start_local=start_local,
            end_local=end_local,
        )
        for batch in batches
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_normalized_frame()

    normalized = pd.concat(frames, ignore_index=True)
    return deduplicate_normalized(normalized)


def normalize_records(
    batch: RawBatch,
    allowed_areas: Iterable[str] = ALLOWED_AREAS,
    start_local: date | None = None,
    end_local: date | None = None,
) -> pd.DataFrame:
    if batch.source_dataset not in SOURCE_SPECS:
        raise ValueError(f"Unsupported EDS source dataset: {batch.source_dataset}")

    spec = SOURCE_SPECS[batch.source_dataset]
    allowed_area_set = set(allowed_areas)
    rows: list[dict[str, Any]] = []

    for index, record in enumerate(batch.records):
        if not _record_in_local_range(
            record,
            spec,
            start_local=start_local,
            end_local=end_local,
        ):
            continue
        _validate_record(record, spec, batch.source_dataset, index)
        area = str(record[spec.area_column])
        if area not in allowed_area_set:
            raise ValueError(
                f"Unexpected price area {area!r} in {batch.source_dataset}; "
                f"allowed areas are {sorted(allowed_area_set)}"
            )

        rows.append(
            {
                "source_dataset": batch.source_dataset,
                "source_time_utc": record[spec.time_column],
                "source_time_local_text": str(record[spec.local_time_column]),
                "area": area,
                "price_dkk_per_mwh": record[spec.price_dkk_column],
                "price_eur_per_mwh": record[spec.price_eur_column],
                "source_resolution_minutes": spec.source_resolution_minutes,
                "raw_batch_id": batch.batch_id,
                "retrieved_at_utc": batch.retrieved_at_utc,
            }
        )

    frame = pd.DataFrame(rows, columns=NORMALIZED_COLUMNS)
    if frame.empty:
        return _empty_normalized_frame()

    frame["source_time_utc"] = pd.to_datetime(frame["source_time_utc"], utc=True)
    frame["price_dkk_per_mwh"] = pd.to_numeric(frame["price_dkk_per_mwh"], errors="raise")
    frame["price_eur_per_mwh"] = pd.to_numeric(frame["price_eur_per_mwh"], errors="raise")
    frame["source_resolution_minutes"] = frame["source_resolution_minutes"].astype("int16")

    if frame[["source_time_utc", "area", "price_dkk_per_mwh", "price_eur_per_mwh"]].isna().any().any():
        raise ValueError(f"Missing normalized values in {batch.source_dataset}")

    _validate_source_local_timestamps(frame, dataset=batch.source_dataset)

    return frame


def deduplicate_normalized(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_normalized_frame()

    key_cols = ["source_dataset", "source_time_utc", "area"]
    batch_key_cols = [*key_cols, "raw_batch_id"]
    value_cols = [
        "source_time_local_text",
        "price_dkk_per_mwh",
        "price_eur_per_mwh",
        "source_resolution_minutes",
    ]
    conflicts: list[str] = []

    for key, group in frame.groupby(batch_key_cols, dropna=False):
        if len(group) <= 1:
            continue
        for column in value_cols:
            if group[column].nunique(dropna=False) > 1:
                conflicts.append(f"{key} differs on {column}")

    if conflicts:
        sample = "; ".join(conflicts[:5])
        raise ValueError(f"Conflicting duplicate normalized EDS rows: {sample}")

    deduped = (
        frame.sort_values(["source_dataset", "area", "source_time_utc", "retrieved_at_utc", "raw_batch_id"])
        .drop_duplicates(key_cols, keep="last")
        .reset_index(drop=True)
    )
    return deduped[NORMALIZED_COLUMNS]


def build_model_ready_panel(
    elspot_normalized: pd.DataFrame,
    day_ahead_normalized: pd.DataFrame,
    *,
    dataset_version: str = DATASET_VERSION,
    allow_incomplete_recent: bool = False,
    required_areas: Iterable[str] | None = ALLOWED_AREAS,
    start_local: date | None = None,
    end_local: date | None = None,
    raw_source_audit: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    elspot_hourly = _prepare_elspot_hourly(elspot_normalized)
    (
        day_ahead_hourly,
        invalid_quarter_hour_group_count,
        invalid_quarter_hour_group_sample,
    ) = _aggregate_day_ahead_to_hourly(
        day_ahead_normalized,
        allow_incomplete_recent=allow_incomplete_recent,
    )

    hourly_frames = [frame for frame in [elspot_hourly, day_ahead_hourly] if not frame.empty]
    panel = pd.concat(hourly_frames, ignore_index=True) if hourly_frames else _empty_hourly_source_frame()
    if panel.empty:
        raise ValueError("No EDS price rows available after source stitching")

    panel = _filter_panel_local_range(panel, start_local=start_local, end_local=end_local)
    if panel.empty:
        raise ValueError("No EDS price rows available after applying local date filters")

    panel = _deduplicate_stitched_panel(panel)
    panel = _add_model_ready_columns(panel, dataset_version=dataset_version)
    shared_coverage_check = validate_area_coverage(panel, required_areas=required_areas)

    missing_hours = find_missing_utc_hours(panel)
    if missing_hours:
        sample = ", ".join(missing_hours[:5])
        raise ValueError(f"Missing hourly observations in stitched panel: {sample}")

    qa = make_qa_report(
        panel,
        dataset_version=dataset_version,
        allow_incomplete_recent=allow_incomplete_recent,
        invalid_quarter_hour_group_count=invalid_quarter_hour_group_count,
        invalid_quarter_hour_group_sample=invalid_quarter_hour_group_sample,
        shared_coverage_check=shared_coverage_check,
        start_local=start_local,
        end_local=end_local,
        raw_source_audit=raw_source_audit,
    )
    return panel[PANEL_COLUMNS], qa


def build_price_panel_from_raw(
    *,
    raw_root: Path,
    normalized_dir: Path,
    model_ready_dir: Path,
    dataset_version: str = DATASET_VERSION,
    allow_incomplete_recent: bool = False,
    required_areas: Iterable[str] | None = ALLOWED_AREAS,
    start_local: date | None = None,
    end_local: date | None = None,
) -> BuildResult:
    elspot_batches = load_raw_batches(raw_root, "Elspotprices")
    day_ahead_batches = load_raw_batches(raw_root, "DayAheadPrices")
    if not elspot_batches and not day_ahead_batches:
        raise ValueError(f"No raw EDS price batches found under {raw_root}")
    raw_source_audit = _make_raw_source_audit(
        {
            "Elspotprices": elspot_batches,
            "DayAheadPrices": day_ahead_batches,
        }
    )
    elspot_normalized = normalize_batches(
        elspot_batches,
        start_local=start_local,
        end_local=end_local,
    )
    day_ahead_normalized = normalize_batches(
        day_ahead_batches,
        start_local=start_local,
        end_local=end_local,
    )
    if required_areas is not None:
        selected_area_set = set(required_areas)
        elspot_normalized = elspot_normalized[
            elspot_normalized["area"].isin(selected_area_set)
        ].reset_index(drop=True)
        day_ahead_normalized = day_ahead_normalized[
            day_ahead_normalized["area"].isin(selected_area_set)
        ].reset_index(drop=True)

    normalized_dir.mkdir(parents=True, exist_ok=True)
    normalized_paths = {
        "Elspotprices": normalized_dir / f"eds_elspotprices_{dataset_version}.parquet",
        "DayAheadPrices": normalized_dir / f"eds_day_ahead_prices_15min_{dataset_version}.parquet",
    }
    elspot_normalized.to_parquet(normalized_paths["Elspotprices"], index=False)
    day_ahead_normalized.to_parquet(normalized_paths["DayAheadPrices"], index=False)

    panel, qa = build_model_ready_panel(
        elspot_normalized,
        day_ahead_normalized,
        dataset_version=dataset_version,
        allow_incomplete_recent=allow_incomplete_recent,
        required_areas=required_areas,
        start_local=start_local,
        end_local=end_local,
        raw_source_audit=raw_source_audit,
    )
    qa["source_metadata_sha256"] = source_metadata_hashes(raw_root)

    model_ready_dir.mkdir(parents=True, exist_ok=True)
    panel_path = model_ready_dir / f"price_panel_hourly_{dataset_version}.parquet"
    qa_path = model_ready_dir / f"price_panel_hourly_{dataset_version}.qa.json"
    panel.to_parquet(panel_path, index=False)
    qa_path.write_text(json.dumps(_json_safe(qa), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return BuildResult(
        panel_path=panel_path,
        qa_path=qa_path,
        normalized_paths=normalized_paths,
        qa=qa,
    )


def find_missing_utc_hours(panel: pd.DataFrame) -> list[str]:
    if panel.empty:
        return []

    missing: list[str] = []
    for area, area_frame in panel.groupby("area"):
        observed = pd.DatetimeIndex(area_frame["ds_utc"].sort_values().unique())
        expected = pd.date_range(observed.min(), observed.max(), freq="h", tz="UTC")
        for timestamp in expected.difference(observed):
            missing.append(f"{area}:{timestamp.isoformat()}")
    return missing


def validate_area_coverage(
    panel: pd.DataFrame,
    *,
    required_areas: Iterable[str] | None,
) -> dict[str, Any]:
    if required_areas is None:
        return {"status": "not_enforced", "required_areas": None}

    required = tuple(required_areas)
    required_set = set(required)
    observed_set = set(panel["area"].dropna().unique().tolist())
    missing_areas = sorted(required_set - observed_set)
    unexpected_areas = sorted(observed_set - required_set)

    if missing_areas or unexpected_areas:
        raise ValueError(
            "Model-ready panel area coverage mismatch. "
            f"required={sorted(required_set)}, observed={sorted(observed_set)}"
        )

    coverage_by_area = {
        area: set(panel.loc[panel["area"] == area, "ds_utc"].tolist())
        for area in required
    }
    reference_area = required[0] if required else None
    reference = coverage_by_area[reference_area] if reference_area else set()
    mismatches: dict[str, dict[str, list[str]]] = {}

    for area, timestamps in coverage_by_area.items():
        missing_from_area = sorted(reference - timestamps)
        extra_in_area = sorted(timestamps - reference)
        if missing_from_area or extra_in_area:
            mismatches[area] = {
                "missing_from_area": [timestamp.isoformat() for timestamp in missing_from_area[:10]],
                "extra_in_area": [timestamp.isoformat() for timestamp in extra_in_area[:10]],
            }

    if mismatches:
        raise ValueError(f"Model-ready panel does not have shared UTC coverage: {mismatches}")

    return {
        "status": "pass",
        "required_areas": list(required),
        "observed_areas": sorted(observed_set),
        "shared_utc_start": min(reference).isoformat() if reference else None,
        "shared_utc_end": max(reference).isoformat() if reference else None,
        "shared_hour_count": len(reference),
    }


def make_qa_report(
    panel: pd.DataFrame,
    *,
    dataset_version: str,
    allow_incomplete_recent: bool,
    invalid_quarter_hour_group_count: int,
    invalid_quarter_hour_group_sample: list[dict[str, Any]],
    shared_coverage_check: dict[str, Any],
    start_local: date | None,
    end_local: date | None,
    raw_source_audit: dict[str, Any] | None,
) -> dict[str, Any]:
    duplicate_key_count = int(panel.duplicated(["area", "ds_utc"]).sum())
    missing_hours = find_missing_utc_hours(panel)
    price = panel["price_dkk_per_mwh"]
    bounded_local_range = start_local is not None or end_local is not None
    bounded_local_range_check = _bounded_local_range_check(panel, start_local=start_local, end_local=end_local)

    return {
        "dataset_version": dataset_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_status": _artifact_status(
            allow_incomplete_recent=allow_incomplete_recent,
            invalid_quarter_hour_group_count=invalid_quarter_hour_group_count,
            bounded_local_range_check=bounded_local_range_check,
        ),
        "allow_incomplete_recent": bool(allow_incomplete_recent),
        "build_scope": "bounded_local_range" if bounded_local_range else "available_raw_history",
        "requested_start_local": start_local.isoformat() if start_local is not None else None,
        "requested_end_local": end_local.isoformat() if end_local is not None else None,
        "observed_start_local": panel["ds_local"].min().date().isoformat(),
        "observed_end_local": panel["ds_local"].max().date().isoformat(),
        "bounded_local_range_check": bounded_local_range_check,
        "raw_source_audit": raw_source_audit or {},
        "source_datasets": sorted(panel["source_dataset"].dropna().unique().tolist()),
        "source_metadata_sha256": {},
        "price_availability_policy": {
            "column": PRICE_AVAILABILITY_COLUMN,
            "publication_local_time": DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
            "timezone": COPENHAGEN_TZ,
            "delivery_day_offset_days": -1,
            "eligibility_operator": "< forecast_origin_utc",
        },
        "target_contract": {
            "columns": TARGET_CONTRACT_COLUMNS,
            "target_definitions": sorted(panel["target_definition"].dropna().unique().tolist()),
            "market_regimes": sorted(panel["market_regime"].dropna().unique().tolist()),
            "native_resolution_minutes": sorted(
                int(value) for value in panel["native_resolution_minutes"].dropna().unique()
            ),
            "target_aggregations": sorted(panel["target_aggregation"].dropna().unique().tolist()),
            "regime_boundary_local": STITCH_BOUNDARY_LOCAL.isoformat(),
        },
        "row_count": int(len(panel)),
        "min_ds_utc": panel["ds_utc"].min().isoformat(),
        "max_ds_utc": panel["ds_utc"].max().isoformat(),
        "areas": sorted(panel["area"].dropna().unique().tolist()),
        "duplicate_key_count": duplicate_key_count,
        "missing_hour_count": len(missing_hours),
        "missing_hour_sample": missing_hours[:10],
        "invalid_quarter_hour_group_count": int(invalid_quarter_hour_group_count),
        "invalid_quarter_hour_group_sample": invalid_quarter_hour_group_sample,
        "negative_price_count": int((price < 0).sum()),
        "null_price_count": int(price.isna().sum()),
        "min_price_dkk_per_mwh": float(price.min()),
        "max_price_dkk_per_mwh": float(price.max()),
        "transition_boundary_check": _transition_boundary_check(panel),
        "shared_utc_coverage_check": shared_coverage_check,
        "dst_day_checks": _dst_day_checks(panel),
    }


def _artifact_status(
    *,
    allow_incomplete_recent: bool,
    invalid_quarter_hour_group_count: int,
    bounded_local_range_check: dict[str, Any],
) -> str:
    if allow_incomplete_recent and invalid_quarter_hour_group_count:
        return "incomplete_live_refresh"
    if bounded_local_range_check["status"] == "fail":
        return "incomplete_bounded_range"
    return "final_historical"


def _bounded_local_range_check(
    panel: pd.DataFrame,
    *,
    start_local: date | None,
    end_local: date | None,
) -> dict[str, Any]:
    if start_local is None and end_local is None:
        return {"status": "not_applicable", "issues": []}

    observed_start_utc = panel["ds_utc"].min()
    observed_end_exclusive_utc = panel["ds_utc"].max() + pd.Timedelta(hours=1)
    issues: list[str] = []

    expected_start_utc = None
    if start_local is not None:
        expected_start_utc = pd.Timestamp(start_local.isoformat(), tz=COPENHAGEN_TZ).tz_convert("UTC")
        if observed_start_utc > expected_start_utc:
            issues.append("observed_start_after_requested_start")

    expected_end_utc = None
    if end_local is not None:
        expected_end_utc = pd.Timestamp(end_local.isoformat(), tz=COPENHAGEN_TZ).tz_convert("UTC")
        if observed_end_exclusive_utc < expected_end_utc:
            issues.append("observed_end_before_requested_end")

    return {
        "status": "fail" if issues else "pass",
        "issues": issues,
        "expected_start_utc": expected_start_utc.isoformat() if expected_start_utc is not None else None,
        "expected_end_exclusive_utc": expected_end_utc.isoformat() if expected_end_utc is not None else None,
        "observed_start_utc": observed_start_utc.isoformat(),
        "observed_end_exclusive_utc": observed_end_exclusive_utc.isoformat(),
    }


def source_metadata_hashes(raw_root: Path) -> dict[str, str | None]:
    metadata_dir = raw_root / "metadata"
    hashes: dict[str, str | None] = {}
    for dataset in SOURCE_SPECS:
        candidates = sorted(metadata_dir.glob(f"{dataset}_*.json"))
        if not candidates:
            hashes[dataset] = None
            continue
        latest = candidates[-1]
        hashes[dataset] = hashlib.sha256(latest.read_bytes()).hexdigest()
    return hashes


def _make_raw_source_audit(batches_by_source: dict[str, list[RawBatch]]) -> dict[str, Any]:
    by_dataset: dict[str, Any] = {}
    total_batch_count = 0
    total_record_count = 0

    for dataset, batches in batches_by_source.items():
        record_count = sum(len(batch.records) for batch in batches)
        total_batch_count += len(batches)
        total_record_count += record_count
        by_dataset[dataset] = {
            "batch_count": len(batches),
            "record_count": record_count,
            "raw_path_sample": [str(batch.raw_path) for batch in batches[:5]],
        }

    return {
        "total_batch_count": total_batch_count,
        "total_record_count": total_record_count,
        "by_dataset": by_dataset,
    }


def _prepare_elspot_hourly(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_hourly_source_frame()

    source = frame.copy()
    source["derived_local"] = source["source_time_utc"].dt.tz_convert(COPENHAGEN_TZ)
    source = source[source["derived_local"] < STITCH_BOUNDARY_LOCAL].copy()
    if source.empty:
        return _empty_hourly_source_frame()

    if not (
        (source["source_time_utc"].dt.minute == 0)
        & (source["source_time_utc"].dt.second == 0)
        & (source["source_time_utc"].dt.microsecond == 0)
    ).all():
        raise ValueError("Elspotprices contains non-hourly timestamps")

    source["ds_utc"] = source["source_time_utc"]
    return source[
        [
            "area",
            "ds_utc",
            "price_dkk_per_mwh",
            "price_eur_per_mwh",
            "source_dataset",
            "source_resolution_minutes",
        ]
    ].reset_index(drop=True)


def _aggregate_day_ahead_to_hourly(
    frame: pd.DataFrame,
    *,
    allow_incomplete_recent: bool,
) -> tuple[pd.DataFrame, int, list[dict[str, Any]]]:
    if frame.empty:
        return _empty_hourly_source_frame(), 0, []

    source = frame.copy()
    source["derived_local"] = source["source_time_utc"].dt.tz_convert(COPENHAGEN_TZ)
    source = source[source["derived_local"] >= STITCH_BOUNDARY_LOCAL].copy()
    if source.empty:
        return _empty_hourly_source_frame(), 0, []

    valid_minutes = source["source_time_utc"].dt.minute.isin([0, 15, 30, 45])
    valid_seconds = (source["source_time_utc"].dt.second == 0) & (source["source_time_utc"].dt.microsecond == 0)
    if not (valid_minutes & valid_seconds).all():
        raise ValueError("DayAheadPrices contains timestamps outside 15-minute boundaries")

    source["hour_utc"] = source["source_time_utc"].dt.floor("h")
    counts = (
        source.groupby(["area", "hour_utc"])["source_time_utc"]
        .nunique()
        .rename("quarter_hour_count")
        .reset_index()
    )
    invalid = counts[counts["quarter_hour_count"] != 4].copy()
    invalid_count = int(len(invalid))
    invalid_sample: list[dict[str, Any]] = []

    if invalid_count:
        if not allow_incomplete_recent:
            raise ValueError(_invalid_quarter_hour_message(invalid))

        latest_hour_by_area = source.groupby("area")["hour_utc"].max().to_dict()
        invalid["is_latest_for_area"] = invalid.apply(
            lambda row: row["hour_utc"] == latest_hour_by_area.get(row["area"]),
            axis=1,
        )
        if not bool(invalid["is_latest_for_area"].all()):
            raise ValueError(_invalid_quarter_hour_message(invalid))

        invalid_sample = _quarter_hour_group_sample(invalid)
        valid_keys = counts[counts["quarter_hour_count"] == 4][["area", "hour_utc"]]
        source = source.merge(valid_keys, on=["area", "hour_utc"], how="inner")

    hourly = (
        source.groupby(["area", "hour_utc"], as_index=False)
        .agg(
            price_dkk_per_mwh=("price_dkk_per_mwh", "mean"),
            price_eur_per_mwh=("price_eur_per_mwh", "mean"),
        )
        .rename(columns={"hour_utc": "ds_utc"})
    )
    hourly["source_dataset"] = "DayAheadPrices"
    hourly["source_resolution_minutes"] = 15
    return hourly[
        [
            "area",
            "ds_utc",
            "price_dkk_per_mwh",
            "price_eur_per_mwh",
            "source_dataset",
            "source_resolution_minutes",
        ]
    ].reset_index(drop=True), invalid_count, invalid_sample


def _deduplicate_stitched_panel(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel

    key_cols = ["area", "ds_utc"]
    value_cols = ["price_dkk_per_mwh", "price_eur_per_mwh", "source_dataset", "source_resolution_minutes"]
    conflicts: list[str] = []
    for key, group in panel.groupby(key_cols, dropna=False):
        if len(group) <= 1:
            continue
        for column in value_cols:
            if group[column].nunique(dropna=False) > 1:
                conflicts.append(f"{key} differs on {column}")
    if conflicts:
        sample = "; ".join(conflicts[:5])
        raise ValueError(f"Conflicting duplicate stitched rows: {sample}")

    return (
        panel.sort_values(["area", "ds_utc", "source_dataset"])
        .drop_duplicates(key_cols, keep="last")
        .reset_index(drop=True)
    )


def _add_model_ready_columns(panel: pd.DataFrame, *, dataset_version: str) -> pd.DataFrame:
    output = panel.copy()
    output["ds_local"] = output["ds_utc"].dt.tz_convert(COPENHAGEN_TZ)
    output["unique_id"] = "day_ahead_price_" + output["area"].astype(str)
    output["local_date"] = output["ds_local"].dt.strftime("%Y-%m-%d")
    output["local_hour"] = output["ds_local"].dt.hour.astype("int8")
    output["local_day_of_week"] = output["ds_local"].dt.dayofweek.astype("int8")
    output["local_month"] = output["ds_local"].dt.month.astype("int8")
    output["is_weekend"] = output["local_day_of_week"].isin([5, 6])
    output["is_dst"] = output["ds_local"].map(lambda value: bool(value.dst()))
    output["utc_offset_hours"] = output["ds_local"].map(
        lambda value: value.utcoffset().total_seconds() / 3600
    )
    output["y"] = output["price_dkk_per_mwh"]
    output["dataset_version"] = dataset_version
    output = add_target_contract(output)
    resolution_mismatch = output["native_resolution_minutes"].ne(
        output["source_resolution_minutes"]
    )
    if bool(resolution_mismatch.any()):
        sample = output.loc[
            resolution_mismatch,
            ["area", "ds_utc", "source_dataset", "source_resolution_minutes", "native_resolution_minutes"],
        ].head(5)
        raise ValueError(
            "Price target contract disagrees with native source resolution:\n"
            + sample.to_string(index=False)
        )
    output = add_price_availability(output)
    return output.sort_values(["area", "ds_utc"]).reset_index(drop=True)


def _filter_panel_local_range(
    panel: pd.DataFrame,
    *,
    start_local: date | None,
    end_local: date | None,
) -> pd.DataFrame:
    if panel.empty or (start_local is None and end_local is None):
        return panel

    output = panel.copy()
    local = output["ds_utc"].dt.tz_convert(COPENHAGEN_TZ)
    if start_local is not None:
        output = output[local >= pd.Timestamp(start_local.isoformat(), tz=COPENHAGEN_TZ)]
        local = output["ds_utc"].dt.tz_convert(COPENHAGEN_TZ)
    if end_local is not None:
        output = output[local < pd.Timestamp(end_local.isoformat(), tz=COPENHAGEN_TZ)]
    return output.reset_index(drop=True)


def _transition_boundary_check(panel: pd.DataFrame) -> dict[str, Any]:
    by_area: dict[str, Any] = {}
    statuses: list[bool] = []

    for area, area_frame in panel.groupby("area"):
        old_rows = area_frame[area_frame["source_dataset"] == "Elspotprices"]
        new_rows = area_frame[area_frame["source_dataset"] == "DayAheadPrices"]
        if old_rows.empty or new_rows.empty:
            by_area[area] = {"status": "not_applicable"}
            continue

        old_max_utc = old_rows["ds_utc"].max()
        new_min_utc = new_rows["ds_utc"].min()
        old_max_local_date = old_rows["ds_local"].max().date().isoformat()
        new_min_local_date = new_rows["ds_local"].min().date().isoformat()
        gap_free = bool(old_max_utc + pd.Timedelta(hours=1) == new_min_utc)
        expected_dates = old_max_local_date == "2025-09-30" and new_min_local_date == "2025-10-01"
        passed = gap_free and expected_dates
        statuses.append(passed)
        by_area[area] = {
            "status": "pass" if passed else "fail",
            "old_max_utc": old_max_utc.isoformat(),
            "new_min_utc": new_min_utc.isoformat(),
            "old_max_local_date": old_max_local_date,
            "new_min_local_date": new_min_local_date,
            "gap_free": gap_free,
        }

    if statuses:
        overall = "pass" if all(statuses) else "fail"
    else:
        overall = "not_applicable"
    return {"status": overall, "areas": by_area}


def _dst_day_checks(panel: pd.DataFrame) -> dict[str, Any]:
    counts = (
        panel.groupby(["area", "local_date"])
        .size()
        .rename("hour_count")
        .reset_index()
    )
    dst_sized_days = counts[counts["hour_count"].isin([23, 25])]

    unexpected: list[dict[str, Any]] = []
    for area, area_counts in counts.groupby("area"):
        complete_candidate = area_counts.iloc[1:-1] if len(area_counts) > 2 else area_counts.iloc[0:0]
        bad = complete_candidate[~complete_candidate["hour_count"].isin([23, 24, 25])]
        unexpected.extend(bad.to_dict(orient="records"))

    return {
        "local_days_with_23_or_25_hours": dst_sized_days.to_dict(orient="records"),
        "unexpected_complete_local_day_counts": unexpected,
    }


def _read_manifest_entries(raw_root: Path, dataset: str) -> list[dict[str, Any]]:
    manifest_path = raw_root / "manifest.jsonl"
    if not manifest_path.exists():
        return []

    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            entry = json.loads(line)
            if entry.get("source_dataset") != dataset:
                continue
            raw_path = str(entry["raw_path"])
            if raw_path in seen_paths:
                continue
            seen_paths.add(raw_path)
            entries.append(entry)
    return entries


def _batch_from_manifest_entry(raw_root: Path, entry: dict[str, Any]) -> RawBatch:
    raw_path = Path(entry["raw_path"])
    if not raw_path.is_absolute():
        raw_path = raw_root / raw_path
    _verify_manifest_hashes(raw_path, entry)
    payload = _read_raw_payload(raw_path)
    return RawBatch(
        source_dataset=str(entry["source_dataset"]),
        batch_id=str(entry["batch_id"]),
        retrieved_at_utc=str(entry.get("retrieved_at_utc", "")),
        raw_path=raw_path,
        records=payload["records"],
    )


def _load_raw_batches_from_glob(raw_root: Path, dataset: str) -> list[RawBatch]:
    batches: list[RawBatch] = []
    for path in sorted((raw_root / dataset).glob("fetched_at=*/start=*_end=*.json")):
        payload = _read_raw_payload(path)
        saved_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        retrieved_at = path.parent.name.removeprefix("fetched_at=")
        batches.append(
            RawBatch(
                source_dataset=dataset,
                batch_id=f"{dataset}_{saved_sha256[:12]}",
                retrieved_at_utc=retrieved_at,
                raw_path=path,
                records=payload["records"],
            )
        )
    return batches


def _read_raw_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Raw EDS payload has no records list: {path}")
    return payload


def _verify_manifest_hashes(raw_path: Path, entry: dict[str, Any]) -> None:
    expected_saved = entry.get("saved_json_sha256")
    expected_response = entry.get("response_sha256")
    if expected_saved and expected_response and expected_saved != expected_response:
        raise ValueError(
            "Raw EDS manifest has inconsistent response and saved hashes for "
            f"{raw_path}: response_sha256={expected_response}, "
            f"saved_json_sha256={expected_saved}"
        )

    expected = expected_saved or expected_response
    if not expected:
        return

    actual = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            "Raw EDS file hash does not match manifest for "
            f"{raw_path}: expected {expected}, got {actual}"
        )


def _record_in_local_range(
    record: dict[str, Any],
    spec: SourceSpec,
    *,
    start_local: date | None,
    end_local: date | None,
) -> bool:
    if start_local is None and end_local is None:
        return True

    local_value = record.get(spec.local_time_column)
    if local_value is None:
        return True

    try:
        local_date = pd.Timestamp(local_value).date()
    except ValueError as exc:
        raise ValueError(
            f"Could not parse source local timestamp {local_value!r} "
            f"for {spec.dataset}"
        ) from exc

    if start_local is not None and local_date < start_local:
        return False
    if end_local is not None and local_date >= end_local:
        return False
    return True


def _validate_record(record: dict[str, Any], spec: SourceSpec, dataset: str, index: int) -> None:
    required = spec.columns
    missing = [column for column in required if record.get(column) is None]
    if missing:
        raise ValueError(f"{dataset} record {index} is missing required columns: {missing}")


def _validate_source_local_timestamps(frame: pd.DataFrame, *, dataset: str) -> None:
    source_local = pd.to_datetime(frame["source_time_local_text"], errors="raise")
    derived_local = frame["source_time_utc"].dt.tz_convert(COPENHAGEN_TZ).dt.tz_localize(None)
    matches = source_local == derived_local
    if bool(matches.all()):
        return

    sample = (
        frame.loc[~matches, ["source_time_utc", "source_time_local_text", "area"]]
        .assign(derived_local=derived_local.loc[~matches].astype(str))
        .head(10)
        .to_dict(orient="records")
    )
    raise ValueError(
        f"{dataset} source local timestamps do not match UTC-derived "
        f"{COPENHAGEN_TZ} timestamps: {sample}"
    )


def _empty_normalized_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=NORMALIZED_COLUMNS)
    frame["source_time_utc"] = pd.to_datetime(frame["source_time_utc"], utc=True)
    return frame


def _empty_hourly_source_frame() -> pd.DataFrame:
    frame = pd.DataFrame(
        columns=[
            "area",
            "ds_utc",
            "price_dkk_per_mwh",
            "price_eur_per_mwh",
            "source_dataset",
            "source_resolution_minutes",
        ]
    )
    frame["ds_utc"] = pd.to_datetime(frame["ds_utc"], utc=True)
    return frame


def _invalid_quarter_hour_message(invalid: pd.DataFrame) -> str:
    sample = _quarter_hour_group_sample(invalid)
    return f"DayAheadPrices has incomplete hourly quarter-hour groups: {sample}"


def _quarter_hour_group_sample(invalid: pd.DataFrame) -> list[dict[str, Any]]:
    if invalid.empty:
        return []
    sample = invalid.head(10).copy()
    sample["hour_utc"] = sample["hour_utc"].map(lambda value: pd.Timestamp(value).isoformat())
    columns = [
        column
        for column in ["area", "hour_utc", "quarter_hour_count", "is_latest_for_area"]
        if column in sample.columns
    ]
    return sample[columns].to_dict(orient="records")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value
