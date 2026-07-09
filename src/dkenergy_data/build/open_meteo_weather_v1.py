from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


SOURCE_PROVIDER = "open_meteo"
SOURCE_PRODUCT = "previous_runs"
DATASET_VERSION = "open_meteo_previous_runs_v1"
OPEN_METEO_MODELS = ("gfs_global", "icon_eu", "metno_nordic")
BASE_VARIABLES = (
    "temperature_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
)
LEAD_TIME_DAYS = (1, 2)
COVERAGE_THRESHOLD = 0.95


@dataclass(frozen=True)
class WeatherLocation:
    location_id: str
    area: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class OpenMeteoRawBatch:
    batch_id: str
    weather_model: str
    location_id: str
    retrieved_at_utc: str
    raw_path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class OpenMeteoBuildResult:
    normalized_path: Path
    area_features_long_path: Path
    qa_path: Path
    qa: dict[str, Any]
    area_features_wide_path: Path | None = None


OPEN_METEO_LOCATION_BASKET: tuple[WeatherLocation, ...] = (
    WeatherLocation("dk1_aalborg", "DK1", 57.0488, 9.9217),
    WeatherLocation("dk1_aarhus", "DK1", 56.1629, 10.2039),
    WeatherLocation("dk1_esbjerg", "DK1", 55.4765, 8.4594),
    WeatherLocation("dk1_odense", "DK1", 55.4038, 10.4024),
    WeatherLocation("dk1_herning", "DK1", 56.1393, 8.9738),
    WeatherLocation("dk2_copenhagen", "DK2", 55.6761, 12.5683),
    WeatherLocation("dk2_holbaek", "DK2", 55.7175, 11.7128),
    WeatherLocation("dk2_naestved", "DK2", 55.2299, 11.7609),
    WeatherLocation("dk2_nykobing_falster", "DK2", 54.7654, 11.8755),
    WeatherLocation("dk2_roenne", "DK2", 55.1009, 14.7066),
)

NORMALIZED_COLUMNS = [
    "source_provider",
    "source_product",
    "weather_model",
    "model",
    "lead_time_days",
    "lead_time_hours",
    "location_id",
    "area",
    "price_area",
    "latitude",
    "longitude",
    "valid_time_utc",
    "valid_time",
    "forecast_available_at_utc",
    "forecast_reference_time",
    "parameter_id",
    "variable",
    "value",
    "unit",
    "raw_batch_id",
    "retrieved_at_utc",
]

AREA_FEATURE_LONG_COLUMNS = [
    "area",
    "price_area",
    "ds_utc",
    "valid_time",
    "weather_model",
    "model",
    "lead_time_days",
    "lead_time_hours",
    "parameter_id",
    "variable",
    "feature_name",
    "value",
    "unit",
    "location_count",
    "expected_location_count",
    "location_coverage_ratio",
    "location_coverage_pass",
    "feature_window_coverage_ratio",
    "feature_group_pass",
    "forecast_available_at_utc",
    "forecast_reference_time",
    "dataset_version",
]


def location_manifest_frame(
    locations: Iterable[WeatherLocation] = OPEN_METEO_LOCATION_BASKET,
) -> pd.DataFrame:
    return pd.DataFrame([location.__dict__ for location in locations])


def load_raw_batches(raw_root: Path) -> list[OpenMeteoRawBatch]:
    entries = _read_manifest_entries(raw_root)
    if entries:
        return [_batch_from_manifest_entry(raw_root, entry) for entry in entries]
    return _load_raw_batches_from_glob(raw_root)


def normalize_batches(
    batches: Iterable[OpenMeteoRawBatch],
    *,
    locations: Iterable[WeatherLocation] = OPEN_METEO_LOCATION_BASKET,
    base_variables: Iterable[str] = BASE_VARIABLES,
    lead_time_days: Iterable[int] = LEAD_TIME_DAYS,
    min_valid_time: str | pd.Timestamp | None = None,
    max_valid_time: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    locations_tuple = tuple(locations)
    base_variables_tuple = tuple(base_variables)
    lead_time_days_tuple = tuple(lead_time_days)
    min_valid_timestamp = _optional_utc_timestamp(min_valid_time)
    max_valid_timestamp = _optional_utc_timestamp(max_valid_time, inclusive_date_end=True)
    frames = [
        normalize_batch(
            batch,
            locations=locations_tuple,
            base_variables=base_variables_tuple,
            lead_time_days=lead_time_days_tuple,
            min_valid_time=min_valid_timestamp,
            max_valid_time=max_valid_timestamp,
        )
        for batch in batches
    ]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return _empty_normalized_frame()
    normalized = pd.concat(frames, ignore_index=True)[NORMALIZED_COLUMNS]
    return deduplicate_normalized(normalized)


def normalize_batch(
    batch: OpenMeteoRawBatch,
    *,
    locations: Iterable[WeatherLocation] = OPEN_METEO_LOCATION_BASKET,
    base_variables: Iterable[str] = BASE_VARIABLES,
    lead_time_days: Iterable[int] = LEAD_TIME_DAYS,
    min_valid_time: str | pd.Timestamp | None = None,
    max_valid_time: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    location_by_id = {location.location_id: location for location in locations}
    if batch.location_id not in location_by_id:
        raise ValueError(f"Unknown Open-Meteo location_id in raw batch: {batch.location_id}")
    location = location_by_id[batch.location_id]

    hourly = batch.payload.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError(f"Open-Meteo payload has no hourly object: {batch.raw_path}")
    times = hourly.get("time")
    if not isinstance(times, list):
        raise ValueError(f"Open-Meteo payload has no hourly time list: {batch.raw_path}")

    valid_time = pd.Series(pd.to_datetime(times, utc=True))
    min_valid_timestamp = _optional_utc_timestamp(min_valid_time)
    max_valid_timestamp = _optional_utc_timestamp(max_valid_time, inclusive_date_end=True)
    valid_mask = pd.Series(True, index=valid_time.index)
    if min_valid_timestamp is not None:
        valid_mask &= valid_time >= min_valid_timestamp
    if max_valid_timestamp is not None:
        valid_mask &= valid_time <= max_valid_timestamp
    if not bool(valid_mask.any()):
        return _empty_normalized_frame()

    valid_time = valid_time.loc[valid_mask].reset_index(drop=True)
    units = batch.payload.get("hourly_units") if isinstance(batch.payload.get("hourly_units"), dict) else {}
    rows: list[pd.DataFrame] = []

    for variable in base_variables:
        for lead_day in lead_time_days:
            raw_key = f"{variable}_previous_day{lead_day}"
            values = hourly.get(raw_key)
            if values is None:
                continue
            if not isinstance(values, list) or len(values) != len(valid_time):
                if isinstance(values, list) and len(values) == len(times):
                    values = pd.Series(values, dtype="object").loc[valid_mask].reset_index(drop=True).tolist()
                else:
                    raise ValueError(f"Open-Meteo hourly field has wrong length: {raw_key}")
            if not values:
                continue
            if len(values) != len(valid_time):
                raise ValueError(f"Open-Meteo hourly field has wrong length: {raw_key}")

            frame = pd.DataFrame(
                {
                    "source_provider": SOURCE_PROVIDER,
                    "source_product": SOURCE_PRODUCT,
                    "weather_model": batch.weather_model,
                    "model": batch.weather_model,
                    "lead_time_days": int(lead_day),
                    "lead_time_hours": int(lead_day) * 24,
                    "location_id": batch.location_id,
                    "area": location.area,
                    "price_area": location.area,
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "valid_time_utc": valid_time.to_numpy(),
                    "valid_time": valid_time.to_numpy(),
                    "forecast_available_at_utc": valid_time - pd.Timedelta(days=int(lead_day)),
                    "forecast_reference_time": valid_time - pd.Timedelta(days=int(lead_day)),
                    "parameter_id": variable,
                    "variable": variable,
                    "value": pd.to_numeric(pd.Series(values, dtype="object"), errors="coerce"),
                    "unit": units.get(raw_key),
                    "raw_batch_id": batch.batch_id,
                    "retrieved_at_utc": batch.retrieved_at_utc,
                }
            )
            rows.append(frame)

    if not rows:
        return _empty_normalized_frame()
    return pd.concat(rows, ignore_index=True)[NORMALIZED_COLUMNS]


def deduplicate_normalized(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return _empty_normalized_frame()

    key_cols = [
        "source_provider",
        "source_product",
        "weather_model",
        "location_id",
        "valid_time_utc",
        "lead_time_days",
        "parameter_id",
    ]
    batch_key_cols = [*key_cols, "raw_batch_id"]
    value_cols = [
        column
        for column in NORMALIZED_COLUMNS
        if column not in {*key_cols, "raw_batch_id", "retrieved_at_utc"}
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
        raise ValueError(f"Conflicting duplicate normalized Open-Meteo rows: {sample}")

    deduped = (
        frame.sort_values(
            [
                "weather_model",
                "location_id",
                "valid_time_utc",
                "lead_time_days",
                "parameter_id",
                "retrieved_at_utc",
                "raw_batch_id",
            ]
        )
        .drop_duplicates(key_cols, keep="last")
        .reset_index(drop=True)
    )
    return deduped[NORMALIZED_COLUMNS]


def build_area_feature_long(
    normalized: pd.DataFrame,
    *,
    locations: Iterable[WeatherLocation] = OPEN_METEO_LOCATION_BASKET,
    dataset_version: str = DATASET_VERSION,
    coverage_threshold: float = COVERAGE_THRESHOLD,
) -> pd.DataFrame:
    if normalized.empty:
        return _empty_area_feature_long_frame()
    if not 0 < coverage_threshold <= 1:
        raise ValueError("coverage_threshold must be in (0, 1]")

    expected_counts = location_manifest_frame(locations).groupby("area").size().to_dict()
    frame = normalized.copy()
    frame["valid_time_utc"] = pd.to_datetime(frame["valid_time_utc"], utc=True)
    frame["forecast_available_at_utc"] = pd.to_datetime(frame["forecast_available_at_utc"], utc=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    expected_hours = (
        frame.groupby(
            [
                "area",
                "weather_model",
                "lead_time_days",
                "parameter_id",
            ],
            as_index=False,
        )
        .agg(expected_valid_hour_count=("valid_time_utc", "nunique"))
    )
    frame = frame.dropna(subset=["value"])

    grouped = (
        frame.groupby(
            [
                "area",
                "valid_time_utc",
                "weather_model",
                "lead_time_days",
                "parameter_id",
            ],
            as_index=False,
        )
        .agg(
            value=("value", "mean"),
            unit=("unit", "first"),
            location_count=("location_id", "nunique"),
            forecast_available_at_utc=("forecast_available_at_utc", "max"),
        )
        .rename(columns={"valid_time_utc": "ds_utc"})
    )
    grouped["expected_location_count"] = grouped["area"].map(expected_counts).astype("int16")
    grouped["location_coverage_ratio"] = (
        grouped["location_count"] / grouped["expected_location_count"]
    )
    grouped["location_coverage_pass"] = grouped["location_coverage_ratio"] >= coverage_threshold
    group_keys = ["area", "weather_model", "lead_time_days", "parameter_id"]
    passing_hours = (
        grouped[grouped["location_coverage_pass"]]
        .groupby(group_keys, as_index=False)
        .agg(valid_hour_count=("ds_utc", "nunique"))
    )
    grouped = grouped.merge(expected_hours, on=group_keys, how="left")
    grouped = grouped.merge(passing_hours, on=group_keys, how="left")
    grouped["valid_hour_count"] = grouped["valid_hour_count"].fillna(0)
    grouped["feature_window_coverage_ratio"] = (
        grouped["valid_hour_count"] / grouped["expected_valid_hour_count"]
    )
    grouped["feature_group_pass"] = (
        grouped["feature_window_coverage_ratio"] >= coverage_threshold
    )
    grouped["feature_name"] = grouped.apply(
        lambda row: weather_feature_name(
            row["weather_model"],
            int(row["lead_time_days"]),
            row["parameter_id"],
        ),
        axis=1,
    )
    grouped["price_area"] = grouped["area"]
    grouped["valid_time"] = grouped["ds_utc"]
    grouped["model"] = grouped["weather_model"]
    grouped["lead_time_hours"] = grouped["lead_time_days"].astype("int16") * 24
    grouped["variable"] = grouped["parameter_id"]
    grouped["forecast_reference_time"] = grouped["forecast_available_at_utc"]
    grouped["dataset_version"] = dataset_version
    return grouped[AREA_FEATURE_LONG_COLUMNS].sort_values(
        ["area", "ds_utc", "weather_model", "lead_time_days", "parameter_id"]
    ).reset_index(drop=True)


def build_area_feature_wide(area_features_long: pd.DataFrame) -> pd.DataFrame:
    if area_features_long.empty:
        return pd.DataFrame(columns=["area", "ds_utc"])
    frame = area_features_long.copy()
    frame["ds_utc"] = pd.to_datetime(frame["ds_utc"], utc=True)

    values = frame.pivot_table(
        index=["area", "ds_utc"],
        columns="feature_name",
        values="value",
        aggfunc="last",
    )
    coverage = frame.pivot_table(
        index=["area", "ds_utc"],
        columns="feature_name",
        values="location_coverage_ratio",
        aggfunc="last",
    ).rename(columns=lambda column: f"{column}_coverage_ratio")
    passes = frame.pivot_table(
        index=["area", "ds_utc"],
        columns="feature_name",
        values="feature_group_pass",
        aggfunc="last",
    ).rename(columns=lambda column: f"{column}_passes_coverage")
    location_passes = frame.pivot_table(
        index=["area", "ds_utc"],
        columns="feature_name",
        values="location_coverage_pass",
        aggfunc="last",
    ).rename(columns=lambda column: f"{column}_passes_location_coverage")
    availability = frame.pivot_table(
        index=["area", "ds_utc"],
        columns="feature_name",
        values="forecast_available_at_utc",
        aggfunc="last",
    ).rename(columns=lambda column: f"{column}_available_at_utc")

    wide = pd.concat([values, coverage, passes, location_passes, availability], axis=1).reset_index()
    wide.columns.name = None
    return wide.sort_values(["area", "ds_utc"]).reset_index(drop=True)


def build_open_meteo_weather_from_raw(
    *,
    raw_root: Path,
    normalized_dir: Path,
    features_dir: Path,
    dataset_version: str = DATASET_VERSION,
    coverage_threshold: float = COVERAGE_THRESHOLD,
    write_wide: bool = False,
    min_valid_time: str | pd.Timestamp | None = None,
    max_valid_time: str | pd.Timestamp | None = None,
) -> OpenMeteoBuildResult:
    batches = load_raw_batches(raw_root)
    if not batches:
        raise ValueError(f"No Open-Meteo raw batches found under {raw_root}")

    normalized = normalize_batches(
        batches,
        min_valid_time=min_valid_time,
        max_valid_time=max_valid_time,
    )
    area_long = build_area_feature_long(
        normalized,
        dataset_version=dataset_version,
        coverage_threshold=coverage_threshold,
    )
    qa = make_qa_report(
        normalized,
        area_long,
        dataset_version=dataset_version,
        coverage_threshold=coverage_threshold,
        raw_batch_count=len(batches),
    )

    normalized_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = normalized_dir / f"open_meteo_previous_runs_{dataset_version}.parquet"
    area_long_path = features_dir / f"weather_open_meteo_area_hourly_long_{dataset_version}.parquet"
    qa_path = features_dir / f"weather_open_meteo_area_hourly_{dataset_version}.qa.json"
    normalized.to_parquet(normalized_path, index=False)
    area_long.to_parquet(area_long_path, index=False)
    area_wide_path = None
    if write_wide:
        area_wide_path = features_dir / f"weather_open_meteo_area_hourly_wide_{dataset_version}.parquet"
        build_area_feature_wide(area_long).to_parquet(area_wide_path, index=False)
    qa_path.write_text(json.dumps(_json_safe(qa), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return OpenMeteoBuildResult(
        normalized_path=normalized_path,
        area_features_long_path=area_long_path,
        qa_path=qa_path,
        qa=qa,
        area_features_wide_path=area_wide_path,
    )


def make_qa_report(
    normalized: pd.DataFrame,
    area_features_long: pd.DataFrame,
    *,
    dataset_version: str,
    coverage_threshold: float,
    raw_batch_count: int,
) -> dict[str, Any]:
    return {
        "dataset_version": dataset_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_provider": SOURCE_PROVIDER,
        "source_product": SOURCE_PRODUCT,
        "raw_batch_count": int(raw_batch_count),
        "normalized_row_count": int(len(normalized)),
        "area_feature_row_count": int(len(area_features_long)),
        "coverage_threshold": float(coverage_threshold),
        "areas": sorted(area_features_long["area"].dropna().unique().tolist()) if not area_features_long.empty else [],
        "weather_models": sorted(normalized["weather_model"].dropna().unique().tolist()) if not normalized.empty else [],
        "lead_time_days": sorted([int(value) for value in normalized["lead_time_days"].dropna().unique().tolist()]) if not normalized.empty else [],
        "parameter_ids": sorted(normalized["parameter_id"].dropna().unique().tolist()) if not normalized.empty else [],
        "min_valid_time_utc": normalized["valid_time_utc"].min().isoformat() if not normalized.empty else None,
        "max_valid_time_utc": normalized["valid_time_utc"].max().isoformat() if not normalized.empty else None,
        "null_value_count": int(normalized["value"].isna().sum()) if not normalized.empty else 0,
        "feature_rows_usable_count": (
            int((area_features_long["feature_group_pass"] & area_features_long["location_coverage_pass"]).sum())
            if not area_features_long.empty
            else 0
        ),
        "feature_rows_total_count": int(len(area_features_long)),
        "feature_groups_passing_count": (
            int(
                area_features_long[
                    [
                        "area",
                        "weather_model",
                        "lead_time_days",
                        "parameter_id",
                        "feature_group_pass",
                    ]
                ]
                .drop_duplicates()["feature_group_pass"]
                .sum()
            )
            if not area_features_long.empty
            else 0
        ),
        "feature_groups_total_count": (
            int(
                len(
                    area_features_long[
                        ["area", "weather_model", "lead_time_days", "parameter_id"]
                    ].drop_duplicates()
                )
            )
            if not area_features_long.empty
            else 0
        ),
    }


def weather_feature_name(weather_model: str, lead_time_days: int, parameter_id: str) -> str:
    return f"weather_{_slug(weather_model)}_lead{lead_time_days}d_{_slug(parameter_id)}"


def _read_manifest_entries(raw_root: Path) -> list[dict[str, Any]]:
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
            if entry.get("source_provider") != SOURCE_PROVIDER:
                continue
            if entry.get("source_product") != SOURCE_PRODUCT:
                continue
            raw_path = str(entry["raw_path"])
            if raw_path in seen_paths:
                continue
            seen_paths.add(raw_path)
            entries.append(entry)
    return entries


def _batch_from_manifest_entry(raw_root: Path, entry: dict[str, Any]) -> OpenMeteoRawBatch:
    raw_path = Path(entry["raw_path"])
    if not raw_path.is_absolute():
        raw_path = raw_root / raw_path
    _verify_manifest_hashes(raw_path, entry)
    payload = _read_raw_payload(raw_path)
    return OpenMeteoRawBatch(
        batch_id=str(entry["batch_id"]),
        weather_model=str(entry["weather_model"]),
        location_id=str(entry["location_id"]),
        retrieved_at_utc=str(entry.get("retrieved_at_utc", "")),
        raw_path=raw_path,
        payload=payload,
    )


def _load_raw_batches_from_glob(raw_root: Path) -> list[OpenMeteoRawBatch]:
    batches: list[OpenMeteoRawBatch] = []
    pattern = "previous_runs/*/*/fetched_at=*/start=*_end=*.json"
    for path in sorted(raw_root.glob(pattern)):
        payload = _read_raw_payload(path)
        saved_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        parts = path.relative_to(raw_root).parts
        weather_model = parts[1]
        location_id = parts[2]
        retrieved_at = path.parent.name.removeprefix("fetched_at=")
        batches.append(
            OpenMeteoRawBatch(
                batch_id=f"open_meteo_{weather_model}_{location_id}_{saved_sha256[:12]}",
                weather_model=weather_model,
                location_id=location_id,
                retrieved_at_utc=retrieved_at,
                raw_path=path,
                payload=payload,
            )
        )
    return batches


def _read_raw_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload.get("hourly"), dict):
        raise ValueError(f"Open-Meteo raw payload has no hourly object: {path}")
    return payload


def _verify_manifest_hashes(raw_path: Path, entry: dict[str, Any]) -> None:
    expected_saved = entry.get("saved_json_sha256")
    expected_response = entry.get("response_sha256")
    if expected_saved and expected_response and expected_saved != expected_response:
        raise ValueError(f"Open-Meteo manifest has inconsistent hashes for {raw_path}")
    expected = expected_saved or expected_response
    if not expected:
        return
    actual = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"Open-Meteo raw file hash does not match manifest for {raw_path}: "
            f"expected {expected}, got {actual}"
        )


def _empty_normalized_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=NORMALIZED_COLUMNS)
    frame["valid_time_utc"] = pd.to_datetime(frame["valid_time_utc"], utc=True)
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)
    frame["forecast_available_at_utc"] = pd.to_datetime(frame["forecast_available_at_utc"], utc=True)
    frame["forecast_reference_time"] = pd.to_datetime(frame["forecast_reference_time"], utc=True)
    return frame


def _empty_area_feature_long_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=AREA_FEATURE_LONG_COLUMNS)
    frame["ds_utc"] = pd.to_datetime(frame["ds_utc"], utc=True)
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], utc=True)
    frame["forecast_available_at_utc"] = pd.to_datetime(frame["forecast_available_at_utc"], utc=True)
    frame["forecast_reference_time"] = pd.to_datetime(frame["forecast_reference_time"], utc=True)
    return frame


def _optional_utc_timestamp(
    value: str | pd.Timestamp | None,
    *,
    inclusive_date_end: bool = False,
) -> pd.Timestamp | None:
    if value is None or value == "":
        return None
    timestamp = pd.Timestamp(value)
    if inclusive_date_end and _is_date_only(value):
        timestamp = timestamp + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _is_date_only(value: str | pd.Timestamp) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"\d{4}-\d{2}-\d{2}", value.strip()))


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


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
