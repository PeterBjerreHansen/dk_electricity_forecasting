from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


DATASET_VERSION = "fundamentals_v1"
ALLOWED_AVAILABILITY_TIME_TYPES = (
    "observed_source_publication",
    "observed_retrieval",
    "provider_schedule_proxy",
    "synthetic_proxy",
)


@dataclass(frozen=True)
class FundamentalParameterSpec:
    unit: str
    hourly_aggregation: str
    entity_types: tuple[str, ...]


PARAMETER_SPECS: dict[str, FundamentalParameterSpec] = {
    "total_load_forecast": FundamentalParameterSpec(
        unit="MW",
        hourly_aggregation="time_weighted_mean",
        entity_types=("bidding_zone",),
    ),
    "wind_generation_forecast": FundamentalParameterSpec(
        unit="MW",
        hourly_aggregation="time_weighted_mean",
        entity_types=("bidding_zone",),
    ),
    "solar_generation_forecast": FundamentalParameterSpec(
        unit="MW",
        hourly_aggregation="time_weighted_mean",
        entity_types=("bidding_zone",),
    ),
    "available_transfer_capacity": FundamentalParameterSpec(
        unit="MW",
        hourly_aggregation="time_weighted_mean",
        entity_types=("directed_border",),
    ),
    "planned_transmission_outage_flag": FundamentalParameterSpec(
        unit="boolean",
        hourly_aggregation="maximum",
        entity_types=("directed_border",),
    ),
}
ALLOWED_PARAMETERS = tuple(PARAMETER_SPECS)

NATIVE_COLUMNS = [
    "source_provider",
    "source_dataset",
    "source_document_id",
    "source_series_id",
    "source_revision",
    "fundamentals_vintage_id",
    "parameter_id",
    "entity_type",
    "area",
    "from_area",
    "to_area",
    "valid_time_start_utc",
    "valid_time_end_utc",
    "native_resolution_minutes",
    "value",
    "unit",
    "forecast_reference_time_utc",
    "forecast_reference_time_type",
    "published_at_utc",
    "available_at_utc",
    "availability_time_type",
    "raw_batch_id",
    "retrieved_at_utc",
]

HOURLY_COLUMNS = [
    "source_provider",
    "source_dataset",
    "source_document_id",
    "source_series_id",
    "source_revision",
    "fundamentals_vintage_id",
    "parameter_id",
    "feature_name",
    "entity_type",
    "area",
    "from_area",
    "to_area",
    "ds_utc",
    "native_resolution_minutes",
    "value",
    "unit",
    "forecast_reference_time_utc",
    "forecast_reference_time_type",
    "published_at_utc",
    "available_at_utc",
    "availability_time_type",
    "raw_batch_id",
    "retrieved_at_utc",
    "hourly_aggregation",
    "interval_count",
    "expected_interval_count",
    "interval_coverage_ratio",
    "interval_coverage_pass",
    "dataset_version",
]
FEATURE_IDENTITY_COLUMNS = [
    "source_provider",
    "parameter_id",
    "entity_type",
    "area",
    "from_area",
    "to_area",
    "unit",
]

_REQUIRED_STRING_COLUMNS = [
    "source_provider",
    "source_dataset",
    "source_document_id",
    "source_series_id",
    "source_revision",
    "fundamentals_vintage_id",
    "parameter_id",
    "entity_type",
    "unit",
    "forecast_reference_time_type",
    "availability_time_type",
    "raw_batch_id",
]
_UTC_COLUMNS = [
    "valid_time_start_utc",
    "valid_time_end_utc",
    "forecast_reference_time_utc",
    "published_at_utc",
    "available_at_utc",
    "retrieved_at_utc",
]
_HOURLY_GROUP_COLUMNS = [
    "source_provider",
    "source_dataset",
    "source_document_id",
    "source_series_id",
    "source_revision",
    "fundamentals_vintage_id",
    "parameter_id",
    "entity_type",
    "area",
    "from_area",
    "to_area",
    "native_resolution_minutes",
    "unit",
    "forecast_reference_time_utc",
    "forecast_reference_time_type",
    "published_at_utc",
    "available_at_utc",
    "availability_time_type",
    "raw_batch_id",
    "retrieved_at_utc",
    "_hour_start_utc",
]


def normalize_native_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate and canonicalize native-resolution point-in-time fundamentals."""

    _require_columns(frame, NATIVE_COLUMNS, "native fundamentals")
    output = frame[NATIVE_COLUMNS].copy()
    for column in _UTC_COLUMNS:
        parsed = pd.to_datetime(output[column], utc=True, errors="coerce")
        invalid = output[column].notna() & parsed.isna()
        if invalid.any():
            raise ValueError(f"native fundamentals has invalid {column}")
        output[column] = parsed

    _validate_required_values(output)
    _validate_parameter_contract(output)
    _validate_time_contract(output)
    _validate_native_intervals(output)
    _validate_duplicate_native_rows(output)

    output["native_resolution_minutes"] = output["native_resolution_minutes"].astype("int16")
    output["value"] = pd.to_numeric(output["value"], errors="raise").astype("float64")
    return output.sort_values(
        [
            "area",
            "from_area",
            "to_area",
            "parameter_id",
            "valid_time_start_utc",
            "forecast_reference_time_utc",
            "available_at_utc",
            "source_revision",
            "fundamentals_vintage_id",
        ],
        na_position="last",
    ).reset_index(drop=True)


def build_hourly_fundamentals(
    native: pd.DataFrame,
    *,
    dataset_version: str = DATASET_VERSION,
) -> pd.DataFrame:
    """Aggregate native intervals within each source vintage to exact UTC hours.

    Revisions and vintages are grouping keys. An incomplete vintage-hour is
    retained for auditability, but its value is null and its coverage flag is
    false.
    """

    normalized = normalize_native_fundamentals(native)
    if normalized.empty:
        return _empty_hourly_frame()

    working = normalized.copy()
    working["_hour_start_utc"] = working["valid_time_start_utc"].dt.floor("h")
    rows = [
        _aggregate_hour(group, key, dataset_version=dataset_version)
        for key, group in working.groupby(
            _HOURLY_GROUP_COLUMNS,
            dropna=False,
            sort=False,
        )
    ]
    hourly = pd.DataFrame(rows, columns=HOURLY_COLUMNS)
    validate_feature_name_uniqueness(hourly)
    return hourly.sort_values(
        [
            "area",
            "from_area",
            "to_area",
            "parameter_id",
            "ds_utc",
            "forecast_reference_time_utc",
            "available_at_utc",
            "source_revision",
            "fundamentals_vintage_id",
        ],
        na_position="last",
    ).reset_index(drop=True)


def fundamental_feature_name(
    parameter_id: str,
    *,
    source_provider: str,
    entity_type: str,
    unit: str,
    area: object = None,
    from_area: object = None,
    to_area: object = None,
) -> str:
    """Return the stable model feature name for one fundamental series."""

    provider = _safe_name(source_provider)
    parameter = _safe_name(parameter_id)
    unit_name = _safe_name(unit)
    if entity_type == "bidding_zone":
        if _is_missing_or_blank_value(area):
            raise ValueError(f"{parameter_id} bidding-zone rows require area")
        geography = _safe_name(area)
    elif entity_type == "directed_border":
        if _is_missing_or_blank_value(from_area) or _is_missing_or_blank_value(
            to_area
        ):
            raise ValueError(
                f"{parameter_id} directed-border rows require from_area and to_area"
            )
        geography = f"{_safe_name(from_area)}_to_{_safe_name(to_area)}"
    else:
        raise ValueError(f"Unsupported fundamentals entity_type: {entity_type!r}")
    return f"fundamental_{provider}_{geography}_{parameter}_{unit_name}"


def validate_feature_name_uniqueness(frame: pd.DataFrame) -> None:
    """Reject canonical-name collisions across distinct source identities."""

    _require_columns(
        frame,
        ["feature_name", *FEATURE_IDENTITY_COLUMNS],
        "fundamentals frame",
    )
    identities = frame[
        ["feature_name", *FEATURE_IDENTITY_COLUMNS]
    ].drop_duplicates()
    collisions = identities[
        identities.duplicated("feature_name", keep=False)
    ]
    if not collisions.empty:
        names = sorted(collisions["feature_name"].unique())
        raise ValueError(
            "Canonical fundamentals feature names collide across source "
            f"identities: {names}"
        )


def _aggregate_hour(
    group: pd.DataFrame,
    key: tuple[Any, ...],
    *,
    dataset_version: str,
) -> dict[str, Any]:
    lineage = dict(zip(_HOURLY_GROUP_COLUMNS, key))
    ordered = group.sort_values("valid_time_start_utc")
    hour_start = pd.Timestamp(lineage.pop("_hour_start_utc"))
    resolution = int(lineage["native_resolution_minutes"])
    expected_count = 60 // resolution
    interval_count = len(ordered)
    coverage_minutes = float(
        (
            ordered["valid_time_end_utc"] - ordered["valid_time_start_utc"]
        ).dt.total_seconds().sum()
        / 60.0
    )
    coverage_ratio = coverage_minutes / 60.0
    contiguous = _intervals_cover_exact_hour(ordered, hour_start)
    coverage_pass = (
        contiguous
        and interval_count == expected_count
        and abs(coverage_ratio - 1.0) < 1e-12
    )

    parameter_id = str(lineage["parameter_id"])
    spec = PARAMETER_SPECS[parameter_id]
    value: float | None
    if not coverage_pass:
        value = None
    elif spec.hourly_aggregation == "maximum":
        value = float(ordered["value"].max())
    else:
        durations = (
            ordered["valid_time_end_utc"] - ordered["valid_time_start_utc"]
        ).dt.total_seconds() / 60.0
        value = float((ordered["value"] * durations).sum() / durations.sum())

    return {
        **{column: lineage[column] for column in _HOURLY_GROUP_COLUMNS[:-1]},
        "feature_name": fundamental_feature_name(
            parameter_id,
            source_provider=str(lineage["source_provider"]),
            entity_type=str(lineage["entity_type"]),
            unit=str(lineage["unit"]),
            area=lineage["area"],
            from_area=lineage["from_area"],
            to_area=lineage["to_area"],
        ),
        "ds_utc": hour_start,
        "value": value,
        "hourly_aggregation": spec.hourly_aggregation,
        "interval_count": interval_count,
        "expected_interval_count": expected_count,
        "interval_coverage_ratio": coverage_ratio,
        "interval_coverage_pass": coverage_pass,
        "dataset_version": dataset_version,
    }


def _validate_required_values(frame: pd.DataFrame) -> None:
    for column in _REQUIRED_STRING_COLUMNS:
        values = frame[column]
        if values.isna().any() or values.astype(str).str.strip().eq("").any():
            raise ValueError(f"native fundamentals has missing or blank {column}")

    for column in [
        "valid_time_start_utc",
        "valid_time_end_utc",
        "forecast_reference_time_utc",
        "available_at_utc",
        "retrieved_at_utc",
    ]:
        if frame[column].isna().any():
            raise ValueError(f"native fundamentals has invalid or missing {column}")

    if frame["value"].isna().any():
        raise ValueError("native fundamentals has missing value")
    numeric_values = pd.to_numeric(frame["value"], errors="coerce")
    if numeric_values.isna().any():
        raise ValueError("native fundamentals value must be numeric")
    if (~numeric_values.map(_is_finite)).any():
        raise ValueError("native fundamentals value must be finite")


def _validate_parameter_contract(frame: pd.DataFrame) -> None:
    unknown_parameters = sorted(set(frame["parameter_id"]) - set(ALLOWED_PARAMETERS))
    if unknown_parameters:
        raise ValueError(f"Unsupported fundamentals parameter_id values: {unknown_parameters}")

    unknown_availability_types = sorted(
        set(frame["availability_time_type"]) - set(ALLOWED_AVAILABILITY_TIME_TYPES)
    )
    if unknown_availability_types:
        raise ValueError(
            "Unsupported fundamentals availability_time_type values: "
            f"{unknown_availability_types}"
        )

    numeric_values = pd.to_numeric(frame["value"], errors="raise")
    for parameter_id, group in frame.groupby("parameter_id", sort=False):
        spec = PARAMETER_SPECS[str(parameter_id)]
        if set(group["unit"]) != {spec.unit}:
            raise ValueError(
                f"{parameter_id} must use unit {spec.unit!r}; "
                f"got {sorted(set(group['unit']))}"
            )
        invalid_entities = sorted(set(group["entity_type"]) - set(spec.entity_types))
        if invalid_entities:
            raise ValueError(
                f"{parameter_id} has unsupported entity_type values: {invalid_entities}"
            )

        values = numeric_values.loc[group.index]
        if parameter_id == "planned_transmission_outage_flag":
            if not values.isin([0.0, 1.0]).all():
                raise ValueError(
                    "planned_transmission_outage_flag values must be 0 or 1"
                )
        elif (values < 0).any():
            raise ValueError(f"{parameter_id} values must be non-negative")

    bidding_zone = frame["entity_type"].eq("bidding_zone")
    if _missing_or_blank(frame.loc[bidding_zone, "area"]).any():
        raise ValueError("bidding_zone fundamentals require area")
    if (
        ~_missing_or_blank(frame.loc[bidding_zone, "from_area"])
        | ~_missing_or_blank(frame.loc[bidding_zone, "to_area"])
    ).any():
        raise ValueError("bidding_zone fundamentals require null from_area and to_area")

    directed_border = frame["entity_type"].eq("directed_border")
    if (
        _missing_or_blank(frame.loc[directed_border, "from_area"]).any()
        or _missing_or_blank(frame.loc[directed_border, "to_area"]).any()
    ):
        raise ValueError(
            "directed_border fundamentals require from_area and to_area"
        )
    if (~_missing_or_blank(frame.loc[directed_border, "area"])).any():
        raise ValueError("directed_border fundamentals require null area")
    if frame.loc[directed_border, "from_area"].eq(
        frame.loc[directed_border, "to_area"]
    ).any():
        raise ValueError("directed_border from_area and to_area must differ")


def _validate_time_contract(frame: pd.DataFrame) -> None:
    if (frame["forecast_reference_time_utc"] > frame["available_at_utc"]).any():
        raise ValueError(
            "forecast_reference_time_utc must not be after available_at_utc"
        )
    published = frame["published_at_utc"].notna()
    if (
        frame.loc[published, "published_at_utc"]
        > frame.loc[published, "retrieved_at_utc"]
    ).any():
        raise ValueError("published_at_utc must not be after retrieved_at_utc")
    if (frame["available_at_utc"] > frame["retrieved_at_utc"]).any():
        raise ValueError("available_at_utc must not be after retrieved_at_utc")

    observed_publication = frame["availability_time_type"].eq(
        "observed_source_publication"
    )
    if frame.loc[observed_publication, "published_at_utc"].isna().any():
        raise ValueError(
            "observed_source_publication rows require published_at_utc"
        )
    if not frame.loc[observed_publication, "available_at_utc"].equals(
        frame.loc[observed_publication, "published_at_utc"]
    ):
        raise ValueError(
            "observed_source_publication requires available_at_utc "
            "to equal published_at_utc"
        )

    observed_retrieval = frame["availability_time_type"].eq("observed_retrieval")
    if not frame.loc[observed_retrieval, "available_at_utc"].equals(
        frame.loc[observed_retrieval, "retrieved_at_utc"]
    ):
        raise ValueError(
            "observed_retrieval requires available_at_utc to equal retrieved_at_utc"
        )


def _validate_native_intervals(frame: pd.DataFrame) -> None:
    resolutions = pd.to_numeric(
        frame["native_resolution_minutes"],
        errors="coerce",
    )
    if resolutions.isna().any() or (resolutions <= 0).any():
        raise ValueError("native_resolution_minutes must be a positive integer")
    if (resolutions % 1 != 0).any() or (60 % resolutions != 0).any():
        raise ValueError("native_resolution_minutes must be an integer divisor of 60")

    durations = (
        frame["valid_time_end_utc"] - frame["valid_time_start_utc"]
    ).dt.total_seconds() / 60.0
    if not durations.equals(resolutions.astype("float64")):
        raise ValueError(
            "Each native interval duration must equal native_resolution_minutes"
        )
    if (
        frame["valid_time_start_utc"].dt.second.ne(0).any()
        or frame["valid_time_end_utc"].dt.second.ne(0).any()
        or frame["valid_time_start_utc"].dt.microsecond.ne(0).any()
        or frame["valid_time_end_utc"].dt.microsecond.ne(0).any()
    ):
        raise ValueError("Native interval boundaries must align to whole minutes")
    end_inside_hour = frame["valid_time_end_utc"] - pd.Timedelta(nanoseconds=1)
    if not frame["valid_time_start_utc"].dt.floor("h").equals(
        end_inside_hour.dt.floor("h")
    ):
        raise ValueError("Native intervals may not cross UTC hour boundaries")


def _validate_duplicate_native_rows(frame: pd.DataFrame) -> None:
    identity_columns = [
        "source_provider",
        "source_dataset",
        "source_series_id",
        "source_revision",
        "fundamentals_vintage_id",
        "parameter_id",
        "entity_type",
        "area",
        "from_area",
        "to_area",
        "valid_time_start_utc",
        "valid_time_end_utc",
    ]
    duplicates = frame.duplicated(identity_columns, keep=False)
    if duplicates.any():
        raise ValueError(
            "Duplicate native fundamentals rows within the same source revision "
            "and vintage"
        )


def _intervals_cover_exact_hour(
    ordered: pd.DataFrame,
    hour_start: pd.Timestamp,
) -> bool:
    if ordered.empty:
        return False
    starts = ordered["valid_time_start_utc"].tolist()
    ends = ordered["valid_time_end_utc"].tolist()
    if starts[0] != hour_start or ends[-1] != hour_start + pd.Timedelta(hours=1):
        return False
    return all(current_end == next_start for current_end, next_start in zip(ends, starts[1:]))


def _safe_name(value: object) -> str:
    text = str(value).strip().lower()
    output = "".join(
        character if character.isalnum() else "_"
        for character in text
    ).strip("_")
    if not output:
        raise ValueError(f"Fundamentals feature-name component is empty: {value!r}")
    return output


def _is_finite(value: object) -> bool:
    numeric = float(value)
    return numeric != float("inf") and numeric != float("-inf")


def _missing_or_blank(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype("string").str.strip().eq("").fillna(True)


def _is_missing_or_blank_value(value: object) -> bool:
    return bool(pd.isna(value) or not str(value).strip())


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    frame_name: str,
) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def _empty_hourly_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=HOURLY_COLUMNS)
