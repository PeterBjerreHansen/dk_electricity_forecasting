from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

from dkenergy_data.build.fundamentals_v1 import (
    ALLOWED_AVAILABILITY_TIME_TYPES,
    ALLOWED_PARAMETERS,
    HOURLY_COLUMNS,
    PARAMETER_SPECS,
    fundamental_feature_name,
    validate_feature_name_uniqueness,
)
from dkenergy_forecast.types import normalize_utc_column, require_columns


FUNDAMENTALS_SELECTION_POLICY = (
    "newest_eligible_reference_then_availability_fail_closed_on_revision_tie"
)
FUNDAMENTALS_CUTOFF_COLUMNS = (
    "information_cutoff_utc",
    "forecast_origin_utc",
)

_LINEAGE_SUFFIX_BY_COLUMN = {
    "parameter_id": "_parameter_id",
    "entity_type": "_entity_type",
    "area": "_source_area",
    "from_area": "_from_area",
    "to_area": "_to_area",
    "unit": "_unit",
    "source_provider": "_source_provider",
    "source_dataset": "_source_dataset",
    "source_document_id": "_source_document_id",
    "source_series_id": "_source_series_id",
    "source_revision": "_source_revision",
    "fundamentals_vintage_id": "_vintage_id",
    "native_resolution_minutes": "_native_resolution_minutes",
    "forecast_reference_time_utc": "_reference_time_utc",
    "forecast_reference_time_type": "_reference_time_type",
    "published_at_utc": "_published_at_utc",
    "available_at_utc": "_available_at_utc",
    "availability_time_type": "_availability_time_type",
    "raw_batch_id": "_raw_batch_id",
    "retrieved_at_utc": "_retrieved_at_utc",
    "hourly_aggregation": "_hourly_aggregation",
    "interval_coverage_ratio": "_interval_coverage_ratio",
    "interval_coverage_pass": "_interval_coverage_pass",
    "dataset_version": "_dataset_version",
}
_DATETIME_LINEAGE_COLUMNS = {
    "forecast_reference_time_utc",
    "published_at_utc",
    "available_at_utc",
    "retrieved_at_utc",
}
_NUMERIC_LINEAGE_COLUMNS = {
    "native_resolution_minutes",
    "interval_coverage_ratio",
}
_BOOLEAN_LINEAGE_COLUMNS = {"interval_coverage_pass"}
_METADATA_SUFFIXES = tuple(_LINEAGE_SUFFIX_BY_COLUMN.values())


def join_fundamentals_features(
    frame: pd.DataFrame,
    hourly_fundamentals: pd.DataFrame,
) -> pd.DataFrame:
    """Join the newest point-in-time eligible fundamental revision.

    ``information_cutoff_utc`` takes precedence when present. A missing cutoff
    value falls back to ``forecast_origin_utc`` for that row. Eligibility is
    inclusive: a revision available exactly at the cutoff is usable.
    """

    require_columns(frame, ["ds_utc"], "frame")
    base = normalize_utc_column(frame, "ds_utc").reset_index(drop=True)
    base["_fundamentals_row_id"] = range(len(base))
    base["_fundamentals_cutoff_utc"] = _resolve_cutoff(base)
    if base.empty:
        return base.drop(
            columns=["_fundamentals_row_id", "_fundamentals_cutoff_utc"]
        )

    fundamentals = _prepare_hourly_fundamentals(hourly_fundamentals)
    relevant = fundamentals[
        fundamentals["ds_utc"].between(
            base["ds_utc"].min(),
            base["ds_utc"].max(),
        )
    ].copy()
    if relevant.empty:
        return base.drop(
            columns=["_fundamentals_row_id", "_fundamentals_cutoff_utc"]
        )

    feature_names = sorted(relevant["feature_name"].unique())
    output = base.copy()
    _initialize_feature_columns(output, feature_names)

    candidates = base[
        [
            "_fundamentals_row_id",
            "ds_utc",
            "_fundamentals_cutoff_utc",
        ]
    ].merge(
        relevant,
        on=["ds_utc"],
        how="left",
    )
    eligible = candidates[
        candidates["feature_name"].notna()
        & candidates["interval_coverage_pass"].fillna(False).astype(bool)
        & candidates["value"].notna()
        & (
            candidates["available_at_utc"]
            <= candidates["_fundamentals_cutoff_utc"]
        )
    ].copy()
    if not eligible.empty:
        selected = _select_newest_eligible_revision(eligible)
        _assign_selected_features(output, selected)

    return output.drop(
        columns=["_fundamentals_row_id", "_fundamentals_cutoff_utc"]
    ).reset_index(drop=True)


def fundamentals_value_columns(frame: pd.DataFrame) -> list[str]:
    """Return model-value columns while excluding per-feature lineage."""

    return [
        column
        for column in frame.columns
        if column.startswith("fundamental_")
        and not column.endswith(_METADATA_SUFFIXES)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def add_residual_load_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add provider- and zone-specific renewable and residual-load forecasts."""

    output = frame.copy()
    required_parameters = {
        "total_load_forecast",
        "wind_generation_forecast",
        "solar_generation_forecast",
    }
    groups: dict[tuple[str, str, str], dict[str, str]] = {}
    for column, identity in _fundamentals_identities(output):
        if (
            identity["entity_type"] != "bidding_zone"
            or identity["parameter_id"] not in required_parameters
        ):
            continue
        key = (
            identity["source_provider"],
            identity["area"],
            identity["unit"],
        )
        groups.setdefault(key, {})[identity["parameter_id"]] = column

    for (provider, area, unit), columns in groups.items():
        if not required_parameters.issubset(columns):
            continue
        renewable = fundamental_feature_name(
            "renewable_generation_forecast",
            source_provider=provider,
            entity_type="bidding_zone",
            unit=unit,
            area=area,
        )
        residual = fundamental_feature_name(
            "residual_load_forecast",
            source_provider=provider,
            entity_type="bidding_zone",
            unit=unit,
            area=area,
        )
        output[renewable] = (
            output[columns["wind_generation_forecast"]]
            + output[columns["solar_generation_forecast"]]
        )
        output[residual] = (
            output[columns["total_load_forecast"]] - output[renewable]
        )
    return output


def add_transfer_capacity_features(
    frame: pd.DataFrame,
    *,
    expected_directed_borders: Mapping[
        str,
        Iterable[tuple[str, str]],
    ],
) -> pd.DataFrame:
    """Add provider/zone totals after validating an explicit border topology."""

    output = frame.copy()
    capacity_identities = [
        (column, identity)
        for column, identity in _fundamentals_identities(output)
        if identity["entity_type"] == "directed_border"
        and identity["parameter_id"] == "available_transfer_capacity"
    ]
    topology = _normalize_directed_border_topology(expected_directed_borders)
    expected_unit = PARAMETER_SPECS["available_transfer_capacity"].unit
    expected_columns = {
        (
            provider,
            from_area,
            to_area,
            expected_unit,
        ): fundamental_feature_name(
            "available_transfer_capacity",
            source_provider=provider,
            entity_type="directed_border",
            unit=expected_unit,
            from_area=from_area,
            to_area=to_area,
        )
        for provider, borders in topology.items()
        for from_area, to_area in borders
    }
    observed_columns: dict[tuple[str, str, str, str], str] = {}
    for column, identity in capacity_identities:
        key = (
            identity["source_provider"],
            identity["from_area"],
            identity["to_area"],
            identity["unit"],
        )
        existing = observed_columns.get(key)
        if existing is not None and existing != column:
            raise ValueError(
                "Transfer-capacity feature identity is represented by multiple "
                f"columns: identity={key}, columns={sorted([existing, column])}"
            )
        observed_columns[key] = column
    missing = sorted(
        expected_columns[key]
        for key in expected_columns.keys() - observed_columns.keys()
    )
    unexpected = sorted(
        observed_columns[key]
        for key in observed_columns.keys() - expected_columns.keys()
    )
    if missing or unexpected:
        raise ValueError(
            "Transfer-capacity features do not match the declared provider "
            f"topology: missing={missing}, unexpected={unexpected}"
        )
    if not expected_columns:
        return output

    directional: dict[
        tuple[str, str, str],
        dict[str, list[str]],
    ] = {}
    for provider, from_area, to_area, unit in sorted(expected_columns):
        column = observed_columns[(provider, from_area, to_area, unit)]
        directional.setdefault(
            (provider, to_area, unit),
            {"import": [], "export": []},
        )["import"].append(column)
        directional.setdefault(
            (provider, from_area, unit),
            {"import": [], "export": []},
        )["export"].append(column)

    for (provider, area, unit), columns_by_direction in directional.items():
        for direction, columns in columns_by_direction.items():
            if not columns:
                continue
            parameter = f"total_available_{direction}_capacity"
            feature_name = fundamental_feature_name(
                parameter,
                source_provider=provider,
                entity_type="bidding_zone",
                unit=unit,
                area=area,
            )
            numeric = output[columns].apply(pd.to_numeric, errors="coerce")
            output[feature_name] = numeric.sum(
                axis=1,
                min_count=len(columns),
            )
    return output


def add_fundamentals_derived_features(
    frame: pd.DataFrame,
    *,
    expected_directed_borders: Mapping[
        str,
        Iterable[tuple[str, str]],
    ],
) -> pd.DataFrame:
    """Add the deterministic v1 residual-load and capacity summaries."""

    return add_transfer_capacity_features(
        add_residual_load_features(frame),
        expected_directed_borders=expected_directed_borders,
    )


def _normalize_directed_border_topology(
    value: Mapping[str, Iterable[tuple[str, str]]],
) -> dict[str, tuple[tuple[str, str], ...]]:
    if not isinstance(value, Mapping):
        raise TypeError(
            "expected_directed_borders must map source providers to directed "
            "border pairs"
        )
    normalized: dict[str, tuple[tuple[str, str], ...]] = {}
    for raw_provider, raw_borders in value.items():
        provider = str(raw_provider).strip()
        if not provider:
            raise ValueError(
                "expected_directed_borders contains a blank source provider"
            )
        borders: list[tuple[str, str]] = []
        for raw_border in raw_borders:
            if (
                not isinstance(raw_border, (tuple, list))
                or len(raw_border) != 2
            ):
                raise ValueError(
                    "Each expected directed border must be a "
                    "(from_area, to_area) pair"
                )
            from_area = str(raw_border[0]).strip()
            to_area = str(raw_border[1]).strip()
            if not from_area or not to_area or from_area == to_area:
                raise ValueError(
                    "Expected directed-border endpoints must be non-blank and "
                    "different"
                )
            borders.append((from_area, to_area))
        if not borders:
            raise ValueError(
                f"Expected directed-border topology for {provider!r} is empty"
            )
        if len(borders) != len(set(borders)):
            raise ValueError(
                f"Expected directed-border topology for {provider!r} contains "
                "duplicates"
            )
        normalized[provider] = tuple(sorted(borders))
    return normalized


def _prepare_hourly_fundamentals(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, HOURLY_COLUMNS, "hourly_fundamentals")
    output = frame[HOURLY_COLUMNS].copy()
    for column in [
        "ds_utc",
        "forecast_reference_time_utc",
        "published_at_utc",
        "available_at_utc",
        "retrieved_at_utc",
    ]:
        parsed = pd.to_datetime(output[column], utc=True, errors="coerce")
        invalid = output[column].notna() & parsed.isna()
        if invalid.any():
            raise ValueError(f"hourly_fundamentals has invalid {column}")
        output[column] = parsed

    for column in [
        "ds_utc",
        "forecast_reference_time_utc",
        "available_at_utc",
        "retrieved_at_utc",
    ]:
        if output[column].isna().any():
            raise ValueError(f"hourly_fundamentals has invalid or missing {column}")
    unknown_parameters = sorted(set(output["parameter_id"]) - set(ALLOWED_PARAMETERS))
    if unknown_parameters:
        raise ValueError(
            f"Unsupported hourly fundamentals parameter_id values: {unknown_parameters}"
        )
    unknown_availability_types = sorted(
        set(output["availability_time_type"])
        - set(ALLOWED_AVAILABILITY_TIME_TYPES)
    )
    if unknown_availability_types:
        raise ValueError(
            "Unsupported hourly fundamentals availability_time_type values: "
            f"{unknown_availability_types}"
        )
    if output.empty:
        return output
    _validate_hourly_parameter_contract(output)
    _validate_hourly_data_contract(output)

    expected_names = output.apply(
        lambda row: fundamental_feature_name(
            str(row["parameter_id"]),
            source_provider=str(row["source_provider"]),
            entity_type=str(row["entity_type"]),
            unit=str(row["unit"]),
            area=row["area"],
            from_area=row["from_area"],
            to_area=row["to_area"],
        ),
        axis=1,
    )
    if not output["feature_name"].equals(expected_names):
        raise ValueError("hourly_fundamentals contains non-canonical feature_name")
    validate_feature_name_uniqueness(output)
    output["value"] = pd.to_numeric(output["value"], errors="coerce")
    return output


def _resolve_cutoff(frame: pd.DataFrame) -> pd.Series:
    has_information_cutoff = "information_cutoff_utc" in frame.columns
    has_origin = "forecast_origin_utc" in frame.columns
    if not has_information_cutoff and not has_origin:
        raise ValueError(
            "frame requires information_cutoff_utc or forecast_origin_utc"
        )

    if has_information_cutoff:
        cutoff = _parse_optional_utc(
            frame["information_cutoff_utc"],
            column="information_cutoff_utc",
        )
    else:
        cutoff = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    if has_origin:
        origin = _parse_optional_utc(
            frame["forecast_origin_utc"],
            column="forecast_origin_utc",
        )
        cutoff = cutoff.fillna(origin)
        later_cutoff = cutoff.notna() & origin.notna() & cutoff.gt(origin)
        if later_cutoff.any():
            raise ValueError(
                "information_cutoff_utc must not be after "
                "forecast_origin_utc for fundamentals features"
            )
    if cutoff.isna().any():
        raise ValueError(
            "Each frame row requires information_cutoff_utc or forecast_origin_utc"
        )
    return cutoff


def _select_newest_eligible_revision(
    eligible: pd.DataFrame,
) -> pd.DataFrame:
    selection_keys = ["_fundamentals_row_id", "feature_name"]
    newest_reference = eligible.groupby(
        selection_keys,
        sort=False,
    )["forecast_reference_time_utc"].transform("max")
    newest = eligible[
        eligible["forecast_reference_time_utc"].eq(newest_reference)
    ].copy()

    newest_availability = newest.groupby(
        selection_keys,
        sort=False,
    )["available_at_utc"].transform("max")
    newest = newest[
        newest["available_at_utc"].eq(newest_availability)
    ].copy()

    revision_identity_columns = [
        "source_revision",
        "fundamentals_vintage_id",
    ]
    identity_counts = (
        newest[
            [
                *selection_keys,
                *revision_identity_columns,
            ]
        ]
        .drop_duplicates()
        .groupby(selection_keys, sort=False)
        .size()
    )
    ambiguous = identity_counts[identity_counts.gt(1)]
    if not ambiguous.empty:
        examples = [
            {
                "row_id": int(row_id),
                "feature_name": str(feature_name),
            }
            for row_id, feature_name in ambiguous.index[:5]
        ]
        raise ValueError(
            "Multiple distinct source revision/vintage identities are tied at "
            "the newest eligible forecast reference and availability "
            f"timestamps: {examples}"
        )

    semantic_columns = [
        column
        for column in HOURLY_COLUMNS
        if column
        not in {
            "feature_name",
            "raw_batch_id",
            "retrieved_at_utc",
        }
    ]
    semantic_counts = (
        newest[
            [
                *selection_keys,
                *semantic_columns,
            ]
        ]
        .drop_duplicates()
        .groupby(selection_keys, sort=False)
        .size()
    )
    conflicting_repeats = semantic_counts[semantic_counts.gt(1)]
    if not conflicting_repeats.empty:
        examples = [
            {
                "row_id": int(row_id),
                "feature_name": str(feature_name),
            }
            for row_id, feature_name in conflicting_repeats.index[:5]
        ]
        raise ValueError(
            "Rows tied at the newest eligible timestamps share a source "
            "revision/vintage identity but disagree on value or semantic "
            f"source lineage: {examples}"
        )

    deterministic_tiebreakers = [
        "retrieved_at_utc",
        "raw_batch_id",
    ]
    return (
        newest.sort_values(
            [
                *selection_keys,
                *deterministic_tiebreakers,
            ],
            kind="stable",
            na_position="first",
        )
        .drop_duplicates(selection_keys, keep="last")
        .reset_index(drop=True)
    )


def _initialize_feature_columns(
    output: pd.DataFrame,
    feature_names: list[str],
) -> None:
    generated_columns = {
        generated
        for feature_name in feature_names
        for generated in [
            feature_name,
            *[
                f"{feature_name}{suffix}"
                for suffix in _LINEAGE_SUFFIX_BY_COLUMN.values()
            ],
        ]
    }
    reserved = generated_columns.intersection(output.columns)
    if reserved:
        raise ValueError(
            "frame already contains fundamentals feature columns: "
            f"{sorted(reserved)}"
        )
    for feature_name in feature_names:
        output[feature_name] = pd.Series(
            float("nan"),
            index=output.index,
            dtype="float64",
        )
        for source_column, suffix in _LINEAGE_SUFFIX_BY_COLUMN.items():
            output_column = f"{feature_name}{suffix}"
            if source_column in _DATETIME_LINEAGE_COLUMNS:
                output[output_column] = pd.Series(
                    pd.NaT,
                    index=output.index,
                    dtype="datetime64[ns, UTC]",
                )
            elif source_column in _NUMERIC_LINEAGE_COLUMNS:
                output[output_column] = pd.Series(
                    float("nan"),
                    index=output.index,
                    dtype="float64",
                )
            elif source_column in _BOOLEAN_LINEAGE_COLUMNS:
                output[output_column] = pd.Series(
                    pd.NA,
                    index=output.index,
                    dtype="boolean",
                )
            else:
                output[output_column] = pd.Series(
                    pd.NA,
                    index=output.index,
                    dtype="object",
                )


def _assign_selected_features(
    output: pd.DataFrame,
    selected: pd.DataFrame,
) -> None:
    for feature_name, rows in selected.groupby("feature_name", sort=False):
        row_ids = rows["_fundamentals_row_id"].astype(int).tolist()
        output.loc[row_ids, feature_name] = rows["value"].astype(float).tolist()
        for source_column, suffix in _LINEAGE_SUFFIX_BY_COLUMN.items():
            output_column = f"{feature_name}{suffix}"
            output.loc[row_ids, output_column] = rows[source_column].tolist()


def _fundamentals_identities(
    frame: pd.DataFrame,
) -> list[tuple[str, dict[str, str]]]:
    identities: list[tuple[str, dict[str, str]]] = []
    for column in fundamentals_value_columns(frame):
        identity = _feature_identity(frame, column)
        if identity is not None:
            identities.append((column, identity))
    return identities


def _feature_identity(
    frame: pd.DataFrame,
    feature_name: str,
) -> dict[str, str] | None:
    required = [
        "parameter_id",
        "entity_type",
        "source_provider",
        "unit",
    ]
    identity: dict[str, str] = {}
    for name in required:
        value = _constant_lineage_value(frame, f"{feature_name}_{name}")
        if value is None:
            return None
        identity[name] = value

    if identity["entity_type"] == "bidding_zone":
        area = _constant_lineage_value(frame, f"{feature_name}_source_area")
        if area is None:
            return None
        identity["area"] = area
    elif identity["entity_type"] == "directed_border":
        from_area = _constant_lineage_value(frame, f"{feature_name}_from_area")
        to_area = _constant_lineage_value(frame, f"{feature_name}_to_area")
        if from_area is None or to_area is None:
            return None
        identity["from_area"] = from_area
        identity["to_area"] = to_area
    return identity


def _constant_lineage_value(
    frame: pd.DataFrame,
    column: str,
) -> str | None:
    if column not in frame.columns:
        return None
    values = frame[column].dropna().astype("string").str.strip()
    values = values[values.ne("")]
    unique = values.unique().tolist()
    if not unique:
        return None
    if len(unique) > 1:
        raise ValueError(f"{column} has conflicting feature lineage: {unique}")
    return str(unique[0])


def _validate_hourly_parameter_contract(frame: pd.DataFrame) -> None:
    for parameter_id, group in frame.groupby("parameter_id", sort=False):
        spec = PARAMETER_SPECS[str(parameter_id)]
        if set(group["unit"]) != {spec.unit}:
            raise ValueError(
                f"{parameter_id} must use unit {spec.unit!r}; "
                f"got {sorted(set(group['unit']))}"
            )
        invalid_entities = sorted(
            set(group["entity_type"]) - set(spec.entity_types)
        )
        if invalid_entities:
            raise ValueError(
                f"{parameter_id} has unsupported entity_type values: "
                f"{invalid_entities}"
            )

    bidding_zone = frame["entity_type"].eq("bidding_zone")
    if _missing_or_blank(frame.loc[bidding_zone, "area"]).any():
        raise ValueError("bidding_zone fundamentals require area")
    if (
        ~_missing_or_blank(frame.loc[bidding_zone, "from_area"])
        | ~_missing_or_blank(frame.loc[bidding_zone, "to_area"])
    ).any():
        raise ValueError(
            "bidding_zone fundamentals require null from_area and to_area"
        )

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


def _validate_hourly_data_contract(frame: pd.DataFrame) -> None:
    coverage = frame["interval_coverage_pass"]
    invalid_coverage = coverage.isna() | ~coverage.isin([True, False])
    if invalid_coverage.any():
        raise ValueError(
            "hourly_fundamentals interval_coverage_pass must contain booleans"
        )
    frame["interval_coverage_pass"] = coverage.astype("boolean")

    numeric_columns = [
        "native_resolution_minutes",
        "value",
        "interval_count",
        "expected_interval_count",
        "interval_coverage_ratio",
    ]
    numeric: dict[str, pd.Series] = {}
    for column in numeric_columns:
        converted = pd.to_numeric(frame[column], errors="coerce")
        invalid = frame[column].notna() & converted.isna()
        if invalid.any():
            raise ValueError(
                f"hourly_fundamentals has invalid numeric {column}"
            )
        numeric[column] = converted

    resolution = numeric["native_resolution_minutes"]
    if (
        resolution.isna().any()
        or resolution.le(0).any()
        or resolution.mod(1).ne(0).any()
        or (60 % resolution).ne(0).any()
    ):
        raise ValueError(
            "hourly_fundamentals native_resolution_minutes must be a positive "
            "integer divisor of 60"
        )
    interval_count = numeric["interval_count"]
    expected_count = numeric["expected_interval_count"]
    if (
        interval_count.isna().any()
        or expected_count.isna().any()
        or interval_count.lt(0).any()
        or interval_count.mod(1).ne(0).any()
        or expected_count.le(0).any()
        or expected_count.mod(1).ne(0).any()
        or expected_count.ne(60 / resolution).any()
    ):
        raise ValueError(
            "hourly_fundamentals has invalid interval counts"
        )
    coverage_ratio = numeric["interval_coverage_ratio"]
    if (
        coverage_ratio.isna().any()
        or coverage_ratio.lt(0).any()
        or coverage_ratio.gt(1).any()
    ):
        raise ValueError(
            "hourly_fundamentals interval_coverage_ratio must be in [0, 1]"
        )

    passed = frame["interval_coverage_pass"].astype(bool)
    value = numeric["value"]
    invalid_pass = passed & (
        value.isna()
        | coverage_ratio.sub(1.0).abs().gt(1e-12)
        | interval_count.ne(expected_count)
    )
    invalid_fail = ~passed & value.notna()
    if invalid_pass.any() or invalid_fail.any():
        raise ValueError(
            "hourly_fundamentals value and coverage fields are inconsistent"
        )

    available = frame["available_at_utc"]
    retrieved = frame["retrieved_at_utc"]
    reference = frame["forecast_reference_time_utc"]
    published = frame["published_at_utc"]
    if reference.gt(available).any() or available.gt(retrieved).any():
        raise ValueError(
            "hourly_fundamentals availability timestamps are inconsistent"
        )
    published_rows = published.notna()
    if published.loc[published_rows].gt(
        retrieved.loc[published_rows]
    ).any():
        raise ValueError(
            "hourly_fundamentals published_at_utc must not be after retrieval"
        )
    observed_publication = frame["availability_time_type"].eq(
        "observed_source_publication"
    )
    if (
        published.loc[observed_publication].isna().any()
        or ~available.loc[observed_publication].eq(
            published.loc[observed_publication]
        ).all()
    ):
        raise ValueError(
            "observed_source_publication requires available_at_utc to equal "
            "published_at_utc"
        )
    observed_retrieval = frame["availability_time_type"].eq(
        "observed_retrieval"
    )
    if ~available.loc[observed_retrieval].eq(
        retrieved.loc[observed_retrieval]
    ).all():
        raise ValueError(
            "observed_retrieval requires available_at_utc to equal "
            "retrieved_at_utc"
        )

    for parameter_id, group in frame.groupby("parameter_id", sort=False):
        values = value.loc[group.index].dropna()
        if parameter_id == "planned_transmission_outage_flag":
            if not values.isin([0.0, 1.0]).all():
                raise ValueError(
                    "planned_transmission_outage_flag values must be 0 or 1"
                )
        elif values.lt(0).any():
            raise ValueError(f"{parameter_id} values must be non-negative")
        expected_aggregation = PARAMETER_SPECS[str(parameter_id)].hourly_aggregation
        if set(group["hourly_aggregation"]) != {expected_aggregation}:
            raise ValueError(
                f"{parameter_id} must use hourly_aggregation "
                f"{expected_aggregation!r}"
            )


def _missing_or_blank(values: pd.Series) -> pd.Series:
    return values.isna() | values.astype("string").str.strip().eq("").fillna(True)


def _parse_optional_utc(
    values: pd.Series,
    *,
    column: str,
) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        raise ValueError(f"frame has invalid {column}")
    return parsed
