from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from dkenergy_forecast.backtesting.horizons import make_danish_delivery_day_horizon
from dkenergy_forecast.features.price_features import (
    PriceFeatureConfig,
    build_price_experiment_frame,
)
from dkenergy_forecast.types import normalize_utc_column, require_columns


HorizonBuilder = Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame]

WEATHER_SELECTION_POLICY = "newest_eligible_per_model_parameter"
WEATHER_CUTOFF_COLUMNS = ("information_cutoff_utc", "forecast_origin_utc")
WEATHER_PARAMETERS = (
    "temperature_2m",
    "wind_speed_10m",
    "wind_speed_100m",
    "shortwave_radiation",
    "cloud_cover",
    "precipitation",
)
# Direction is intentionally excluded. Its arithmetic area/model mean is not a
# valid circular statistic, and the current Chronos release does not need it.
EXCLUDED_WEATHER_PARAMETERS = ("wind_direction_10m",)


def join_weather_features(
    frame: pd.DataFrame,
    area_features_long: pd.DataFrame,
    *,
    require_feature_group_pass: bool = True,
) -> pd.DataFrame:
    """Join the newest eligible weather value under stable feature names.

    The source table may contain several lead-specific forecasts for the same
    model, parameter, area, and valid hour. For each target row we select the
    newest forecast available at that row's information cutoff. The lead is
    retained as metadata instead of being encoded in the model covariate name.
    """

    require_columns(frame, ["area", "ds_utc"], "frame")
    require_columns(area_features_long, ["area", "ds_utc"], "area_features_long")
    _weather_cutoff_column(frame)
    base = normalize_utc_column(frame, "ds_utc")
    if base.empty:
        return base.reset_index(drop=True)
    weather_ds = pd.to_datetime(area_features_long["ds_utc"], utc=True)
    relevant = area_features_long[
        area_features_long["area"].isin(base["area"].unique())
        & weather_ds.between(base["ds_utc"].min(), base["ds_utc"].max())
    ].copy()
    weather = _prepare_weather_features(
        relevant,
        require_feature_group_pass=require_feature_group_pass,
    )
    return _join_prepared_weather_features(base, weather)


def _prepare_weather_features(
    area_features_long: pd.DataFrame,
    *,
    require_feature_group_pass: bool = True,
) -> pd.DataFrame:
    require_columns(
        area_features_long,
        [
            "area",
            "ds_utc",
            "feature_name",
            "value",
            "location_coverage_ratio",
            "location_coverage_pass",
            "feature_group_pass",
            "forecast_available_at_utc",
        ],
        "area_features_long",
    )
    weather = normalize_utc_column(area_features_long, "ds_utc")
    weather = normalize_utc_column(weather, "forecast_available_at_utc")
    if {"weather_model", "parameter_id"}.issubset(weather.columns):
        weather["_canonical_feature_name"] = (
            "weather_"
            + weather["weather_model"].astype(str)
            + "_"
            + weather["parameter_id"].astype(str)
        )
        weather["_weather_parameter"] = weather["parameter_id"].astype(str)
    else:
        weather["_canonical_feature_name"] = weather["feature_name"].map(
            canonical_weather_feature_name
        )
        weather["_weather_parameter"] = weather["feature_name"].map(
            _weather_parameter
        )
    weather = weather[
        ~weather["_weather_parameter"].isin(EXCLUDED_WEATHER_PARAMETERS)
    ].copy()
    if weather.empty:
        return weather.reset_index(drop=True)
    if "lead_time_days" not in weather.columns:
        weather["lead_time_days"] = weather["feature_name"].map(
            _weather_lead_time_days
        )
    if "forecast_reference_time" not in weather.columns:
        weather["forecast_reference_time"] = weather["forecast_available_at_utc"]
    weather = normalize_utc_column(weather, "forecast_reference_time")
    if "forecast_reference_time_type" not in weather.columns:
        weather["forecast_reference_time_type"] = "legacy_unspecified"
    if "forecast_reference_time_is_observed" not in weather.columns:
        weather["forecast_reference_time_is_observed"] = pd.NA
    if "forecast_availability_time_type" not in weather.columns:
        weather["forecast_availability_time_type"] = "legacy_unspecified"
    if "weather_vintage_id" not in weather.columns:
        weather["weather_vintage_id"] = (
            "legacy_unspecified:"
            + weather["_canonical_feature_name"].astype(str)
            + ":"
            + weather["forecast_reference_time"].dt.strftime("%Y%m%dT%H%M%SZ")
        )
    if require_feature_group_pass:
        weather = weather[
            weather["feature_group_pass"] & weather["location_coverage_pass"]
        ].copy()
    return weather.reset_index(drop=True)


def _join_prepared_weather_features(
    frame: pd.DataFrame,
    weather: pd.DataFrame,
) -> pd.DataFrame:
    require_columns(frame, ["area", "ds_utc"], "frame")
    base = normalize_utc_column(frame, "ds_utc")
    cutoff_column = _weather_cutoff_column(base)
    base = normalize_utc_column(base, cutoff_column).reset_index(drop=True)
    base["_weather_row_id"] = range(len(base))
    if base.empty:
        return base.drop(columns=["_weather_row_id"])
    weather = weather[
        weather["area"].isin(base["area"].unique())
        & weather["ds_utc"].between(base["ds_utc"].min(), base["ds_utc"].max())
    ].copy()

    merged = base[["_weather_row_id", "area", "ds_utc", cutoff_column]].merge(
        weather,
        on=["area", "ds_utc"],
        how="left",
    )
    eligible = merged[
        merged["feature_name"].notna()
        & (merged["forecast_available_at_utc"] <= merged[cutoff_column])
    ].copy()

    output = base.copy()
    if not eligible.empty:
        eligible = (
            eligible.sort_values(
                [
                    "_weather_row_id",
                    "_canonical_feature_name",
                    "forecast_reference_time",
                    "forecast_available_at_utc",
                    "lead_time_days",
                ]
            )
            .drop_duplicates(
                ["_weather_row_id", "_canonical_feature_name"],
                keep="last",
            )
            .reset_index(drop=True)
        )
        value_wide = eligible.pivot_table(
            index="_weather_row_id",
            columns="_canonical_feature_name",
            values="value",
            aggfunc="last",
        )
        coverage_wide = eligible.pivot_table(
            index="_weather_row_id",
            columns="_canonical_feature_name",
            values="location_coverage_ratio",
            aggfunc="last",
        ).rename(columns=lambda column: f"{column}_coverage_ratio")
        availability_wide = eligible.pivot_table(
            index="_weather_row_id",
            columns="_canonical_feature_name",
            values="forecast_available_at_utc",
            aggfunc="last",
        ).rename(columns=lambda column: f"{column}_available_at_utc")
        reference_time_wide = eligible.pivot_table(
            index="_weather_row_id",
            columns="_canonical_feature_name",
            values="forecast_reference_time",
            aggfunc="last",
        ).rename(columns=lambda column: f"{column}_reference_time_utc")
        vintage_wide = eligible.pivot_table(
            index="_weather_row_id",
            columns="_canonical_feature_name",
            values="weather_vintage_id",
            aggfunc="last",
        ).rename(columns=lambda column: f"{column}_vintage_id")
        reference_type_wide = eligible.pivot_table(
            index="_weather_row_id",
            columns="_canonical_feature_name",
            values="forecast_reference_time_type",
            aggfunc="last",
        ).rename(columns=lambda column: f"{column}_reference_time_type")
        reference_observed_wide = eligible.pivot_table(
            index="_weather_row_id",
            columns="_canonical_feature_name",
            values="forecast_reference_time_is_observed",
            aggfunc="last",
            dropna=False,
        ).rename(columns=lambda column: f"{column}_reference_time_is_observed")
        availability_type_wide = eligible.pivot_table(
            index="_weather_row_id",
            columns="_canonical_feature_name",
            values="forecast_availability_time_type",
            aggfunc="last",
        ).rename(columns=lambda column: f"{column}_availability_time_type")
        selected_lead_wide = eligible.pivot_table(
            index="_weather_row_id",
            columns="_canonical_feature_name",
            values="lead_time_days",
            aggfunc="last",
        ).rename(columns=lambda column: f"{column}_selected_lead_time_days")
        source_feature_wide = eligible.pivot_table(
            index="_weather_row_id",
            columns="_canonical_feature_name",
            values="feature_name",
            aggfunc="last",
        ).rename(columns=lambda column: f"{column}_source_feature_name")
        wide = pd.concat(
            [
                value_wide,
                coverage_wide,
                availability_wide,
                reference_time_wide,
                vintage_wide,
                reference_type_wide,
                reference_observed_wide,
                availability_type_wide,
                selected_lead_wide,
                source_feature_wide,
            ],
            axis=1,
        )
        wide.columns.name = None
        output = output.merge(wide, left_on="_weather_row_id", right_index=True, how="left")

    return output.drop(columns=["_weather_row_id"]).reset_index(drop=True)


def build_weather_experiment_frame(
    panel: pd.DataFrame,
    origins: pd.DataFrame,
    area_features_long: pd.DataFrame,
    *,
    price_feature_config: PriceFeatureConfig | None = None,
    horizon_builder: HorizonBuilder = make_danish_delivery_day_horizon,
    add_ensemble_features: bool = True,
    add_derived_features: bool = True,
) -> pd.DataFrame:
    """Build price + availability-safe weather feature rows for backtests."""

    require_columns(origins, ["forecast_origin_utc"], "origins")
    panel_utc = normalize_utc_column(panel, "ds_utc")
    origins_utc = normalize_utc_column(origins, "forecast_origin_utc")
    price_features = build_price_experiment_frame(
        panel_utc,
        origins_utc,
        config=price_feature_config,
        horizon_builder=horizon_builder,
    )
    if price_features.empty:
        return pd.DataFrame()

    output = join_weather_features(price_features, area_features_long)
    if add_ensemble_features:
        output = add_weather_ensemble_features(output)
    if add_derived_features:
        output = add_weather_derived_features(output)
    return output.reset_index(drop=True)


def weather_value_columns(frame: pd.DataFrame) -> list[str]:
    metadata_suffixes = (
        "_coverage_ratio",
        "_available_at_utc",
        "_reference_time_utc",
        "_vintage_id",
        "_reference_time_type",
        "_reference_time_is_observed",
        "_availability_time_type",
        "_selected_lead_time_days",
        "_source_feature_name",
    )
    return [
        column
        for column in frame.columns
        if column.startswith("weather_")
        and not column.endswith(metadata_suffixes)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def add_weather_ensemble_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    value_columns = [
        column
        for column in weather_value_columns(output)
        if not column.startswith("weather_ensemble_")
    ]
    groups: dict[str, list[str]] = {}

    for column in value_columns:
        parsed = _parse_canonical_weather_feature_column(column)
        if parsed is None:
            continue
        _model, parameter = parsed
        groups.setdefault(parameter, []).append(column)

    for parameter, columns in groups.items():
        if len(columns) < 2:
            continue
        prefix = f"weather_ensemble_{parameter}"
        enough_models = output[columns].notna().sum(axis=1) >= 2
        output[f"{prefix}_mean"] = output[columns].mean(axis=1, skipna=True).where(
            enough_models
        )
        output[f"{prefix}_min"] = output[columns].min(axis=1, skipna=True).where(
            enough_models
        )
        output[f"{prefix}_max"] = output[columns].max(axis=1, skipna=True).where(
            enough_models
        )
        output[f"{prefix}_spread"] = output[f"{prefix}_max"] - output[f"{prefix}_min"]

    return output


def add_weather_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add physically motivated weather transforms without touching availability columns."""

    output = frame.copy()
    output = _add_weather_physics_features(output)
    output = _add_weather_area_spread_features(output)
    return output


def _add_weather_physics_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    groups = _weather_columns_by_model(output)

    for model, columns in groups.items():
        prefix = f"weather_{model}"
        direction = columns.get("wind_direction_10m")
        speed_10m = columns.get("wind_speed_10m")
        speed_100m = columns.get("wind_speed_100m")
        shortwave = columns.get("shortwave_radiation")
        cloud = columns.get("cloud_cover")

        if direction is not None:
            radians = np.deg2rad(output[direction].astype(float) % 360)
            output[f"{prefix}_wind_direction_10m_sin"] = np.sin(radians)
            output[f"{prefix}_wind_direction_10m_cos"] = np.cos(radians)
            if speed_10m is not None:
                output[f"{prefix}_wind10_u"] = -output[speed_10m] * np.sin(radians)
                output[f"{prefix}_wind10_v"] = -output[speed_10m] * np.cos(radians)

        if speed_10m is not None and speed_100m is not None:
            output[f"{prefix}_wind_shear_100m_minus_10m"] = (
                output[speed_100m] - output[speed_10m]
            )

        if shortwave is not None and cloud is not None:
            cloud_fraction = output[cloud] / 100.0
            output[f"{prefix}_shortwave_x_cloud_cover"] = (
                output[shortwave] * cloud_fraction
            )
            output[f"{prefix}_shortwave_x_clear_sky_proxy"] = (
                output[shortwave] * (1 - cloud_fraction)
            )

    return output

def _add_weather_area_spread_features(frame: pd.DataFrame) -> pd.DataFrame:
    if not {"area", "ds_utc"}.issubset(frame.columns):
        return frame

    output = frame.copy()
    cutoff_column = _weather_cutoff_column(output)
    key_columns = [cutoff_column, "ds_utc"]
    source_columns = [
        column
        for column in weather_value_columns(output)
        if "_dk1_minus_dk2" not in column
    ]
    if not source_columns:
        return output

    wide = output.pivot_table(
        index=key_columns,
        columns="area",
        values=source_columns,
        aggfunc="last",
    )
    if "DK1" not in wide.columns.get_level_values("area") or "DK2" not in wide.columns.get_level_values("area"):
        return output

    spread_columns: dict[str, pd.Series] = {}
    for column in source_columns:
        if (column, "DK1") not in wide.columns or (column, "DK2") not in wide.columns:
            continue
        spread_columns[f"{column}_dk1_minus_dk2"] = wide[(column, "DK1")] - wide[(column, "DK2")]

    if not spread_columns:
        return output

    spread = pd.DataFrame(spread_columns, index=wide.index).reset_index()
    output = output.merge(spread, on=key_columns, how="left")

    return output


def _weather_columns_by_model(frame: pd.DataFrame) -> dict[str, dict[str, str]]:
    groups: dict[str, dict[str, str]] = {}
    for column in weather_value_columns(frame):
        parsed = _parse_canonical_weather_feature_column(column)
        if parsed is None:
            continue
        model, parameter = parsed
        if parameter not in WEATHER_PARAMETERS:
            continue
        groups.setdefault(model, {})[parameter] = column
    return groups


def _parse_weather_feature_column(column: str) -> tuple[str, str, str] | None:
    # weather_<model>_lead<Nd>_<parameter>
    marker = "_lead"
    if not column.startswith("weather_") or marker not in column:
        return None
    rest = column.removeprefix("weather_")
    model, lead_and_parameter = rest.split(marker, 1)
    lead, _, parameter = lead_and_parameter.partition("_")
    if not lead or not parameter:
        return None
    return model, f"lead{lead}", parameter


def _parse_canonical_weather_feature_column(column: str) -> tuple[str, str] | None:
    if not column.startswith("weather_") or column.startswith("weather_ensemble_"):
        return None
    body = column.removeprefix("weather_")
    for parameter in sorted((*WEATHER_PARAMETERS, *EXCLUDED_WEATHER_PARAMETERS), key=len, reverse=True):
        suffix = f"_{parameter}"
        if body.endswith(suffix):
            model = body[: -len(suffix)]
            if model:
                return model, parameter
    return None


def canonical_weather_feature_name(feature_name: object) -> str:
    """Return the stable model/parameter name for a lead-specific source name."""

    value = str(feature_name)
    parsed = _parse_weather_feature_column(value)
    if parsed is None:
        return value
    model, _lead, parameter = parsed
    return f"weather_{model}_{parameter}"


def _weather_parameter(feature_name: object) -> str | None:
    parsed = _parse_weather_feature_column(str(feature_name))
    if parsed is not None:
        return parsed[2]
    canonical = _parse_canonical_weather_feature_column(str(feature_name))
    return canonical[1] if canonical is not None else None


def _weather_lead_time_days(feature_name: object) -> int | None:
    parsed = _parse_weather_feature_column(str(feature_name))
    if parsed is None:
        return None
    lead = parsed[1].removeprefix("lead").removesuffix("d")
    return int(lead) if lead.isdigit() else None


def _weather_cutoff_column(frame: pd.DataFrame) -> str:
    for column in WEATHER_CUTOFF_COLUMNS:
        if column in frame.columns:
            return column
    raise ValueError(
        "frame is missing a weather information cutoff column; expected one of "
        f"{list(WEATHER_CUTOFF_COLUMNS)}"
    )
