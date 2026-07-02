from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from dkenergy_forecast.backtesting.horizons import make_danish_delivery_day_horizon
from dkenergy_forecast.features.price_features import (
    PriceFeatureConfig,
    build_price_feature_frame,
)
from dkenergy_forecast.types import normalize_utc_column, require_columns, to_utc_timestamp


HorizonBuilder = Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame]


def join_weather_features(
    frame: pd.DataFrame,
    area_features_long: pd.DataFrame,
    *,
    require_feature_group_pass: bool = True,
) -> pd.DataFrame:
    """Join availability-masked weather features to a future/feature frame."""

    require_columns(frame, ["area", "ds_utc", "forecast_origin_utc"], "frame")
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

    base = normalize_utc_column(frame, "ds_utc")
    base = normalize_utc_column(base, "forecast_origin_utc").reset_index(drop=True)
    base["_weather_row_id"] = range(len(base))

    weather = normalize_utc_column(area_features_long, "ds_utc")
    weather = normalize_utc_column(weather, "forecast_available_at_utc")
    if require_feature_group_pass:
        weather = weather[
            weather["feature_group_pass"] & weather["location_coverage_pass"]
        ].copy()

    merged = base[["_weather_row_id", "area", "ds_utc", "forecast_origin_utc"]].merge(
        weather,
        on=["area", "ds_utc"],
        how="left",
    )
    eligible = merged[
        merged["feature_name"].notna()
        & (merged["forecast_available_at_utc"] <= merged["forecast_origin_utc"])
    ].copy()

    output = base.copy()
    if not eligible.empty:
        value_wide = eligible.pivot_table(
            index="_weather_row_id",
            columns="feature_name",
            values="value",
            aggfunc="last",
        )
        coverage_wide = eligible.pivot_table(
            index="_weather_row_id",
            columns="feature_name",
            values="location_coverage_ratio",
            aggfunc="last",
        ).rename(columns=lambda column: f"{column}_coverage_ratio")
        availability_wide = eligible.pivot_table(
            index="_weather_row_id",
            columns="feature_name",
            values="forecast_available_at_utc",
            aggfunc="last",
        ).rename(columns=lambda column: f"{column}_available_at_utc")
        wide = pd.concat([value_wide, coverage_wide, availability_wide], axis=1)
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
) -> pd.DataFrame:
    """Build price + availability-safe weather feature rows for backtests."""

    require_columns(origins, ["forecast_origin_utc"], "origins")
    panel_utc = normalize_utc_column(panel, "ds_utc")
    origins_utc = normalize_utc_column(origins, "forecast_origin_utc")
    frames: list[pd.DataFrame] = []

    for origin in origins_utc["forecast_origin_utc"].sort_values().drop_duplicates():
        origin = to_utc_timestamp(origin)
        future = horizon_builder(panel_utc, origin)
        price_features = build_price_feature_frame(
            panel_utc,
            future,
            forecast_origin_utc=origin,
            include_target=True,
            config=price_feature_config,
        )
        with_weather = join_weather_features(price_features, area_features_long)
        frames.append(with_weather)

    if not frames:
        return pd.DataFrame()

    output = pd.concat(frames, ignore_index=True)
    if add_ensemble_features:
        output = add_weather_ensemble_features(output)
    return output.reset_index(drop=True)


def weather_value_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.columns
        if column.startswith("weather_")
        and not column.endswith("_coverage_ratio")
        and not column.endswith("_available_at_utc")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]


def add_weather_ensemble_features(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    value_columns = [
        column
        for column in weather_value_columns(output)
        if not column.startswith("weather_ensemble_")
    ]
    groups: dict[tuple[str, str], list[str]] = {}

    for column in value_columns:
        parsed = _parse_weather_feature_column(column)
        if parsed is None:
            continue
        _model, lead, parameter = parsed
        groups.setdefault((lead, parameter), []).append(column)

    for (lead, parameter), columns in groups.items():
        if len(columns) < 2:
            continue
        prefix = f"weather_ensemble_{lead}_{parameter}"
        output[f"{prefix}_mean"] = output[columns].mean(axis=1, skipna=True)
        output[f"{prefix}_min"] = output[columns].min(axis=1, skipna=True)
        output[f"{prefix}_max"] = output[columns].max(axis=1, skipna=True)
        output[f"{prefix}_spread"] = output[f"{prefix}_max"] - output[f"{prefix}_min"]

    return output


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
