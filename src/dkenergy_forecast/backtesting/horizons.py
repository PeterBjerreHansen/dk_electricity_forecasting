from __future__ import annotations

from datetime import timedelta

import pandas as pd

from dkenergy_forecast.types import (
    COPENHAGEN_TZ,
    KNOWN_FUTURE_COLUMNS,
    TARGET_LEAKAGE_COLUMNS,
    add_copenhagen_calendar,
    add_horizon_column,
    copenhagen_timestamp,
    ensure_price_availability,
    normalize_utc_column,
    parse_local_time,
    require_columns,
    to_utc_timestamp,
)


def make_daily_origins(
    panel: pd.DataFrame,
    start: object,
    end: object,
    *,
    at_hour_utc: int = 10,
) -> pd.DataFrame:
    if not 0 <= at_hour_utc <= 23:
        raise ValueError("at_hour_utc must be between 0 and 23")

    start_midnight = to_utc_timestamp(start).normalize()
    end_midnight = to_utc_timestamp(end).normalize()
    if end_midnight <= start_midnight:
        return pd.DataFrame({"forecast_origin_utc": pd.Series(dtype="datetime64[ns, UTC]")})

    dates = pd.date_range(
        start=start_midnight,
        end=end_midnight - pd.Timedelta(days=1),
        freq="D",
        tz="UTC",
    )
    return pd.DataFrame({"forecast_origin_utc": dates + pd.Timedelta(hours=at_hour_utc)})


def make_local_daily_origins(
    panel: pd.DataFrame,
    start: object,
    end: object,
    *,
    forecast_local_time: str = "12:00",
) -> pd.DataFrame:
    local_time = parse_local_time(forecast_local_time)
    start_midnight = to_utc_timestamp(start).tz_convert(COPENHAGEN_TZ).normalize()
    end_midnight = to_utc_timestamp(end).tz_convert(COPENHAGEN_TZ).normalize()
    if end_midnight <= start_midnight:
        return pd.DataFrame({"forecast_origin_utc": pd.Series(dtype="datetime64[ns, UTC]")})

    dates = pd.date_range(
        start=start_midnight,
        end=end_midnight - pd.Timedelta(days=1),
        freq="D",
        tz=COPENHAGEN_TZ,
    )
    origins = pd.DatetimeIndex(
        [copenhagen_timestamp(date.date(), local_time) for date in dates]
    )
    return pd.DataFrame({"forecast_origin_utc": origins.tz_convert("UTC")})


def make_next_utc_hours_horizon(
    panel: pd.DataFrame,
    forecast_origin_utc: object,
    *,
    hours: int = 24,
) -> pd.DataFrame:
    if hours <= 0:
        raise ValueError("hours must be positive")
    origin = to_utc_timestamp(forecast_origin_utc)
    timestamps = pd.date_range(
        start=origin + pd.Timedelta(hours=1),
        periods=hours,
        freq="h",
        tz="UTC",
    )
    return _make_horizon_for_timestamps(panel, origin, timestamps)


def make_danish_delivery_day_horizon(
    panel: pd.DataFrame,
    forecast_origin_utc: object,
    *,
    delivery_date_local: object | None = None,
    days_ahead: int = 1,
) -> pd.DataFrame:
    origin = to_utc_timestamp(forecast_origin_utc)
    if delivery_date_local is None:
        origin_local_date = origin.tz_convert(COPENHAGEN_TZ).date()
        delivery_date = origin_local_date + timedelta(days=days_ahead)
    else:
        delivery_date = pd.Timestamp(delivery_date_local).date()

    start_local = pd.Timestamp(delivery_date).tz_localize(COPENHAGEN_TZ)
    end_local = pd.Timestamp(delivery_date + timedelta(days=1)).tz_localize(COPENHAGEN_TZ)
    start_utc = start_local.tz_convert("UTC")
    end_utc = end_local.tz_convert("UTC")
    timestamps = pd.date_range(
        start=start_utc,
        end=end_utc - pd.Timedelta(hours=1),
        freq="h",
        tz="UTC",
    )
    return _make_horizon_for_timestamps(panel, origin, timestamps)


def _make_horizon_for_timestamps(
    panel: pd.DataFrame,
    forecast_origin_utc: pd.Timestamp,
    timestamps: pd.DatetimeIndex,
) -> pd.DataFrame:
    require_columns(panel, ["unique_id", "area", "ds_utc"], "panel")
    panel_utc = ensure_price_availability(normalize_utc_column(panel, "ds_utc"))
    unique_ids = panel_utc[["unique_id", "area"]].drop_duplicates().reset_index(drop=True)
    times = pd.DataFrame({"ds_utc": timestamps})
    horizon = unique_ids.merge(times, how="cross")
    horizon["forecast_origin_utc"] = forecast_origin_utc

    metadata_cols = [
        column
        for column in KNOWN_FUTURE_COLUMNS
        if column in panel_utc.columns and column not in {"forecast_origin_utc", "horizon"}
    ]
    metadata = panel_utc[metadata_cols].drop_duplicates(["unique_id", "ds_utc"])
    horizon = horizon.merge(
        metadata,
        on=["unique_id", "ds_utc"],
        how="left",
        suffixes=("", "_panel"),
    )
    for column in ["area", "dataset_version"]:
        panel_column = f"{column}_panel"
        if panel_column in horizon.columns:
            horizon[column] = horizon[column].combine_first(horizon[panel_column])
            horizon = horizon.drop(columns=[panel_column])

    if "dataset_version" in panel_utc.columns:
        dataset_version = panel_utc.groupby("unique_id")["dataset_version"].last()
        horizon["dataset_version"] = horizon["dataset_version"].combine_first(
            horizon["unique_id"].map(dataset_version)
        )

    horizon = add_copenhagen_calendar(horizon)
    horizon = add_horizon_column(horizon)
    horizon = horizon.drop(columns=[column for column in TARGET_LEAKAGE_COLUMNS if column in horizon.columns])
    columns = [column for column in KNOWN_FUTURE_COLUMNS if column in horizon.columns]
    return horizon[columns].reset_index(drop=True)
