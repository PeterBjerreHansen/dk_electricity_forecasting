from __future__ import annotations

from datetime import time
from typing import Iterable, Protocol

import pandas as pd


COPENHAGEN_TZ = "Europe/Copenhagen"
DEFAULT_PRICE_PUBLICATION_LOCAL_TIME = "12:00"
PRICE_AVAILABILITY_COLUMN = "price_available_at_utc"

PANEL_REQUIRED_COLUMNS = [
    "unique_id",
    "ds_utc",
    "ds_local",
    "area",
    "y",
    "dataset_version",
    PRICE_AVAILABILITY_COLUMN,
]

PREDICTION_REQUIRED_COLUMNS = [
    "unique_id",
    "ds_utc",
    "forecast_origin_utc",
    "horizon",
    "model_name",
    "model_version",
    "y_pred",
]

KNOWN_FUTURE_COLUMNS = [
    "unique_id",
    "ds_utc",
    "forecast_origin_utc",
    "horizon",
    "area",
    "ds_local",
    "local_date",
    "local_hour",
    "local_day_of_week",
    "local_month",
    "is_weekend",
    "is_dst",
    "utc_offset_hours",
    "dataset_version",
    PRICE_AVAILABILITY_COLUMN,
]

TARGET_LEAKAGE_COLUMNS = ["y", "price_dkk_per_mwh", "price_eur_per_mwh"]


class ForecastModel(Protocol):
    model_name: str
    model_version: str

    def fit(self, history: pd.DataFrame) -> "ForecastModel":
        ...

    def predict(
        self,
        future: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        ...


def require_columns(frame: pd.DataFrame, columns: Iterable[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing required columns: {missing}")


def to_utc_timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def normalize_utc_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    output = frame.copy()
    output[column] = pd.to_datetime(output[column], utc=True).astype("datetime64[ns, UTC]")
    return output


def parse_local_time(value: str | time = DEFAULT_PRICE_PUBLICATION_LOCAL_TIME) -> time:
    if isinstance(value, time):
        return value
    parts = str(value).split(":")
    if len(parts) not in {2, 3}:
        raise ValueError(f"Local time must use HH:MM or HH:MM:SS format; got {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) == 3 else 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"Invalid local time: {value!r}")
    return time(hour=hour, minute=minute, second=second)


def add_price_availability(
    frame: pd.DataFrame,
    *,
    publication_local_time: str | time = DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
    column: str = PRICE_AVAILABILITY_COLUMN,
) -> pd.DataFrame:
    """Add deterministic day-ahead price publication timestamps."""

    require_columns(frame, ["ds_utc"], "frame")
    output = normalize_utc_column(frame, "ds_utc")
    local_time = parse_local_time(publication_local_time)
    local_dates = pd.to_datetime(
        output["ds_utc"].dt.tz_convert(COPENHAGEN_TZ).dt.strftime("%Y-%m-%d")
    )
    available_naive = (
        local_dates
        - pd.Timedelta(days=1)
        + pd.Timedelta(hours=local_time.hour, minutes=local_time.minute, seconds=local_time.second)
    )
    output[column] = available_naive.dt.tz_localize(COPENHAGEN_TZ).dt.tz_convert("UTC")
    return output


def ensure_price_availability(
    frame: pd.DataFrame,
    *,
    publication_local_time: str | time = DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
    column: str = PRICE_AVAILABILITY_COLUMN,
) -> pd.DataFrame:
    if column in frame.columns:
        return normalize_utc_column(frame, column)
    return add_price_availability(
        frame,
        publication_local_time=publication_local_time,
        column=column,
    )


def price_available_before_mask(
    frame: pd.DataFrame,
    forecast_origin_utc: object,
    *,
    column: str = PRICE_AVAILABILITY_COLUMN,
) -> pd.Series:
    require_columns(frame, [column], "frame")
    origin = to_utc_timestamp(forecast_origin_utc)
    return pd.to_datetime(frame[column], utc=True) < origin


def filter_price_history_available_before(
    frame: pd.DataFrame,
    forecast_origin_utc: object,
    *,
    publication_local_time: str | time = DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
    column: str = PRICE_AVAILABILITY_COLUMN,
) -> pd.DataFrame:
    prepared = ensure_price_availability(
        frame,
        publication_local_time=publication_local_time,
        column=column,
    )
    return prepared.loc[
        price_available_before_mask(prepared, forecast_origin_utc, column=column)
    ].copy()


def add_copenhagen_calendar(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, ["ds_utc"], "frame")
    output = normalize_utc_column(frame, "ds_utc")
    output["ds_local"] = output["ds_utc"].dt.tz_convert(COPENHAGEN_TZ)
    output["local_date"] = output["ds_local"].dt.strftime("%Y-%m-%d")
    output["local_hour"] = output["ds_local"].dt.hour.astype("int8")
    output["local_day_of_week"] = output["ds_local"].dt.dayofweek.astype("int8")
    output["local_month"] = output["ds_local"].dt.month.astype("int8")
    output["is_weekend"] = output["local_day_of_week"].isin([5, 6])
    output["is_dst"] = output["ds_local"].map(lambda value: bool(value.dst()))
    output["utc_offset_hours"] = output["ds_local"].map(
        lambda value: value.utcoffset().total_seconds() / 3600
    )
    return output


def add_horizon_column(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, ["unique_id", "forecast_origin_utc", "ds_utc"], "forecast frame")
    output = normalize_utc_column(frame, "ds_utc")
    output = normalize_utc_column(output, "forecast_origin_utc")
    output = output.sort_values(["forecast_origin_utc", "unique_id", "ds_utc"]).reset_index(drop=True)
    output["horizon"] = (
        output.groupby(["forecast_origin_utc", "unique_id"], sort=False)
        .cumcount()
        .add(1)
        .astype("int16")
    )
    return output
