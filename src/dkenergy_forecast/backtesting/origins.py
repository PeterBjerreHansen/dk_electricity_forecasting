from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from dkenergy_forecast.backtesting.horizons import (
    make_daily_origins,
    make_danish_delivery_day_horizon,
)
from dkenergy_forecast.types import normalize_utc_column, require_columns


HorizonBuilder = Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame]


def choose_recent_complete_daily_origins(
    panel: pd.DataFrame,
    *,
    days: int,
    at_hour_utc: int = 10,
    max_origins: int = 0,
    min_history_days: int = 90,
    holdout_days: int = 2,
    horizon_builder: HorizonBuilder = make_danish_delivery_day_horizon,
) -> pd.DataFrame:
    """Choose recent daily origins whose full horizon fits inside the panel."""

    if days <= 0:
        raise ValueError("days must be positive")
    if max_origins < 0:
        raise ValueError("max_origins must be non-negative")
    if min_history_days < 0:
        raise ValueError("min_history_days must be non-negative")
    if holdout_days < 0:
        raise ValueError("holdout_days must be non-negative")

    require_columns(panel, ["unique_id", "area", "ds_utc"], "panel")
    panel_utc = normalize_utc_column(panel, "ds_utc")
    min_origin = (panel_utc["ds_utc"].min() + pd.Timedelta(days=min_history_days)).normalize()
    max_origin = (panel_utc["ds_utc"].max() - pd.Timedelta(days=holdout_days)).normalize()
    start = max(min_origin, max_origin - pd.Timedelta(days=days))
    end = max_origin + pd.Timedelta(days=1)
    origins = make_daily_origins(panel_utc, start=start, end=end, at_hour_utc=at_hour_utc)

    valid_origins: list[pd.Timestamp] = []
    for origin in origins["forecast_origin_utc"]:
        horizon = horizon_builder(panel_utc, origin)
        if horizon.empty:
            continue
        if horizon["ds_utc"].min() >= panel_utc["ds_utc"].min() and horizon["ds_utc"].max() <= panel_utc["ds_utc"].max():
            valid_origins.append(origin)

    selected = valid_origins[-max_origins:] if max_origins > 0 else valid_origins
    if not selected:
        raise ValueError("No valid forecast origins fit inside the panel range.")
    return pd.DataFrame({"forecast_origin_utc": selected})
