from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from dkenergy_forecast.backtesting.horizons import (
    make_daily_origins,
    make_danish_delivery_day_horizon,
)
from dkenergy_forecast.types import (
    add_horizon_column,
    normalize_utc_column,
    require_columns,
    to_utc_timestamp,
)


HorizonBuilder = Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame]


@dataclass(frozen=True)
class PriceFeatureConfig:
    """Leakage-safe EDS-only feature recipe for tabular price models."""

    lag_hours: tuple[int, ...] = (24, 48, 168)
    rolling_windows_hours: tuple[int, ...] = (24, 168, 24 * 28)
    seasonal_lookback_days: int = 56
    spread_lag_hours: tuple[int, ...] = (24, 168)
    base_features: tuple[str, ...] = (
        "area",
        "local_hour",
        "local_day_of_week",
        "local_month",
        "is_weekend",
        "is_dst",
        "utc_offset_hours",
    )
    categorical_features: tuple[str, ...] = ("area",)

    def __post_init__(self) -> None:
        _require_positive("lag_hours", self.lag_hours)
        _require_positive("rolling_windows_hours", self.rolling_windows_hours)
        _require_positive("spread_lag_hours", self.spread_lag_hours)
        if self.seasonal_lookback_days <= 0:
            raise ValueError("seasonal_lookback_days must be positive")

    @property
    def lag_feature_columns(self) -> list[str]:
        return [f"lag_{lag}h" for lag in self.lag_hours]

    @property
    def rolling_feature_columns(self) -> list[str]:
        return [
            f"rolling_mean_{window}h_asof_origin"
            for window in self.rolling_windows_hours
        ] + [
            f"rolling_median_{window}h_asof_origin"
            for window in self.rolling_windows_hours
        ]

    @property
    def seasonal_feature_columns(self) -> list[str]:
        return ["seasonal_median_local_hour", "seasonal_median_hour_weekend"]

    @property
    def spread_feature_columns(self) -> list[str]:
        return [
            f"dk1_minus_dk2_lag_{lag}h"
            for lag in self.spread_lag_hours
        ]

    @property
    def feature_columns(self) -> list[str]:
        return (
            list(self.base_features)
            + self.lag_feature_columns
            + self.rolling_feature_columns
            + self.seasonal_feature_columns
            + self.spread_feature_columns
        )


def build_origin_feature_frame(
    panel: pd.DataFrame,
    forecast_origin_utc: object,
    *,
    include_target: bool = False,
    config: PriceFeatureConfig | None = None,
    horizon_builder: HorizonBuilder = make_danish_delivery_day_horizon,
) -> pd.DataFrame:
    """Build features for one origin using a horizon builder.

    This helper is convenient for day-ahead development notebooks. For scripts
    or backtests that already have an explicit future frame, use
    ``build_price_feature_frame`` directly.
    """

    config = config or PriceFeatureConfig()
    panel_utc = _prepare_panel(panel, config)
    origin = to_utc_timestamp(forecast_origin_utc)
    future = horizon_builder(panel_utc, origin)
    return build_price_feature_frame(
        panel_utc,
        future,
        forecast_origin_utc=origin,
        include_target=include_target,
        config=config,
    )


def build_price_feature_frame(
    panel: pd.DataFrame,
    future: pd.DataFrame,
    *,
    forecast_origin_utc: object | None = None,
    include_target: bool = False,
    config: PriceFeatureConfig | None = None,
) -> pd.DataFrame:
    """Build an EDS-only feature matrix for an explicit future frame.

    The function never uses target rows at or after ``forecast_origin_utc`` as
    features. If ``include_target`` is true, the target is joined only as an
    output column for training or evaluation.
    """

    config = config or PriceFeatureConfig()
    panel_utc = _prepare_panel(panel, config)
    future_utc = _prepare_future(future, config)
    origin = _resolve_single_origin(future_utc, forecast_origin_utc)

    history = panel_utc[panel_utc["ds_utc"] < origin].copy()
    features = _base_feature_frame(future_utc, config)

    features = _add_lag_features(features, history, origin, config)
    features = _add_rolling_features(features, history, origin, config)
    features = _add_seasonal_features(features, history, origin, config)
    features = _add_spread_features(features, history, origin, config)

    for column in ["is_weekend", "is_dst"]:
        if column in features.columns:
            features[column] = features[column].astype("int8")

    if include_target:
        actuals = panel_utc[["unique_id", "ds_utc", "y"]].drop_duplicates(
            ["unique_id", "ds_utc"]
        )
        features = features.merge(actuals, on=["unique_id", "ds_utc"], how="left")

    return features.reset_index(drop=True)


def build_training_matrix(
    panel: pd.DataFrame,
    forecast_origin_utc: object,
    *,
    training_origin_days: int = 70,
    at_hour_utc: int = 10,
    config: PriceFeatureConfig | None = None,
    horizon_builder: HorizonBuilder = make_danish_delivery_day_horizon,
    require_complete_horizons_before_origin: bool = True,
) -> pd.DataFrame:
    """Build historic origin/horizon examples available before one origin."""

    if training_origin_days <= 0:
        raise ValueError("training_origin_days must be positive")

    config = config or PriceFeatureConfig()
    panel_utc = _prepare_panel(panel, config)
    origin = to_utc_timestamp(forecast_origin_utc)
    train_origin_end = origin.normalize()
    train_origin_start = train_origin_end - pd.Timedelta(days=training_origin_days)
    train_origins = make_daily_origins(
        panel_utc,
        start=train_origin_start,
        end=train_origin_end,
        at_hour_utc=at_hour_utc,
    )

    frames: list[pd.DataFrame] = []
    for train_origin in train_origins["forecast_origin_utc"]:
        horizon = horizon_builder(panel_utc, train_origin)
        if horizon.empty:
            continue
        if require_complete_horizons_before_origin and horizon["ds_utc"].max() >= origin:
            continue
        if horizon["ds_utc"].min() < panel_utc["ds_utc"].min():
            continue

        frame = build_price_feature_frame(
            panel_utc,
            horizon,
            forecast_origin_utc=train_origin,
            include_target=True,
            config=config,
        )
        frames.append(frame)

    if not frames:
        raise ValueError(f"No training feature rows available before {origin.isoformat()}")

    training = pd.concat(frames, ignore_index=True)
    return training.dropna(subset=["y"]).reset_index(drop=True)


def _prepare_panel(panel: pd.DataFrame, config: PriceFeatureConfig) -> pd.DataFrame:
    require_columns(panel, ["unique_id", "area", "ds_utc", "y"], "panel")
    prepared = normalize_utc_column(panel, "ds_utc")
    required_calendar = [
        column
        for column in ["local_hour", "local_day_of_week", "local_month", "is_weekend", "is_dst", "utc_offset_hours"]
        if column in config.base_features
    ]
    require_columns(prepared, required_calendar, "panel")
    return prepared.sort_values(["area", "ds_utc"]).reset_index(drop=True)


def _prepare_future(future: pd.DataFrame, config: PriceFeatureConfig) -> pd.DataFrame:
    require_columns(future, ["unique_id", "area", "ds_utc", "forecast_origin_utc"], "future")
    prepared = normalize_utc_column(future, "ds_utc")
    prepared = normalize_utc_column(prepared, "forecast_origin_utc")
    if "horizon" not in prepared.columns:
        prepared = add_horizon_column(prepared)
    require_columns(prepared, config.base_features, "future")
    return prepared.sort_values(["forecast_origin_utc", "unique_id", "ds_utc"]).reset_index(drop=True)


def _resolve_single_origin(
    future: pd.DataFrame,
    forecast_origin_utc: object | None,
) -> pd.Timestamp:
    if forecast_origin_utc is None:
        origins = future["forecast_origin_utc"].drop_duplicates().tolist()
        if len(origins) != 1:
            raise ValueError(
                "future must contain exactly one forecast_origin_utc when "
                "forecast_origin_utc is not supplied"
            )
        return to_utc_timestamp(origins[0])
    origin = to_utc_timestamp(forecast_origin_utc)
    future_origins = future["forecast_origin_utc"].drop_duplicates()
    if not bool((future_origins == origin).all()):
        raise ValueError("future forecast_origin_utc values do not match forecast_origin_utc")
    return origin


def _base_feature_frame(future: pd.DataFrame, config: PriceFeatureConfig) -> pd.DataFrame:
    metadata_columns = [
        "unique_id",
        "area",
        "ds_utc",
        "ds_local",
        "local_date",
        "forecast_origin_utc",
        "horizon",
        "dataset_version",
    ]
    columns = _ordered_existing(metadata_columns + list(config.base_features), future.columns)
    return future[columns].copy()


def _add_lag_features(
    features: pd.DataFrame,
    history: pd.DataFrame,
    origin: pd.Timestamp,
    config: PriceFeatureConfig,
) -> pd.DataFrame:
    lookup = history[["area", "ds_utc", "y"]].rename(
        columns={"ds_utc": "feature_time", "y": "feature_y"}
    )
    output = features.copy()
    for lag in config.lag_hours:
        lagged = output[["area", "ds_utc"]].copy()
        lagged["feature_time"] = lagged["ds_utc"] - pd.Timedelta(hours=lag)
        lagged = lagged.merge(lookup, on=["area", "feature_time"], how="left")
        output[f"lag_{lag}h"] = lagged["feature_y"].where(
            lagged["feature_time"] < origin
        ).to_numpy()
    return output


def _add_rolling_features(
    features: pd.DataFrame,
    history: pd.DataFrame,
    origin: pd.Timestamp,
    config: PriceFeatureConfig,
) -> pd.DataFrame:
    output = features.copy()
    for window in config.rolling_windows_hours:
        summary = (
            history.loc[history["ds_utc"] >= origin - pd.Timedelta(hours=window)]
            .groupby("area")["y"]
            .agg(["mean", "median"])
            .rename(
                columns={
                    "mean": f"rolling_mean_{window}h_asof_origin",
                    "median": f"rolling_median_{window}h_asof_origin",
                }
            )
            .reset_index()
        )
        output = output.merge(summary, on="area", how="left")
    return output


def _add_seasonal_features(
    features: pd.DataFrame,
    history: pd.DataFrame,
    origin: pd.Timestamp,
    config: PriceFeatureConfig,
) -> pd.DataFrame:
    output = features.copy()
    seasonal_history = history.loc[
        history["ds_utc"] >= origin - pd.Timedelta(days=config.seasonal_lookback_days)
    ].copy()

    hour_summary = (
        seasonal_history.groupby(["area", "local_hour"])["y"]
        .median()
        .rename("seasonal_median_local_hour")
        .reset_index()
    )
    hour_weekend_summary = (
        seasonal_history.groupby(["area", "local_hour", "is_weekend"])["y"]
        .median()
        .rename("seasonal_median_hour_weekend")
        .reset_index()
    )
    output = output.merge(hour_summary, on=["area", "local_hour"], how="left")
    return output.merge(
        hour_weekend_summary,
        on=["area", "local_hour", "is_weekend"],
        how="left",
    )


def _add_spread_features(
    features: pd.DataFrame,
    history: pd.DataFrame,
    origin: pd.Timestamp,
    config: PriceFeatureConfig,
) -> pd.DataFrame:
    output = features.copy()
    for lag in config.spread_lag_hours:
        spread = _spread_lookup(history, lag)
        output = output.merge(
            spread,
            left_on="ds_utc",
            right_on="target_ds_utc",
            how="left",
        ).drop(columns=["target_ds_utc"])
        feature_time = output["ds_utc"] - pd.Timedelta(hours=lag)
        output[f"dk1_minus_dk2_lag_{lag}h"] = output[
            f"dk1_minus_dk2_lag_{lag}h"
        ].where(feature_time < origin)
    return output


def _spread_lookup(history: pd.DataFrame, lag_hours: int) -> pd.DataFrame:
    column = f"dk1_minus_dk2_lag_{lag_hours}h"
    if history.empty:
        return pd.DataFrame({"target_ds_utc": pd.Series(dtype="datetime64[ns, UTC]"), column: []})

    wide = history.pivot_table(index="ds_utc", columns="area", values="y", aggfunc="last")
    if {"DK1", "DK2"}.issubset(wide.columns):
        spread = (wide["DK1"] - wide["DK2"]).rename(column).reset_index()
    else:
        spread = pd.DataFrame({"ds_utc": pd.Series(dtype="datetime64[ns, UTC]"), column: []})
    spread["target_ds_utc"] = spread["ds_utc"] + pd.Timedelta(hours=lag_hours)
    return spread[["target_ds_utc", column]]


def _ordered_existing(columns: list[str], available: Any) -> list[str]:
    available_set = set(available)
    output: list[str] = []
    for column in columns:
        if column in available_set and column not in output:
            output.append(column)
    return output


def _require_positive(name: str, values: tuple[int, ...]) -> None:
    if not values:
        raise ValueError(f"{name} must not be empty")
    if any(value <= 0 for value in values):
        raise ValueError(f"{name} values must be positive")
