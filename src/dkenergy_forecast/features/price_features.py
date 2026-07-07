from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import pandas as pd

from dkenergy_forecast.backtesting.horizons import (
    make_daily_origins,
    make_danish_delivery_day_horizon,
    make_local_daily_origins,
)
from dkenergy_forecast.statistics import weighted_median
from dkenergy_forecast.types import (
    PRICE_AVAILABILITY_COLUMN,
    add_horizon_column,
    ensure_price_availability,
    filter_price_history_available_before,
    normalize_utc_column,
    price_available_before_mask,
    require_columns,
    to_utc_timestamp,
)


HorizonBuilder = Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame]
WEIGHTED_MEDIAN_BASELINE_COLUMN = "baseline_wdwe_weighted_median_y_pred"


@dataclass(frozen=True)
class PriceFeatureConfig:
    """Leakage-safe EDS-only feature recipe for tabular price models."""

    lag_hours: tuple[int, ...] = (24, 48, 168)
    rolling_windows_hours: tuple[int, ...] = (24, 168, 24 * 28)
    seasonal_lookback_days: int = 56
    spread_lag_hours: tuple[int, ...] = (24, 168)
    include_weighted_median_baseline: bool = True
    weighted_median_baseline_column: str = WEIGHTED_MEDIAN_BASELINE_COLUMN
    weekday_weighted_median_lookback_days: int = 42
    weekday_weighted_median_half_life_days: float = 4.0
    weekday_weighted_median_floor: float | None = 0.10
    weekend_weighted_median_lookback_days: int = 56
    weekend_weighted_median_half_life_days: float = 28.0
    weekend_weighted_median_floor: float | None = 0.20
    weighted_median_min_periods: int = 4
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
        for name, value in [
            ("weekday_weighted_median_lookback_days", self.weekday_weighted_median_lookback_days),
            ("weekend_weighted_median_lookback_days", self.weekend_weighted_median_lookback_days),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in [
            ("weekday_weighted_median_half_life_days", self.weekday_weighted_median_half_life_days),
            ("weekend_weighted_median_half_life_days", self.weekend_weighted_median_half_life_days),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in [
            ("weekday_weighted_median_floor", self.weekday_weighted_median_floor),
            ("weekend_weighted_median_floor", self.weekend_weighted_median_floor),
        ]:
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.weighted_median_min_periods <= 0:
            raise ValueError("weighted_median_min_periods must be positive")

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
    def baseline_feature_columns(self) -> list[str]:
        if not self.include_weighted_median_baseline:
            return []
        return [self.weighted_median_baseline_column]

    @property
    def feature_columns(self) -> list[str]:
        return (
            list(self.base_features)
            + self.lag_feature_columns
            + self.rolling_feature_columns
            + self.seasonal_feature_columns
            + self.baseline_feature_columns
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

    return _build_price_feature_frame_prepared(
        panel_utc,
        future_utc,
        forecast_origin_utc=origin,
        include_target=include_target,
        config=config,
    )


def _build_price_feature_frame_prepared(
    panel_utc: pd.DataFrame,
    future_utc: pd.DataFrame,
    *,
    forecast_origin_utc: pd.Timestamp,
    include_target: bool,
    config: PriceFeatureConfig,
) -> pd.DataFrame:
    origin = to_utc_timestamp(forecast_origin_utc)
    history = filter_price_history_available_before(panel_utc, origin)
    features = _base_feature_frame(future_utc, config)

    features = _add_lag_features(features, history, origin, config)
    features = _add_rolling_features(features, history, origin, config)
    features = _add_seasonal_features(features, history, origin, config)
    features = _add_weighted_median_baseline_feature(features, history, origin, config)
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


def build_price_experiment_frame(
    panel: pd.DataFrame,
    origins: pd.DataFrame,
    *,
    config: PriceFeatureConfig | None = None,
    horizon_builder: HorizonBuilder = make_danish_delivery_day_horizon,
) -> pd.DataFrame:
    """Build price feature rows for several rolling origins with targets joined."""

    require_columns(origins, ["forecast_origin_utc"], "origins")
    config = config or PriceFeatureConfig()
    panel_utc = _prepare_panel(panel, config)
    origins_utc = normalize_utc_column(origins, "forecast_origin_utc")
    origin_config = (
        replace(config, include_weighted_median_baseline=False)
        if config.include_weighted_median_baseline
        else config
    )
    frames: list[pd.DataFrame] = []

    for origin in origins_utc["forecast_origin_utc"].sort_values().drop_duplicates():
        origin = to_utc_timestamp(origin)
        future = horizon_builder(panel_utc, origin)
        if future.empty:
            continue
        future_utc = _prepare_future(future, origin_config)
        frame = _build_price_feature_frame_prepared(
            panel_utc,
            future_utc,
            forecast_origin_utc=origin,
            include_target=True,
            config=origin_config,
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    if config.include_weighted_median_baseline:
        output = add_weighted_median_baseline_feature(output, panel_utc, config=config)
    return output


def add_weighted_median_baseline_feature(
    frame: pd.DataFrame,
    panel: pd.DataFrame,
    *,
    config: PriceFeatureConfig | None = None,
) -> pd.DataFrame:
    """Add the robust weekday/weekend weighted-median baseline as a feature column."""

    config = config or PriceFeatureConfig()
    if not config.include_weighted_median_baseline:
        return frame.copy()

    require_columns(frame, ["unique_id", "ds_utc", "forecast_origin_utc", "local_hour", "is_weekend"], "frame")
    panel_utc = _prepare_panel(panel, config)
    output = normalize_utc_column(frame, "ds_utc")
    output = normalize_utc_column(output, "forecast_origin_utc")
    output[config.weighted_median_baseline_column] = _weighted_median_baseline_predictions(
        output,
        panel_utc,
        config,
    )
    return output


def build_training_matrix(
    panel: pd.DataFrame,
    forecast_origin_utc: object,
    *,
    training_origin_days: int = 70,
    at_hour_utc: int | None = None,
    forecast_local_time: str = "12:00",
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
    origin_config = (
        replace(config, include_weighted_median_baseline=False)
        if config.include_weighted_median_baseline
        else config
    )
    train_origin_end = origin
    train_origin_start = train_origin_end - pd.Timedelta(days=training_origin_days)
    if at_hour_utc is None:
        train_origins = make_local_daily_origins(
            panel_utc,
            start=train_origin_start,
            end=train_origin_end,
            forecast_local_time=forecast_local_time,
        )
    else:
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
        horizon_with_availability = ensure_price_availability(horizon)
        if require_complete_horizons_before_origin and not bool(
            price_available_before_mask(horizon_with_availability, origin).all()
        ):
            continue
        if horizon["ds_utc"].min() < panel_utc["ds_utc"].min():
            continue

        frame = build_price_feature_frame(
            panel_utc,
            horizon,
            forecast_origin_utc=train_origin,
            include_target=True,
            config=origin_config,
        )
        frames.append(frame)

    if not frames:
        raise ValueError(f"No training feature rows available before {origin.isoformat()}")

    training = pd.concat(frames, ignore_index=True)
    if config.include_weighted_median_baseline:
        training = add_weighted_median_baseline_feature(training, panel_utc, config=config)
    return training.dropna(subset=["y"]).reset_index(drop=True)


def _prepare_panel(panel: pd.DataFrame, config: PriceFeatureConfig) -> pd.DataFrame:
    require_columns(panel, ["unique_id", "area", "ds_utc", "y"], "panel")
    prepared = ensure_price_availability(normalize_utc_column(panel, "ds_utc"))
    required_calendar = _required_calendar_columns(config)
    require_columns(prepared, required_calendar, "panel")
    return prepared.sort_values(["area", "ds_utc"]).reset_index(drop=True)


def _prepare_future(future: pd.DataFrame, config: PriceFeatureConfig) -> pd.DataFrame:
    require_columns(future, ["unique_id", "area", "ds_utc", "forecast_origin_utc"], "future")
    prepared = ensure_price_availability(normalize_utc_column(future, "ds_utc"))
    prepared = normalize_utc_column(prepared, "forecast_origin_utc")
    if "horizon" not in prepared.columns:
        prepared = add_horizon_column(prepared)
    require_columns(prepared, list(config.base_features) + _baseline_calendar_columns(config), "future")
    return prepared.sort_values(["forecast_origin_utc", "unique_id", "ds_utc"]).reset_index(drop=True)


def _required_calendar_columns(config: PriceFeatureConfig) -> list[str]:
    columns = [
        column
        for column in ["local_hour", "local_day_of_week", "local_month", "is_weekend", "is_dst", "utc_offset_hours"]
        if column in config.base_features
    ]
    return _ordered_existing(columns + _baseline_calendar_columns(config), columns + _baseline_calendar_columns(config))


def _baseline_calendar_columns(config: PriceFeatureConfig) -> list[str]:
    if not config.include_weighted_median_baseline:
        return []
    return ["local_hour", "is_weekend"]


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
        PRICE_AVAILABILITY_COLUMN,
    ]
    columns = _ordered_existing(metadata_columns + list(config.base_features), future.columns)
    return future[columns].copy()


def _add_lag_features(
    features: pd.DataFrame,
    history: pd.DataFrame,
    origin: pd.Timestamp,
    config: PriceFeatureConfig,
) -> pd.DataFrame:
    lookup = history[["area", "ds_utc", "y", PRICE_AVAILABILITY_COLUMN]].rename(
        columns={
            "ds_utc": "feature_time",
            "y": "feature_y",
            PRICE_AVAILABILITY_COLUMN: "feature_available_at",
        }
    )
    output = features.copy()
    for lag in config.lag_hours:
        lagged = output[["area", "ds_utc"]].copy()
        lagged["feature_time"] = lagged["ds_utc"] - pd.Timedelta(hours=lag)
        lagged = lagged.merge(lookup, on=["area", "feature_time"], how="left")
        output[f"lag_{lag}h"] = lagged["feature_y"].where(
            pd.to_datetime(lagged["feature_available_at"], utc=True) < origin
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
        window_history = (
            history.sort_values(["area", "ds_utc"])
            .groupby("area", group_keys=False)
            .tail(window)
        )
        summary = (
            window_history
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
    hour_groups = _seasonal_history_groups(history, ["area", "local_hour"])
    hour_weekend_groups = _seasonal_history_groups(history, ["area", "local_hour", "is_weekend"])
    lookback = pd.Timedelta(days=config.seasonal_lookback_days)
    hour_values: list[float | None] = []
    hour_weekend_values: list[float | None] = []

    for row in output.itertuples(index=False):
        target_time = to_utc_timestamp(getattr(row, "ds_utc"))
        window_start = target_time - lookback
        hour_values.append(
            _median_from_target_relative_window(
                hour_groups.get((getattr(row, "area"), int(getattr(row, "local_hour")))),
                target_time=target_time,
                window_start=window_start,
            )
        )
        hour_weekend_values.append(
            _median_from_target_relative_window(
                hour_weekend_groups.get(
                    (
                        getattr(row, "area"),
                        int(getattr(row, "local_hour")),
                        bool(getattr(row, "is_weekend")),
                    )
                ),
                target_time=target_time,
                window_start=window_start,
            )
        )

    output["seasonal_median_local_hour"] = hour_values
    output["seasonal_median_hour_weekend"] = hour_weekend_values
    return output


def _add_weighted_median_baseline_feature(
    features: pd.DataFrame,
    history: pd.DataFrame,
    origin: pd.Timestamp,
    config: PriceFeatureConfig,
) -> pd.DataFrame:
    if not config.include_weighted_median_baseline:
        return features

    require_columns(features, ["unique_id", "local_hour", "is_weekend"], "features")
    require_columns(history, ["unique_id", "local_hour", "is_weekend"], "history")
    output = features.copy()
    output[config.weighted_median_baseline_column] = _weighted_median_baseline_predictions(
        output,
        history,
        config,
    )
    return output


def _weighted_median_baseline_predictions(
    future: pd.DataFrame,
    history: pd.DataFrame,
    config: PriceFeatureConfig,
) -> list[float | None]:
    groups = _weighted_median_history_groups(history)

    predictions: list[float | None] = []
    for row in future.itertuples(index=False):
        is_weekend = bool(getattr(row, "is_weekend"))
        group = groups.get((getattr(row, "unique_id"), int(getattr(row, "local_hour")), is_weekend))
        if group is None:
            predictions.append(None)
            continue

        lookback_days, half_life_days, floor = _weighted_median_parameters(config, is_weekend)
        origin = to_utc_timestamp(getattr(row, "forecast_origin_utc"))
        target_time = to_utc_timestamp(getattr(row, "ds_utc"))
        window = group[
            (pd.to_datetime(group[PRICE_AVAILABILITY_COLUMN], utc=True) < origin)
            & (group["ds_utc"] < target_time)
            & (group["ds_utc"] >= target_time - pd.Timedelta(days=lookback_days))
        ].dropna(subset=["y"])
        if len(window) < config.weighted_median_min_periods:
            predictions.append(None)
            continue

        age_days = (target_time - window["ds_utc"]) / pd.Timedelta(days=1)
        weights = 0.5 ** (age_days / float(half_life_days))
        if floor is not None:
            floor_value = float(floor)
            weights = floor_value + (1 - floor_value) * weights
        predictions.append(weighted_median(window["y"], weights))

    return predictions


def _weighted_median_history_groups(
    history: pd.DataFrame,
) -> dict[tuple[object, int, bool], pd.DataFrame]:
    groups: dict[tuple[object, int, bool], pd.DataFrame] = {}
    for key, frame in history.groupby(["unique_id", "local_hour", "is_weekend"], dropna=False, sort=False):
        unique_id, local_hour, is_weekend = key
        groups[(unique_id, int(local_hour), bool(is_weekend))] = (
            frame[["ds_utc", "y", PRICE_AVAILABILITY_COLUMN]]
            .sort_values("ds_utc")
            .reset_index(drop=True)
        )
    return groups


def _weighted_median_parameters(
    config: PriceFeatureConfig,
    is_weekend: bool,
) -> tuple[int, float, float | None]:
    if is_weekend:
        return (
            config.weekend_weighted_median_lookback_days,
            config.weekend_weighted_median_half_life_days,
            config.weekend_weighted_median_floor,
        )
    return (
        config.weekday_weighted_median_lookback_days,
        config.weekday_weighted_median_half_life_days,
        config.weekday_weighted_median_floor,
    )


def _add_spread_features(
    features: pd.DataFrame,
    history: pd.DataFrame,
    origin: pd.Timestamp,
    config: PriceFeatureConfig,
) -> pd.DataFrame:
    output = features.copy()
    for lag in config.spread_lag_hours:
        column = f"dk1_minus_dk2_lag_{lag}h"
        feature_time = output["ds_utc"] - pd.Timedelta(hours=lag)
        needed_times = pd.DataFrame({"feature_time": feature_time.drop_duplicates()})
        source = history.loc[
            history["ds_utc"].isin(needed_times["feature_time"]),
            ["ds_utc", "area", "y"],
        ]
        if source.empty:
            output[column] = pd.NA
            continue
        wide = source.pivot_table(index="ds_utc", columns="area", values="y", aggfunc="last")
        if {"DK1", "DK2"}.issubset(wide.columns):
            spread = (wide["DK1"] - wide["DK2"]).rename(column).reset_index()
            spread = spread.rename(columns={"ds_utc": "feature_time"})
            lagged = needed_times.merge(spread, on="feature_time", how="left")
        else:
            lagged = needed_times.copy()
            lagged[column] = pd.NA
        lagged = lagged.set_index("feature_time")
        output[column] = feature_time.map(lagged[column])
    return output


def _seasonal_history_groups(
    history: pd.DataFrame,
    key_columns: list[str],
) -> dict[tuple[object, ...], pd.DataFrame]:
    groups: dict[tuple[object, ...], pd.DataFrame] = {}
    if history.empty:
        return groups
    for key, frame in history.groupby(key_columns, dropna=False, sort=False):
        group_key = key if isinstance(key, tuple) else (key,)
        groups[group_key] = frame[["ds_utc", "y"]].sort_values("ds_utc").reset_index(drop=True)
    return groups


def _median_from_target_relative_window(
    group: pd.DataFrame | None,
    *,
    target_time: pd.Timestamp,
    window_start: pd.Timestamp,
) -> float | None:
    if group is None or group.empty:
        return None
    window = group[
        (group["ds_utc"] < target_time)
        & (group["ds_utc"] >= window_start)
    ].dropna(subset=["y"])
    if window.empty:
        return None
    return float(window["y"].median())


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
