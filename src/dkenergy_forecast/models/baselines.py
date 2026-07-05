from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from dkenergy_forecast.statistics import weighted_median
from dkenergy_forecast.types import (
    add_horizon_column,
    normalize_utc_column,
    require_columns,
)


Fallback = Literal["last_available"] | None
WeightFamily = Literal["equal", "linear", "linear_floor", "exponential"]


@dataclass
class LagNaive:
    lag_hours: int
    fallback: Fallback = None

    model_name: str = "lag_naive"
    model_version: str = "v1"

    def __post_init__(self) -> None:
        if self.lag_hours <= 0:
            raise ValueError("lag_hours must be positive")
        if self.fallback not in (None, "last_available"):
            raise ValueError("fallback must be None or 'last_available'")
        self._history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> "LagNaive":
        self._history = _prepare_history(history)
        return self

    def predict(
        self,
        future: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        future_base = _prepare_future(future)
        history_frame = _resolve_history(history, self._history)

        lookup = history_frame.rename(
            columns={"ds_utc": "lagged_ds_utc", "y": "lagged_y"}
        )[["unique_id", "lagged_ds_utc", "lagged_y"]]
        output = future_base.copy()
        output["lagged_ds_utc"] = output["ds_utc"] - pd.Timedelta(hours=self.lag_hours)
        output = output.merge(
            lookup,
            on=["unique_id", "lagged_ds_utc"],
            how="left",
        )
        known_before_origin = output["lagged_ds_utc"] < output["forecast_origin_utc"]
        output["y_pred"] = output["lagged_y"].where(known_before_origin)

        if self.fallback == "last_available":
            missing = output["y_pred"].isna()
            if bool(missing.any()):
                output.loc[missing, "y_pred"] = [
                    _last_available_y(history_frame, row.unique_id, row.forecast_origin_utc)
                    for row in output.loc[missing, ["unique_id", "forecast_origin_utc"]].itertuples(index=False)
                ]

        return _finalize_predictions(output, self.model_name, self.model_version)


@dataclass
class SeasonalRollingMedian:
    lookback_days: int = 28
    seasonal_keys: tuple[str, ...] = ("local_hour",)
    min_periods: int = 7

    model_name: str = "seasonal_rolling_median"
    model_version: str = "v1"

    def __post_init__(self) -> None:
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if self.min_periods <= 0:
            raise ValueError("min_periods must be positive")
        if not self.seasonal_keys:
            raise ValueError("seasonal_keys must not be empty")
        self._history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> "SeasonalRollingMedian":
        self._history = _prepare_history(history)
        return self

    def predict(
        self,
        future: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        future_base = _prepare_future(future)
        require_columns(future_base, self.seasonal_keys, "future")
        history_frame = _resolve_history(history, self._history)
        require_columns(history_frame, self.seasonal_keys, "history")

        lookback = pd.Timedelta(days=self.lookback_days)
        predictions: list[float | None] = []
        for row in future_base.itertuples(index=False):
            candidates = history_frame[
                (history_frame["unique_id"] == row.unique_id)
                & (history_frame["ds_utc"] < row.forecast_origin_utc)
                & (history_frame["ds_utc"] >= row.forecast_origin_utc - lookback)
            ]
            for key in self.seasonal_keys:
                candidates = candidates[candidates[key] == getattr(row, key)]

            if len(candidates) < self.min_periods:
                predictions.append(None)
            else:
                predictions.append(float(candidates["y"].median()))

        output = future_base.copy()
        output["y_pred"] = predictions
        return _finalize_predictions(output, self.model_name, self.model_version)


@dataclass
class WeightedSeasonalMedian:
    lookback_days: int = 56
    seasonal_keys: tuple[str, ...] = ("local_hour", "is_weekend")
    min_periods: int = 4
    weight_family: WeightFamily = "equal"
    half_life_days: float | None = None
    floor: float | None = None

    model_name: str = "weighted_seasonal_median"
    model_version: str = "v1"

    def __post_init__(self) -> None:
        if self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        if self.min_periods <= 0:
            raise ValueError("min_periods must be positive")
        if not self.seasonal_keys:
            raise ValueError("seasonal_keys must not be empty")
        if self.weight_family not in ("equal", "linear", "linear_floor", "exponential"):
            raise ValueError(
                "weight_family must be one of "
                "'equal', 'linear', 'linear_floor', or 'exponential'"
            )
        if self.weight_family == "exponential" and (
            self.half_life_days is None or self.half_life_days <= 0
        ):
            raise ValueError("half_life_days must be positive for exponential weights")
        if self.floor is not None and not 0 <= self.floor <= 1:
            raise ValueError("floor must be between 0 and 1")
        if self.weight_family == "linear_floor" and self.floor is None:
            raise ValueError("floor must be supplied for linear_floor weights")
        self._history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> "WeightedSeasonalMedian":
        self._history = _prepare_history(history)
        return self

    def predict(
        self,
        future: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        future_base = _prepare_future(future)
        require_columns(future_base, self.seasonal_keys, "future")
        history_frame = _resolve_history(history, self._history)
        require_columns(history_frame, self.seasonal_keys, "history")

        grouped_history = _seasonal_history_groups(history_frame, self.seasonal_keys)
        predictions: list[float | None] = []

        for row in future_base.itertuples(index=False):
            key = _seasonal_group_key(row, self.seasonal_keys)
            candidates = grouped_history.get(key)
            if candidates is None:
                predictions.append(None)
                continue

            lookback = pd.Timedelta(days=self.lookback_days)
            window_start = row.forecast_origin_utc - lookback
            window = candidates[
                (candidates["ds_utc"] < row.forecast_origin_utc)
                & (candidates["ds_utc"] >= window_start)
            ]
            prediction = self._predict_from_window(
                window,
                row.forecast_origin_utc,
                lookback_days=self.lookback_days,
            )
            predictions.append(prediction)

        output = future_base.copy()
        output["y_pred"] = predictions
        return _finalize_predictions(output, self.model_name, self.model_version)

    def _predict_from_window(
        self,
        window: pd.DataFrame,
        forecast_origin_utc: pd.Timestamp,
        *,
        lookback_days: int,
    ) -> float | None:
        if window.empty:
            return None

        values = window[["ds_utc", "y"]].dropna(subset=["y"]).copy()
        if values.empty:
            return None

        if self.weight_family == "equal":
            if len(values) < self.min_periods:
                return None
            return float(values["y"].median())

        age_days = (
            forecast_origin_utc - values["ds_utc"]
        ) / pd.Timedelta(days=1)
        weights = self._weights_for_age_days(age_days, lookback_days=lookback_days)
        positive_weight = weights > 0
        if int(positive_weight.sum()) < self.min_periods:
            return None

        return weighted_median(
            values.loc[positive_weight, "y"],
            weights.loc[positive_weight],
        )

    def _weights_for_age_days(
        self,
        age_days: pd.Series,
        *,
        lookback_days: int,
    ) -> pd.Series:
        if self.weight_family == "linear":
            return (1 - age_days / lookback_days).clip(lower=0)
        if self.weight_family == "linear_floor":
            floor = float(self.floor)
            linear = (1 - age_days / lookback_days).clip(lower=0)
            return floor + (1 - floor) * linear
        if self.weight_family == "exponential":
            half_life_days = float(self.half_life_days)
            weights = 0.5 ** (age_days / half_life_days)
            if self.floor is not None:
                floor = float(self.floor)
                weights = floor + (1 - floor) * weights
            return weights
        raise ValueError(f"Unsupported weight_family: {self.weight_family}")


@dataclass
class WeekdayWeekendWeightedMedian:
    weekday_lookback_days: int = 42
    weekday_half_life_days: float = 4
    weekday_floor: float | None = 0.10
    weekend_lookback_days: int = 56
    weekend_half_life_days: float = 28
    weekend_floor: float | None = 0.20
    seasonal_keys: tuple[str, ...] = ("local_hour", "is_weekend")
    min_periods: int = 4

    model_name: str = "weekday_weekend_weighted_median"
    model_version: str = "v1"

    def __post_init__(self) -> None:
        for name, value in [
            ("weekday_lookback_days", self.weekday_lookback_days),
            ("weekend_lookback_days", self.weekend_lookback_days),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in [
            ("weekday_half_life_days", self.weekday_half_life_days),
            ("weekend_half_life_days", self.weekend_half_life_days),
        ]:
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in [
            ("weekday_floor", self.weekday_floor),
            ("weekend_floor", self.weekend_floor),
        ]:
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.min_periods <= 0:
            raise ValueError("min_periods must be positive")
        if "is_weekend" not in self.seasonal_keys:
            raise ValueError("seasonal_keys must include is_weekend")
        self._history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> "WeekdayWeekendWeightedMedian":
        self._history = _prepare_history(history)
        return self

    def predict(
        self,
        future: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        future_base = _prepare_future(future)
        require_columns(future_base, self.seasonal_keys, "future")
        history_frame = _resolve_history(history, self._history)
        require_columns(history_frame, self.seasonal_keys, "history")

        grouped_history = _seasonal_history_groups(history_frame, self.seasonal_keys)
        predictions: list[float | None] = []

        for row in future_base.itertuples(index=False):
            key = _seasonal_group_key(row, self.seasonal_keys)
            candidates = grouped_history.get(key)
            if candidates is None:
                predictions.append(None)
                continue

            lookback_days, half_life_days, floor = self._parameters_for_row(row)
            window_start = row.forecast_origin_utc - pd.Timedelta(days=lookback_days)
            window = candidates[
                (candidates["ds_utc"] < row.forecast_origin_utc)
                & (candidates["ds_utc"] >= window_start)
            ]
            predictions.append(
                self._predict_from_window(
                    window,
                    row.forecast_origin_utc,
                    half_life_days=half_life_days,
                    floor=floor,
                )
            )

        output = future_base.copy()
        output["y_pred"] = predictions
        return _finalize_predictions(output, self.model_name, self.model_version)

    def _parameters_for_row(self, row: object) -> tuple[int, float, float | None]:
        if bool(getattr(row, "is_weekend")):
            return (
                self.weekend_lookback_days,
                self.weekend_half_life_days,
                self.weekend_floor,
            )
        return (
            self.weekday_lookback_days,
            self.weekday_half_life_days,
            self.weekday_floor,
        )

    def _predict_from_window(
        self,
        window: pd.DataFrame,
        forecast_origin_utc: pd.Timestamp,
        *,
        half_life_days: float,
        floor: float | None,
    ) -> float | None:
        if window.empty:
            return None
        values = window[["ds_utc", "y"]].dropna(subset=["y"]).copy()
        if len(values) < self.min_periods:
            return None

        age_days = (forecast_origin_utc - values["ds_utc"]) / pd.Timedelta(days=1)
        weights = 0.5 ** (age_days / float(half_life_days))
        if floor is not None:
            floor_value = float(floor)
            weights = floor_value + (1 - floor_value) * weights
        return weighted_median(values["y"], weights)


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    require_columns(history, ["unique_id", "ds_utc", "y"], "history")
    prepared = normalize_utc_column(history, "ds_utc")
    return prepared.sort_values(["unique_id", "ds_utc"]).reset_index(drop=True)


def _prepare_future(future: pd.DataFrame) -> pd.DataFrame:
    require_columns(future, ["unique_id", "ds_utc", "forecast_origin_utc"], "future")
    prepared = normalize_utc_column(future, "ds_utc")
    prepared = normalize_utc_column(prepared, "forecast_origin_utc")
    if "horizon" not in prepared.columns:
        prepared = add_horizon_column(prepared)
    return prepared


def _resolve_history(
    history: pd.DataFrame | None,
    fitted_history: pd.DataFrame | None,
) -> pd.DataFrame:
    if history is not None:
        return _prepare_history(history)
    if fitted_history is not None:
        return fitted_history
    raise ValueError("history must be supplied either through fit() or predict()")


def _seasonal_history_groups(
    history: pd.DataFrame,
    seasonal_keys: tuple[str, ...],
) -> dict[tuple[object, ...], pd.DataFrame]:
    key_columns = ["unique_id", *seasonal_keys]
    groups: dict[tuple[object, ...], pd.DataFrame] = {}
    for key, frame in history.groupby(key_columns, dropna=False, sort=False):
        group_key = key if isinstance(key, tuple) else (key,)
        groups[group_key] = frame[["ds_utc", "y"]].sort_values("ds_utc").reset_index(drop=True)
    return groups


def _seasonal_group_key(row: object, seasonal_keys: tuple[str, ...]) -> tuple[object, ...]:
    return (getattr(row, "unique_id"), *[getattr(row, key) for key in seasonal_keys])


def _last_available_y(
    history: pd.DataFrame,
    unique_id: str,
    forecast_origin_utc: pd.Timestamp,
) -> float | None:
    candidates = history[
        (history["unique_id"] == unique_id)
        & (history["ds_utc"] < forecast_origin_utc)
    ]
    if candidates.empty:
        return None
    return float(candidates.sort_values("ds_utc").iloc[-1]["y"])


def _finalize_predictions(
    frame: pd.DataFrame,
    model_name: str,
    model_version: str,
) -> pd.DataFrame:
    output = frame.copy()
    output["model_name"] = model_name
    output["model_version"] = model_version
    return output[
        [
            "unique_id",
            "ds_utc",
            "forecast_origin_utc",
            "horizon",
            "model_name",
            "model_version",
            "y_pred",
        ]
    ].reset_index(drop=True)
