from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dkenergy_forecast.features.price_features import (
    PriceFeatureConfig,
    build_price_feature_frame,
    build_training_matrix,
)
from dkenergy_forecast.types import add_horizon_column, normalize_utc_column, require_columns


DEFAULT_QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}


@dataclass
class CatBoostQuantileModel:
    """Rolling-origin CatBoost quantile model for EDS-only price features."""

    feature_config: PriceFeatureConfig = field(default_factory=PriceFeatureConfig)
    quantiles: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_QUANTILES))
    training_origin_days: int = 70
    at_hour_utc: int = 10
    iterations: int = 250
    depth: int = 6
    learning_rate: float = 0.05
    l2_leaf_reg: float = 8.0
    random_seed: int = 42
    verbose: bool = False

    model_name: str = "catboost_quantile"
    model_version: str = "v1"

    def __post_init__(self) -> None:
        if self.training_origin_days <= 0:
            raise ValueError("training_origin_days must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if not self.quantiles:
            raise ValueError("quantiles must not be empty")
        seen_quantile_values: set[float] = set()
        for name, value in self.quantiles.items():
            if not name.startswith("q"):
                raise ValueError(f"Quantile column names must start with 'q': {name}")
            if not 0 < value < 1:
                raise ValueError(f"Quantile value must be between 0 and 1: {name}={value}")
            if value in seen_quantile_values:
                raise ValueError(f"Duplicate quantile value is ambiguous: {value}")
            seen_quantile_values.add(value)
        self.quantiles = dict(sorted(self.quantiles.items(), key=lambda item: item[1]))
        self._history: pd.DataFrame | None = None
        self.last_models_: dict[str, Any] = {}
        self.last_feature_columns_: list[str] = []
        self.last_raw_quantile_crossing_rate_: float | None = None

    def fit(self, history: pd.DataFrame) -> "CatBoostQuantileModel":
        self._history = _prepare_history(history)
        return self

    def predict(
        self,
        future: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        CatBoostRegressor, Pool = _load_catboost()
        history_frame = _resolve_history(history, self._history)
        future_frame = _prepare_future(future)

        outputs: list[pd.DataFrame] = []
        self.last_models_ = {}
        for origin, origin_future in future_frame.groupby("forecast_origin_utc", sort=True):
            train_frame = build_training_matrix(
                history_frame,
                origin,
                training_origin_days=self.training_origin_days,
                at_hour_utc=self.at_hour_utc,
                config=self.feature_config,
            )
            predict_frame = build_price_feature_frame(
                history_frame,
                origin_future,
                forecast_origin_utc=origin,
                include_target=False,
                config=self.feature_config,
            )

            feature_columns = self.feature_config.feature_columns
            if train_frame.empty:
                raise ValueError(
                    "No CatBoost training rows available before forecast origin "
                    f"{pd.Timestamp(origin).isoformat()} after dropping missing targets"
                )

            train_pool = Pool(
                train_frame[feature_columns],
                label=train_frame["y"],
                cat_features=list(self.feature_config.categorical_features),
            )
            predict_pool = Pool(
                predict_frame[feature_columns],
                cat_features=list(self.feature_config.categorical_features),
            )

            origin_models: dict[str, Any] = {}
            origin_predictions = predict_frame[
                [
                    column
                    for column in [
                        "unique_id",
                        "ds_utc",
                        "forecast_origin_utc",
                        "horizon",
                        "area",
                        "ds_local",
                        "local_date",
                        "dataset_version",
                    ]
                    if column in predict_frame.columns
                ]
            ].copy()
            for column_name, alpha in self.quantiles.items():
                model = CatBoostRegressor(
                    loss_function=f"Quantile:alpha={alpha}",
                    iterations=self.iterations,
                    depth=self.depth,
                    learning_rate=self.learning_rate,
                    l2_leaf_reg=self.l2_leaf_reg,
                    random_seed=self.random_seed,
                    verbose=self.verbose,
                    allow_writing_files=False,
                )
                model.fit(train_pool)
                origin_models[column_name] = model
                origin_predictions[column_name] = model.predict(predict_pool)

            raw_crossing_rate = _quantile_crossing_rate(origin_predictions, list(self.quantiles))
            origin_predictions = _repair_quantile_crossing(origin_predictions, list(self.quantiles))
            origin_predictions["raw_quantile_crossing_rate"] = raw_crossing_rate
            origin_predictions["model_name"] = self.model_name
            origin_predictions["model_version"] = self.model_version
            origin_predictions["y_pred"] = _median_prediction(origin_predictions, self.quantiles)
            outputs.append(origin_predictions)
            self.last_models_[pd.Timestamp(origin).isoformat()] = origin_models
            self.last_feature_columns_ = feature_columns
            self.last_raw_quantile_crossing_rate_ = raw_crossing_rate

        if not outputs:
            return pd.DataFrame(
                columns=[
                    "unique_id",
                    "ds_utc",
                    "forecast_origin_utc",
                    "horizon",
                    "model_name",
                    "model_version",
                    "y_pred",
                    *self.quantiles.keys(),
                    "raw_quantile_crossing_rate",
                ]
            )

        output = pd.concat(outputs, ignore_index=True)
        ordered_columns = [
            "unique_id",
            "ds_utc",
            "forecast_origin_utc",
            "horizon",
            "model_name",
            "model_version",
            "y_pred",
            *self.quantiles.keys(),
            "raw_quantile_crossing_rate",
        ]
        optional_columns = [
            column
            for column in ["area", "ds_local", "local_date", "dataset_version"]
            if column in output.columns
        ]
        return output[ordered_columns + optional_columns].reset_index(drop=True)

    def feature_importance_frame(self) -> pd.DataFrame:
        """Return feature importances for the most recent fitted backend models."""

        rows: list[dict[str, object]] = []
        for origin, origin_models in self.last_models_.items():
            for quantile_column, model in origin_models.items():
                if not hasattr(model, "get_feature_importance"):
                    continue
                importances = model.get_feature_importance()
                for feature_name, importance in zip(self.last_feature_columns_, importances):
                    rows.append(
                        {
                            "forecast_origin_utc": origin,
                            "quantile": quantile_column,
                            "feature": feature_name,
                            "importance": float(importance),
                        }
                    )

        return pd.DataFrame(
            rows,
            columns=[
                "forecast_origin_utc",
                "quantile",
                "feature",
                "importance",
            ],
        )


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    require_columns(history, ["unique_id", "area", "ds_utc", "y"], "history")
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


def _load_catboost() -> tuple[Any, Any]:
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise ImportError(
            "CatBoostQuantileModel requires the optional CatBoost dependency. "
            'Install it with `pip install -e ".[catboost]"` or `pip install catboost>=1.2`.'
        ) from exc
    return CatBoostRegressor, Pool


def _quantile_crossing_rate(frame: pd.DataFrame, quantile_columns: list[str]) -> float:
    if len(quantile_columns) < 2:
        return 0.0
    values = frame[quantile_columns].to_numpy()
    crossings = np.diff(values, axis=1) < 0
    return float(crossings.any(axis=1).mean())


def _repair_quantile_crossing(frame: pd.DataFrame, quantile_columns: list[str]) -> pd.DataFrame:
    if len(quantile_columns) < 2:
        return frame
    output = frame.copy()
    output[quantile_columns] = np.sort(output[quantile_columns].to_numpy(), axis=1)
    return output


def _median_prediction(frame: pd.DataFrame, quantiles: dict[str, float]) -> pd.Series:
    if "q50" in frame.columns:
        return frame["q50"]
    closest_column = min(quantiles, key=lambda column: abs(quantiles[column] - 0.5))
    return frame[closest_column]
