from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from dkenergy_forecast.features import tabular_feature_columns_for_set
from dkenergy_forecast.features.price_features import (
    WEIGHTED_MEDIAN_BASELINE_COLUMN,
    PriceFeatureConfig,
    build_price_feature_frame,
    build_training_matrix,
)
from dkenergy_forecast.types import (
    ForecastModel,
    normalize_utc_column,
    require_columns,
    to_utc_timestamp,
)


TargetMode = Literal["direct", "residual_baseline"]


DEFAULT_CATBOOST_PARAMS: dict[str, Any] = {
    "loss_function": "Quantile:alpha=0.5",
    "eval_metric": "MAE",
    "iterations": 500,
    "depth": 5,
    "learning_rate": 0.03,
    "l2_leaf_reg": 50.0,
    "random_strength": 8.0,
    "bootstrap_type": "Bernoulli",
    "subsample": 0.85,
    "border_count": 64,
    "boosting_type": "Plain",
    "leaf_estimation_iterations": 2,
    "random_seed": 42,
    "has_time": True,
    "verbose": False,
    "allow_writing_files": False,
}


@dataclass(frozen=True)
class CatBoostExperimentConfig:
    """Source-controlled CatBoost comparison policy.

    The fixed parameters make notebook comparisons reproducible; this model is
    not part of live publication.
    """

    feature_set: str = "price_full_engineered"
    target_mode: TargetMode = "residual_baseline"
    training_origin_days: int = 365
    at_hour_utc: int | None = None
    forecast_local_time: str = "12:00"
    recency_half_life_days: float | None = 720.0
    recency_floor: float | None = None
    residual_baseline_column: str = WEIGHTED_MEDIAN_BASELINE_COLUMN
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_CATBOOST_PARAMS))
    price_feature_config: PriceFeatureConfig = field(default_factory=PriceFeatureConfig)


CATBOOST_EXPERIMENT_CONFIG = CatBoostExperimentConfig()


@dataclass
class ExperimentalCatBoostDayAhead(ForecastModel):
    config: CatBoostExperimentConfig = field(
        default_factory=lambda: CATBOOST_EXPERIMENT_CONFIG
    )
    model_name: str = "experimental_catboost_day_ahead"
    model_version: str = "v1"

    def __post_init__(self) -> None:
        if self.config.training_origin_days <= 0:
            raise ValueError("training_origin_days must be positive")
        if self.config.target_mode not in {"direct", "residual_baseline"}:
            raise ValueError("target_mode must be 'direct' or 'residual_baseline'")
        if self.config.recency_half_life_days is not None and self.config.recency_half_life_days <= 0:
            raise ValueError("recency_half_life_days must be positive when supplied")
        if self.config.recency_floor is not None and not 0 <= self.config.recency_floor <= 1:
            raise ValueError("recency_floor must be between 0 and 1")
        self._history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> "ExperimentalCatBoostDayAhead":
        self._history = _prepare_history(history)
        return self

    def predict(
        self,
        future: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        history_frame = _resolve_history(history, self._history)
        future_frame = _prepare_future(future)
        origin = _single_forecast_origin(future_frame)

        training = build_training_matrix(
            history_frame,
            origin,
            training_origin_days=self.config.training_origin_days,
            at_hour_utc=self.config.at_hour_utc,
            forecast_local_time=self.config.forecast_local_time,
            config=self.config.price_feature_config,
        )
        future_features = build_price_feature_frame(
            history_frame,
            future_frame,
            forecast_origin_utc=origin,
            include_target=False,
            config=self.config.price_feature_config,
        )

        training = _drop_unusable_training_rows(
            training,
            target_mode=self.config.target_mode,
            residual_baseline_column=self.config.residual_baseline_column,
        )
        _require_residual_baseline_if_needed(
            future_features,
            target_mode=self.config.target_mode,
            residual_baseline_column=self.config.residual_baseline_column,
        )

        feature_columns = tabular_feature_columns_for_set(
            training,
            self.config.feature_set,
            price_feature_config=self.config.price_feature_config,
        )
        if not feature_columns:
            raise ValueError(f"No usable CatBoost feature columns for {self.config.feature_set!r}")
        missing_future_features = sorted(set(feature_columns) - set(future_features.columns))
        if missing_future_features:
            raise ValueError(f"Future CatBoost feature frame is missing columns: {missing_future_features}")

        CatBoostRegressor, Pool = load_catboost()
        cat_features = [column for column in ["area"] if column in feature_columns]
        params = _production_params(self.config.params)
        model = CatBoostRegressor(**params)
        train_pool = Pool(
            training[feature_columns],
            label=_target_values(
                training,
                target_mode=self.config.target_mode,
                residual_baseline_column=self.config.residual_baseline_column,
            ),
            weight=_recency_sample_weights(
                training,
                reference_origin=origin,
                half_life_days=self.config.recency_half_life_days,
                floor=self.config.recency_floor,
            ),
            cat_features=cat_features,
        )
        model.fit(train_pool)
        prediction = model.predict(
            Pool(
                future_features[feature_columns],
                cat_features=cat_features,
            )
        )
        prediction = pd.Series(prediction, index=future_features.index, dtype="float64")
        if self.config.target_mode == "residual_baseline":
            prediction = prediction + future_features[self.config.residual_baseline_column].astype(float)

        output = future_features[
            ["unique_id", "ds_utc", "forecast_origin_utc", "horizon"]
        ].copy()
        output["model_name"] = self.model_name
        output["model_version"] = self.model_version
        output["y_pred"] = prediction.to_numpy()
        return output.reset_index(drop=True)


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    require_columns(history, ["unique_id", "area", "ds_utc", "y"], "history")
    return normalize_utc_column(history, "ds_utc").sort_values(["unique_id", "ds_utc"]).reset_index(drop=True)


def _prepare_future(future: pd.DataFrame) -> pd.DataFrame:
    require_columns(future, ["unique_id", "area", "ds_utc", "forecast_origin_utc"], "future")
    output = normalize_utc_column(future, "ds_utc")
    output = normalize_utc_column(output, "forecast_origin_utc")
    return output.sort_values(["forecast_origin_utc", "unique_id", "ds_utc"]).reset_index(drop=True)


def _resolve_history(
    history: pd.DataFrame | None,
    fitted_history: pd.DataFrame | None,
) -> pd.DataFrame:
    if history is not None:
        return _prepare_history(history)
    if fitted_history is not None:
        return fitted_history
    raise ValueError("history must be supplied either through fit() or predict()")


def _single_forecast_origin(future: pd.DataFrame) -> pd.Timestamp:
    origins = future["forecast_origin_utc"].drop_duplicates().tolist()
    if len(origins) != 1:
            raise ValueError("ExperimentalCatBoostDayAhead expects exactly one forecast_origin_utc")
    return to_utc_timestamp(origins[0])


def _drop_unusable_training_rows(
    frame: pd.DataFrame,
    *,
    target_mode: TargetMode,
    residual_baseline_column: str,
) -> pd.DataFrame:
    columns = ["y"]
    if target_mode == "residual_baseline":
        columns.append(residual_baseline_column)
    output = frame.dropna(subset=columns).copy()
    if output.empty:
        raise ValueError("No usable CatBoost training rows after dropping missing target/baseline values")
    return output


def _require_residual_baseline_if_needed(
    frame: pd.DataFrame,
    *,
    target_mode: TargetMode,
    residual_baseline_column: str,
) -> None:
    if target_mode != "residual_baseline":
        return
    require_columns(frame, [residual_baseline_column], "future features")
    missing = int(frame[residual_baseline_column].isna().sum())
    if missing:
        raise ValueError(
            f"Residual CatBoost requires complete {residual_baseline_column}; "
            f"missing_rows={missing}"
        )


def _target_values(
    frame: pd.DataFrame,
    *,
    target_mode: TargetMode,
    residual_baseline_column: str,
) -> pd.Series:
    if target_mode == "direct":
        return frame["y"]
    return frame["y"] - frame[residual_baseline_column]


def _production_params(params: dict[str, Any]) -> dict[str, Any]:
    output = dict(params)
    output.setdefault("loss_function", "Quantile:alpha=0.5")
    output.setdefault("eval_metric", "MAE")
    output.setdefault("random_seed", 42)
    output.setdefault("verbose", False)
    output.setdefault("allow_writing_files", False)
    return output


def _recency_sample_weights(
    frame: pd.DataFrame,
    *,
    reference_origin: pd.Timestamp,
    half_life_days: float | None,
    floor: float | None = None,
) -> pd.Series | None:
    if half_life_days is None:
        return None
    timestamps = pd.to_datetime(frame["forecast_origin_utc"], utc=True)
    age_days = (reference_origin - timestamps) / pd.Timedelta(days=1)
    weights = 0.5 ** (age_days.clip(lower=0) / float(half_life_days))
    if floor is not None:
        weights = float(floor) + (1 - float(floor)) * weights
    return weights.astype(float)


def ensure_catboost_available() -> None:
    load_catboost()


def load_catboost() -> tuple[Any, Any]:
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise ImportError(
            "Production CatBoost requires CatBoost. Install it with "
            '`pip install -e ".[catboost]"`, `pip install -e ".[tuning]"`, '
            "or `pip install catboost>=1.2`."
        ) from exc
    return CatBoostRegressor, Pool
