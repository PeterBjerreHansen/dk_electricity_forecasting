from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from dkenergy_forecast.backtesting.horizons import (
    make_daily_origins,
    make_danish_delivery_day_horizon,
    make_local_daily_origins,
)
from dkenergy_forecast.backtesting.origins import choose_recent_complete_daily_origins
from dkenergy_forecast.backtesting.rolling_origin import rolling_origin_backtest
from dkenergy_forecast.evaluation.point_metrics import bias, mae, rmse
from dkenergy_forecast.features.price_features import (
    build_price_feature_frame,
    build_training_matrix,
)
from dkenergy_forecast.models.catboost_experimental import (
    CATBOOST_EXPERIMENT_CONFIG,
    CatBoostExperimentConfig,
    FittedCatBoost,
    fit_catboost_regressor,
    predict_catboost_regressor,
)
from dkenergy_forecast.models.registry import (
    WEIGHTED_MEDIAN_MODEL_LABEL,
    baseline_model_specs,
)
from dkenergy_forecast.types import (
    PRICE_AVAILABILITY_COLUMN,
    ForecastModel,
    copenhagen_timestamp,
    to_utc_timestamp,
)


FORECAST_LOCAL_TIME = "10:00"
CANDIDATE_MODEL_LABEL = "catboost_price_v1"

# The research CatBoost recipe, with training origins at the same 10:00 local
# wall-clock time as the daily forecast so train and serve see the same
# information set.
DATABRICKS_CATBOOST_CONFIG = replace(
    CATBOOST_EXPERIMENT_CONFIG,
    forecast_local_time=FORECAST_LOCAL_TIME,
)


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str
    candidate_mae: float
    reference_mae: float


def forecast_origin_for(run_date_local: object, forecast_local_time: str = FORECAST_LOCAL_TIME) -> pd.Timestamp:
    """The UTC forecast origin for a Copenhagen run date."""

    return to_utc_timestamp(copenhagen_timestamp(run_date_local, forecast_local_time))


def choose_backtest_origins(
    panel: pd.DataFrame,
    *,
    days: int = 28,
    max_origins: int = 14,
    forecast_local_time: str = FORECAST_LOCAL_TIME,
) -> pd.DataFrame:
    origins = choose_recent_complete_daily_origins(
        panel,
        days=days,
        max_origins=max_origins,
        forecast_local_time=forecast_local_time,
    )
    if origins.empty:
        raise ValueError("No complete recent origins available for the backtest")
    return origins


def build_training_pool(
    panel: pd.DataFrame,
    first_origin: pd.Timestamp,
    last_origin: pd.Timestamp,
    config: CatBoostExperimentConfig = DATABRICKS_CATBOOST_CONFIG,
) -> pd.DataFrame:
    """Training examples for every origin from ``first_origin`` to ``last_origin``.

    Building features costs about a second per training origin, and a year of
    origins per model fit made the weekly backtest take hours. Each row's
    features depend only on its own origin, so the rows can be built once and
    sliced per origin with ``training_rows_for_origin``.
    """

    span_days = int(np.ceil((last_origin - first_origin) / pd.Timedelta(days=1)))
    return build_training_matrix(
        panel,
        last_origin,
        training_origin_days=config.training_origin_days + span_days,
        at_hour_utc=config.at_hour_utc,
        forecast_local_time=config.forecast_local_time,
        config=config.price_feature_config,
    )


def training_rows_for_origin(
    pool: pd.DataFrame,
    panel: pd.DataFrame,
    origin: pd.Timestamp,
    config: CatBoostExperimentConfig = DATABRICKS_CATBOOST_CONFIG,
) -> pd.DataFrame:
    """The rows ``build_training_matrix(panel, origin)`` would build, taken from the pool."""

    start = origin - pd.Timedelta(days=config.training_origin_days)
    if config.at_hour_utc is None:
        allowed = make_local_daily_origins(
            panel, start=start, end=origin, forecast_local_time=config.forecast_local_time
        )
    else:
        allowed = make_daily_origins(panel, start=start, end=origin, at_hour_utc=config.at_hour_utc)
    rows = pool[pool["forecast_origin_utc"].isin(allowed["forecast_origin_utc"])]
    horizon_known = rows.groupby("forecast_origin_utc")[PRICE_AVAILABILITY_COLUMN].transform("max") < origin
    rows = rows[horizon_known].reset_index(drop=True)
    if rows.empty:
        raise ValueError(f"No training feature rows available before {origin.isoformat()}")
    return rows


def train_candidate(
    panel: pd.DataFrame,
    origin: pd.Timestamp,
    config: CatBoostExperimentConfig = DATABRICKS_CATBOOST_CONFIG,
    *,
    pool: pd.DataFrame | None = None,
) -> tuple[FittedCatBoost, pd.DataFrame]:
    """Fit the CatBoost candidate on every complete day available before ``origin``."""

    if pool is None:
        pool = build_training_pool(panel, origin, origin, config)
    training = training_rows_for_origin(pool, panel, origin, config)
    return fit_catboost_regressor(training, origin=origin, config=config), training


def backtest_candidate(
    panel: pd.DataFrame,
    origins: pd.DataFrame,
    pool: pd.DataFrame,
    config: CatBoostExperimentConfig = DATABRICKS_CATBOOST_CONFIG,
) -> pd.DataFrame:
    """Rolling-origin backtest of the candidate, refitting at every origin.

    Equivalent to ``rolling_origin_backtest`` with ``ExperimentalCatBoostDayAhead``
    but reuses the shared training pool.
    """

    def factory() -> ForecastModel:
        return _PooledCatBoost(pool=pool, panel=panel, config=config)

    predictions = rolling_origin_backtest(factory, panel, origins, make_danish_delivery_day_horizon)
    predictions["model_label"] = CANDIDATE_MODEL_LABEL
    return predictions


def baseline_model_factories() -> dict[str, Callable[[], ForecastModel]]:
    """The repository's three diagnostic baselines, including the fallback."""

    return {
        label: spec.factory
        for label, spec in baseline_model_specs().items()
        if spec.factory is not None
    }


def backtest_models(
    panel: pd.DataFrame,
    factories: Mapping[str, Callable[[], ForecastModel]],
    origins: pd.DataFrame,
) -> pd.DataFrame:
    """Rolling-origin backtest of each model on the same origins."""

    frames = []
    for label, factory in factories.items():
        predictions = rolling_origin_backtest(
            factory,
            panel,
            origins,
            make_danish_delivery_day_horizon,
        )
        predictions["model_label"] = label
        frames.append(predictions)
    return pd.concat(frames, ignore_index=True)


@dataclass
class _PooledCatBoost:
    pool: pd.DataFrame
    panel: pd.DataFrame
    config: CatBoostExperimentConfig
    model_name: str = CANDIDATE_MODEL_LABEL
    model_version: str = "candidate"

    def fit(self, history: pd.DataFrame) -> "_PooledCatBoost":
        return self

    def predict(self, future: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
        if history is None:
            raise ValueError("history is required")
        origin = to_utc_timestamp(future["forecast_origin_utc"].iloc[0])
        fitted, _ = train_candidate(self.panel, origin, self.config, pool=self.pool)
        features = build_price_feature_frame(
            history,
            future,
            forecast_origin_utc=origin,
            include_target=False,
            config=self.config.price_feature_config,
        )
        output = features[["unique_id", "ds_utc", "forecast_origin_utc", "horizon"]].copy()
        output["model_name"] = self.model_name
        output["model_version"] = self.model_version
        output["y_pred"] = predict_catboost_regressor(fitted, features).to_numpy()
        return output.reset_index(drop=True)


def summarize_backtest(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, group in predictions.groupby("model_label", sort=True):
        rows.append(
            {
                "model_label": label,
                "mae": mae(group),
                "rmse": rmse(group),
                "bias": bias(group),
                "rows": int(group["y"].notna().sum()),
                "origins": int(group["forecast_origin_utc"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def promotion_decision(
    summary: pd.DataFrame,
    *,
    candidate_label: str = CANDIDATE_MODEL_LABEL,
    reference_label: str = WEIGHTED_MEDIAN_MODEL_LABEL,
    min_relative_improvement: float = 0.0,
) -> PromotionDecision:
    """Promote only if the candidate beats the fallback on the same origins.

    Comparing against the fallback (rather than the previous champion's old
    backtest) keeps the gate a paired comparison on identical rows.
    """

    by_label = summary.set_index("model_label")["mae"]
    candidate = float(by_label[candidate_label])
    reference = float(by_label[reference_label])
    threshold = reference * (1 - min_relative_improvement)
    if candidate < threshold:
        return PromotionDecision(
            True,
            f"{candidate_label} MAE {candidate:.1f} beats {reference_label} {reference:.1f}",
            candidate,
            reference,
        )
    margin = f" by {min_relative_improvement:.0%}" if min_relative_improvement else ""
    return PromotionDecision(
        False,
        f"{candidate_label} MAE {candidate:.1f} does not beat {reference_label} {reference:.1f}{margin}",
        candidate,
        reference,
    )
