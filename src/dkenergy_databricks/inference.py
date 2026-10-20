from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from dkenergy_forecast.backtesting.horizons import make_danish_delivery_day_horizon
from dkenergy_forecast.features.price_features import build_price_feature_frame
from dkenergy_forecast.models.registry import (
    WEIGHTED_MEDIAN_MODEL_LABEL,
    baseline_model_specs,
)
from dkenergy_forecast.types import (
    TARGET_LEAKAGE_COLUMNS,
    filter_price_history_available_before,
    normalize_utc_column,
)
from dkenergy_databricks.training import DATABRICKS_CATBOOST_CONFIG


FORECAST_COLUMNS = [
    "run_id",
    "forecast_origin_utc",
    "unique_id",
    "area",
    "ds_utc",
    "horizon",
    "model_name",
    "model_version",
    "y_pred",
    "is_published",
    "created_at_utc",
]


@dataclass(frozen=True)
class ForecastInputs:
    history: pd.DataFrame
    future: pd.DataFrame
    features: pd.DataFrame


@dataclass(frozen=True)
class ChampionPrediction:
    model_name: str
    model_version: str
    y_pred: np.ndarray | None
    error: str | None = None


def prepare_forecast_inputs(panel: pd.DataFrame, origin: pd.Timestamp) -> ForecastInputs:
    """History known at ``origin``, tomorrow's horizon, and its model features."""

    history = filter_price_history_available_before(panel, origin)
    future = make_danish_delivery_day_horizon(panel, origin)
    future["information_cutoff_utc"] = origin
    future = future.drop(columns=[c for c in TARGET_LEAKAGE_COLUMNS if c in future.columns])
    features = build_price_feature_frame(
        history,
        future,
        forecast_origin_utc=origin,
        include_target=False,
        config=DATABRICKS_CATBOOST_CONFIG.price_feature_config,
    )
    return ForecastInputs(history=history, future=future, features=features)


def predict_champion(
    load_model: Callable[[], object],
    features: pd.DataFrame,
    *,
    model_name: str,
    model_version: str,
) -> ChampionPrediction:
    """Run the registered champion, reporting rather than raising on failure."""

    try:
        model = load_model()
        output = model.predict(features)
        values = np.asarray(output["y_pred"] if isinstance(output, pd.DataFrame) else output, dtype=float)
    except Exception as exc:  # noqa: BLE001 - any model failure means fallback
        return ChampionPrediction(model_name, model_version, None, error=f"{type(exc).__name__}: {exc}")
    if len(values) != len(features) or not np.isfinite(values).all():
        return ChampionPrediction(
            model_name,
            model_version,
            None,
            error="champion returned missing or non-finite predictions",
        )
    return ChampionPrediction(model_name, model_version, values)


def baseline_predictions(inputs: ForecastInputs) -> pd.DataFrame:
    frames = []
    for label, spec in baseline_model_specs().items():
        if spec.factory is None:
            continue
        model = spec.factory().fit(inputs.history)
        predictions = model.predict(inputs.future, history=inputs.history)
        predictions["model_name"] = label
        predictions["model_version"] = "fixed"
        frames.append(predictions)
    return pd.concat(frames, ignore_index=True)


def assemble_forecasts(
    inputs: ForecastInputs,
    champion: ChampionPrediction | None,
    baselines: pd.DataFrame,
    *,
    run_id: str,
    created_at_utc: datetime | None = None,
) -> pd.DataFrame:
    """All model rows for one origin; exactly one model is marked published.

    The champion is published when it produced a full, finite forecast;
    otherwise the fixed weighted median is, as in the AWS pipeline.
    """

    keys = ["unique_id", "ds_utc", "forecast_origin_utc", "horizon"]
    area_lookup = inputs.future[["unique_id", "ds_utc", "area"]]
    frames = []
    if champion is not None and champion.y_pred is not None:
        champion_rows = inputs.features[keys].copy()
        champion_rows["model_name"] = champion.model_name
        champion_rows["model_version"] = champion.model_version
        champion_rows["y_pred"] = champion.y_pred
        frames.append(champion_rows)
        published_model = champion.model_name
    else:
        published_model = WEIGHTED_MEDIAN_MODEL_LABEL
    frames.append(baselines[keys + ["model_name", "model_version", "y_pred"]])

    output = pd.concat(frames, ignore_index=True)
    output = normalize_utc_column(output, "ds_utc")
    output = normalize_utc_column(output, "forecast_origin_utc")
    output = output.merge(normalize_utc_column(area_lookup, "ds_utc"), on=["unique_id", "ds_utc"], how="left")
    output["is_published"] = output["model_name"] == published_model
    output["run_id"] = run_id
    output["created_at_utc"] = pd.Timestamp(created_at_utc or datetime.now(timezone.utc)).tz_convert("UTC")
    output["y_pred"] = output["y_pred"].astype(float)
    return output[FORECAST_COLUMNS].sort_values(["model_name", "unique_id", "ds_utc"]).reset_index(drop=True)
