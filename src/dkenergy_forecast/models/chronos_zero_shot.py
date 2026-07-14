from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dkenergy_forecast.types import (
    ensure_price_availability,
    filter_price_history_available_before,
    normalize_utc_column,
    require_columns,
)


@dataclass(frozen=True)
class ChronosZeroShotConfig:
    model_id: str = "amazon/chronos-2"
    context_length: int = 24 * 365
    quantile_levels: tuple[float, float, float] = (0.10, 0.50, 0.90)
    device_map: str = "cpu"
    torch_dtype: str | None = None


DEFAULT_CHRONOS_ZERO_SHOT_CONFIG = ChronosZeroShotConfig()


@dataclass
class ChronosZeroShotDayAhead:
    config: ChronosZeroShotConfig = DEFAULT_CHRONOS_ZERO_SHOT_CONFIG
    pipeline: Any | None = None
    model_name: str = "chronos_zero_shot_day_ahead"
    model_version: str = "v1"

    def __post_init__(self) -> None:
        if self.config.context_length <= 0:
            raise ValueError("context_length must be positive")
        if tuple(self.config.quantile_levels) != (0.10, 0.50, 0.90):
            raise ValueError("ChronosZeroShotDayAhead currently expects q10/q50/q90 quantiles")
        self._history: pd.DataFrame | None = None

    def fit(self, history: pd.DataFrame) -> "ChronosZeroShotDayAhead":
        self._history = _prepare_history(history)
        return self

    def predict(
        self,
        future: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        history_frame = _resolve_history(history, self._history)
        future_frame = _prepare_future(future)
        grouped_future = [
            frame.sort_values("ds_utc").reset_index(drop=True)
            for _, frame in future_frame.groupby("unique_id", sort=False)
        ]
        contexts = [
            _history_context(
                filter_price_history_available_before(
                    history_frame,
                    frame["forecast_origin_utc"].iloc[0],
                ),
                unique_id=str(frame["unique_id"].iloc[0]),
                context_length=self.config.context_length,
            )
            for frame in grouped_future
        ]
        prediction_length = max(len(frame) for frame in grouped_future)
        quantiles = _predict_quantiles(
            self._pipeline(),
            contexts=contexts,
            prediction_length=prediction_length,
            quantile_levels=self.config.quantile_levels,
        )

        frames: list[pd.DataFrame] = []
        for series_index, frame in enumerate(grouped_future):
            horizon_count = len(frame)
            q_values = quantiles[series_index, :horizon_count, :]
            output = frame[["unique_id", "ds_utc", "forecast_origin_utc", "horizon"]].copy()
            output["model_name"] = self.model_name
            output["model_version"] = self.model_version
            output["q10"] = q_values[:, 0]
            output["q50"] = q_values[:, 1]
            output["q90"] = q_values[:, 2]
            output["y_pred"] = output["q50"]
            frames.append(output)

        return pd.concat(frames, ignore_index=True).sort_values(["unique_id", "ds_utc"]).reset_index(drop=True)

    def _pipeline(self) -> Any:
        if self.pipeline is None:
            self.pipeline = load_chronos_pipeline(self.config)
        return self.pipeline


def ensure_chronos_zero_shot_available() -> None:
    _chronos_pipeline_class()


def load_chronos_pipeline(config: ChronosZeroShotConfig) -> Any:
    pipeline_cls = _chronos_pipeline_class()
    kwargs: dict[str, Any] = {"device_map": config.device_map}
    if config.torch_dtype:
        kwargs["torch_dtype"] = config.torch_dtype
    return pipeline_cls.from_pretrained(config.model_id, **kwargs)


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    require_columns(history, ["unique_id", "ds_utc", "y"], "history")
    return (
        ensure_price_availability(normalize_utc_column(history, "ds_utc"))
        .sort_values(["unique_id", "ds_utc"])
        .reset_index(drop=True)
    )


def _prepare_future(future: pd.DataFrame) -> pd.DataFrame:
    require_columns(future, ["unique_id", "ds_utc", "forecast_origin_utc", "horizon"], "future")
    output = normalize_utc_column(future, "ds_utc")
    output = normalize_utc_column(output, "forecast_origin_utc")
    return output.sort_values(["unique_id", "ds_utc"]).reset_index(drop=True)


def _resolve_history(
    history: pd.DataFrame | None,
    fitted_history: pd.DataFrame | None,
) -> pd.DataFrame:
    if history is not None:
        return _prepare_history(history)
    if fitted_history is not None:
        return fitted_history
    raise ValueError("history must be supplied either through fit() or predict()")


def _history_context(
    history: pd.DataFrame,
    *,
    unique_id: str,
    context_length: int,
) -> list[float]:
    values = (
        history.loc[history["unique_id"].astype(str).eq(unique_id)]
        .sort_values("ds_utc")["y"]
        .dropna()
        .tail(context_length)
        .astype(float)
        .tolist()
    )
    if not values:
        raise ValueError(f"No Chronos history values available for {unique_id!r}")
    return values


def _predict_quantiles(
    pipeline: Any,
    *,
    contexts: list[list[float]],
    prediction_length: int,
    quantile_levels: tuple[float, float, float],
) -> np.ndarray:
    levels = list(quantile_levels)
    if hasattr(pipeline, "predict_quantiles"):
        raw = pipeline.predict_quantiles(
            context=contexts,
            prediction_length=prediction_length,
            quantile_levels=levels,
        )
        return _coerce_quantile_array(raw, series_count=len(contexts), prediction_length=prediction_length)

    if not hasattr(pipeline, "predict"):
        raise TypeError("Chronos pipeline must provide predict_quantiles(...) or predict(...)")
    samples = pipeline.predict(
        contexts,
        prediction_length=prediction_length,
    )
    sample_array = np.asarray(samples, dtype=float)
    if sample_array.ndim != 3:
        raise ValueError(
            "Chronos predict(...) output must have shape "
            "(series, samples, prediction_length)"
        )
    return np.quantile(sample_array, levels, axis=1).transpose(1, 2, 0)


def _coerce_quantile_array(
    raw: Any,
    *,
    series_count: int,
    prediction_length: int,
) -> np.ndarray:
    if isinstance(raw, tuple):
        candidates = [np.asarray(item, dtype=float) for item in raw]
        arrays = [item for item in candidates if item.ndim == 3]
        if not arrays:
            raise ValueError("Chronos predict_quantiles(...) tuple did not contain a 3D quantile array")
        array = arrays[0]
    elif isinstance(raw, dict):
        array = np.stack([raw["q10"], raw["q50"], raw["q90"]], axis=-1).astype(float)
    else:
        array = np.asarray(raw, dtype=float)

    if array.shape == (series_count, prediction_length, 3):
        return array
    if array.shape == (series_count, 3, prediction_length):
        return array.transpose(0, 2, 1)
    raise ValueError(
        "Chronos quantile output must have shape "
        f"({series_count}, {prediction_length}, 3) or ({series_count}, 3, {prediction_length}); "
        f"got {array.shape}"
    )


def _import_chronos() -> Any:
    try:
        import chronos
    except ImportError as exc:
        raise ImportError(
            "Chronos zero-shot comparison models require the optional Chronos dependency. "
            'Install it with `pip install -e ".[chronos]"` or '
            "`pip install 'chronos-forecasting>=2.2' 'peft>=0.18.1'`."
        ) from exc
    return chronos


def _chronos_pipeline_class() -> Any:
    chronos = _import_chronos()
    pipeline_cls = (
        getattr(chronos, "BaseChronosPipeline", None)
        or getattr(chronos, "ChronosPipeline", None)
    )
    if pipeline_cls is None:
        raise ImportError(
            "The installed chronos package does not expose BaseChronosPipeline "
            "or ChronosPipeline."
        )
    return pipeline_cls
