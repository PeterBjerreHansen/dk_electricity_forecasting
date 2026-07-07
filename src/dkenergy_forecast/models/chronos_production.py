from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from dkenergy_forecast.features.weather_features import (
    add_weather_derived_features,
    add_weather_ensemble_features,
    join_weather_features,
    weather_value_columns,
)
from dkenergy_forecast.types import (
    add_copenhagen_calendar,
    ensure_price_availability,
    filter_price_history_available_before,
    normalize_utc_column,
    require_columns,
    to_utc_timestamp,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHRONOS_LORA_ARTIFACT_PATH = (
    PROJECT_ROOT / "artifacts" / "models" / "chronos2_lora_calendar_weather_ctx1024_v1"
)
DEFAULT_WEATHER_FEATURES_LONG_PATH = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "weather_open_meteo_area_hourly_long_v1.parquet"
)
CHRONOS_LORA_ARTIFACT_SCHEMA_VERSION = 1

CALENDAR_COVARIATES = [
    "local_hour",
    "local_day_of_week",
    "local_month",
    "is_weekend",
    "is_dst",
    "utc_offset_hours",
]


@dataclass(frozen=True)
class ChronosProductionConfig:
    model_id: str = "amazon/chronos-2"
    context_length: int = 24 * 365
    quantile_levels: tuple[float, float, float] = (0.10, 0.50, 0.90)
    device_map: str = "cpu"
    torch_dtype: str | None = None


PRODUCTION_CHRONOS_CONFIG = ChronosProductionConfig()


@dataclass(frozen=True)
class Chronos2LoRAWeatherConfig:
    model_artifact_path: str | Path = DEFAULT_CHRONOS_LORA_ARTIFACT_PATH
    base_model_id: str = "amazon/chronos-2"
    context_length: int = 1024
    prediction_length: int = 36
    quantile_levels: tuple[float, float, float] = (0.10, 0.50, 0.90)
    device_map: str = "cpu"
    torch_dtype: str | None = None
    weather_features_long_path: str | Path = DEFAULT_WEATHER_FEATURES_LONG_PATH
    weather_covariate_mode: Literal["all", "raw", "ensemble", "ensemble_mean"] = "all"
    batch_size: int = 128
    cross_learning: bool = False
    add_weather_ensemble_features: bool = True
    add_weather_derived_features: bool = False


PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG = Chronos2LoRAWeatherConfig()


@dataclass
class Chronos2ProductionFrames:
    context_df: pd.DataFrame
    future_df: pd.DataFrame
    horizon_metadata: pd.DataFrame
    prediction_length: int
    covariates: list[str]


@dataclass
class ChronosZeroShotDayAhead:
    config: ChronosProductionConfig = PRODUCTION_CHRONOS_CONFIG
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


@dataclass
class Chronos2LoRAWeatherDayAhead:
    config: Chronos2LoRAWeatherConfig = PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG
    pipeline: Any | None = None
    model_name: str = "chronos2_lora_calendar_weather_day_ahead"
    model_version: str = "ctx1024_v1"

    def __post_init__(self) -> None:
        if self.config.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.config.prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if tuple(self.config.quantile_levels) != (0.10, 0.50, 0.90):
            raise ValueError("Chronos2LoRAWeatherDayAhead currently expects q10/q50/q90 quantiles")
        self._history: pd.DataFrame | None = None
        self._artifact_manifest: dict[str, Any] | None = None

    def fit(self, history: pd.DataFrame) -> "Chronos2LoRAWeatherDayAhead":
        self._history = _prepare_weather_history(history)
        return self

    def predict(
        self,
        future: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        history_frame = _resolve_weather_history(history, self._history)
        future_frame = _prepare_weather_future(future)
        covariates = self.artifact_covariates()
        weather = load_weather_features(self.config.weather_features_long_path)
        frames = build_lora_weather_prediction_frames(
            history_frame,
            future_frame,
            weather=weather,
            covariates=covariates,
            config=self.config,
        )
        pred_df = self._predict_df(frames)
        scored = frames.horizon_metadata.merge(
            pred_df[["unique_id", "ds_utc", "y_pred", "q10", "q50", "q90"]],
            on=["unique_id", "ds_utc"],
            how="left",
        )
        if scored[["y_pred", "q10", "q50", "q90"]].isna().any().any():
            missing = int(scored["y_pred"].isna().sum())
            raise ValueError(f"Chronos LoRA predictions are missing delivery rows: missing={missing}")
        scored["model_name"] = self.model_name
        scored["model_version"] = self.model_version
        return scored[
            [
                "unique_id",
                "ds_utc",
                "forecast_origin_utc",
                "horizon",
                "model_name",
                "model_version",
                "q10",
                "q50",
                "q90",
                "y_pred",
            ]
        ].sort_values(["unique_id", "ds_utc"]).reset_index(drop=True)

    def artifact_manifest(self) -> dict[str, Any]:
        if self._artifact_manifest is None:
            self._artifact_manifest = load_lora_artifact_manifest(self.config.model_artifact_path)
        return self._artifact_manifest

    def artifact_covariates(self) -> list[str]:
        manifest = self.artifact_manifest()
        covariates = manifest.get("covariates")
        if not isinstance(covariates, list) or not covariates or not all(isinstance(item, str) for item in covariates):
            raise ValueError("Chronos LoRA artifact manifest must contain a non-empty string list `covariates`")
        return list(covariates)

    def _pipeline(self) -> Any:
        if self.pipeline is None:
            self.pipeline = load_chronos_lora_pipeline(self.config)
        return self.pipeline

    def _predict_df(self, frames: Chronos2ProductionFrames) -> pd.DataFrame:
        pipeline = self._pipeline()
        if not hasattr(pipeline, "predict_df"):
            raise TypeError("Chronos-2 LoRA production pipeline must provide predict_df(...)")
        raw = pipeline.predict_df(
            frames.context_df,
            future_df=frames.future_df,
            prediction_length=frames.prediction_length,
            quantile_levels=list(self.config.quantile_levels),
            id_column="item_id",
            timestamp_column="timestamp",
            target="target",
            batch_size=self.config.batch_size,
            context_length=self.config.context_length,
            cross_learning=self.config.cross_learning,
            freq="h",
        )
        output = rename_chronos_prediction_columns(raw).rename(columns={"item_id": "unique_id"})
        require_columns(output, ["unique_id", "timestamp", "y_pred", "q10", "q50", "q90"], "Chronos predictions")
        output["ds_utc"] = from_chronos_timestamp(output["timestamp"])
        return output


def build_lora_weather_prediction_frames(
    history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    weather: pd.DataFrame,
    covariates: list[str],
    config: Chronos2LoRAWeatherConfig,
) -> Chronos2ProductionFrames:
    history_frame = _prepare_weather_history(history)
    future_frame = _prepare_weather_future(future)
    origin = _single_forecast_origin(future_frame)
    history_available = filter_price_history_available_before(history_frame, origin)
    if history_available.empty:
        raise ValueError(f"No Chronos history rows available before {origin.isoformat()}")

    context = history_available.groupby("unique_id", group_keys=False).tail(config.context_length).copy()
    context_lengths = context.groupby("unique_id")["ds_utc"].size()
    if int(context_lengths.min()) < config.context_length:
        raise ValueError(
            "Insufficient Chronos history available before "
            f"{origin.isoformat()}: minimum rows per series is {int(context_lengths.min())}"
        )

    last_by_id = context.groupby("unique_id")["ds_utc"].max()
    if last_by_id.nunique() != 1:
        raise ValueError(f"Chronos context series do not share a final timestamp before {origin.isoformat()}")
    last_context_ts = last_by_id.iloc[0]

    prediction_timestamps = pd.date_range(
        start=last_context_ts + pd.Timedelta(hours=1),
        end=future_frame["ds_utc"].max(),
        freq="h",
        tz="UTC",
    )
    if prediction_timestamps.empty:
        raise ValueError("Chronos prediction timestamp range is empty")

    ids = future_frame[["unique_id", "area"]].drop_duplicates("unique_id")
    full_future = ids.merge(pd.DataFrame({"ds_utc": prediction_timestamps}), how="cross")
    full_future["forecast_origin_utc"] = origin
    full_future = add_copenhagen_calendar(full_future)

    context_full = context.copy()
    context_full["forecast_origin_utc"] = origin
    context_full = add_copenhagen_calendar(context_full)
    context_full = add_weather_covariates(context_full, weather, config=config, expected_covariates=covariates)
    full_future = add_weather_covariates(full_future, weather, config=config, expected_covariates=covariates)

    _require_covariate_columns(context_full, covariates, "Chronos context frame")
    _require_covariate_columns(full_future, covariates, "Chronos future frame")
    _require_weather_signal(context_full, covariates, "Chronos context frame")
    _require_weather_signal(full_future, covariates, "Chronos future frame")

    context_full = normalize_covariate_dtypes(context_full, covariates)
    full_future = normalize_covariate_dtypes(full_future, covariates)
    context_full["item_id"] = context_full["unique_id"]
    context_full["timestamp"] = to_chronos_timestamp(context_full["ds_utc"])
    full_future["item_id"] = full_future["unique_id"]
    full_future["timestamp"] = to_chronos_timestamp(full_future["ds_utc"])

    context_df = context_full[["item_id", "timestamp", "y", *covariates]].rename(columns={"y": "target"})
    future_df = full_future[["item_id", "timestamp", *covariates]]
    horizon_metadata = future_frame[["unique_id", "ds_utc", "forecast_origin_utc", "horizon"]].copy()
    return Chronos2ProductionFrames(
        context_df=context_df.sort_values(["item_id", "timestamp"]).reset_index(drop=True),
        future_df=future_df.sort_values(["item_id", "timestamp"]).reset_index(drop=True),
        horizon_metadata=horizon_metadata.sort_values(["unique_id", "ds_utc"]).reset_index(drop=True),
        prediction_length=len(prediction_timestamps),
        covariates=list(covariates),
    )


def load_lora_artifact_manifest(model_artifact_path: str | Path) -> dict[str, Any]:
    artifact_path = Path(model_artifact_path)
    manifest_path = artifact_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing Chronos LoRA artifact manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    schema_version = manifest.get("artifact_schema_version")
    if schema_version != CHRONOS_LORA_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Chronos LoRA artifact schema version: "
            f"{schema_version!r}; expected {CHRONOS_LORA_ARTIFACT_SCHEMA_VERSION}"
        )
    return manifest


def load_weather_features(path: str | Path) -> pd.DataFrame:
    weather_path = Path(path)
    if not weather_path.exists():
        raise FileNotFoundError(f"Missing Chronos weather feature file: {weather_path}")
    return pd.read_parquet(weather_path)


def weather_artifact_summary(path: str | Path) -> dict[str, Any]:
    weather_path = Path(path)
    if not weather_path.exists():
        return {
            "weather_features_long_path": str(weather_path),
            "weather_features_exists": False,
        }
    weather = load_weather_features(weather_path)
    summary: dict[str, Any] = {
        "weather_features_long_path": str(weather_path),
        "weather_features_exists": True,
        "weather_feature_rows": int(len(weather)),
    }
    if "ds_utc" in weather.columns:
        summary["weather_max_ds_utc"] = pd.to_datetime(weather["ds_utc"], utc=True).max()
    if "forecast_available_at_utc" in weather.columns:
        summary["weather_max_forecast_available_at_utc"] = pd.to_datetime(
            weather["forecast_available_at_utc"],
            utc=True,
        ).max()
    return summary


def add_weather_covariates(
    frame: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    config: Chronos2LoRAWeatherConfig,
    expected_covariates: list[str] | None = None,
) -> pd.DataFrame:
    enriched = join_weather_features(frame, weather)
    if config.add_weather_ensemble_features:
        enriched = add_weather_ensemble_features(enriched)
    if config.add_weather_derived_features:
        enriched = add_weather_derived_features(enriched)
    if expected_covariates is not None:
        enriched = materialize_unavailable_weather_covariates(
            enriched,
            weather,
            expected_covariates,
        )
    return enriched


def materialize_unavailable_weather_covariates(
    frame: pd.DataFrame,
    weather: pd.DataFrame,
    covariates: list[str],
) -> pd.DataFrame:
    output = frame.copy()
    if "feature_name" not in weather.columns:
        return output
    source_features = set(weather["feature_name"].dropna().astype(str))
    for column in covariates:
        if column in output.columns or not column.startswith("weather_"):
            continue
        if column in source_features or _ensemble_covariate_has_source(column, source_features):
            output[column] = np.nan
    return output


def _ensemble_covariate_has_source(column: str, source_features: set[str]) -> bool:
    if not column.startswith("weather_ensemble_"):
        return False
    body = column.removeprefix("weather_ensemble_")
    for suffix in ("_mean", "_min", "_max", "_spread"):
        if not body.endswith(suffix):
            continue
        lead_and_parameter = body[: -len(suffix)]
        lead, _, parameter = lead_and_parameter.partition("_")
        if not lead.startswith("lead") or not parameter:
            return False
        matching_sources = [
            feature
            for feature in source_features
            if feature.startswith("weather_")
            and not feature.startswith("weather_ensemble_")
            and feature.endswith(f"_{lead}_{parameter}")
        ]
        return len(matching_sources) >= 2
    return False


def selected_weather_columns(frame: pd.DataFrame, mode: str) -> list[str]:
    columns = weather_value_columns(frame)
    if mode == "raw":
        return [column for column in columns if not column.startswith("weather_ensemble_")]
    if mode == "ensemble":
        return [column for column in columns if column.startswith("weather_ensemble_")]
    if mode == "ensemble_mean":
        return [
            column
            for column in columns
            if column.startswith("weather_ensemble_") and column.endswith("_mean")
        ]
    if mode == "all":
        return columns
    raise ValueError(f"Unknown weather covariate mode: {mode!r}")


def normalize_covariate_dtypes(frame: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in covariates:
        if pd.api.types.is_bool_dtype(output[column]):
            output[column] = output[column].astype("int8")
    return output


def rename_chronos_prediction_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    output = predictions.copy()
    rename_map = {}
    for level in [0.1, 0.5, 0.9]:
        quantile_name = f"q{int(round(level * 100))}"
        candidates = [str(level), f"{level:.1f}", f"{level:.2f}"]
        for candidate in candidates:
            if candidate in output.columns:
                rename_map[candidate] = quantile_name
                break
    output = output.rename(columns=rename_map)
    if "predictions" in output.columns:
        output["y_pred"] = output["predictions"]
    elif "q50" in output.columns:
        output["y_pred"] = output["q50"]
    else:
        raise ValueError("Chronos output did not include predictions or q50")
    return output


def to_chronos_timestamp(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series, utc=True)
    return values.dt.tz_localize(None)


def from_chronos_timestamp(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series)
    if values.dt.tz is None:
        return values.dt.tz_localize("UTC")
    return values.dt.tz_convert("UTC")


def _prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    require_columns(history, ["unique_id", "ds_utc", "y"], "history")
    return (
        ensure_price_availability(normalize_utc_column(history, "ds_utc"))
        .sort_values(["unique_id", "ds_utc"])
        .reset_index(drop=True)
    )


def _prepare_weather_history(history: pd.DataFrame) -> pd.DataFrame:
    require_columns(history, ["unique_id", "area", "ds_utc", "y"], "history")
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


def _prepare_weather_future(future: pd.DataFrame) -> pd.DataFrame:
    require_columns(future, ["unique_id", "area", "ds_utc", "forecast_origin_utc", "horizon"], "future")
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


def _resolve_weather_history(
    history: pd.DataFrame | None,
    fitted_history: pd.DataFrame | None,
) -> pd.DataFrame:
    if history is not None:
        return _prepare_weather_history(history)
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


def _single_forecast_origin(future: pd.DataFrame) -> pd.Timestamp:
    origins = future["forecast_origin_utc"].drop_duplicates().tolist()
    if len(origins) != 1:
        raise ValueError("Chronos production model expects exactly one forecast_origin_utc")
    return to_utc_timestamp(origins[0])


def _require_covariate_columns(frame: pd.DataFrame, covariates: list[str], frame_name: str) -> None:
    missing = [column for column in covariates if column not in frame.columns]
    if missing:
        raise ValueError(f"{frame_name} is missing Chronos artifact covariates: {missing[:20]}")


def _require_weather_signal(frame: pd.DataFrame, covariates: list[str], frame_name: str) -> None:
    weather_columns = [column for column in covariates if column.startswith("weather_")]
    if not weather_columns:
        return
    has_signal = frame[weather_columns].notna().any(axis=1)
    if bool((~has_signal).any()):
        sample_columns = [column for column in ["unique_id", "area", "ds_utc", "forecast_origin_utc"] if column in frame]
        sample = frame.loc[~has_signal, sample_columns].head(5).to_dict(orient="records")
        raise ValueError(f"{frame_name} has rows with no availability-safe weather signal: {sample}")


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


def ensure_chronos_available() -> None:
    _chronos_pipeline_class()


def load_chronos_pipeline(config: ChronosProductionConfig) -> Any:
    pipeline_cls = _chronos_pipeline_class()
    kwargs: dict[str, Any] = {"device_map": config.device_map}
    if config.torch_dtype:
        kwargs["torch_dtype"] = config.torch_dtype
    return pipeline_cls.from_pretrained(config.model_id, **kwargs)


def load_chronos_lora_pipeline(config: Chronos2LoRAWeatherConfig) -> Any:
    artifact_path = Path(config.model_artifact_path)
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing Chronos LoRA model artifact directory: {artifact_path}")
    pipeline_cls = _chronos2_pipeline_class()
    kwargs: dict[str, Any] = {"device_map": config.device_map}
    if config.torch_dtype:
        kwargs["torch_dtype"] = config.torch_dtype
    return pipeline_cls.from_pretrained(str(artifact_path), **kwargs)


def _import_chronos() -> Any:
    try:
        import chronos
    except ImportError as exc:
        raise ImportError(
            "Chronos production models require the optional Chronos dependency. "
            'Install it with `pip install -e ".[chronos]"` or '
            "`pip install 'chronos-forecasting[extras]>=2.2'`."
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


def _chronos2_pipeline_class() -> Any:
    chronos = _import_chronos()
    pipeline_cls = getattr(chronos, "Chronos2Pipeline", None)
    if pipeline_cls is None:
        raise ImportError(
            "Chronos-2 LoRA production models require a chronos package exposing "
            "Chronos2Pipeline. Install `chronos-forecasting[extras]>=2.2`."
        )
    return pipeline_cls
