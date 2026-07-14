from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from dkenergy_forecast.layout import CHRONOS_LORA_WEATHER_MODEL_LABEL
from dkenergy_forecast.models.baselines import (
    LagNaive,
    SeasonalRollingMedian,
    WeekdayWeekendWeightedMedian,
)
from dkenergy_forecast.models.chronos_production import (
    PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG,
    Chronos2LoRAWeatherDayAhead,
    ensure_chronos_available,
)
from dkenergy_forecast.types import ForecastModel


WEIGHTED_MEDIAN_MODEL_LABEL = "weighted_median_v1"
SAME_HOUR_LAST_WEEK_MODEL_LABEL = "same_hour_last_week"
ROLLING_MEDIAN_MODEL_LABEL = "rolling_median_local_hour_28d"


@dataclass(frozen=True)
class ProductionModelSpec:
    """The small amount of metadata needed to construct a model."""

    label: str
    family: str
    factory: Callable[[], ForecastModel] | None
    description: str
    required_extra: str | None = None
    emits_quantiles: bool = False
    requires_weather: bool = False
    dependency_check: Callable[[], None] | None = None


def production_model_specs() -> dict[str, ProductionModelSpec]:
    """Models allowed on the live path: Chronos and its fixed fallback."""

    return {
        CHRONOS_LORA_WEATHER_MODEL_LABEL: _chronos_spec(),
        WEIGHTED_MEDIAN_MODEL_LABEL: _weighted_median_spec(),
    }


def baseline_model_specs() -> dict[str, ProductionModelSpec]:
    """Compact baselines used for diagnostics and model development."""

    return {
        SAME_HOUR_LAST_WEEK_MODEL_LABEL: ProductionModelSpec(
            label=SAME_HOUR_LAST_WEEK_MODEL_LABEL,
            family="baseline",
            factory=lambda: LagNaive(lag_hours=168),
            description="Same UTC hour one week earlier.",
        ),
        ROLLING_MEDIAN_MODEL_LABEL: ProductionModelSpec(
            label=ROLLING_MEDIAN_MODEL_LABEL,
            family="baseline",
            factory=lambda: SeasonalRollingMedian(
                lookback_days=28,
                seasonal_keys=("local_hour",),
                min_periods=7,
            ),
            description="Local-hour rolling median over the previous 28 days.",
        ),
        WEIGHTED_MEDIAN_MODEL_LABEL: _weighted_median_spec(),
    }


def _weighted_median_spec() -> ProductionModelSpec:
    return ProductionModelSpec(
        label=WEIGHTED_MEDIAN_MODEL_LABEL,
        family="baseline",
        factory=lambda: WeekdayWeekendWeightedMedian(
            weekday_lookback_days=42,
            weekday_half_life_days=4,
            weekday_floor=0.10,
            weekend_lookback_days=56,
            weekend_half_life_days=28,
            weekend_floor=0.20,
            seasonal_keys=("local_hour", "is_weekend"),
            min_periods=4,
        ),
        description="Fixed weekday/weekend recency-weighted median baseline.",
    )


def _chronos_spec() -> ProductionModelSpec:
    return ProductionModelSpec(
        label=CHRONOS_LORA_WEATHER_MODEL_LABEL,
        family="chronos",
        factory=lambda: Chronos2LoRAWeatherDayAhead(PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG),
        description="Chronos-2 LoRA with calendar and point-in-time weather covariates.",
        required_extra="chronos",
        emits_quantiles=True,
        requires_weather=True,
        dependency_check=ensure_chronos_available,
    )


def baseline_model_factories(*, include_optional: bool = False) -> dict[str, Callable[[], ForecastModel]]:
    factories = {
        label: spec.factory
        for label, spec in baseline_model_specs().items()
        if spec.factory is not None
    }
    if include_optional:
        from dkenergy_forecast.models.comparison_registry import comparison_baseline_model_factories

        factories.update(comparison_baseline_model_factories())
    return factories


def default_production_model_labels() -> list[str]:
    """Backward-compatible view of the single primary production model."""

    return [CHRONOS_LORA_WEATHER_MODEL_LABEL]


def latest_publish_model_factories(
    labels: list[str] | None = None,
    *,
    weather_features_long_path: str | Path | None = None,
    chronos_model_artifact_path: str | Path | None = None,
) -> dict[str, Callable[[], ForecastModel]]:
    """Construct explicitly requested live-path models.

    Production orchestration requests the configured primary first and the fixed
    fallback only after a primary failure. Diagnostics may request both.
    """

    specs = production_model_specs()
    selected = labels or default_production_model_labels()
    missing = sorted(set(selected) - set(specs))
    if missing:
        raise ValueError(
            f"Unknown production model label(s): {missing}; available={sorted(specs)}"
        )

    factories: dict[str, Callable[[], ForecastModel]] = {}
    for label in selected:
        spec = specs[label]
        if spec.dependency_check is not None:
            spec.dependency_check()
        if spec.factory is None:
            raise ValueError(f"Production model {label!r} has no factory")
        if label == CHRONOS_LORA_WEATHER_MODEL_LABEL:
            config = PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG
            if weather_features_long_path is not None:
                config = replace(config, weather_features_long_path=weather_features_long_path)
            if chronos_model_artifact_path is not None:
                config = replace(config, model_artifact_path=chronos_model_artifact_path)
            factories[label] = lambda config=config: Chronos2LoRAWeatherDayAhead(config)
        else:
            factories[label] = spec.factory
    return factories
