from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path

from dkenergy_forecast.models.baselines import (
    LagNaive,
    WeekdayWeekendWeightedMedian,
)
from dkenergy_forecast.models.chronos_production import (
    PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG,
    Chronos2LoRAWeatherDayAhead,
    ensure_chronos_available,
)
from dkenergy_forecast.types import ForecastModel


@dataclass(frozen=True)
class ProductionModelSpec:
    label: str
    family: str
    default_enabled: bool
    supports_latest_publish: bool
    factory: Callable[[], ForecastModel] | None
    description: str
    required_extra: str | None = None
    emits_quantiles: bool = False
    requires_weather: bool = False
    dependency_check: Callable[[], None] | None = None


def production_model_specs() -> dict[str, ProductionModelSpec]:
    return {
        **baseline_production_model_specs(),
        **chronos_production_model_specs(),
    }


def baseline_production_model_specs() -> dict[str, ProductionModelSpec]:
    return {
        "same_hour_last_week": ProductionModelSpec(
            label="same_hour_last_week",
            family="baseline",
            default_enabled=True,
            supports_latest_publish=True,
            factory=lambda: LagNaive(lag_hours=168),
            description="Lag-naive baseline using the same UTC hour from one week earlier.",
        ),
        "median_weekday_exp_hl4_floor10_42d__median_weekend_exp_hl28_floor20_56d": ProductionModelSpec(
            label="median_weekday_exp_hl4_floor10_42d__median_weekend_exp_hl28_floor20_56d",
            family="baseline",
            default_enabled=True,
            supports_latest_publish=True,
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
            description=(
                "Weekday/weekend weighted median baseline: weekday "
                "42d half-life 4d floor 10%, weekend 56d half-life 28d floor 20%."
            ),
        ),
    }


def chronos_production_model_specs() -> dict[str, ProductionModelSpec]:
    return {
        "chronos2_lora_calendar_weather_ctx1024_v1": ProductionModelSpec(
            label="chronos2_lora_calendar_weather_ctx1024_v1",
            family="chronos",
            default_enabled=True,
            supports_latest_publish=True,
            factory=lambda: Chronos2LoRAWeatherDayAhead(PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG),
            description=(
                "Chronos-2 LoRA day-ahead probabilistic model with calendar and "
                "Open-Meteo weather covariates."
            ),
            required_extra="chronos",
            emits_quantiles=True,
            requires_weather=True,
            dependency_check=ensure_chronos_available,
        ),
    }


def baseline_model_factories(*, include_optional: bool = False) -> dict[str, Callable[[], ForecastModel]]:
    factories = {
        label: spec.factory
        for label, spec in production_model_specs().items()
        if spec.family == "baseline"
        and spec.factory is not None
        and (include_optional or spec.default_enabled)
    }
    if include_optional:
        from dkenergy_forecast.models.comparison_registry import comparison_baseline_model_factories

        factories.update(comparison_baseline_model_factories())
    return factories


def default_production_model_labels() -> list[str]:
    return [
        label
        for label, spec in production_model_specs().items()
        if spec.default_enabled
    ]


def latest_publish_model_factories(
    labels: list[str] | None = None,
    *,
    weather_features_long_path: str | Path | None = None,
    chronos_model_artifact_path: str | Path | None = None,
) -> dict[str, Callable[[], ForecastModel]]:
    specs = production_model_specs()
    selected = labels or default_production_model_labels()
    missing = sorted(set(selected) - set(specs))
    if missing:
        raise ValueError(
            "Unknown production model label(s): "
            f"{missing}; available={sorted(specs)}"
        )

    unsupported = sorted(
        label
        for label in selected
        if not specs[label].supports_latest_publish or specs[label].factory is None
    )
    if unsupported:
        raise ValueError(
            "The selected model label(s) are registered but not yet wired into "
            "latest-forecast publishing: "
            f"{unsupported}. Run their backtest scripts or add a publish adapter first."
        )

    for label in selected:
        dependency_check = specs[label].dependency_check
        if dependency_check is not None:
            dependency_check()

    factories: dict[str, Callable[[], ForecastModel]] = {}
    for label in selected:
        spec = specs[label]
        if spec.factory is None:
            continue
        if label == "chronos2_lora_calendar_weather_ctx1024_v1":
            config = PRODUCTION_CHRONOS_LORA_WEATHER_CONFIG
            if weather_features_long_path is not None:
                config = replace(config, weather_features_long_path=weather_features_long_path)
            if chronos_model_artifact_path is not None:
                config = replace(config, model_artifact_path=chronos_model_artifact_path)
            factories[label] = lambda config=config: Chronos2LoRAWeatherDayAhead(config)
        else:
            factories[label] = spec.factory
    return factories
