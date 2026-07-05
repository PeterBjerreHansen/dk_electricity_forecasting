from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dkenergy_forecast.models.baselines import (
    LagNaive,
    SeasonalRollingMedian,
    WeekdayWeekendWeightedMedian,
)
from dkenergy_forecast.models.catboost_production import (
    PRODUCTION_CATBOOST_CONFIG,
    ProductionCatBoostDayAhead,
    ensure_catboost_available,
)
from dkenergy_forecast.models.chronos_production import (
    ChronosZeroShotDayAhead,
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
    dependency_check: Callable[[], None] | None = None


def production_model_specs() -> dict[str, ProductionModelSpec]:
    return {
        **baseline_production_model_specs(),
        **catboost_production_model_specs(),
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
        "rolling_median_local_hour_28d": ProductionModelSpec(
            label="rolling_median_local_hour_28d",
            family="baseline",
            default_enabled=False,
            supports_latest_publish=True,
            factory=lambda: SeasonalRollingMedian(
                lookback_days=28,
                seasonal_keys=("local_hour",),
                min_periods=7,
            ),
            description="Local-hour seasonal rolling median over the previous 28 days.",
        ),
        "rolling_median_hour_weekend_56d": ProductionModelSpec(
            label="rolling_median_hour_weekend_56d",
            family="baseline",
            default_enabled=True,
            supports_latest_publish=True,
            factory=lambda: SeasonalRollingMedian(
                lookback_days=56,
                seasonal_keys=("local_hour", "is_weekend"),
                min_periods=4,
            ),
            description="Local-hour/weekend seasonal rolling median over the previous 56 days.",
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


def catboost_production_model_specs() -> dict[str, ProductionModelSpec]:
    return {
        "catboost_price_manual_v1": ProductionModelSpec(
            label="catboost_price_manual_v1",
            family="catboost",
            default_enabled=True,
            supports_latest_publish=True,
            factory=lambda: ProductionCatBoostDayAhead(PRODUCTION_CATBOOST_CONFIG),
            description=(
                "Manually selected fixed-parameter CatBoost day-ahead price model "
                f"({PRODUCTION_CATBOOST_CONFIG.feature_set}, "
                f"{PRODUCTION_CATBOOST_CONFIG.target_mode}, "
                f"{PRODUCTION_CATBOOST_CONFIG.training_origin_days}d training window)."
            ),
            required_extra="catboost",
            emits_quantiles=False,
            dependency_check=ensure_catboost_available,
        ),
    }


def chronos_production_model_specs() -> dict[str, ProductionModelSpec]:
    return {
        "chronos_zero_shot_v1": ProductionModelSpec(
            label="chronos_zero_shot_v1",
            family="chronos",
            default_enabled=False,
            supports_latest_publish=True,
            factory=lambda: ChronosZeroShotDayAhead(),
            description="Chronos zero-shot day-ahead probabilistic forecast.",
            required_extra="chronos",
            emits_quantiles=True,
            dependency_check=ensure_chronos_available,
        ),
    }


def baseline_model_factories(*, include_optional: bool = False) -> dict[str, Callable[[], ForecastModel]]:
    return {
        label: spec.factory
        for label, spec in production_model_specs().items()
        if spec.family == "baseline"
        and spec.factory is not None
        and (include_optional or spec.default_enabled)
    }


def default_production_model_labels() -> list[str]:
    return [
        label
        for label, spec in production_model_specs().items()
        if spec.default_enabled
    ]


def latest_publish_model_factories(
    labels: list[str] | None = None,
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

    return {label: specs[label].factory for label in selected if specs[label].factory is not None}
