from __future__ import annotations

from collections.abc import Callable

from dkenergy_forecast.models.baselines import SeasonalRollingMedian
from dkenergy_forecast.models.catboost_experimental import (
    CATBOOST_EXPERIMENT_CONFIG,
    ExperimentalCatBoostDayAhead,
    ensure_catboost_available,
)
from dkenergy_forecast.models.chronos_zero_shot import (
    ChronosZeroShotDayAhead,
    ensure_chronos_zero_shot_available,
)
from dkenergy_forecast.models.registry import (
    ROLLING_MEDIAN_MODEL_LABEL,
    ProductionModelSpec,
    baseline_model_specs,
)
from dkenergy_forecast.types import ForecastModel


def comparison_model_specs() -> dict[str, ProductionModelSpec]:
    """Notebook/smoke comparison models that are not part of production publishing."""

    return {
        **comparison_baseline_model_specs(),
        **comparison_catboost_model_specs(),
        **comparison_chronos_model_specs(),
    }


def comparison_baseline_model_specs() -> dict[str, ProductionModelSpec]:
    return {
        ROLLING_MEDIAN_MODEL_LABEL: baseline_model_specs()[
            ROLLING_MEDIAN_MODEL_LABEL
        ],
        "rolling_median_hour_weekend_56d": ProductionModelSpec(
            label="rolling_median_hour_weekend_56d",
            family="baseline",
            factory=lambda: SeasonalRollingMedian(
                lookback_days=56,
                seasonal_keys=("local_hour", "is_weekend"),
                min_periods=4,
            ),
            description="Comparison local-hour/weekend seasonal rolling median over the previous 56 days.",
        ),
    }


def comparison_catboost_model_specs() -> dict[str, ProductionModelSpec]:
    return {
        "catboost_price_manual_v1": ProductionModelSpec(
            label="catboost_price_manual_v1",
            family="catboost",
            factory=lambda: ExperimentalCatBoostDayAhead(CATBOOST_EXPERIMENT_CONFIG),
            description=(
                "Manually selected fixed-parameter CatBoost day-ahead price model "
                f"({CATBOOST_EXPERIMENT_CONFIG.feature_set}, "
                f"{CATBOOST_EXPERIMENT_CONFIG.target_mode}, "
                f"{CATBOOST_EXPERIMENT_CONFIG.training_origin_days}d training window)."
            ),
            required_extra="catboost",
            emits_quantiles=False,
            dependency_check=ensure_catboost_available,
        ),
    }


def comparison_chronos_model_specs() -> dict[str, ProductionModelSpec]:
    return {
        "chronos_zero_shot_v1": ProductionModelSpec(
            label="chronos_zero_shot_v1",
            family="chronos",
            factory=lambda: ChronosZeroShotDayAhead(),
            description="Chronos zero-shot day-ahead probabilistic forecast.",
            required_extra="chronos",
            emits_quantiles=True,
            dependency_check=ensure_chronos_zero_shot_available,
        ),
    }


def comparison_model_factories(labels: list[str] | None = None) -> dict[str, Callable[[], ForecastModel]]:
    specs = comparison_model_specs()
    selected = labels or list(specs)
    missing = sorted(set(selected) - set(specs))
    if missing:
        raise ValueError(f"Unknown comparison model label(s): {missing}; available={sorted(specs)}")

    for label in selected:
        dependency_check = specs[label].dependency_check
        if dependency_check is not None:
            dependency_check()

    return {
        label: specs[label].factory
        for label in selected
        if specs[label].factory is not None
    }


def comparison_baseline_model_factories() -> dict[str, Callable[[], ForecastModel]]:
    return {
        label: spec.factory
        for label, spec in comparison_baseline_model_specs().items()
        if spec.factory is not None
    }
