"""Stable forecasting models used by the live path.

Experimental CatBoost and zero-shot Chronos implementations remain available
from their explicit modules and through ``comparison_registry``.
"""

from dkenergy_forecast.models.baselines import (
    LagNaive,
    SeasonalRollingMedian,
    WeekdayWeekendWeightedMedian,
    WeightedSeasonalMedian,
)
from dkenergy_forecast.models.chronos_production import (
    Chronos2LoRAWeatherConfig,
    Chronos2LoRAWeatherDayAhead,
)
from dkenergy_forecast.models.registry import (
    ProductionModelSpec,
    baseline_model_factories,
    default_production_model_labels,
    latest_publish_model_factories,
    production_model_specs,
)

__all__ = [
    "Chronos2LoRAWeatherConfig",
    "Chronos2LoRAWeatherDayAhead",
    "LagNaive",
    "ProductionModelSpec",
    "SeasonalRollingMedian",
    "WeekdayWeekendWeightedMedian",
    "WeightedSeasonalMedian",
    "baseline_model_factories",
    "default_production_model_labels",
    "latest_publish_model_factories",
    "production_model_specs",
]
