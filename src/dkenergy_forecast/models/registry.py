from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from dkenergy_forecast.models.baselines import LagNaive, SeasonalRollingMedian
from dkenergy_forecast.models.catboost_quantile import CatBoostQuantileModel
from dkenergy_forecast.types import ForecastModel


@dataclass(frozen=True)
class ProductionModelSpec:
    label: str
    family: str
    default_enabled: bool
    supports_latest_publish: bool
    factory: Callable[[], ForecastModel] | None
    description: str


def production_model_specs() -> dict[str, ProductionModelSpec]:
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
            default_enabled=True,
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
        "catboost_quantile": ProductionModelSpec(
            label="catboost_quantile",
            family="catboost_price",
            default_enabled=False,
            supports_latest_publish=True,
            factory=lambda: CatBoostQuantileModel(),
            description="Optional EDS-only CatBoost q10/q50/q90 model.",
        ),
        "weather_catboost_gfs_global": _weather_catboost_spec(
            "weather_catboost_gfs_global",
            "GFS Global weather CatBoost feature-set model.",
        ),
        "weather_catboost_icon_eu": _weather_catboost_spec(
            "weather_catboost_icon_eu",
            "ICON-EU weather CatBoost feature-set model.",
        ),
        "weather_catboost_metno_nordic": _weather_catboost_spec(
            "weather_catboost_metno_nordic",
            "MET Norway Nordic weather CatBoost feature-set model.",
        ),
        "weather_catboost_all_weather": _weather_catboost_spec(
            "weather_catboost_all_weather",
            "All raw Open-Meteo weather feature CatBoost model.",
        ),
        "weather_catboost_ensemble": _weather_catboost_spec(
            "weather_catboost_ensemble",
            "Cross-provider weather ensemble summary CatBoost model.",
        ),
    }


def baseline_model_factories() -> dict[str, Callable[[], ForecastModel]]:
    return {
        label: spec.factory
        for label, spec in production_model_specs().items()
        if spec.family == "baseline" and spec.factory is not None
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

    return {label: specs[label].factory for label in selected if specs[label].factory is not None}


def _weather_catboost_spec(label: str, description: str) -> ProductionModelSpec:
    return ProductionModelSpec(
        label=label,
        family="weather_catboost",
        default_enabled=False,
        supports_latest_publish=False,
        factory=None,
        description=description,
    )
