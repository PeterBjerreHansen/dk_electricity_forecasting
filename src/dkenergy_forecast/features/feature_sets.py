from __future__ import annotations

import pandas as pd

from dkenergy_forecast.features.price_features import PriceFeatureConfig
from dkenergy_forecast.features.weather_features import weather_value_columns


TABULAR_METADATA_COLUMNS = {
    "unique_id",
    "area",
    "ds_utc",
    "ds_local",
    "local_date",
    "forecast_origin_utc",
    "horizon",
    "y",
    "price_available_at_utc",
    "dataset_version",
    "price_dkk_per_mwh",
    "price_eur_per_mwh",
    "source_dataset",
    "source_resolution_minutes",
}

DEFAULT_PRICE_FEATURE_CONFIG = PriceFeatureConfig()
ROBUST_BASELINE_COLUMN = DEFAULT_PRICE_FEATURE_CONFIG.weighted_median_baseline_column
CALENDAR_PRICE_COLUMNS = tuple(
    column for column in DEFAULT_PRICE_FEATURE_CONFIG.base_features if column != "area"
)
PRICE_LAG_COLUMNS = tuple(DEFAULT_PRICE_FEATURE_CONFIG.lag_feature_columns)
PRICE_ROLLING_COLUMNS = tuple(DEFAULT_PRICE_FEATURE_CONFIG.rolling_feature_columns)
PRICE_SEASONAL_COLUMNS = tuple(DEFAULT_PRICE_FEATURE_CONFIG.seasonal_feature_columns)
PRICE_SPREAD_COLUMNS = tuple(DEFAULT_PRICE_FEATURE_CONFIG.spread_feature_columns)
PRICE_FULL_ENGINEERED_COLUMNS = tuple(DEFAULT_PRICE_FEATURE_CONFIG.feature_columns)
PRICE_BASELINE_CALENDAR_COLUMNS = (
    "area",
    *DEFAULT_PRICE_FEATURE_CONFIG.baseline_feature_columns,
    *CALENDAR_PRICE_COLUMNS,
)
PRICE_LAGS_CALENDAR_COLUMNS = (
    "area",
    *CALENDAR_PRICE_COLUMNS,
    *PRICE_LAG_COLUMNS,
    *PRICE_SPREAD_COLUMNS,
)


def tabular_feature_columns_for_set(
    frame: pd.DataFrame,
    feature_set: str,
    *,
    price_feature_config: PriceFeatureConfig | None = None,
) -> list[str]:
    """Return feature columns for the named CatBoost tabular feature set."""

    config = price_feature_config or DEFAULT_PRICE_FEATURE_CONFIG
    weather_columns = weather_value_columns(frame)
    price_columns = _numeric_existing(tuple(config.feature_columns), frame)
    if feature_set == "price_full_engineered":
        return price_columns
    if feature_set == "price_baseline_calendar":
        return _numeric_existing(_baseline_calendar_columns(config), frame)
    if feature_set == "price_lags_calendar":
        return _numeric_existing(_lags_calendar_columns(config), frame)
    if feature_set == "all_weather":
        selected_weather = [
            column for column in weather_columns if not column.startswith("weather_ensemble_")
        ]
        return price_columns + selected_weather if selected_weather else []
    if feature_set == "all_weather_plus_ensemble":
        return price_columns + weather_columns if weather_columns else []
    if feature_set == "ensemble":
        selected_weather = [
            column for column in weather_columns if column.startswith("weather_ensemble_")
        ]
        return price_columns + selected_weather if selected_weather else []

    raise ValueError(f"Unknown CatBoost feature set: {feature_set}")


def _baseline_calendar_columns(config: PriceFeatureConfig) -> tuple[str, ...]:
    return (
        "area",
        *config.baseline_feature_columns,
        *(column for column in config.base_features if column != "area"),
    )


def _lags_calendar_columns(config: PriceFeatureConfig) -> tuple[str, ...]:
    return (
        "area",
        *(column for column in config.base_features if column != "area"),
        *config.lag_feature_columns,
        *config.spread_feature_columns,
    )


def _numeric_existing(columns: tuple[str, ...], frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in columns
        if column in frame.columns
        and (
            column == "area"
            or (
                column not in TABULAR_METADATA_COLUMNS
                and not column.startswith("weather_")
                and pd.api.types.is_numeric_dtype(frame[column])
            )
        )
    ]
