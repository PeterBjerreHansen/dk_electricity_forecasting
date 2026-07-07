from __future__ import annotations

import pandas as pd

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

ROBUST_BASELINE_COLUMN = "baseline_wdwe_weighted_median_y_pred"
CALENDAR_PRICE_COLUMNS = (
    "local_hour",
    "local_day_of_week",
    "local_month",
    "is_weekend",
    "is_dst",
    "utc_offset_hours",
)
PRICE_LAG_COLUMNS = (
    "lag_24h",
    "lag_48h",
    "lag_168h",
)
PRICE_ROLLING_COLUMNS = (
    "rolling_mean_24h_asof_origin",
    "rolling_median_24h_asof_origin",
    "rolling_mean_168h_asof_origin",
    "rolling_median_168h_asof_origin",
    "rolling_mean_672h_asof_origin",
    "rolling_median_672h_asof_origin",
)
PRICE_SEASONAL_COLUMNS = (
    "seasonal_median_local_hour",
    "seasonal_median_hour_weekend",
)
PRICE_SPREAD_COLUMNS = (
    "dk1_minus_dk2_lag_24h",
    "dk1_minus_dk2_lag_168h",
)
PRICE_FULL_ENGINEERED_COLUMNS = (
    "area",
    *CALENDAR_PRICE_COLUMNS,
    *PRICE_LAG_COLUMNS,
    *PRICE_ROLLING_COLUMNS,
    *PRICE_SEASONAL_COLUMNS,
    ROBUST_BASELINE_COLUMN,
    *PRICE_SPREAD_COLUMNS,
)
PRICE_BASELINE_CALENDAR_COLUMNS = (
    "area",
    ROBUST_BASELINE_COLUMN,
    *CALENDAR_PRICE_COLUMNS,
)
PRICE_LAGS_CALENDAR_COLUMNS = (
    "area",
    *CALENDAR_PRICE_COLUMNS,
    *PRICE_LAG_COLUMNS,
    *PRICE_SPREAD_COLUMNS,
)


def tabular_feature_columns_for_set(frame: pd.DataFrame, feature_set: str) -> list[str]:
    """Return feature columns for the named CatBoost tabular feature set."""

    weather_columns = weather_value_columns(frame)
    price_columns = _numeric_existing(PRICE_FULL_ENGINEERED_COLUMNS, frame)
    if feature_set == "price_full_engineered":
        return price_columns
    if feature_set == "price_baseline_calendar":
        return _numeric_existing(PRICE_BASELINE_CALENDAR_COLUMNS, frame)
    if feature_set == "price_lags_calendar":
        return _numeric_existing(PRICE_LAGS_CALENDAR_COLUMNS, frame)
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
