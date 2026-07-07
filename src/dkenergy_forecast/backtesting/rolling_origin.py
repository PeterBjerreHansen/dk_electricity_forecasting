from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from dkenergy_forecast.types import (
    PREDICTION_REQUIRED_COLUMNS,
    TARGET_LEAKAGE_COLUMNS,
    ForecastModel,
    ensure_price_availability,
    filter_price_history_available_before,
    normalize_utc_column,
    require_columns,
)


METADATA_JOIN_COLUMNS = [
    "unique_id",
    "ds_utc",
    "y",
    "area",
    "ds_local",
    "local_date",
    "local_hour",
    "local_day_of_week",
    "local_month",
    "is_weekend",
    "is_dst",
    "utc_offset_hours",
    "price_available_at_utc",
    "dataset_version",
]


def rolling_origin_backtest(
    model_factory: Callable[[], ForecastModel],
    panel: pd.DataFrame,
    origins: pd.DataFrame,
    horizon_builder: Callable[[pd.DataFrame, pd.Timestamp], pd.DataFrame],
    min_train_rows: int | None = None,
) -> pd.DataFrame:
    require_columns(panel, ["unique_id", "ds_utc", "y"], "panel")
    require_columns(origins, ["forecast_origin_utc"], "origins")
    panel_utc = (
        ensure_price_availability(normalize_utc_column(panel, "ds_utc"))
        .sort_values(["unique_id", "ds_utc"])
        .reset_index(drop=True)
    )
    origins_utc = normalize_utc_column(origins, "forecast_origin_utc")

    outputs: list[pd.DataFrame] = []
    for origin in origins_utc["forecast_origin_utc"].sort_values().drop_duplicates():
        history = filter_price_history_available_before(panel_utc, origin)
        if min_train_rows is not None and len(history) < min_train_rows:
            raise ValueError(
                "Not enough training rows before forecast origin "
                f"{origin.isoformat()}: {len(history)} < {min_train_rows}"
            )

        model = model_factory()
        model.fit(history)
        future = horizon_builder(panel_utc, origin)
        future_for_model = future.drop(
            columns=[column for column in TARGET_LEAKAGE_COLUMNS if column in future.columns],
            errors="ignore",
        )
        predictions = model.predict(future_for_model, history=history)
        require_columns(predictions, PREDICTION_REQUIRED_COLUMNS, "predictions")
        predictions = normalize_utc_column(predictions, "ds_utc")
        predictions = normalize_utc_column(predictions, "forecast_origin_utc")
        _validate_prediction_keys(predictions, future_for_model, origin)
        outputs.append(_join_actuals_and_metadata(predictions, panel_utc, future_for_model))

    if not outputs:
        return pd.DataFrame(columns=PREDICTION_REQUIRED_COLUMNS + ["y"])
    return pd.concat(outputs, ignore_index=True)


def _validate_prediction_keys(
    predictions: pd.DataFrame,
    future: pd.DataFrame,
    forecast_origin_utc: pd.Timestamp,
) -> None:
    key_cols = ["unique_id", "ds_utc", "forecast_origin_utc"]
    require_columns(future, key_cols, "future")

    expected = (
        normalize_utc_column(future[key_cols], "ds_utc")
        .pipe(normalize_utc_column, "forecast_origin_utc")
        .sort_values(key_cols)
        .reset_index(drop=True)
    )
    observed = predictions[key_cols].sort_values(key_cols).reset_index(drop=True)

    duplicate_count = int(observed.duplicated(key_cols).sum())
    if duplicate_count:
        raise ValueError(
            "Model predictions contain duplicate "
            f"(unique_id, ds_utc, forecast_origin_utc) rows before {forecast_origin_utc.isoformat()}: "
            f"{duplicate_count}"
        )

    if len(observed) != len(expected) or not observed.equals(expected):
        expected_keys = set(map(tuple, expected.to_numpy()))
        observed_keys = set(map(tuple, observed.to_numpy()))
        missing = sorted(expected_keys - observed_keys)[:5]
        extra = sorted(observed_keys - expected_keys)[:5]
        raise ValueError(
            "Model predictions do not match the requested future frame for "
            f"{forecast_origin_utc.isoformat()}: expected {len(expected)} rows, "
            f"got {len(observed)} rows, missing_sample={missing}, extra_sample={extra}"
        )


def _join_actuals_and_metadata(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    future: pd.DataFrame,
) -> pd.DataFrame:
    key_cols = ["unique_id", "ds_utc"]
    metadata_cols = [column for column in METADATA_JOIN_COLUMNS if column in panel.columns]
    future_metadata_cols = [
        column
        for column in METADATA_JOIN_COLUMNS
        if column != "y" and column in future.columns
    ]
    panel_metadata = panel[metadata_cols].drop_duplicates(key_cols)
    if future_metadata_cols:
        future_metadata = future[future_metadata_cols].drop_duplicates(key_cols)
        actuals = future_metadata.merge(
            panel_metadata,
            on=key_cols,
            how="left",
            suffixes=("", "_panel"),
        )
        for column in metadata_cols:
            if column in key_cols:
                continue
            panel_column = f"{column}_panel"
            if panel_column in actuals.columns:
                missing = actuals[column].isna() & actuals[panel_column].notna()
                if bool(missing.any()):
                    actuals.loc[missing, column] = actuals.loc[missing, panel_column]
                actuals = actuals.drop(columns=[panel_column])
    else:
        actuals = panel_metadata

    output = predictions.drop(
        columns=[
            column
            for column in metadata_cols
            if column not in {"unique_id", "ds_utc"} and column in predictions.columns
        ],
        errors="ignore",
    )
    return output.merge(actuals, on=["unique_id", "ds_utc"], how="left")
