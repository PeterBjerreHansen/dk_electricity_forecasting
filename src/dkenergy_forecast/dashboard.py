from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from dkenergy_forecast.types import normalize_utc_column


COPENHAGEN_TIMEZONE = "Europe/Copenhagen"

MODEL_FAMILY_NAMES = {
    "chronos": "Chronos 2 LoRA Weather",
    "weighted_median": "Weighted Rolling Median",
    "rolling_median": "Rolling Median",
    "last_week": "Last Week Baseline",
}

MODEL_FAMILY_ORDER = {
    "chronos": 0,
    "weighted_median": 1,
    "rolling_median": 2,
    "last_week": 3,
}

DASHBOARD_HISTORY_DAYS = 30
DASHBOARD_ARCHIVE_DAYS = DASHBOARD_HISTORY_DAYS + 2
FORECAST_HISTORY_COLUMNS = (
    "area",
    "ds_utc",
    "forecast_origin_utc",
    "model_label",
    "model_release_id",
    "q10",
    "q50",
    "q90",
    "y",
    "y_pred",
    "run_id",
)


def canonical_model_family(label: object) -> str:
    """Return the stable dashboard identity for a model artifact label."""
    if pd.isna(label):
        return ""
    value = str(label).lower()
    if "chronos" in value:
        return "chronos"
    if "weighted_median" in value or "median_weekday" in value:
        return "weighted_median"
    if "rolling_median" in value:
        return "rolling_median"
    if value == "same_hour_last_week" or "last_week" in value:
        return "last_week"
    return value


def prepare_dashboard_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize prediction timestamps and add stable display columns."""
    if frame.empty:
        return frame.copy()
    output = normalize_utc_column(frame, "ds_utc")
    if "forecast_origin_utc" in output:
        output = normalize_utc_column(output, "forecast_origin_utc")
    if "model_label" not in output and "model_name" in output:
        output["model_label"] = output["model_name"]
    output["model_family"] = output["model_label"].map(canonical_model_family)
    output["model"] = output.apply(_dashboard_model_name, axis=1)
    output["display_time"] = output["ds_utc"].dt.tz_convert(COPENHAGEN_TIMEZONE)
    output["delivery_date"] = output["display_time"].dt.date
    return output


def combine_prediction_history(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine evaluated artifacts, preferring later frames for duplicate rows."""
    prepared: list[pd.DataFrame] = []
    for priority, frame in enumerate(frames):
        if frame.empty or not {"ds_utc", "area", "y", "y_pred"}.issubset(frame.columns):
            continue
        item = prepare_dashboard_predictions(frame)
        item = item[item["y"].notna() & item["y_pred"].notna()].copy()
        item["_priority"] = priority
        prepared.append(item)
    if not prepared:
        return pd.DataFrame()

    output = pd.concat(prepared, ignore_index=True, sort=False)
    output = output.sort_values("_priority").drop_duplicates(
        ["area", "ds_utc", "model_family"],
        keep="last",
    )
    return (
        output.drop(columns="_priority")
        .sort_values(["model_family", "area", "ds_utc"])
        .reset_index(drop=True)
    )


def hero_series(
    latest_predictions: pd.DataFrame,
    history: pd.DataFrame,
    *,
    area: str,
    model_family: str = "chronos",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the evaluated day before the latest forecast for one area and model."""
    latest = prepare_dashboard_predictions(latest_predictions)
    if latest.empty:
        return pd.DataFrame(), pd.DataFrame()
    forecast = latest[
        (latest["area"] == area)
        & (latest["model_family"] == model_family)
        & latest["y_pred"].notna()
    ].copy()
    if forecast.empty:
        return pd.DataFrame(), pd.DataFrame()
    forecast_date = max(forecast["delivery_date"])
    forecast = forecast[forecast["delivery_date"] == forecast_date].sort_values(
        "ds_utc"
    )

    evaluated = history[
        (history["area"] == area)
        & (history["model_family"] == model_family)
        & history["y"].notna()
        & history["y_pred"].notna()
        & (history["delivery_date"] < forecast_date)
    ].copy()
    if evaluated.empty:
        return pd.DataFrame(), forecast
    target_date = pd.Timestamp(forecast_date) - pd.Timedelta(days=1)
    available_dates = pd.Series(evaluated["delivery_date"].unique())
    exact = available_dates[available_dates == target_date.date()]
    evaluated_date = exact.iloc[0] if not exact.empty else max(available_dates)
    evaluated = evaluated[evaluated["delivery_date"] == evaluated_date].sort_values(
        "ds_utc"
    )
    return evaluated, forecast


def recent_model_history(
    history: pd.DataFrame,
    *,
    area: str,
    days: int = 30,
) -> pd.DataFrame:
    """Keep the most recent delivery dates independently for every model."""
    if history.empty:
        return history.copy()
    area_history = history[history["area"] == area].copy()
    selected: list[pd.DataFrame] = []
    for _, model_frame in area_history.groupby("model_family", sort=False):
        dates = sorted(model_frame["delivery_date"].dropna().unique())[-days:]
        selected.append(model_frame[model_frame["delivery_date"].isin(dates)])
    if not selected:
        return pd.DataFrame()
    output = pd.concat(selected, ignore_index=True, sort=False)
    output["model_order"] = output["model_family"].map(MODEL_FAMILY_ORDER).fillna(99)
    return output.sort_values(["model_order", "model", "ds_utc"]).drop(
        columns="model_order"
    )


def update_forecast_history(
    existing: pd.DataFrame,
    new_predictions: pd.DataFrame,
    price_panel: pd.DataFrame,
    *,
    archive_days: int = DASHBOARD_ARCHIVE_DAYS,
) -> pd.DataFrame:
    """Merge registered forecasts with newly available official prices.

    The archive retains a small number of unevaluated delivery days as well as
    the 30 evaluated days shown on the public page. Forecast values are never
    recomputed here; only the observed ``y`` column is refreshed from the
    latest price panel.
    """

    if archive_days < DASHBOARD_HISTORY_DAYS:
        raise ValueError(
            f"archive_days must be at least {DASHBOARD_HISTORY_DAYS}"
        )

    frames = [frame for frame in (existing, new_predictions) if not frame.empty]
    if not frames:
        return pd.DataFrame()

    prepared: list[pd.DataFrame] = []
    for priority, frame in enumerate(frames):
        item = prepare_dashboard_predictions(frame)
        if not {"area", "ds_utc", "model_family", "y_pred"}.issubset(item.columns):
            continue
        item = item[item["y_pred"].notna()].copy()
        item["_priority"] = priority
        prepared.append(item)
    if not prepared:
        return pd.DataFrame()

    history = pd.concat(prepared, ignore_index=True, sort=False)
    history = history.sort_values("_priority").drop_duplicates(
        ["area", "ds_utc", "model_family"],
        keep="last",
    )

    panel = normalize_utc_column(price_panel, "ds_utc")
    if "area" not in panel and "unique_id" in panel:
        panel["area"] = panel["unique_id"]
    if {"area", "ds_utc", "y"}.issubset(panel.columns):
        actuals = (
            panel[["area", "ds_utc", "y"]]
            .dropna(subset=["y"])
            .drop_duplicates(["area", "ds_utc"], keep="last")
            .rename(columns={"y": "_official_y"})
        )
        history = history.merge(actuals, on=["area", "ds_utc"], how="left")
        prior_y = history["y"] if "y" in history else pd.Series(
            pd.NA,
            index=history.index,
            dtype="object",
        )
        history["y"] = history["_official_y"].where(
            history["_official_y"].notna(),
            prior_y,
        )
        history = history.drop(columns="_official_y")

    selected: list[pd.DataFrame] = []
    for _, frame in history.groupby(["area", "model_family"], sort=False):
        dates = sorted(frame["delivery_date"].dropna().unique())[-archive_days:]
        selected.append(frame[frame["delivery_date"].isin(dates)])
    if not selected:
        return pd.DataFrame()

    output = pd.concat(selected, ignore_index=True, sort=False)
    return canonical_forecast_history(
        output.drop(columns=["_priority", "display_time", "delivery_date"], errors="ignore")
        .sort_values(["model_family", "area", "ds_utc"])
        .reset_index(drop=True)
    )


def canonical_forecast_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the compact, stable parquet schema used by daily dashboard runs."""

    if frame.empty:
        return pd.DataFrame(columns=FORECAST_HISTORY_COLUMNS)
    columns = [column for column in FORECAST_HISTORY_COLUMNS if column in frame]
    output = normalize_utc_column(frame[list(columns)].copy(), "ds_utc")
    if "forecast_origin_utc" in output:
        output = normalize_utc_column(output, "forecast_origin_utc")
    for column in ["q10", "q50", "q90", "y", "y_pred"]:
        if column in output:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    for column in ["model_label", "model_release_id", "run_id"]:
        if column in output:
            output[column] = output[column].astype("string")
    return output.reset_index(drop=True)


def evaluated_dashboard_history(
    history: pd.DataFrame,
    *,
    days: int = DASHBOARD_HISTORY_DAYS,
) -> pd.DataFrame:
    """Return the latest evaluated delivery days for each model and area."""

    evaluated = combine_prediction_history([history])
    if evaluated.empty:
        return evaluated
    return pd.concat(
        [
            recent_model_history(evaluated, area=area, days=days)
            for area in sorted(evaluated["area"].dropna().unique())
        ],
        ignore_index=True,
    )


def dashboard_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    """Serialize only public dashboard columns to JSON-compatible records."""

    if frame.empty:
        return []
    columns = [
        column
        for column in [
            "area",
            "ds_utc",
            "ds_local",
            "local_date",
            "horizon",
            "model_label",
            "model_release_id",
            "q10",
            "q50",
            "q90",
            "y",
            "y_pred",
        ]
        if column in frame.columns
    ]
    output = frame[columns].copy()
    for column in ["ds_utc", "ds_local", "local_date"]:
        if column in output:
            output[column] = output[column].map(_isoformat_or_none)
    output = output.astype(object).where(pd.notna(output), None)
    return output.to_dict(orient="records")


def _dashboard_model_name(row: pd.Series) -> str:
    family = row["model_family"]
    if family in MODEL_FAMILY_NAMES:
        return MODEL_FAMILY_NAMES[family]
    return str(row["model_label"]).replace("_", " ").title()


def _isoformat_or_none(value: object) -> object:
    if pd.isna(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)
