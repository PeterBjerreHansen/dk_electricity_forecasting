from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from dkenergy_forecast.types import normalize_utc_column


SCORE_COLUMNS = [
    "forecast_origin_utc",
    "model_name",
    "model_version",
    "area",
    "is_published",
    "rows",
    "mae",
    "rmse",
    "bias",
]


@dataclass(frozen=True)
class CheckResult:
    check: str
    subject: str
    status: str  # "ok", "alert" or "insufficient_data"
    value: float | None
    threshold: float
    detail: str


def score_forecasts(forecasts: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Per origin, model and area error once the actual prices are known."""

    actuals = normalize_utc_column(panel[["unique_id", "ds_utc", "y"]], "ds_utc")
    joined = normalize_utc_column(forecasts, "ds_utc").merge(actuals, on=["unique_id", "ds_utc"], how="inner")
    joined = joined.dropna(subset=["y", "y_pred"])
    if joined.empty:
        return pd.DataFrame(columns=SCORE_COLUMNS)
    joined["error"] = joined["y_pred"] - joined["y"]
    keys = ["forecast_origin_utc", "model_name", "model_version", "area", "is_published"]
    scores = (
        joined.groupby(keys, dropna=False)
        .agg(
            rows=("error", "size"),
            mae=("error", lambda e: float(np.abs(e).mean())),
            rmse=("error", lambda e: float(np.sqrt(np.square(e).mean()))),
            bias=("error", "mean"),
        )
        .reset_index()
    )
    return scores[SCORE_COLUMNS]


def degradation_check(
    scores: pd.DataFrame,
    *,
    reference_model: str,
    window_origins: int = 7,
    max_ratio: float = 1.0,
) -> CheckResult:
    """Alert when the published model is worse than the fallback lately.

    Only origins where both were scored count, so the comparison is paired.
    """

    subject = "published_vs_" + reference_model
    if scores.empty:
        return CheckResult("degradation", subject, "insufficient_data", None, max_ratio, "no scored forecasts")
    per_origin = (
        scores.assign(abs_error=scores["mae"] * scores["rows"])
        .groupby(["forecast_origin_utc", "model_name", "is_published"])[["abs_error", "rows"]]
        .sum()
        .reset_index()
    )
    published = per_origin[per_origin["is_published"]].set_index("forecast_origin_utc")
    reference = per_origin[per_origin["model_name"] == reference_model].set_index("forecast_origin_utc")
    shared = published.index.intersection(reference.index).sort_values()[-window_origins:]
    if len(shared) < window_origins:
        return CheckResult(
            "degradation", subject, "insufficient_data", None, max_ratio,
            f"{len(shared)} of {window_origins} paired origins scored",
        )
    published_mae = published.loc[shared, "abs_error"].sum() / published.loc[shared, "rows"].sum()
    reference_mae = reference.loc[shared, "abs_error"].sum() / reference.loc[shared, "rows"].sum()
    ratio = float(published_mae / reference_mae)
    status = "alert" if ratio > max_ratio else "ok"
    return CheckResult(
        "degradation", subject, status, ratio, max_ratio,
        f"published MAE {published_mae:.1f} vs {reference_model} {reference_mae:.1f} "
        f"over the last {window_origins} origins",
    )


def population_stability_index(reference: pd.Series, current: pd.Series, *, bins: int = 10) -> float:
    """PSI of ``current`` against decile bins of ``reference``."""

    reference = reference.dropna().astype(float)
    current = current.dropna().astype(float)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    expected = np.histogram(reference, edges)[0] / len(reference)
    actual = np.histogram(current, edges)[0] / len(current)
    expected = np.clip(expected, 1e-6, None)
    actual = np.clip(actual, 1e-6, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def price_drift_checks(
    panel: pd.DataFrame,
    *,
    as_of: pd.Timestamp,
    recent_days: int = 7,
    reference_days: int = 90,
    threshold: float = 0.25,
) -> list[CheckResult]:
    """Compare the last week's price distribution with the previous quarter, per area.

    A PSI above 0.25 is the conventional "significant shift" level. For prices
    this flags regime changes (a gas shock, a windy month) that warrant a
    retrain or a look at the model, not a data error.
    """

    frame = normalize_utc_column(panel, "ds_utc")
    recent_start = as_of - pd.Timedelta(days=recent_days)
    reference_start = recent_start - pd.Timedelta(days=reference_days)
    results = []
    for area, group in frame.groupby("area", sort=True):
        recent = group.loc[(group["ds_utc"] >= recent_start) & (group["ds_utc"] < as_of), "y"]
        reference = group.loc[(group["ds_utc"] >= reference_start) & (group["ds_utc"] < recent_start), "y"]
        if len(recent) < 24 * recent_days // 2 or len(reference) < 24 * 7:
            results.append(CheckResult("price_drift_psi", str(area), "insufficient_data", None, threshold, "too few rows"))
            continue
        psi = population_stability_index(reference, recent)
        results.append(
            CheckResult(
                "price_drift_psi",
                str(area),
                "alert" if psi > threshold else "ok",
                psi,
                threshold,
                f"last {recent_days}d vs previous {reference_days}d; "
                f"mean {recent.mean():.0f} vs {reference.mean():.0f} DKK/MWh",
            )
        )
    return results


def checks_frame(checks: list[CheckResult], *, checked_at_utc: pd.Timestamp) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(check) for check in checks])
    frame.insert(0, "checked_at_utc", checked_at_utc)
    frame["value"] = frame["value"].astype(float)
    return frame
