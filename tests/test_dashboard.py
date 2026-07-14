from __future__ import annotations

import pandas as pd

from dkenergy_forecast.dashboard import (
    canonical_model_family,
    combine_prediction_history,
    dashboard_records,
    evaluated_dashboard_history,
    hero_series,
    recent_model_history,
    update_forecast_history,
)


def test_model_aliases_share_stable_dashboard_families() -> None:
    assert canonical_model_family("chronos_weather") == "chronos"
    assert (
        canonical_model_family("chronos2_lora_calendar_weather_ctx1024_v1") == "chronos"
    )
    assert canonical_model_family("same_hour_last_week") == "last_week"
    assert (
        canonical_model_family("median_weekday_exp_hl4_floor10_42d")
        == "weighted_median"
    )


def test_combined_history_prefers_later_artifact_for_duplicate_model_hour() -> None:
    older = pd.DataFrame([_row("2026-07-06T00:00:00Z", "chronos_weather", y_pred=10.0)])
    newer = pd.DataFrame(
        [
            _row(
                "2026-07-06T00:00:00Z",
                "chronos2_lora_calendar_weather_ctx1024_v1",
                y_pred=12.0,
            )
        ]
    )

    combined = combine_prediction_history([older, newer])

    assert len(combined) == 1
    assert combined.iloc[0]["y_pred"] == 12.0
    assert combined.iloc[0]["model"] == "Chronos 2 LoRA Weather"


def test_hero_uses_previous_delivery_day_and_latest_forecast_day() -> None:
    history = combine_prediction_history(
        [
            pd.DataFrame(
                [
                    _row("2026-07-05T22:00:00Z", "chronos_weather", y_pred=8.0),
                    _row("2026-07-06T21:00:00Z", "chronos_weather", y_pred=9.0),
                ]
            )
        ]
    )
    latest = pd.DataFrame(
        [
            _row(
                "2026-07-06T22:00:00Z",
                "chronos2_lora_calendar_weather_ctx1024_v1",
                y_pred=11.0,
            ),
            _row(
                "2026-07-07T21:00:00Z",
                "chronos2_lora_calendar_weather_ctx1024_v1",
                y_pred=12.0,
            ),
        ]
    ).drop(columns="y")

    evaluated, forecast = hero_series(latest, history, area="DK1")

    assert evaluated["delivery_date"].unique().tolist() == [
        pd.Timestamp("2026-07-06").date()
    ]
    assert forecast["delivery_date"].unique().tolist() == [
        pd.Timestamp("2026-07-07").date()
    ]


def test_recent_history_keeps_thirty_days_per_model() -> None:
    rows = []
    for day in pd.date_range("2026-05-01", periods=35, tz="UTC"):
        rows.append(_row(day.isoformat(), "chronos_weather", y_pred=10.0))
    for day in pd.date_range("2026-06-01", periods=4, tz="UTC"):
        rows.append(_row(day.isoformat(), "same_hour_last_week", y_pred=11.0))
    history = combine_prediction_history([pd.DataFrame(rows)])

    recent = recent_model_history(history, area="DK1", days=30)

    counts = recent.groupby("model_family")["delivery_date"].nunique().to_dict()
    assert counts == {"chronos": 30, "last_week": 4}


def test_forecast_history_preserves_predictions_and_refreshes_actuals() -> None:
    pending = pd.DataFrame(
        [
            {
                **_row("2026-07-06T22:00:00Z", "chronos_weather", y_pred=11.0),
                "y": None,
            }
        ]
    )
    panel = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": ["2026-07-06T22:00:00Z"],
            "y": [12.5],
        }
    )

    history = update_forecast_history(pd.DataFrame(), pending, panel)

    assert history.loc[0, "y_pred"] == 11.0
    assert history.loc[0, "y"] == 12.5
    assert canonical_model_family(history.loc[0, "model_label"]) == "chronos"


def test_evaluated_history_and_records_exclude_pending_and_private_columns() -> None:
    history = pd.DataFrame(
        [
            _row("2026-07-05T22:00:00Z", "chronos_weather", y_pred=11.0),
            {
                **_row("2026-07-06T22:00:00Z", "chronos_weather", y_pred=12.0),
                "y": None,
                "private_path": "/models/private.bin",
            },
        ]
    )

    evaluated = evaluated_dashboard_history(history)
    records = dashboard_records(evaluated)

    assert len(records) == 1
    assert records[0]["ds_utc"] == "2026-07-05T22:00:00+00:00"
    assert "private_path" not in records[0]


def _row(ds_utc: str, model_label: str, *, y_pred: float) -> dict[str, object]:
    return {
        "ds_utc": ds_utc,
        "forecast_origin_utc": "2026-07-01T10:00:00Z",
        "area": "DK1",
        "model_label": model_label,
        "y": 10.0,
        "y_pred": y_pred,
    }
