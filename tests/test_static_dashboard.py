from __future__ import annotations

import json
import re
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from dkenergy_forecast.static_dashboard import build_static_dashboard


COPENHAGEN = ZoneInfo("Europe/Copenhagen")


def _delivery_rows(
    delivery_date: str,
    *,
    area: str = "DK1",
    model_label: str = "chronos_weather",
    release_id: str | None = "release_1",
    with_actual: bool = False,
    with_interval: bool = True,
) -> list[dict[str, object]]:
    day = date.fromisoformat(delivery_date)
    start = datetime.combine(day, time.min, tzinfo=COPENHAGEN).astimezone(timezone.utc)
    end = datetime.combine(
        day + timedelta(days=1), time.min, tzinfo=COPENHAGEN
    ).astimezone(timezone.utc)
    hours = int((end - start).total_seconds() // 3600)
    rows: list[dict[str, object]] = []
    for horizon in range(hours):
        timestamp = start + timedelta(hours=horizon)
        prediction = 100.0 + horizon
        row: dict[str, object] = {
            "area": area,
            "ds_utc": timestamp.isoformat(),
            "local_date": delivery_date,
            "horizon": horizon + 1,
            "model_label": model_label,
            "model_release_id": release_id,
            "y_pred": prediction,
        }
        if with_interval:
            row.update(q10=prediction - 10, q50=prediction, q90=prediction + 10)
        if with_actual:
            row["y"] = prediction + 5
        rows.append(row)
    return rows


def _payload(
    *,
    delivery_date: str = "2026-07-02",
    release_id: str | None = "release_1",
    model_label: str = "chronos_weather",
    with_interval: bool = True,
    areas: tuple[str, ...] = ("DK1", "DK2"),
) -> dict:
    return {
        "generated_at_utc": "2026-07-13T21:14:18Z",
        "run": {
            "run_id": "replay_demo",
            "run_kind": "replay",
            "delivery_date_local": delivery_date,
            "forecast_origin_utc": "2026-07-01T08:00:00Z",
            "forecast_status": "primary",
            "git_commit": "abc1234",
            "artifact_paths": {"private": "/var/lib/private"},
            "model": {
                "published_model": model_label,
                "primary_model": "chronos_weather",
                "model_release_id": release_id,
            },
        },
        "predictions": [
            row
            for area in areas
            for row in _delivery_rows(
                delivery_date,
                area=area,
                model_label=model_label,
                release_id=release_id,
                with_interval=with_interval,
            )
        ],
    }


def _embedded_data(html: str) -> dict:
    match = re.search(r"const DATA = (.*?);\n", html)
    assert match is not None
    return json.loads(match.group(1))


def test_static_dashboard_is_self_contained_and_omits_private_fields() -> None:
    html = build_static_dashboard(_payload())

    assert html.startswith("<!doctype html>")
    assert "replay_demo" in html
    assert "chronos_weather" in html
    assert "https://" not in html
    assert "/var/lib/private" not in html
    assert "const DATA" in html


def test_static_dashboard_escapes_script_terminators() -> None:
    payload = _payload()
    payload["run"]["run_id"] = "</script><script>alert(1)</script>"

    html = build_static_dashboard(payload)

    assert "</script><script>alert(1)</script>" not in html
    assert "\\u003c/script\\u003e" in html


def test_static_dashboard_requires_predictions() -> None:
    payload = _payload()
    payload["predictions"] = []

    with pytest.raises(ValueError, match="at least one prediction"):
        build_static_dashboard(payload)


def test_static_dashboard_requires_both_production_areas() -> None:
    with pytest.raises(ValueError, match="must contain exactly DK1 and DK2"):
        build_static_dashboard(_payload(areas=("DK1",)))


def test_static_dashboard_requires_an_explicit_model_release() -> None:
    with pytest.raises(ValueError, match="missing model_release_id"):
        build_static_dashboard(_payload(release_id=None))


def test_static_dashboard_embeds_exact_previous_day_for_same_release() -> None:
    history = _delivery_rows(
        "2026-07-01",
        with_actual=True,
        release_id="release_1",
    )

    html = build_static_dashboard(_payload(), history_predictions=history)
    outlook = _embedded_data(html)["outlook"]["DK1"]

    assert outlook["evaluated_date"] == "2026-07-01"
    assert len(outlook["evaluated"]) == 24
    assert outlook["show_interval"] is True
    assert "Recent model performance" in html
    assert "/private/model.bin" not in html


def test_static_dashboard_never_joins_a_stale_history_day() -> None:
    history = _delivery_rows(
        "2026-07-06",
        with_actual=True,
        release_id="release_1",
    )

    html = build_static_dashboard(
        _payload(delivery_date="2026-07-15"),
        history_predictions=history,
    )
    outlook = _embedded_data(html)["outlook"]["DK1"]

    assert outlook["evaluated_date"] is None
    assert outlook["evaluated"] == []
    assert len(outlook["forecast"]) == 24


def test_static_dashboard_never_joins_a_different_model_release() -> None:
    history = _delivery_rows(
        "2026-07-14",
        with_actual=True,
        release_id="old_release",
    )

    html = build_static_dashboard(
        _payload(delivery_date="2026-07-15", release_id="current_release"),
        history_predictions=history,
    )

    assert _embedded_data(html)["outlook"]["DK1"]["evaluated"] == []


def test_static_dashboard_rejects_incomplete_adjacent_history() -> None:
    history = _delivery_rows("2026-07-14", with_actual=True)[:-1]

    with pytest.raises(ValueError, match="complete 2026-07-14 delivery grid"):
        build_static_dashboard(
            _payload(delivery_date="2026-07-15"),
            history_predictions=history,
        )


def test_static_dashboard_validates_each_price_area_independently() -> None:
    payload = _payload(areas=("DK1", "DK2"))
    payload["predictions"] = [
        row
        for row in payload["predictions"]
        if not (row["area"] == "DK2" and row["horizon"] == 24)
    ]

    with pytest.raises(ValueError, match="delivery grid for DK2"):
        build_static_dashboard(payload)


def test_static_dashboard_rejects_duplicate_forecast_hours() -> None:
    payload = _payload()
    payload["predictions"][-1] = payload["predictions"][0].copy()

    with pytest.raises(ValueError, match="duplicate timestamps for DK1"):
        build_static_dashboard(payload)


def test_static_dashboard_rejects_unordered_intervals() -> None:
    payload = _payload()
    payload["predictions"][0]["q10"] = 200.0

    with pytest.raises(ValueError, match="unordered intervals"):
        build_static_dashboard(payload)


@pytest.mark.parametrize(
    ("delivery_date", "expected_hours"),
    [("2026-03-29", 23), ("2026-10-25", 25)],
)
def test_static_dashboard_accepts_complete_dst_delivery_days(
    delivery_date: str,
    expected_hours: int,
) -> None:
    html = build_static_dashboard(_payload(delivery_date=delivery_date))

    assert len(_embedded_data(html)["outlook"]["DK1"]["forecast"]) == expected_hours


def test_static_dashboard_accepts_a_point_only_fallback_forecast() -> None:
    payload = _payload(
        release_id="weighted_median_v1",
        model_label="weighted_median_v1",
        with_interval=False,
    )
    payload["run"]["forecast_status"] = "degraded"

    html = build_static_dashboard(payload)

    assert "Fallback forecast" in html
    assert "weighted_median_v1" in html
    assert _embedded_data(html)["outlook"]["DK1"]["show_interval"] is False


def test_static_dashboard_keeps_the_outlook_focused_and_marks_timing() -> None:
    html = build_static_dashboard(
        _payload(),
        history_predictions=_delivery_rows("2026-07-01", with_actual=True),
    )

    assert "Forecast avg · DKK/MWh" not in html
    assert "Forecast min · DKK/MWh" not in html
    assert "Forecast max · DKK/MWh" not in html
    assert "Previous MAE · DKK/MWh" not in html
    assert "scale.x(evaluated.length-1)" in html
    assert "Forecast made" in html
    assert 'shiftedPolyline(rows,"y_pred",0,scale)' in html
    assert 'shiftedPolyline(forecast,"y_pred",evaluated.length,scale)' not in html
