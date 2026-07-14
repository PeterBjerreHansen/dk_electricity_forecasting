from __future__ import annotations

import pytest

from dkenergy_forecast.static_dashboard import build_static_dashboard


def _payload() -> dict:
    return {
        "generated_at_utc": "2026-07-13T21:14:18Z",
        "run": {
            "run_id": "replay_demo",
            "run_kind": "replay",
            "delivery_date_local": "2026-07-02",
            "forecast_origin_utc": "2026-07-01T08:00:00Z",
            "forecast_status": "primary",
            "git_commit": "abc1234",
            "artifact_paths": {"private": "/var/lib/private"},
            "model": {
                "published_model": "chronos_weather",
                "primary_model": "chronos_weather",
                "model_release_id": "release_1",
            },
        },
        "predictions": [
            {
                "area": "DK1",
                "ds_utc": "2026-07-01T22:00:00Z",
                "horizon": 1,
                "model_label": "chronos_weather",
                "q10": 90.0,
                "q50": 100.0,
                "q90": 110.0,
                "y_pred": 100.0,
                "actual_price": 105.0,
                "ignored": "</script><script>alert(1)</script>",
            }
        ],
    }


def test_static_dashboard_is_self_contained_and_omits_private_fields() -> None:
    html = build_static_dashboard(_payload())

    assert html.startswith("<!doctype html>")
    assert "replay_demo" in html
    assert "chronos_weather" in html
    assert "https://" not in html
    assert "/var/lib/private" not in html
    assert "ignored" not in html
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


def test_static_dashboard_embeds_evaluated_history_without_extra_fields() -> None:
    history = [
        {
            "area": "DK1",
            "ds_utc": "2026-06-30T22:00:00Z",
            "local_date": "2026-07-01",
            "model_label": "chronos_weather",
            "q10": 80.0,
            "q50": 100.0,
            "q90": 120.0,
            "y": 105.0,
            "y_pred": 100.0,
            "private_path": "/private/model.bin",
        }
    ]

    html = build_static_dashboard(_payload(), history_predictions=history)

    assert "2026-06-30T22:00:00Z" in html
    assert "Recent model performance" in html
    assert "Official day-ahead price" in html
    assert "/private/model.bin" not in html


def test_static_dashboard_accepts_a_point_only_fallback_forecast() -> None:
    payload = _payload()
    payload["run"]["forecast_status"] = "degraded"
    payload["run"]["model"]["published_model"] = "weighted_median_v1"
    payload["predictions"] = [
        {
            "area": "DK1",
            "ds_utc": "2026-07-01T22:00:00Z",
            "model_label": "weighted_median_v1",
            "y_pred": 100.0,
        }
    ]

    html = build_static_dashboard(payload)

    assert "Fallback forecast" in html
    assert "weighted_median_v1" in html
