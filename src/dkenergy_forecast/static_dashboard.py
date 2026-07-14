from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from dkenergy_forecast.dashboard import (
    COPENHAGEN_TIMEZONE,
    canonical_model_family,
)


COPENHAGEN = ZoneInfo(COPENHAGEN_TIMEZONE)
PRODUCTION_AREAS = ("DK1", "DK2")


RUN_FIELDS = (
    "run_id",
    "run_kind",
    "delivery_date_local",
    "forecast_origin_utc",
    "information_cutoff_utc",
    "created_at_utc",
    "generated_at_utc",
    "forecast_status",
    "status",
    "git_commit",
)
MODEL_FIELDS = (
    "published_model",
    "primary_model",
    "fallback_model",
    "model_release_id",
    "model_artifact_sha256",
)
PREDICTION_FIELDS = (
    "area",
    "ds_utc",
    "ds_local",
    "local_date",
    "horizon",
    "model_label",
    "model_release_id",
    "forecast_status",
    "q10",
    "q50",
    "q90",
    "y",
    "y_pred",
    "actual_price",
)


def build_static_dashboard(
    payload: Mapping[str, Any],
    *,
    title: str = "Danish Electricity Price Forecasts",
    history_predictions: Iterable[Mapping[str, Any]] | None = None,
) -> str:
    public_payload = _public_payload(payload, history_predictions=history_predictions)
    encoded = json.dumps(
        public_payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    encoded = (
        encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    )
    return (
        _TEMPLATE.replace("__TITLE_JSON__", json.dumps(title, ensure_ascii=False))
        .replace("__TITLE_HTML__", _html_text(title))
        .replace("__DATA__", encoded)
    )


def _public_payload(
    payload: Mapping[str, Any],
    *,
    history_predictions: Iterable[Mapping[str, Any]] | None,
) -> dict[str, Any]:
    run = payload.get("run")
    predictions = payload.get("predictions")
    if not isinstance(run, Mapping):
        raise ValueError("Dashboard payload must contain a run object")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Dashboard payload must contain at least one prediction")

    public_run = {field: _json_safe(run.get(field)) for field in RUN_FIELDS}
    model = run.get("model")
    public_run["model"] = (
        {field: _json_safe(model.get(field)) for field in MODEL_FIELDS}
        if isinstance(model, Mapping)
        else {}
    )

    public_predictions = [
        _public_prediction(row, require_interval=False) for row in predictions
    ]
    public_predictions.sort(
        key=lambda row: (str(row["area"]), int(row.get("horizon") or 0))
    )

    if history_predictions is None:
        candidate_history = (
            payload.get("history_predictions")
            or payload.get("recent_predictions")
            or []
        )
    else:
        candidate_history = list(history_predictions)
    if not isinstance(candidate_history, list):
        raise ValueError("Dashboard history must be a list of prediction objects")
    public_history = [
        _public_prediction(row, require_interval=False)
        for row in candidate_history
        if isinstance(row, Mapping) and row.get("y_pred") is not None
    ]
    public_history.sort(
        key=lambda row: (str(row["area"]), str(row["ds_utc"]), str(row["model_label"]))
    )

    outlook = _build_outlook(public_run, public_predictions, public_history)

    return {
        "generated_at_utc": _json_safe(payload.get("generated_at_utc")),
        "run": public_run,
        "predictions": public_predictions,
        "history": public_history,
        "outlook": outlook,
    }


def _public_prediction(
    prediction: Mapping[str, Any],
    *,
    require_interval: bool,
) -> dict[str, Any]:
    required = {"area", "ds_utc", "y_pred"}
    if require_interval:
        required |= {"q10", "q90"}
    missing = sorted(required - set(prediction))
    if missing:
        raise ValueError(f"Prediction is missing required fields: {missing}")
    return {field: _json_safe(prediction.get(field)) for field in PREDICTION_FIELDS}


def _build_outlook(
    run: Mapping[str, Any],
    predictions: list[dict[str, Any]],
    history: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build the validated two-day hero dataset used by the public page.

    The two sides may only be joined when the evaluated forecast is for the
    immediately preceding Danish delivery day and was produced by the same
    model release. Missing history therefore yields a truthful forecast-only
    chart instead of visually compressing an arbitrary calendar gap.
    """

    model = run.get("model")
    model = model if isinstance(model, Mapping) else {}
    published_model = str(model.get("published_model") or "")
    production_family = canonical_model_family(published_model)
    if not production_family:
        production_family = canonical_model_family(predictions[0].get("model_label"))
    release_id = _optional_text(model.get("model_release_id"))
    if release_id is None:
        raise ValueError("Dashboard run is missing model_release_id")
    delivery_date_text = _optional_text(run.get("delivery_date_local"))
    if delivery_date_text is None:
        raise ValueError("Dashboard run is missing delivery_date_local")
    try:
        forecast_date = date.fromisoformat(delivery_date_text)
    except ValueError as exc:
        raise ValueError(
            f"Dashboard run has invalid delivery_date_local: {delivery_date_text!r}"
        ) from exc

    production_predictions = [
        row
        for row in predictions
        if canonical_model_family(row.get("model_label")) == production_family
    ]
    if not production_predictions:
        raise ValueError(
            f"Dashboard contains no predictions for production model {published_model!r}"
        )

    areas = sorted({str(row.get("area")) for row in production_predictions})
    if set(areas) != set(PRODUCTION_AREAS):
        missing = sorted(set(PRODUCTION_AREAS) - set(areas))
        extra = sorted(set(areas) - set(PRODUCTION_AREAS))
        raise ValueError(
            "Dashboard production forecast must contain exactly DK1 and DK2: "
            f"missing={missing}, extra={extra}"
        )
    outlook: dict[str, dict[str, Any]] = {}
    previous_date = forecast_date - timedelta(days=1)
    for area in areas:
        forecast = sorted(
            [row for row in production_predictions if str(row.get("area")) == area],
            key=lambda row: _utc_timestamp(row.get("ds_utc")),
        )
        forecast_interval = _validate_delivery_grid(
            forecast,
            delivery_date=forecast_date,
            area=area,
            segment="forecast",
            require_actual=False,
        )
        _validate_release(
            forecast,
            expected_release_id=release_id,
            area=area,
            segment="forecast",
        )

        evaluated_candidates = [
            row
            for row in history
            if str(row.get("area")) == area
            and canonical_model_family(row.get("model_label")) == production_family
            and _delivery_date(row) == previous_date
            and _matches_release(row, release_id=release_id)
            and _finite(_actual(row))
            and _finite(row.get("y_pred"))
        ]
        evaluated = sorted(
            evaluated_candidates,
            key=lambda row: _utc_timestamp(row.get("ds_utc")),
        )
        evaluated_interval = False
        if evaluated:
            evaluated_interval = _validate_delivery_grid(
                evaluated,
                delivery_date=previous_date,
                area=area,
                segment="evaluated forecast",
                require_actual=True,
            )
            _validate_release(
                evaluated,
                expected_release_id=release_id,
                area=area,
                segment="evaluated forecast",
            )

        outlook[area] = {
            "forecast_date": forecast_date.isoformat(),
            "evaluated_date": previous_date.isoformat() if evaluated else None,
            "forecast": forecast,
            "evaluated": evaluated,
            "show_interval": forecast_interval
            and (not evaluated or evaluated_interval),
        }
    return outlook


def _validate_delivery_grid(
    rows: list[dict[str, Any]],
    *,
    delivery_date: date,
    area: str,
    segment: str,
    require_actual: bool,
) -> bool:
    if not rows:
        raise ValueError(f"Dashboard {segment} is empty for {area}")

    observed = [_utc_timestamp(row.get("ds_utc")) for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError(
            f"Dashboard {segment} contains duplicate timestamps for {area}"
        )
    expected = _expected_delivery_hours(delivery_date)
    if set(observed) != set(expected):
        missing = len(set(expected) - set(observed))
        extra = len(set(observed) - set(expected))
        raise ValueError(
            f"Dashboard {segment} must contain the complete {delivery_date} "
            f"delivery grid for {area}: expected {len(expected)} hourly rows, "
            f"got {len(observed)} (missing={missing}, extra={extra})"
        )

    for row in rows:
        observed_date = _delivery_date(row)
        if observed_date != delivery_date:
            raise ValueError(
                f"Dashboard {segment} date mismatch for {area}: "
                f"expected {delivery_date}, got {observed_date}"
            )
        if not _finite(row.get("y_pred")):
            raise ValueError(
                f"Dashboard {segment} contains a non-finite prediction for {area}"
            )
        if require_actual and not _finite(_actual(row)):
            raise ValueError(
                f"Dashboard {segment} contains a missing actual price for {area}"
            )

    return _validate_intervals(rows, area=area, segment=segment)


def _validate_intervals(
    rows: list[dict[str, Any]],
    *,
    area: str,
    segment: str,
) -> bool:
    interval_fields = ("q10", "q50", "q90")
    complete = [
        all(_finite(row.get(field)) for field in interval_fields) for row in rows
    ]
    present = [
        any(_finite(row.get(field)) for field in interval_fields) for row in rows
    ]
    if any(present) and not all(complete):
        raise ValueError(
            f"Dashboard {segment} intervals must be complete for every row in {area}"
        )
    if not any(present):
        return False

    for row in rows:
        q10 = float(row["q10"])
        q50 = float(row["q50"])
        q90 = float(row["q90"])
        y_pred = float(row["y_pred"])
        if not q10 <= q50 <= q90 or not q10 <= y_pred <= q90:
            raise ValueError(
                f"Dashboard {segment} contains unordered intervals for {area}"
            )
    return True


def _validate_release(
    rows: list[dict[str, Any]],
    *,
    expected_release_id: str,
    area: str,
    segment: str,
) -> None:
    observed = {_optional_text(row.get("model_release_id")) for row in rows}
    if observed != {expected_release_id}:
        raise ValueError(
            f"Dashboard {segment} release mismatch for {area}: "
            f"expected {expected_release_id!r}, got {sorted(str(value) for value in observed)!r}"
        )


def _matches_release(
    row: Mapping[str, Any],
    *,
    release_id: str,
) -> bool:
    return _optional_text(row.get("model_release_id")) == release_id


def _expected_delivery_hours(delivery_date: date) -> list[datetime]:
    start = datetime.combine(delivery_date, time.min, tzinfo=COPENHAGEN).astimezone(
        timezone.utc
    )
    end = datetime.combine(
        delivery_date + timedelta(days=1),
        time.min,
        tzinfo=COPENHAGEN,
    ).astimezone(timezone.utc)
    count = int((end - start).total_seconds() // 3600)
    return [start + timedelta(hours=hour) for hour in range(count)]


def _utc_timestamp(value: object) -> datetime:
    text = _optional_text(value)
    if text is None:
        raise ValueError("Dashboard prediction is missing ds_utc")
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"Dashboard prediction has invalid ds_utc: {text!r}") from exc
    if parsed.tzinfo is None:
        raise ValueError(
            f"Dashboard prediction timestamp must be timezone-aware: {text!r}"
        )
    return parsed.astimezone(timezone.utc)


def _delivery_date(row: Mapping[str, Any]) -> date:
    timestamp_date = _utc_timestamp(row.get("ds_utc")).astimezone(COPENHAGEN).date()
    local_date = _optional_text(row.get("local_date"))
    if local_date is not None:
        try:
            declared = date.fromisoformat(local_date[:10])
        except ValueError as exc:
            raise ValueError(
                f"Dashboard prediction has invalid local_date: {local_date!r}"
            ) from exc
        if declared != timestamp_date:
            raise ValueError(
                "Dashboard prediction local_date does not match ds_utc: "
                f"{declared} != {timestamp_date}"
            )
    return timestamp_date


def _actual(row: Mapping[str, Any]) -> object:
    return row.get("y") if _finite(row.get("y")) else row.get("actual_price")


def _finite(value: object) -> bool:
    if value is None or value == "":
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _html_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <meta name="description" content="Daily day-ahead electricity price forecasts for the Danish DK1 and DK2 bidding zones.">
  <title>__TITLE_HTML__</title>
  <style>
    :root {
      --ink: #172033;
      --muted: #667085;
      --paper: #f7f8fa;
      --card: #ffffff;
      --actual: #172b4d;
      --previous-forecast: #d97706;
      --previous-band: #f6c679;
      --new-forecast: #0f766e;
      --new-band: #5eead4;
      --forecast: var(--previous-forecast);
      --grid: #e2e7ef;
      --line: #d4dae4;
      --shadow: 0 14px 40px rgba(23, 32, 51, .07);
    }
    * { box-sizing: border-box; }
    body { margin: 0; color: var(--ink); background: var(--paper); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { width: min(1160px, calc(100% - 32px)); margin: 0 auto; padding: 48px 0 72px; }
    header { margin-bottom: 28px; }
    h1 { margin: 0 0 8px; font-size: clamp(2.25rem, 5vw, 4rem); line-height: 1.05; letter-spacing: -.045em; }
    h2 { margin: 0; font-size: 1.55rem; letter-spacing: -.025em; }
    h3 { margin: 0; font-size: 1.12rem; }
    .lede, .muted { color: var(--muted); }
    .lede { margin: 0; font-size: 1.05rem; line-height: 1.6; }
    .panel { border: 1px solid var(--line); background: var(--card); box-shadow: var(--shadow); }
    .panel { margin-top: 18px; padding: 24px; border-radius: 18px; }
    .panel-head { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 18px; }
    .outlook-context { display: flex; flex-wrap: wrap; justify-content: flex-end; gap: 8px 22px; color: var(--muted); font-size: .78rem; }
    .outlook-context strong { display: block; margin-top: 3px; color: var(--ink); font-size: .92rem; }
    .section-head { margin: 46px 0 16px; }
    .tabs { display: flex; gap: 8px; }
    button { border: 1px solid var(--line); border-radius: 999px; padding: 8px 16px; background: white; color: var(--ink); font: inherit; font-weight: 700; cursor: pointer; }
    button.active { border-color: var(--ink); background: var(--ink); color: white; }
    .notice { margin: 18px 0; padding: 13px 16px; border: 1px solid #edb38f; border-radius: 12px; background: #fff5ec; color: #773a18; }
    .chart-wrap { width: 100%; overflow-x: auto; }
    svg { display: block; min-width: 760px; width: 100%; height: auto; }
    .legend { display: flex; flex-wrap: wrap; gap: 18px; margin-top: 12px; color: var(--muted); font-size: .84rem; }
    .swatch { display: inline-block; width: 22px; height: 3px; margin-right: 7px; vertical-align: middle; border-radius: 2px; }
    .performance { display: grid; gap: 18px; }
    .performance .panel { margin-top: 0; }
    .model-metrics { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; margin: 16px 0 12px; }
    .mini-metric { padding: 10px 12px; border-radius: 10px; background: #f6f7f9; }
    .mini-label { color: var(--muted); font-size: .7rem; font-weight: 700; }
    .mini-value { margin-top: 4px; font-size: 1.08rem; font-weight: 700; }
    details { margin-top: 20px; }
    summary { cursor: pointer; font-weight: 700; }
    .table-wrap { margin-top: 14px; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: .88rem; }
    th, td { padding: 10px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-size: .7rem; letter-spacing: .05em; text-transform: uppercase; }
    footer { margin-top: 28px; color: var(--muted); font-size: .82rem; line-height: 1.6; }
    footer a { color: var(--ink); text-decoration-color: var(--line); text-underline-offset: 3px; }
    footer a:hover { text-decoration-color: var(--ink); }
    code { padding: .1rem .3rem; border-radius: 5px; background: #eef1f5; }
    @media (max-width: 820px) {
      main { padding-top: 30px; }
      .model-metrics { grid-template-columns: repeat(3, 1fr); }
      .panel { padding: 18px; }
      .panel-head { align-items: flex-start; flex-direction: column; }
      .outlook-context { justify-content: flex-start; }
    }
    @media (max-width: 520px) { .model-metrics { grid-template-columns: 1fr 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>__TITLE_HTML__</h1>
      <p class="lede">Electricity is traded one day ahead, with a separate market price for every delivery hour. Each morning, before the day-ahead auction, this page produces and displays a new forecast for tomorrow and logs performance over the past 30 days for a battery of forecasting models.</p>
      <div id="notice" class="notice" hidden></div>
    </header>
    <div id="area-tabs" class="tabs" aria-label="Price area"></div>
    <section class="panel">
      <div class="panel-head">
        <h2 id="outlook-title">Latest outlook</h2>
        <div class="outlook-context" aria-label="Forecast context">
          <span>Forecast day<strong id="outlook-date"></strong></span>
          <span>Production model<strong id="outlook-model"></strong></span>
        </div>
      </div>
      <div id="hero-chart" class="chart-wrap"></div>
      <div class="legend">
        <span><i class="swatch" style="background:var(--actual)"></i>Official day-ahead price</span>
        <span><i class="swatch" style="background:var(--previous-forecast)"></i>Previous forecast · today</span>
        <span><i class="swatch" style="background:var(--new-forecast)"></i>New forecast · tomorrow</span>
        <span id="interval-legend"><i class="swatch" style="height:10px;background:linear-gradient(90deg,var(--previous-band) 50%,var(--new-band) 50%)"></i>10–90% intervals</span>
      </div>
    </section>
    <div class="section-head"><h2>Recent model performance</h2></div>
    <section id="performance" class="performance"></section>
    <details class="panel">
      <summary>Forecast values and run metadata</summary>
      <div class="table-wrap"><table><thead><tr><th>Local hour</th><th>Forecast</th><th>p10</th><th>p50</th><th>p90</th></tr></thead><tbody id="forecast-table"></tbody></table></div>
      <footer id="provenance"></footer>
    </details>
    <footer>Source code, methodology, and model documentation are available on <a href="https://github.com/PeterBjerreHansen/dk_electricity_forecasting">GitHub</a>.</footer>
  </main>
  <script>
    const PAGE_TITLE = __TITLE_JSON__;
    const DATA = __DATA__;
    const esc = value => String(value ?? "—").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
    const finite = value => value !== null && value !== "" && Number.isFinite(Number(value));
    const number = (value, digits=0) => finite(value) ? new Intl.NumberFormat("en-GB", {minimumFractionDigits:digits, maximumFractionDigits:digits}).format(Number(value)) : "—";
    const localHour = value => new Intl.DateTimeFormat("en-GB", {timeZone:"Europe/Copenhagen", weekday:"short", day:"2-digit", month:"short", hour:"2-digit", minute:"2-digit", hour12:false}).format(new Date(value));
    const shortHour = value => new Intl.DateTimeFormat("en-GB", {timeZone:"Europe/Copenhagen", hour:"2-digit", minute:"2-digit", hour12:false}).format(new Date(value));
    const dateLabel = value => value ? new Intl.DateTimeFormat("en-GB", {timeZone:"Europe/Copenhagen", day:"2-digit", month:"short", year:"numeric"}).format(new Date(value)) : "—";
    const deliveryLabel = value => value ? new Intl.DateTimeFormat("en-GB", {timeZone:"UTC", day:"2-digit", month:"short", year:"numeric"}).format(new Date(`${value}T12:00:00Z`)) : "—";
    const markerDateTime = value => {
      if (!value) return "—";
      const parts = Object.fromEntries(new Intl.DateTimeFormat("en-GB", {timeZone:"Europe/Copenhagen", weekday:"short", day:"2-digit", month:"short", year:"numeric", hour:"2-digit", minute:"2-digit", hour12:false}).formatToParts(new Date(value)).map(part => [part.type, part.value]));
      return `${parts.hour}:${parts.minute} ${parts.weekday} ${parts.day} ${parts.month} ${parts.year}`;
    };
    const deliveryDate = row => {
      if (row.local_date) return String(row.local_date);
      const parts = Object.fromEntries(new Intl.DateTimeFormat("en-GB", {timeZone:"Europe/Copenhagen", year:"numeric", month:"2-digit", day:"2-digit"}).formatToParts(new Date(row.ds_utc)).map(part => [part.type, part.value]));
      return `${parts.year}-${parts.month}-${parts.day}`;
    };
    const actual = row => finite(row.y) ? Number(row.y) : finite(row.actual_price) ? Number(row.actual_price) : null;
    const family = label => {
      const value = String(label || "").toLowerCase();
      if (value.includes("chronos")) return "chronos";
      if (value.includes("weighted_median") || value.includes("median_weekday")) return "weighted_median";
      if (value.includes("rolling_median")) return "rolling_median";
      if (value === "same_hour_last_week" || value.includes("last_week")) return "last_week";
      return value;
    };
    const modelNames = {chronos:"Chronos 2 LoRA Weather", weighted_median:"Weighted Rolling Median", rolling_median:"Rolling Median", last_week:"Last Week Baseline"};
    const modelOrder = {chronos:0, weighted_median:1, rolling_median:2, last_week:3};
    const run = DATA.run || {};
    const model = run.model || {};
    const inferredProductionLabel = DATA.predictions.some(row => family(row.model_label) === "chronos") ? "chronos" : DATA.predictions[0]?.model_label;
    const productionFamily = family(model.published_model || inferredProductionLabel);
    const productionName = modelNames[productionFamily] || model.published_model || "Production model";
    document.title = PAGE_TITLE;
    if (run.run_kind === "replay") {
      const notice = document.getElementById("notice");
      notice.hidden = false;
      notice.innerHTML = `<strong>Historical replay.</strong> This page demonstrates the deployment layout and is not the current live market forecast.`;
    } else if (run.forecast_status === "degraded") {
      const notice = document.getElementById("notice");
      notice.hidden = false;
      notice.innerHTML = `<strong>Fallback forecast.</strong> The primary Chronos run failed, so this outlook uses ${esc(productionName)}.`;
    }
    const areas = [...new Set(DATA.predictions.map(row => row.area))].sort();
    let selectedArea = areas[0];
    const tabs = document.getElementById("area-tabs");
    function renderTabs() {
      tabs.innerHTML = areas.map(area => `<button type="button" data-area="${esc(area)}" aria-pressed="${area === selectedArea}" class="${area === selectedArea ? "active" : ""}">${esc(area)}</button>`).join("");
      tabs.querySelectorAll("button").forEach(button => button.addEventListener("click", () => { selectedArea = button.dataset.area; render(); }));
    }

    function heroData() {
      const outlook = DATA.outlook?.[selectedArea];
      if (!outlook) return {evaluated:[], forecast:[], forecastDate:null, showInterval:false};
      return {
        evaluated: outlook.evaluated || [],
        forecast: outlook.forecast || [],
        forecastDate: outlook.forecast_date || null,
        showInterval: Boolean(outlook.show_interval),
      };
    }

    function niceStep(span, targetIntervals=6) {
      const rough = Math.max(span, 1) / targetIntervals;
      const power = 10 ** Math.floor(Math.log10(rough));
      const ratio = rough / power;
      const multiple = ratio <= 1 ? 1 : ratio <= 2 ? 2 : ratio <= 2.5 ? 2.5 : ratio <= 5 ? 5 : 10;
      return multiple * power;
    }

    function scales(rows, valueKeys, width, height, padding) {
      const values = rows.flatMap(row => valueKeys.map(key => key === "actual" ? actual(row) : row[key])).filter(finite).map(Number);
      const dataLow = Math.min(...values), dataHigh = Math.max(...values);
      let low = Math.min(dataLow, 0), high = Math.max(dataHigh, 0);
      const margin = Math.max((high - low) * .05, 10);
      const step = niceStep(high - low + 2 * margin);
      low = dataLow >= 0 ? 0 : Math.floor((low - margin) / step) * step;
      high = dataHigh <= 0 ? 0 : Math.ceil((high + margin) / step) * step;
      const ticks = Array.from({length:Math.round((high-low)/step)+1}, (_,index) => low + index * step);
      return {
        x: index => padding.left + index * (width - padding.left - padding.right) / Math.max(rows.length - 1, 1),
        y: value => padding.top + (high - Number(value)) * (height - padding.top - padding.bottom) / Math.max(high - low, 1),
        low, high, ticks,
      };
    }

    function polyline(rows, key, x, y) {
      return rows.map((row, index) => ({row,index})).filter(item => finite(key === "actual" ? actual(item.row) : item.row[key])).map(item => `${x(item.index).toFixed(1)},${y(key === "actual" ? actual(item.row) : item.row[key]).toFixed(1)}`).join(" ");
    }

    function gridSvg(scale, width, height, padding) {
      return [...scale.ticks].reverse().map(value => {
        const py = scale.y(value);
        return `<line x1="${padding.left}" x2="${width-padding.right}" y1="${py}" y2="${py}" stroke="#e2e7ef"/><text x="${padding.left-10}" y="${py+4}" text-anchor="end" fill="#667085" font-size="12">${number(value)}</text>`;
      }).join("");
    }

    function intervalBand(rows, offset, scale, fill) {
      if (!rows.length) return "";
      const upper = rows.map((row,index) => `${scale.x(offset+index)},${scale.y(row.q90)}`);
      const lower = rows.map((row,index) => `${scale.x(offset+index)},${scale.y(row.q10)}`).reverse();
      return `<polygon points="${upper.concat(lower).join(" ")}" fill="${fill}" opacity=".35"/>`;
    }

    function shiftedPolyline(rows, key, offset, scale) {
      return rows.map((row,index) => ({row,index:offset+index}))
        .filter(item => finite(key === "actual" ? actual(item.row) : item.row[key]))
        .map(item => `${scale.x(item.index).toFixed(1)},${scale.y(key === "actual" ? actual(item.row) : item.row[key]).toFixed(1)}`)
        .join(" ");
    }

    function heroSvg(evaluated, forecast, showInterval) {
      const rows = [...evaluated, ...forecast];
      if (!rows.length) return `<p class="muted">No production forecast is available.</p>`;
      const width = 1040, height = 420, padding = {left:70,right:24,top:24,bottom:66};
      const scale = scales(rows, showInterval ? ["actual","y_pred","q10","q90"] : ["actual","y_pred"], width, height, padding);
      const newForecast = evaluated.length ? [evaluated[evaluated.length-1], ...forecast] : forecast;
      const newForecastOffset = Math.max(evaluated.length-1, 0);
      const bands = showInterval ? intervalBand(evaluated,0,scale,"#f6c679") + intervalBand(newForecast,newForecastOffset,scale,"#5eead4") : "";
      const labels = rows.map((row,index) => index % 6 === 0 ? `<text x="${scale.x(index)}" y="${height-20}" text-anchor="middle" transform="rotate(-30 ${scale.x(index)} ${height-20})" fill="#667085" font-size="11">${esc(localHour(row.ds_utc))}</text>` : "").join("");
      const boundaryX = evaluated.length ? scale.x(evaluated.length-1) : null;
      const forecastBegins = forecast[0]?.ds_utc;
      const separator = boundaryX === null ? "" : `<line x1="${boundaryX}" x2="${boundaryX}" y1="${padding.top}" y2="${height-padding.bottom}" stroke="#7d8799" stroke-width="1.5" stroke-dasharray="6 5"/><text x="${boundaryX+8}" y="${padding.top+14}" fill="#667085" font-size="12" font-weight="700">New forecast begins</text><text x="${boundaryX+8}" y="${padding.top+30}" fill="#7d8799" font-size="10">${esc(markerDateTime(forecastBegins))}</text>`;
      const originIndex = run.forecast_origin_utc ? (new Date(run.forecast_origin_utc) - new Date(rows[0].ds_utc)) / 3600000 : null;
      const originX = Number.isFinite(originIndex) && originIndex >= 0 && originIndex <= evaluated.length-1 ? scale.x(originIndex) : null;
      const originMarker = originX === null ? "" : `<line x1="${originX}" x2="${originX}" y1="${padding.top}" y2="${height-padding.bottom}" stroke="#7d8799" stroke-width="1.5" stroke-dasharray="6 5"/><text x="${originX+8}" y="${padding.top+14}" fill="#667085" font-size="12" font-weight="700">New forecast made</text><text x="${originX+8}" y="${padding.top+30}" fill="#7d8799" font-size="10">${esc(markerDateTime(run.forecast_origin_utc))}</text>`;
      document.getElementById("interval-legend").hidden = !showInterval;
      return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(selectedArea)} evaluated and forecast prices">${gridSvg(scale,width,height,padding)}${bands}<polyline points="${shiftedPolyline(evaluated,"y_pred",0,scale)}" fill="none" stroke="#d97706" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/><polyline points="${shiftedPolyline(newForecast,"y_pred",newForecastOffset,scale)}" fill="none" stroke="#0f766e" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/><polyline points="${shiftedPolyline(evaluated,"actual",0,scale)}" fill="none" stroke="#172b4d" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/>${originMarker}${separator}${labels}<text x="18" y="${height/2}" transform="rotate(-90 18 ${height/2})" text-anchor="middle" fill="#667085" font-size="12">DKK / MWh</text></svg>`;
    }

    function performanceSvg(rows) {
      const width = 1040, height = 270, padding = {left:70,right:24,top:16,bottom:48};
      const scale = scales(rows, ["actual","y_pred"], width, height, padding);
      const labels = rows.map((row,index) => index % Math.max(Math.floor(rows.length / 6), 1) === 0 ? `<text x="${scale.x(index)}" y="${height-17}" text-anchor="middle" fill="#667085" font-size="11">${esc(deliveryDate(row).slice(5))}</text>` : "").join("");
      return `<svg viewBox="0 0 ${width} ${height}" role="img">${gridSvg(scale,width,height,padding)}<polyline points="${polyline(rows,"actual",scale.x,scale.y)}" fill="none" stroke="#172b4d" stroke-width="2"/><polyline points="${polyline(rows,"y_pred",scale.x,scale.y)}" fill="none" stroke="#d97706" stroke-width="2"/>${labels}</svg>`;
    }

    function modelHistory() {
      const areaRows = DATA.history.filter(row => row.area === selectedArea && finite(actual(row)) && finite(row.y_pred));
      const families = [...new Set(areaRows.map(row => family(row.model_label)))].sort((a,b) => (modelOrder[a] ?? 99) - (modelOrder[b] ?? 99));
      const container = document.getElementById("performance");
      if (!families.length) { container.innerHTML = `<p class="muted">No evaluated history is available.</p>`; return; }
      container.innerHTML = families.map(modelFamily => {
        let familyRows = areaRows.filter(row => family(row.model_label) === modelFamily);
        if (modelFamily === productionFamily && model.model_release_id) {
          familyRows = familyRows.filter(row => row.model_release_id === model.model_release_id);
        }
        if (!familyRows.length) return "";
        const dates = [...new Set(familyRows.map(deliveryDate))].sort().slice(-30);
        const rows = familyRows.filter(row => dates.includes(deliveryDate(row))).sort((a,b) => new Date(a.ds_utc) - new Date(b.ds_utc));
        const errors = rows.map(row => Number(row.y_pred) - actual(row));
        const mae = errors.reduce((sum,value) => sum + Math.abs(value), 0) / errors.length;
        const rmse = Math.sqrt(errors.reduce((sum,value) => sum + value * value, 0) / errors.length);
        const bias = errors.reduce((sum,value) => sum + value, 0) / errors.length;
        const intervalRows = rows.filter(row => finite(row.q10) && finite(row.q90));
        const coverage = intervalRows.length ? intervalRows.filter(row => actual(row) >= Number(row.q10) && actual(row) <= Number(row.q90)).length / intervalRows.length : null;
        const metrics = [["Days",dates.length],["MAE",number(mae,1)],["RMSE",number(rmse,1)],["Bias",number(bias,1)],["80% coverage",coverage === null ? "—" : `${number(coverage*100)}%`]];
        return `<article class="panel"><h3>${esc(modelNames[modelFamily] || modelFamily)}</h3><div class="model-metrics">${metrics.map(([label,value]) => `<div class="mini-metric"><div class="mini-label">${esc(label)}</div><div class="mini-value">${esc(value)}</div></div>`).join("")}</div><div class="chart-wrap">${performanceSvg(rows)}</div><div class="legend"><span><i class="swatch" style="background:var(--actual)"></i>Official day-ahead price</span><span><i class="swatch" style="background:var(--forecast)"></i>Predicted price</span></div></article>`;
      }).join("");
    }

    function render() {
      renderTabs();
      const {evaluated, forecast, forecastDate, showInterval} = heroData();
      document.getElementById("outlook-title").textContent = `Latest outlook · ${selectedArea}`;
      document.getElementById("outlook-date").textContent = deliveryLabel(forecastDate);
      document.getElementById("outlook-model").textContent = productionName;
      document.getElementById("hero-chart").innerHTML = heroSvg(evaluated, forecast, showInterval);
      modelHistory();
      document.getElementById("forecast-table").innerHTML = forecast.map(row => `<tr><td>${esc(localHour(row.ds_utc))}</td><td>${number(row.y_pred)}</td><td>${number(row.q10)}</td><td>${number(row.q50)}</td><td>${number(row.q90)}</td></tr>`).join("");
      const origin = run.forecast_origin_utc ? `${dateLabel(run.forecast_origin_utc)} ${shortHour(run.forecast_origin_utc)}` : "—";
      document.getElementById("provenance").innerHTML = `Run <code>${esc(run.run_id)}</code> · origin ${esc(origin)} Copenhagen · status ${esc(run.forecast_status || run.status)} · source commit <code>${esc((run.git_commit || "").slice(0,12))}</code>. Prices are DKK/MWh.`;
    }
    render();
  </script>
</body>
</html>
"""
