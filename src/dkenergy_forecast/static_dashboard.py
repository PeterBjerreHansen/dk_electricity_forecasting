from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


RUN_FIELDS = (
    "run_id",
    "run_kind",
    "delivery_date_local",
    "forecast_origin_utc",
    "information_cutoff_utc",
    "generated_at_utc",
    "forecast_status",
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
    "horizon",
    "model_label",
    "model_release_id",
    "forecast_status",
    "q10",
    "q50",
    "q90",
    "y_pred",
    "actual_price",
)


def build_static_dashboard(payload: Mapping[str, Any], *, title: str = "Danish Electricity Forecasts") -> str:
    public_payload = _public_payload(payload)
    encoded = json.dumps(public_payload, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    return (
        _TEMPLATE.replace("__TITLE_JSON__", json.dumps(title, ensure_ascii=False))
        .replace("__TITLE_HTML__", _html_text(title))
        .replace("__DATA__", encoded)
    )


def _public_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    run = payload.get("run")
    predictions = payload.get("predictions")
    if not isinstance(run, Mapping):
        raise ValueError("Dashboard payload must contain a run object")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Dashboard payload must contain at least one prediction")

    public_run = {field: run.get(field) for field in RUN_FIELDS}
    model = run.get("model")
    public_run["model"] = (
        {field: model.get(field) for field in MODEL_FIELDS}
        if isinstance(model, Mapping)
        else {}
    )
    weather = run.get("weather")
    public_run["weather"] = {
        "weather_features_exists": weather.get("weather_features_exists"),
        "weather_dataset_versions": weather.get("weather_dataset_versions"),
    } if isinstance(weather, Mapping) else {}

    public_predictions: list[dict[str, Any]] = []
    for prediction in predictions:
        if not isinstance(prediction, Mapping):
            raise ValueError("Every prediction must be an object")
        missing = sorted({"area", "ds_utc", "y_pred", "q10", "q90"} - set(prediction))
        if missing:
            raise ValueError(f"Prediction is missing required fields: {missing}")
        public_predictions.append({field: prediction.get(field) for field in PREDICTION_FIELDS})

    public_predictions.sort(key=lambda row: (str(row["area"]), int(row.get("horizon") or 0)))
    return {
        "generated_at_utc": payload.get("generated_at_utc"),
        "run": public_run,
        "predictions": public_predictions,
    }


def _html_text(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="light">
  <title>__TITLE_HTML__</title>
  <style>
    :root {
      --ink: #14213d;
      --muted: #60708f;
      --paper: #f6f4ee;
      --card: rgba(255, 255, 255, .86);
      --blue: #1768ac;
      --blue-soft: #dcecf8;
      --orange: #ef8354;
      --grid: #d9dfeb;
      --shadow: 0 18px 60px rgba(20, 33, 61, .10);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 15% 0%, #d8ecf5 0, transparent 34rem),
        radial-gradient(circle at 100% 20%, #f7dfcd 0, transparent 30rem),
        var(--paper);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { width: min(1180px, calc(100% - 32px)); margin: 0 auto; padding: 56px 0 72px; }
    .eyebrow { color: var(--blue); font-size: .78rem; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
    h1 { max-width: 760px; margin: 10px 0 12px; font-family: Georgia, serif; font-size: clamp(2.4rem, 6vw, 5.2rem); font-weight: 500; line-height: .98; letter-spacing: -.045em; }
    .lede { max-width: 740px; margin: 0; color: var(--muted); font-size: 1.08rem; line-height: 1.65; }
    .notice { margin: 24px 0; padding: 14px 18px; border: 1px solid #edb38f; border-radius: 14px; background: #fff1e8; color: #773a18; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 34px 0; }
    .metric, .panel { border: 1px solid rgba(20, 33, 61, .10); background: var(--card); box-shadow: var(--shadow); backdrop-filter: blur(12px); }
    .metric { min-height: 120px; padding: 20px; border-radius: 18px; }
    .metric-label { color: var(--muted); font-size: .75rem; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; }
    .metric-value { margin-top: 10px; font-family: Georgia, serif; font-size: 1.55rem; overflow-wrap: anywhere; }
    .panel { margin-top: 18px; padding: 24px; border-radius: 22px; }
    .panel-head { display: flex; align-items: end; justify-content: space-between; gap: 16px; margin-bottom: 18px; }
    h2 { margin: 0; font-family: Georgia, serif; font-size: 1.8rem; font-weight: 500; }
    .tabs { display: flex; gap: 8px; }
    button { border: 1px solid var(--grid); border-radius: 999px; padding: 9px 16px; background: white; color: var(--ink); font: inherit; font-weight: 700; cursor: pointer; }
    button.active { border-color: var(--blue); background: var(--blue); color: white; }
    .chart-wrap { width: 100%; overflow-x: auto; }
    svg { display: block; min-width: 760px; width: 100%; height: auto; }
    .legend { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 12px; color: var(--muted); font-size: .86rem; }
    .swatch { display: inline-block; width: 22px; height: 3px; margin-right: 7px; vertical-align: middle; border-radius: 2px; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; font-size: .9rem; }
    th, td { padding: 11px 10px; border-bottom: 1px solid var(--grid); text-align: right; white-space: nowrap; }
    th:first-child, td:first-child { text-align: left; }
    th { color: var(--muted); font-size: .72rem; letter-spacing: .07em; text-transform: uppercase; }
    footer { margin-top: 28px; color: var(--muted); font-size: .84rem; line-height: 1.6; }
    code { padding: .12rem .35rem; border-radius: 5px; background: rgba(20,33,61,.07); }
    @media (max-width: 820px) {
      main { padding-top: 34px; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .panel { padding: 18px; }
      .panel-head { align-items: start; flex-direction: column; }
    }
    @media (max-width: 480px) { .metrics { grid-template-columns: 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="eyebrow">Day-ahead model · Denmark</div>
      <h1 id="page-title">__TITLE_HTML__</h1>
      <p class="lede">A compact view of hourly DK1 and DK2 electricity-price forecasts, uncertainty intervals, and reproducible model provenance.</p>
      <div id="notice" class="notice" hidden></div>
    </header>
    <section id="metrics" class="metrics" aria-label="Forecast summary"></section>
    <section class="panel">
      <div class="panel-head">
        <div><div class="eyebrow">Forecast profile</div><h2>Hourly price path</h2></div>
        <div id="area-tabs" class="tabs" aria-label="Price area"></div>
      </div>
      <div id="chart" class="chart-wrap"></div>
      <div class="legend">
        <span><i class="swatch" style="background:#1768ac"></i>Median forecast</span>
        <span><i class="swatch" style="background:#ef8354"></i>Observed price</span>
        <span><i class="swatch" style="height:10px;background:#dcecf8"></i>10–90% interval</span>
      </div>
    </section>
    <section class="panel">
      <div class="panel-head"><div><div class="eyebrow">Details</div><h2>Hourly values</h2></div></div>
      <div class="table-wrap"><table><thead><tr><th>Hour</th><th>Median</th><th>q10</th><th>q90</th><th>Observed</th></tr></thead><tbody id="forecast-table"></tbody></table></div>
    </section>
    <footer id="provenance"></footer>
  </main>
  <script>
    const PAGE_TITLE = __TITLE_JSON__;
    const DATA = __DATA__;
    const esc = value => String(value ?? "—").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
    const finite = value => Number.isFinite(Number(value));
    const price = value => finite(value) ? new Intl.NumberFormat("da-DK", {maximumFractionDigits: 0}).format(Number(value)) : "—";
    const localHour = value => new Intl.DateTimeFormat("en-GB", {timeZone:"Europe/Copenhagen", hour:"2-digit", minute:"2-digit", hour12:false}).format(new Date(value));
    const dateLabel = value => new Intl.DateTimeFormat("en-GB", {timeZone:"Europe/Copenhagen", dateStyle:"medium"}).format(new Date(value));
    const run = DATA.run;
    const model = run.model || {};
    document.title = PAGE_TITLE;
    document.getElementById("metrics").innerHTML = [
      ["Delivery day", run.delivery_date_local || "—"],
      ["Published model", model.published_model || "—"],
      ["Forecast rows", DATA.predictions.length],
      ["Model release", model.model_release_id || "—"]
    ].map(([label, value]) => `<article class="metric"><div class="metric-label">${esc(label)}</div><div class="metric-value">${esc(value)}</div></article>`).join("");
    if (run.run_kind !== "live") {
      const notice = document.getElementById("notice");
      notice.hidden = false;
      notice.innerHTML = `<strong>Historical replay.</strong> This is a deployment demonstration for ${esc(run.delivery_date_local)} and is not the current live market forecast.`;
    }
    const areas = [...new Set(DATA.predictions.map(row => row.area))].sort();
    let selectedArea = areas[0];
    const tabs = document.getElementById("area-tabs");
    function renderTabs() {
      tabs.innerHTML = areas.map(area => `<button type="button" data-area="${esc(area)}" aria-pressed="${area === selectedArea}" class="${area === selectedArea ? "active" : ""}">${esc(area)}</button>`).join("");
      tabs.querySelectorAll("button").forEach(button => button.addEventListener("click", () => { selectedArea = button.dataset.area; render(); }));
    }
    function chartSvg(rows) {
      const width = 980, height = 330, left = 66, right = 22, top = 18, bottom = 46;
      const values = rows.flatMap(row => [row.q10, row.q90, row.y_pred, row.actual_price]).filter(finite).map(Number);
      let low = Math.min(...values), high = Math.max(...values);
      const margin = Math.max((high - low) * .12, 10); low -= margin; high += margin;
      const x = index => left + index * (width - left - right) / Math.max(rows.length - 1, 1);
      const y = value => top + (high - Number(value)) * (height - top - bottom) / Math.max(high - low, 1);
      const points = (key, source = rows) => source.filter(row => finite(row[key])).map((row, index) => `${x(rows.indexOf(row)).toFixed(1)},${y(row[key]).toFixed(1)}`).join(" ");
      const upper = rows.map((row, index) => `${x(index).toFixed(1)},${y(row.q90).toFixed(1)}`);
      const lower = rows.map((row, index) => `${x(index).toFixed(1)},${y(row.q10).toFixed(1)}`).reverse();
      const grid = Array.from({length:5}, (_, index) => {
        const value = high - index * (high - low) / 4, py = y(value);
        return `<line x1="${left}" x2="${width-right}" y1="${py}" y2="${py}" stroke="#d9dfeb"/><text x="${left-10}" y="${py+4}" text-anchor="end" fill="#60708f" font-size="12">${price(value)}</text>`;
      }).join("");
      const labels = rows.map((row, index) => index % 4 === 0 ? `<text x="${x(index)}" y="${height-16}" text-anchor="middle" fill="#60708f" font-size="12">${localHour(row.ds_utc)}</text>` : "").join("");
      return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(selectedArea)} hourly forecast chart">${grid}<polygon points="${upper.concat(lower).join(" ")}" fill="#dcecf8" opacity=".85"/><polyline points="${points("y_pred")}" fill="none" stroke="#1768ac" stroke-width="4" stroke-linejoin="round" stroke-linecap="round"/><polyline points="${points("actual_price")}" fill="none" stroke="#ef8354" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"/>${labels}<text x="18" y="${height/2}" transform="rotate(-90 18 ${height/2})" text-anchor="middle" fill="#60708f" font-size="12">DKK / MWh</text></svg>`;
    }
    function render() {
      renderTabs();
      const rows = DATA.predictions.filter(row => row.area === selectedArea).sort((a,b) => Number(a.horizon) - Number(b.horizon));
      document.getElementById("chart").innerHTML = chartSvg(rows);
      document.getElementById("forecast-table").innerHTML = rows.map(row => `<tr><td>${esc(localHour(row.ds_utc))}</td><td>${price(row.y_pred)}</td><td>${price(row.q10)}</td><td>${price(row.q90)}</td><td>${price(row.actual_price)}</td></tr>`).join("");
    }
    const origin = run.forecast_origin_utc ? `${dateLabel(run.forecast_origin_utc)} ${localHour(run.forecast_origin_utc)}` : "—";
    document.getElementById("provenance").innerHTML = `Run <code>${esc(run.run_id)}</code> · origin ${esc(origin)} Copenhagen · status ${esc(run.forecast_status)} · source commit <code>${esc((run.git_commit || "").slice(0,12))}</code>. Prices are DKK/MWh. The shaded band is the model's q10–q90 interval.`;
    render();
  </script>
</body>
</html>
'''
