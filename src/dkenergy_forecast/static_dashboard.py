from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any


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
    title: str = "Danish Electricity Forecasts",
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

    return {
        "generated_at_utc": _json_safe(payload.get("generated_at_utc")),
        "run": public_run,
        "predictions": public_predictions,
        "history": public_history,
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
  <title>__TITLE_HTML__</title>
  <style>
    :root {
      --ink: #172033;
      --muted: #667085;
      --paper: #f7f8fa;
      --card: #ffffff;
      --actual: #172b4d;
      --forecast: #d97706;
      --band: #f6c679;
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
    .lede { margin: 0; font-size: 1rem; }
    .metrics { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; margin: 24px 0; }
    .metric, .panel { border: 1px solid var(--line); background: var(--card); box-shadow: var(--shadow); }
    .metric { min-height: 104px; padding: 18px; border-radius: 14px; }
    .metric.compact { min-height: 88px; box-shadow: none; }
    .metric-label { color: var(--muted); font-size: .75rem; font-weight: 700; letter-spacing: .04em; }
    .metric-value { margin-top: 9px; font-size: 1.75rem; line-height: 1.1; letter-spacing: -.035em; }
    .panel { margin-top: 18px; padding: 24px; border-radius: 18px; }
    .panel-head { display: flex; align-items: center; justify-content: space-between; gap: 18px; margin-bottom: 18px; }
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
    code { padding: .1rem .3rem; border-radius: 5px; background: #eef1f5; }
    @media (max-width: 820px) {
      main { padding-top: 30px; }
      .metrics { grid-template-columns: repeat(2, 1fr); }
      .model-metrics { grid-template-columns: repeat(3, 1fr); }
      .panel { padding: 18px; }
      .panel-head { align-items: flex-start; flex-direction: column; }
    }
    @media (max-width: 520px) { .metrics, .model-metrics { grid-template-columns: 1fr 1fr; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>__TITLE_HTML__</h1>
      <p class="lede">Day-ahead electricity-price forecasts for DK1 and DK2</p>
      <div id="notice" class="notice" hidden></div>
    </header>
    <div id="area-tabs" class="tabs" aria-label="Price area"></div>
    <section id="top-metrics" class="metrics" aria-label="Forecast status"></section>
    <section class="panel">
      <div class="panel-head"><h2 id="outlook-title">Latest outlook</h2></div>
      <div id="hero-chart" class="chart-wrap"></div>
      <div class="legend">
        <span><i class="swatch" style="background:var(--actual)"></i>Official day-ahead price</span>
        <span id="forecast-legend"><i class="swatch" style="background:var(--forecast)"></i>Production forecast</span>
        <span id="interval-legend"><i class="swatch" style="height:10px;background:var(--band)"></i>10–90% interval</span>
      </div>
      <div id="forecast-metrics" class="metrics"></div>
    </section>
    <div class="section-head"><h2>Recent model performance</h2></div>
    <section id="performance" class="performance"></section>
    <details class="panel">
      <summary>Forecast values and run metadata</summary>
      <div class="table-wrap"><table><thead><tr><th>Local hour</th><th>Forecast</th><th>p10</th><th>p50</th><th>p90</th></tr></thead><tbody id="forecast-table"></tbody></table></div>
      <footer id="provenance"></footer>
    </details>
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
    const utcDateLabel = value => value ? new Intl.DateTimeFormat("en-GB", {timeZone:"UTC", day:"2-digit", month:"short"}).format(new Date(value)) : "—";
    const deliveryLabel = value => value ? new Intl.DateTimeFormat("en-GB", {timeZone:"UTC", day:"2-digit", month:"short", year:"numeric"}).format(new Date(`${value}T12:00:00Z`)) : "—";
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
    document.getElementById("forecast-legend").innerHTML = `<i class="swatch" style="background:var(--forecast)"></i>${esc(productionName)} forecast`;

    const areas = [...new Set(DATA.predictions.map(row => row.area))].sort();
    let selectedArea = areas[0];
    const tabs = document.getElementById("area-tabs");
    function renderTabs() {
      tabs.innerHTML = areas.map(area => `<button type="button" data-area="${esc(area)}" aria-pressed="${area === selectedArea}" class="${area === selectedArea ? "active" : ""}">${esc(area)}</button>`).join("");
      tabs.querySelectorAll("button").forEach(button => button.addEventListener("click", () => { selectedArea = button.dataset.area; render(); }));
    }

    function heroData() {
      const allForecasts = DATA.predictions.filter(row => row.area === selectedArea && family(row.model_label) === productionFamily && finite(row.y_pred));
      const forecastDates = [...new Set(allForecasts.map(deliveryDate))].sort();
      const forecastDate = forecastDates.at(-1);
      const forecast = allForecasts.filter(row => deliveryDate(row) === forecastDate).sort((a,b) => new Date(a.ds_utc) - new Date(b.ds_utc));
      const allHistory = DATA.history.filter(row => row.area === selectedArea && family(row.model_label) === productionFamily && finite(actual(row)) && finite(row.y_pred) && deliveryDate(row) < forecastDate);
      const historyDates = [...new Set(allHistory.map(deliveryDate))].sort();
      const evaluatedDate = historyDates.at(-1);
      const evaluated = allHistory.filter(row => deliveryDate(row) === evaluatedDate).sort((a,b) => new Date(a.ds_utc) - new Date(b.ds_utc));
      return {evaluated, forecast, evaluatedDate, forecastDate};
    }

    function scales(rows, valueKeys, width, height, padding) {
      const values = rows.flatMap(row => valueKeys.map(key => key === "actual" ? actual(row) : row[key])).filter(finite).map(Number);
      let low = Math.min(...values), high = Math.max(...values);
      const margin = Math.max((high - low) * .1, 10); low -= margin; high += margin;
      return {
        x: index => padding.left + index * (width - padding.left - padding.right) / Math.max(rows.length - 1, 1),
        y: value => padding.top + (high - Number(value)) * (height - padding.top - padding.bottom) / Math.max(high - low, 1),
        low, high,
      };
    }

    function polyline(rows, key, x, y) {
      return rows.map((row, index) => ({row,index})).filter(item => finite(key === "actual" ? actual(item.row) : item.row[key])).map(item => `${x(item.index).toFixed(1)},${y(key === "actual" ? actual(item.row) : item.row[key]).toFixed(1)}`).join(" ");
    }

    function gridSvg(scale, width, height, padding) {
      return Array.from({length:5}, (_, index) => {
        const value = scale.high - index * (scale.high - scale.low) / 4, py = scale.y(value);
        return `<line x1="${padding.left}" x2="${width-padding.right}" y1="${py}" y2="${py}" stroke="#e2e7ef"/><text x="${padding.left-10}" y="${py+4}" text-anchor="end" fill="#667085" font-size="12">${number(value)}</text>`;
      }).join("");
    }

    function heroSvg(evaluated, forecast) {
      const rows = [...evaluated, ...forecast];
      if (!rows.length) return `<p class="muted">No production forecast is available.</p>`;
      const width = 1040, height = 420, padding = {left:70,right:24,top:24,bottom:66};
      const completeInterval = evaluated.length > 0 && forecast.length > 0 && rows.every(row => finite(row.q10) && finite(row.q90));
      const scale = scales(rows, completeInterval ? ["actual","y_pred","q10","q90"] : ["actual","y_pred"], width, height, padding);
      const band = completeInterval ? `<polygon points="${rows.map((row,index) => `${scale.x(index)},${scale.y(row.q90)}`).concat(rows.map((row,index) => `${scale.x(index)},${scale.y(row.q10)}`).reverse()).join(" ")}" fill="#f6c679" opacity=".35"/>` : "";
      const labels = rows.map((row,index) => index % 6 === 0 ? `<text x="${scale.x(index)}" y="${height-20}" text-anchor="middle" transform="rotate(-30 ${scale.x(index)} ${height-20})" fill="#667085" font-size="11">${esc(localHour(row.ds_utc))}</text>` : "").join("");
      const boundaryX = scale.x(evaluated.length);
      document.getElementById("interval-legend").hidden = !completeInterval;
      return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(selectedArea)} evaluated and forecast prices">${gridSvg(scale,width,height,padding)}${band}<polyline points="${polyline(rows,"y_pred",scale.x,scale.y)}" fill="none" stroke="#d97706" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/><polyline points="${polyline(rows,"actual",scale.x,scale.y)}" fill="none" stroke="#172b4d" stroke-width="3" stroke-linejoin="round" stroke-linecap="round"/><line x1="${boundaryX}" x2="${boundaryX}" y1="${padding.top}" y2="${height-padding.bottom}" stroke="#7d8799" stroke-width="1.5" stroke-dasharray="6 5"/><text x="${boundaryX+8}" y="${padding.top+15}" fill="#667085" font-size="12">Forecast begins</text>${labels}<text x="18" y="${height/2}" transform="rotate(-90 18 ${height/2})" text-anchor="middle" fill="#667085" font-size="12">DKK / MWh</text></svg>`;
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
        const familyRows = areaRows.filter(row => family(row.model_label) === modelFamily);
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

    function metricCards(target, values, compact=false) {
      document.getElementById(target).innerHTML = values.map(([label,value]) => `<article class="metric ${compact ? "compact" : ""}"><div class="metric-label">${esc(label)}</div><div class="metric-value">${esc(value)}</div></article>`).join("");
    }

    function render() {
      renderTabs();
      const {evaluated, forecast, evaluatedDate, forecastDate} = heroData();
      document.getElementById("outlook-title").textContent = `Latest outlook · ${selectedArea}`;
      const generated = run.created_at_utc || run.generated_at_utc || DATA.generated_at_utc;
      metricCards("top-metrics", [["Evaluated day", deliveryLabel(evaluatedDate)],["Forecast day",deliveryLabel(forecastDate)],["Production model",productionName],["Generated · UTC",utcDateLabel(generated)]]);
      document.getElementById("hero-chart").innerHTML = heroSvg(evaluated, forecast);
      const forecastValues = forecast.map(row => Number(row.y_pred)).filter(Number.isFinite);
      const errors = evaluated.map(row => Number(row.y_pred) - actual(row));
      const mae = errors.length ? errors.reduce((sum,value) => sum + Math.abs(value), 0) / errors.length : null;
      metricCards("forecast-metrics", [["Forecast avg · DKK/MWh",number(forecastValues.reduce((a,b)=>a+b,0)/forecastValues.length)],["Forecast min · DKK/MWh",number(Math.min(...forecastValues))],["Forecast max · DKK/MWh",number(Math.max(...forecastValues))],["Previous MAE · DKK/MWh",number(mae)]], true);
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
