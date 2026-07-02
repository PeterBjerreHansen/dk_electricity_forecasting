# Danish Energy Forecasts

Notebook-friendly data and forecasting pipeline for Danish electricity markets.

The first target is hourly DK1/DK2 day-ahead electricity prices. The current
stack is intentionally small: preserve raw source data, build a clean
model-ready price panel, run leakage-safe baselines, and then add optional
CatBoost quantile models.

## Install

```bash
pip install -e .
```

Optional notebook and CatBoost extras:

```bash
pip install -e ".[notebooks]"
pip install -e ".[catboost]"
```

## Data Pipeline

Fetch raw Energi Data Service price batches:

```bash
python scripts/fetch_eds_prices.py --areas DK1 DK2
```

Build the hourly model-ready panel:

```bash
python scripts/build_price_panel.py --dataset-version v1
```

Main output:

```text
data/model_ready/price_panel_hourly_v1.parquet
data/model_ready/price_panel_hourly_v1.qa.json
```

`data/` and `results/` are intentionally ignored. A fresh clone should run the
fetch step before `scripts/build_price_panel.py`; committed code does not include
the raw EDS batches or generated result artifacts. The fetch script preserves
raw HTTP response bytes and skips only existing raw batch files that are valid
JSON with a `records` list and, when a manifest hash is present, matching bytes.

The EDS processing contract is documented in:

```text
docs/data-processing/energi_data_service_v1.md
```

## Forecasting

Run the official EDS-only baseline backtest:

```bash
python scripts/run_baseline_backtest.py
```

Run the optional CatBoost quantile backtest after installing CatBoost:

```bash
python scripts/run_catboost_backtest.py
```

Backtest outputs are written under `results/`. Scripted backtests now write the
production-facing score artifact as `model_scores.parquet` and keep
`metrics.parquet` as a compatibility alias.

Publish a file-based forecast run for the production-learning dashboard path:

```bash
python scripts/run_publish_forecast.py
```

This writes an immutable run under `artifacts/forecast_runs/<run_id>/`, updates
`results/latest_forecast/predictions.parquet`, writes
`results/recent_scores/model_scores.parquet`, and exports
`app_data/forecast_dashboard.json`.

The forecasting contract is documented in:

```text
docs/forecasting/forecasting_library_contract_v1.md
```

## Notebooks

Notebooks live in `notebooks/`:

1. EDS processing walkthrough,
2. EDS price-panel EDA,
3. baseline model development,
4. CatBoost model development.

The notebooks are explanatory workspaces. Canonical reusable logic lives in
`src/` and `scripts/`.

## Weather

Open-Meteo Previous Runs is the MVP forecast-weather source. Fetch and build it
separately from the canonical EDS price panel:

```bash
python scripts/fetch_open_meteo_previous_runs.py --start 2024-07-01 --end YYYY-MM-DD
python scripts/build_open_meteo_weather_features.py
python scripts/build_weather_experiment_frame.py
```

Run the optional weather-feature CatBoost ablation after installing CatBoost:

```bash
python scripts/run_weather_feature_backtest.py
```

Weather outputs are experiment artifacts under `data/features/`; they do not
modify `data/model_ready/price_panel_hourly_v1.parquet`. The Open-Meteo
processing contract lives in:

```text
docs/data-processing/open_meteo_weather_v1.md
```

DMI direct weather archiving is still planned as a separate provenance track.
The DMI research note lives in:

```text
docs/data-processing/dmi_weather_v1_plan.md
```
