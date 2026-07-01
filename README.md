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

Outputs are written under `results/`.

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

DMI weather is planned but not yet joined into the price panel. The current
weather research and v1 plan live in:

```text
docs/data-processing/dmi_weather_v1_plan.md
```
