# Notebooks

This directory is for notebook-first data science work.

The canonical EDS pipeline still lives in package code and CLI scripts. These
notebooks are for inspection, explanation, and exploratory analysis:

1. `01_eds_processing_walkthrough.ipynb` explains the v1 EDS processing choices
   with tiny source-shaped examples.
2. `02_eds_price_panel_eda.ipynb` analyzes the built
   `data/model_ready/price_panel_hourly_v1.parquet` panel and its QA report,
   or the focused `v1_eda` panel when the full-history artifact has not been
   built yet.
3. `03_baseline_model_development.ipynb` develops the first local forecasting
   baselines with rolling-origin backtests, diagnostics, and plots. It can run
   on the real panel or a clearly labeled synthetic teaching panel when the
   real data has not been built yet.
4. `04_catboost_model_development.ipynb` builds leakage-safe tabular features,
   compares CatBoost-ready origins against the local baselines, and trains
   CatBoost quantile models when the optional `catboost` dependency is
   installed.

Reusable versions of the CatBoost feature and model logic now live in:

```text
src/dkenergy_forecast/features/price_features.py
src/dkenergy_forecast/models/catboost_quantile.py
scripts/run_catboost_backtest.py
```

The notebook remains useful for explanation and inspection, but scripts should
prefer the package code.

Run the canonical data build before the EDA notebook:

```bash
python scripts/fetch_eds_prices.py --areas DK1 DK2
python scripts/build_price_panel.py --dataset-version v1
```

Install optional notebook and CatBoost dependencies only when you need them:

```bash
pip install -e ".[notebooks]"
pip install -e ".[catboost]"
```

The current interpreted EDA pass uses a focused real-data artifact:

```text
data/model_ready/price_panel_hourly_v1_eda.parquet
data/model_ready/price_panel_hourly_v1_eda.qa.json
```

It covers Danish local delivery time from 2024-01-01 through 2026-07-01 and
includes the 2025-10-01 EDS source transition. The local canonical
`price_panel_hourly_v1.parquet` may also be built from this focused raw slice
for early modeling work; do not treat that as the final full-history backfill.
Generated EDA figures and small summary tables live in:

```text
notebooks/figures/
```
