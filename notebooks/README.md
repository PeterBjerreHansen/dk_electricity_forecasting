# Notebooks

This directory is for notebook-first data science work.

The canonical EDS pipeline still lives in package code and production-oriented
CLI scripts. These notebooks are the active data-science laboratory: exploratory
model tuning, interpretation, and didactic development happen here.

1. `01_eds_processing_walkthrough.ipynb` explains the v1 EDS processing choices
   with tiny source-shaped examples.
2. `02_eds_price_panel_eda.ipynb` analyzes the built
   `data/model_ready/price_panel_hourly_v1.parquet` panel and its QA report,
   or the focused `v1_eda` panel when the full-history artifact has not been
   built yet.
3. `03_baseline_model_development.ipynb` develops the first local forecasting
   baselines as a diagnostic story: autocorrelation motivates lag rules, lag
   variance motivates robust rolling medians, and recency-vs-sample-size
   motivates weighted weekday/weekend medians. It can run on the real panel or
   a clearly labeled synthetic teaching panel when the real data has not been
   built yet.
4. `04_open_meteo_weather_eda.ipynb` inspects Open-Meteo Previous Runs weather
   features before modeling: raw variable completeness, v1 coverage gates,
   forecast-origin availability masking, DK1/DK2 area behavior, model-to-model
   differences, and price-plus-weather experiment-frame checks.
5. `05_catboost_model_development.ipynb` runs the full CatBoost development
   loop in-notebook: price-only direct/residual models, feature ablations,
   recency weighting, scheduled retuning/retraining, weather-enhanced
   candidates, and final price-only/weather-conditioned candidate selection.
Reusable versions of the CatBoost feature and validation logic live in:

```text
src/dkenergy_forecast/features/price_features.py
src/dkenergy_forecast/features/weather_features.py
src/dkenergy_forecast/tuning/catboost_common.py
src/dkenergy_forecast/tuning/catboost_validation.py
```

Exploratory CatBoost tuning belongs in notebook 05. Scripts are retained for
data builds, production-style backtests, and publishing workflows.
Notebook winners do not auto-promote; production model selection lives in the
source-controlled registry and fixed model configs.

Run the canonical data build before the EDA notebook:

```bash
python scripts/fetch_eds_prices.py --areas DK1 DK2
python scripts/build_price_panel.py --dataset-version v1
```

Install optional notebook and CatBoost dependencies only when you need them:

```bash
pip install -e ".[notebooks,tuning]"
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
EDA plots are displayed inline when notebooks are run. Generated figures and
small summary tables are treated as local notebook artifacts and are not
committed.

The current weather EDA and CatBoost development pass uses the canonical local
Open-Meteo backfill when present:

```text
data/raw/open_meteo/
data/normalized/open_meteo_previous_runs_open_meteo_previous_runs_v1.parquet
data/features/weather_open_meteo_area_hourly_long_open_meteo_previous_runs_v1.parquet
data/features/weather_experiment_frame_backtest.parquet
```

The local backfill covers 2024-07-01 through 2026-06-30 for GFS, ICON-EU,
MET Norway Nordic, and all ten DK1/DK2 coordinate-basket points. It is still a
local experiment artifact; weather is not written into the canonical EDS price
panel.

Use `scripts/build_weather_backtest_frame.py --frame-kind recent` for a short
diagnostic frame and `--frame-kind backtest` for the standard offline modeling
frame.

The focused Open-Meteo EDA slice remains useful as a small fallback artifact:

```text
data/raw/open_meteo_eda/
data/normalized/open_meteo_previous_runs_open_meteo_previous_runs_v1_eda.parquet
data/features/weather_open_meteo_area_hourly_long_open_meteo_previous_runs_v1_eda.parquet
data/features/weather_open_meteo_area_hourly_open_meteo_previous_runs_v1_eda.qa.json
```

It covers 2025-01-01 through 2025-01-14. Treat it as a source-shape EDA sample,
not as the weather-model comparison artifact. Weather notebooks also display
plots inline and should not commit generated figure or summary-table exports.
