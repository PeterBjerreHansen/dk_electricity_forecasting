# Danish Energy Forecasts

Production-oriented reference skeleton for Danish day-ahead electricity price
forecasting. The project keeps the core simple: public API ingestion, immutable
raw responses, normalized parquet tables, leakage-safe rolling-origin
backtests, baseline forecasts, optional weather-feature boosting experiments,
one manually selected CatBoost production model, optional Chronos experiments,
and a minimal artifact-only Streamlit dashboard.

## Setup

```bash
python -m pip install -e ".[dev,app,catboost]"
cp .env.example .env
python -m pytest
```

For a dashboard-only install, `python -m pip install -e ".[app]"` is enough.
Optional modeling extras:

```bash
python -m pip install -e ".[tuning]"
python -m pip install -e ".[chronos]"
python -m pip install -e ".[lightgbm]"
```

The default workflow is file based and writes ignored runtime artifacts under
`data/`, `results/`, `artifacts/`, and `app_data/`.

## Data Sources

Price data comes from Energi Data Service. The ingestion script archives raw
JSON bytes and metadata for DK1/DK2, then builds an hourly model-ready panel:

```bash
python scripts/fetch_eds_prices.py --areas DK1 DK2
python scripts/build_price_panel.py --allow-incomplete-recent
```

Open-Meteo Previous Runs provides representative DK1/DK2 weather forecast
features. Raw API payloads are stored before normalized long and wide parquet
feature tables are built:

```bash
python scripts/fetch_open_meteo_previous_runs.py --start 2024-07-01 --end YYYY-MM-DD
python scripts/build_open_meteo_weather_features.py
```

Weather outputs preserve the production timing/provenance fields
`forecast_reference_time`, `valid_time`, `lead_time_hours`, `model`, `variable`,
and `price_area`, while retaining existing project columns such as
`forecast_available_at_utc`, `ds_utc`, `weather_model`, and `parameter_id`.

## Architecture

```text
src/dkenergy_data/
  sources/      API clients and raw response writers
  build/        raw-to-normalized and model-ready parquet builders

src/dkenergy_forecast/
  backtesting/  origin and horizon builders plus rolling-origin execution
  features/     price and availability-safe weather feature joins
  models/       baselines, fixed CatBoost adapter, optional Chronos adapter,
                and the production model registry
  evaluation/   point and probabilistic forecast-accuracy metrics
  publishing/   immutable run artifacts and latest dashboard exports

scripts/
  fetch_*.py, build_*.py, run_*_backtest.py, run_publish_forecast.py
  run_daily_pipeline.py

app/
  streamlit_app.py
```

Raw API responses are append-only under `data/raw/`. Normalized source tables
are written to `data/normalized/`. Model-ready price and weather feature tables
are written to `data/model_ready/` and `data/features/`.

## Forecasting

Forecasting is split into a development path and a production path. Notebooks
can search policies and model families freely; production publishing uses only
the source-controlled model registry.

List the production registry:

```bash
python scripts/run_publish_forecast.py --list-models
```

Run baseline diagnostics and publish the default production forecast artifacts:

```bash
python scripts/run_baseline_backtest.py --allow-incomplete-panel
python scripts/run_publish_forecast.py --allow-incomplete-panel
```

The default latest-forecast set is:
`same_hour_last_week`, `rolling_median_hour_weekend_56d`, and
`median_weekday_exp_hl4_floor10_42d__median_weekend_exp_hl28_floor20_56d`,
plus the manually selected `catboost_price_manual_v1` adapter.

`chronos_zero_shot_v1` is registered but disabled by default. Selecting it
requires the optional Chronos extra and an explicit model list.

Weighted median recency experiments are baseline modes. The default command
stays compact; heavier grids are explicit. The weekday/weekend diagnostic tunes
lookback horizons, calendar-day exponential half-lives, and optional weight
floors, while keeping equal-weight medians as references:

```bash
python scripts/run_baseline_backtest.py --weighted-median-grid common
python scripts/run_baseline_backtest.py --weighted-median-grid weekday-weekend
```

Weather-feature boosting is developed in notebooks. The reusable code lives in
feature builders and CatBoost tuning helpers; exploratory Optuna grids and
rolling validation are intentionally run from the modeling notebooks:

```bash
python scripts/build_weather_backtest_frame.py --frame-kind backtest --allow-incomplete-panel
jupyter notebook notebooks/05_catboost_model_development.ipynb
```

For a short diagnostic weather frame, use:

```bash
python scripts/build_weather_backtest_frame.py --frame-kind recent --allow-incomplete-panel
```

These frames are historical modeling/backtest artifacts. They are not used by
the latest-forecast publisher. Larger windows are explicit, for
example `--frame-kind custom --days 730 --output-path data/features/weather_experiment_frame_backtest_730d.parquet`.

CatBoost development remains notebook-first, but production has one fixed
adapter: `catboost_price_manual_v1`. It has source-code parameters, no Optuna,
and builds leakage-safe price features internally. Notebook 05 can motivate a
new policy, but promotion is manual: edit the registry/config, run a smoke
publish, and commit the change.

## Evaluation

Rolling-origin evaluation uses forecast origins that occur before the target
delivery window. Horizon builders strip target columns before model prediction;
actuals are joined only after predictions are produced. Tests cover UTC/local
alignment, DST day lengths, weather forecast availability masking, and leakage
prevention.

Backtest outputs use compact ignored run folders. Notebook runs default to
summary or diagnostic artifacts; use the audit level only when you need a full
debug trail.

```text
results/<run>/run_manifest.json
results/<run>/model_scores.parquet
results/<run>/policy_scores.parquet              # CatBoost policy notebooks
results/<run>/final_policy_selections.parquet    # CatBoost policy notebooks
results/<run>/predictions.parquet                # diagnostic notebook output
results/<run>/tuning_trials.jsonl                # CatBoost Optuna trials
results/<run>/experiment_runs.jsonl              # Chronos experiment log
```

Metrics include MAE, RMSE, bias, and optional quantile interval coverage/width.

## Daily Job

Run the file-based daily pipeline. By default this refreshes prices, runs the
baseline backtest, and publishes the registry-default forecast artifacts used
by Streamlit:

```bash
python scripts/run_daily_pipeline.py
```

Inspect the commands without running them:

```bash
make dry-run
```

Useful switches:

```bash
python scripts/run_daily_pipeline.py --with-weather
python scripts/run_daily_pipeline.py --skip-backtest
python scripts/run_daily_pipeline.py --strict-panel
```

`--with-weather` also refreshes Open-Meteo experiment artifacts. Current latest
forecast publishing uses price-only production models, so weather is not part
of the default daily update path.

In production, schedule that command from cron, GitHub Actions, Airflow, or a
container scheduler with persistent volumes mounted for `data/`, `results/`,
`artifacts/`, and `app_data/`.

## Dashboard

Publish latest forecast artifacts, then run Streamlit:

```bash
make publish
make dashboard
```

The dashboard reads:

```text
data/model_ready/price_panel_hourly_v1.parquet
results/latest_forecast/predictions.parquet
results/recent_scores/model_scores.parquet
app_data/forecast_dashboard.json
```

It shows DK1/DK2 actual day-ahead prices, the next available forecast,
point/quantile forecast rows when present, and recent backtest metrics.

## Docker

```bash
docker compose build
docker compose up dashboard
docker compose --profile jobs run --rm pipeline
```

The Compose services mount local `data/`, `results/`, `artifacts/`, and
`app_data/` directories so runs survive container restarts. The image installs
the dashboard and production CatBoost extras because CatBoost is part of the
default publish registry.

## CI

GitHub Actions runs `python -m pytest` on Python 3.10, 3.11, and 3.12. The
workflow lives in `.github/workflows/ci.yml`.
