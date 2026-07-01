# Danish Probabilistic Electricity Forecasting Project

Build a small but extensible forecasting stack for Danish electricity markets.

The first target is **hourly DK1/DK2 day-ahead electricity prices**. Load forecasting can be added later. The project should prioritize clean data access, leakage-free backtesting, and simple strong baselines before adding complex models.

## Data Sources

Start with:

1. **Energi Data Service**

   * Main electricity data source.
   * Use for DK1/DK2 day-ahead prices and later load/production/consumption data.
   * For the first price dataset, stitch discontinued `Elspotprices` history through 2025-09-30 with `DayAheadPrices` from 2025-10-01 onward.
   * Data-processing decisions for v1 live in `docs/data-processing/energi_data_service_v1.md`.

2. **DMI weather data**

   * Add only after the first EDS-only baseline works.
   * Use temperature, wind, cloud/solar proxies, etc.

3. **ENTSO-E**

   * Do not implement initially.
   * Add later for neighboring market features if the basic pipeline works.

## Storage Pipeline

Use a simple layered data flow:

```text
raw API JSON
→ normalized parquet
→ model-ready panel tables
→ forecasts/results
```

The forecasting code should not call external APIs directly. It should only consume model-ready parquet files.

The initial forecasting library contract lives in
`docs/forecasting/forecasting_library_contract_v1.md`.

## Initial Modeling Scope

Implement these models first:

1. Seasonal naive baselines:

   * same hour yesterday
   * same hour last week
   * rolling median for same hour

2. LightGBM or CatBoost quantile models:

   * predict p10 / p50 / p90
   * use calendar, lag, rolling, and DK1/DK2 spread features

Do not implement neural models until the data pipeline and backtesting are stable.

## Evaluation

Use rolling-origin backtesting.

For each forecast origin, train only on data available before that origin and
predict an explicit future horizon, usually the next Danish delivery day or
next 24 UTC hours for initial baselines.

Track:

* MAE
* RMSE
* pinball loss
* p10–p90 interval coverage
* average interval width
* simple value metric: did the model identify the cheapest hours?

## Rough Repo Structure

```text
dk-electricity-forecasting/
  README.md
  pyproject.toml

  configs/
    experiments/
      price_seasonal_naive.yaml
      price_lgbm_quantile.yaml

  data/
    raw/
    normalized/
    features/
    model_ready/

  src/
    dkenergy_data/
      sources/
        energidataservice.py
        dmi.py
      normalize/
        prices.py
        weather.py
      features/
        calendar.py
        lags.py
        rolling.py
      build/
        build_price_panel.py

    dkenergy_forecast/
      models/
        baselines.py
        lightgbm_quantile.py
        catboost_quantile.py
      backtesting/
        rolling_origin.py
      evaluation/
        point_metrics.py
        probabilistic_metrics.py
        value_metrics.py

  scripts/
    fetch_eds_prices.py
    build_price_panel.py
    run_backtest.py
    evaluate_run.py

  results/
```

## First Milestones

### Milestone 1: EDS price panel

Fetch DK1/DK2 price data from Energi Data Service, normalize it, and save:

```text
data/model_ready/price_panel_hourly_v1.parquet
```

The table should contain:

```text
unique_id, ds_utc, ds_local, y, area
```

plus basic calendar features.

### Milestone 2: Baseline backtest

Implement rolling-origin backtesting and evaluate:

* same hour yesterday
* same hour last week
* rolling median same hour

Save predictions and metrics to `results/`.

### Milestone 3: CatBoost quantile model

Add lag and rolling features, then train quantile models for:

```text
q10, q50, q90
```

Save predictions, metrics, and feature importance.

### Milestone 4: Add DMI weather

Only after the EDS-only model works, add weather features and compare performance against the EDS-only model.

## Important Constraints

* Use UTC internally.
* Keep local Danish time as a feature.
* Avoid future leakage in all lag/rolling features.
* Do not mix realtime and settlement data silently.
* Store every backtest’s config, predictions, metrics, dataset version, and git commit.
* Reconciliation should only be added later for additive targets such as load or net-load, not for prices.
