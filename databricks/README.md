# Databricks deployment

This folder runs the DK1/DK2 day-ahead price forecaster on Databricks:
raw API responses land in a Unity Catalog volume, Delta tables hold the
bronze, silver and gold layers, the model is trained and tracked with MLflow
and registered in Unity Catalog, and two serverless jobs run it on a schedule.
Everything is declared in a Databricks Asset Bundle (`databricks.yml`), so one
command deploys it.

It reuses the repository's library code unchanged: the same panel builder,
leakage-safe price features, rolling-origin backtest and baselines that the
AWS pipeline used. The Databricks layer is the orchestration, storage, model
registry and monitoring around them.

## Architecture

```text
                      daily_forecast job (10:15 Copenhagen)
┌──────────────┐   ┌───────────────┐   ┌──────────────┐   ┌──────────┐   ┌─────────┐
│ land_prices  │ → │ bronze_ingest │ → │ silver_panel │ → │ forecast │ → │ monitor │
└──────────────┘   └───────────────┘   └──────────────┘   └──────────┘   └─────────┘
 EDS API → volume    new raw batches     latest revision    @champion,     scores +
 /Volumes/.../       → Delta, one row    per quarter-hour   weighted-      drift and
 landing             per API record      → hourly panel     median         degradation
                                                            fallback       checks

                      weekly_retrain job (Monday 06:00)
┌───────────────────────────────────────────────────────────────────────────────┐
│ train_register: backtest candidate vs baselines on the same recent origins,   │
│ train on all history, log to MLflow, register in UC, move @champion if it     │
│ beats the fallback                                                            │
└───────────────────────────────────────────────────────────────────────────────┘
```

| Layer | Table | Contents |
|---|---|---|
| Landing | volume `landing` | Raw EDS JSON responses and their manifest, byte for byte |
| Bronze | `bronze_eds_price_records` | One row per API record, with batch id, fetch time and source file |
| Silver | `silver_price_panel_hourly` | Canonical hourly DK1/DK2 panel with calendar and availability columns |
| Silver | `silver_price_panel_qa` | The builder's QA report for every rebuild |
| Gold | `gold_forecasts` | Every model's forecast per origin; exactly one row set has `is_published` |
| Gold | `gold_backtest_predictions` | Backtest predictions behind each training run, keyed by MLflow run id |
| Gold | `gold_forecast_scores` | Live MAE, RMSE and bias per origin, model and area |
| Gold | `gold_monitoring_checks` | Drift and degradation checks with status and detail |
| Model | `dk_day_ahead_price_catboost` | Registered model versions; the `@champion` alias is what the forecast uses |

## Design choices

**Training and serving are separate.** The research CatBoost model refits on
every call. Here the weekly job fits once, logs the CatBoost file and its
column contract as an MLflow pyfunc model, and the daily job loads
`models:/…@champion`. The pyfunc takes the same feature frame as training, so
there is one feature code path for both.

**Features are built once per training run.** Building a year of training
features takes about a second per origin, and the research backtest rebuilt
them for every backtest origin. The weekly job builds one pool of training
rows covering all backtest origins and the final fit, then slices it per
origin. A test checks that the slices and the resulting predictions are
identical to the per-origin build.

**Promotion is a paired comparison.** A new version only gets `@champion` if
its rolling-origin backtest MAE beats the weighted-median fallback on the same
recent origins (`promotion_decision` in `src/dkenergy_databricks/training.py`).
Comparing against the previous champion's backtest would mix different
windows. Every version is registered with its backtest MAE as a tag, so a
rejected candidate is still inspectable.

**The fallback is always computed.** The daily task writes the three
baselines next to the champion. If the champion is missing, fails to load, or
returns non-finite values, the weighted median is published instead and the
task logs why. This is the same rule as the AWS pipeline.

**Monitoring compares against what we would have done anyway.** The
degradation check alerts when the published model has a higher MAE than the
fallback over the last seven scored origins. The drift check computes the
population stability index of the last week's prices against the previous 90
days per area; above 0.25 is flagged. Pass `--fail-on-alert` to the monitor
task to turn alerts into job failures, so a failure notification on the job
reaches you.

**Spark where it pays off.** Bronze grows by one revision of the current
month every day. The silver task keeps only the newest revision of each
quarter-hour with a window function in Spark, so the single-node pandas panel
builder only sees one row per price.

**Time is explicit.** Delta stores instants, so the session runs in UTC, the
Copenhagen-local timestamp is rebuilt on read, and the forecast origin is
10:00 Copenhagen wall-clock time on every run date, including across DST.

## Setup on Databricks Free Edition

1. Sign up at [databricks.com/learn/free-edition](https://www.databricks.com/learn/free-edition).
   The workspace comes with a catalog called `workspace`, which is the bundle's
   default.
2. Install the [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install)
   and log in: `databricks auth login --host https://<your-workspace>.cloud.databricks.com`.
3. Install [uv](https://docs.astral.sh/uv/) if you have not; the bundle builds
   the wheel with it.
4. From the repository root:

   ```bash
   databricks bundle validate
   databricks bundle deploy          # dev target: prefixed names, schedules paused
   databricks bundle run daily_forecast
   ```

   The first run publishes the weighted-median fallback because no model is
   registered yet.

5. Train and register the first model, then forecast again:

   ```bash
   databricks bundle run weekly_retrain
   databricks bundle run daily_forecast
   ```

6. Open **Catalog → workspace → dev_<you>_dk_energy** for the tables and
   model, **Experiments** for the MLflow runs, and paste the queries in
   `databricks/sql/monitoring_queries.sql` into the SQL editor (change the
   schema name) to build a dashboard.

When it works in dev, `databricks bundle deploy -t prod` deploys the same
jobs with unprefixed names and live schedules.

### If the price API is blocked

Free Edition limits outbound internet access from serverless compute to a set
of trusted domains. If `land_prices` logs that Energi Data Service is
unreachable, land the raw files from GitHub instead:

1. In the repository settings, add the variable `DATABRICKS_HOST` and the
   secret `DATABRICKS_TOKEN` (a personal access token from **Settings →
   Developer**).
2. Run the **Databricks** workflow with the `deploy` action once.
3. Set the variable `DATABRICKS_LAND_FROM_GITHUB` to `true`. Every morning the
   workflow fetches the prices on a GitHub runner, copies them to
   `/Volumes/workspace/dk_energy/landing/energi_data_service` and runs the
   prod `daily_forecast` job. The `land_prices` task then continues with the
   files already in the volume.

## Running the code locally

The Databricks modules are plain pandas and are tested without Spark:

```bash
uv sync --locked --extra dev --extra databricks
uv run --locked --extra dev --extra databricks python -m pytest tests/test_databricks_pipeline.py
```

The tests generate synthetic EDS responses in both source formats and cover
landing, bronze, the silver rebuild (including price revisions), the Delta
round trip, the backtest and promotion gate, the logged MLflow model,
fallback publication and the monitoring checks. CI also builds the bundle
wheel and checks the bundle YAML against the CLI's JSON schema.

## Files

```text
databricks.yml                     bundle: wheel artifact, variables, dev/prod targets
databricks/resources/schema.yml    Unity Catalog schema and landing volume
databricks/resources/jobs.yml      the two serverless jobs
databricks/jobs/*.py               thin task entry points (read Delta → library → write Delta)
databricks/sql/                    dashboard queries
src/dkenergy_databricks/           ingest, silver, training, MLflow model, inference, monitoring
.github/workflows/databricks.yml   deploy on push to main; optional daily landing
```

## Limits and next steps

- The model uses price history and calendar features only. The weather
  features from the AWS pipeline (Open-Meteo previous runs) are the obvious
  next source to land in bronze.
- The Chronos-2 LoRA model from the AWS pipeline is not ported; its adapter is
  a local artifact and it needs more compute than Free Edition gives.
- Forecasts are batch-written to Delta. Free Edition limits model serving
  endpoints, so there is no real-time endpoint; the registered pyfunc could be
  served as one on a paid workspace without code changes.
