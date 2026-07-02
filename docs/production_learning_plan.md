# Production Forecast Dashboard Learning Plan

## Summary

This plan uses the Danish electricity forecasting project as a hands-on path
from data science workflows to a production-style forecast dashboard. The end
state is a local, backend-backed system that publishes batch forecast artifacts,
registers metadata, serves results through an API, and renders DK1/DK2 forecast
curves in a frontend.

The goal is not live inference. The goal is batch forecast publishing: generate
durable forecast artifacts, track provenance and scores, serve them through an
API, and display them in a dashboard.

## Final Setup

The local production-style stack should include:

- Docker Compose for local orchestration.
- Postgres for forecast run metadata and model scores.
- A FastAPI backend for serving forecast data to the dashboard.
- A worker process for forecast generation, scoring, and dashboard export.
- A scheduler process for periodic local runs.
- A React/Vite frontend for the forecast dashboard.

Core local artifacts:

```text
results/latest_forecast/predictions.parquet
results/recent_scores/model_scores.parquet
app_data/forecast_dashboard.json
artifacts/forecast_runs/<run_id>/manifest.json
```

The first production-learning worker should start with the default baseline
registry only. Optional CatBoost labels are registered for later promotion, but
they are not enabled by default:

```text
same_hour_last_week
rolling_median_local_hour_28d
rolling_median_hour_weekend_56d
```

Weather CatBoost labels are registered as optional/backtest-validated model
families. They should not be selected for latest-forecast publishing until a
dedicated weather publish adapter has been added:

```text
weather_catboost_gfs_global
weather_catboost_icon_eu
weather_catboost_metno_nordic
weather_catboost_all_weather
weather_catboost_ensemble
```

Core database tables:

- `forecast_runs`: run id, forecast origin, created time, model labels, status,
  artifact paths, dataset version, and git commit.
- `model_scores`: run id, model label, area, MAE, RMSE, bias, coverage, and
  interval width.

Core API endpoints:

- `GET /health`
- `GET /api/forecast-runs`
- `GET /api/latest-forecast`
- `GET /api/recent-scores?days=7`
- `GET /api/dashboard`

Core dashboard views:

- Latest forecast round with DK1/DK2 p10-p50-p90 curves.
- Last 7 days predicted vs actual by forecast origin.
- Model toggles for CatBoost, weather CatBoost, rolling median, and
  same-hour-last-week.
- Scoreboard with MAE, RMSE, bias, coverage, and interval width.

## 4-5 Week Learning Path

### Week 1: Production Mental Model And Artifact Discipline

- Learn the difference between notebooks, batch jobs, artifacts, metadata, and
  APIs.
- Think in immutable forecast runs: every run has inputs, outputs, status, and
  provenance.
- Define the artifact contract for `predictions.parquet`,
  `model_scores.parquet`, and `forecast_dashboard.json`.
- Run the existing price panel and backtest scripts.
- Inspect parquet outputs and manually sketch one forecast run manifest.

### Week 2: Backend Foundations

- Learn FastAPI basics: routes, response models, dependency injection, and error
  handling.
- Learn Postgres basics for ML products: metadata tables, migrations,
  timestamps, and indexes.
- Add a local API that can return mocked latest forecast data before connecting
  real artifacts.
- Add database migrations for `forecast_runs` and `model_scores`.
- Practice by registering a fake run in Postgres and fetching it through
  `/api/forecast-runs`.

### Week 3: Forecast Worker And Real Artifacts

- Add worker scripts for latest forecast generation, recent scoring, dashboard
  JSON export, and forecast run registration.
- Connect the worker to the existing model code instead of duplicating modeling
  logic.
- Keep latest forecast generation separate from recent completed-origin
  scoring. The latest forecast may not have actual prices yet, while recent
  scores should be computed from origins whose delivery horizons are complete.
- Write each forecast run under a unique `run_id`.
- Update `results/latest_forecast/` as a convenience pointer to the latest
  successful forecast.
- Practice by running the worker locally, confirming Postgres metadata, checking
  parquet artifacts, and calling `/api/latest-forecast`.

### Week 4: Frontend Dashboard

- Learn React/Vite dashboard basics: API fetching, loading states, error states,
  chart components, and model toggles.
- Build charts around the API response shape, not local files.
- Implement DK1/DK2 area selection.
- Implement model toggles.
- Show p10-p90 bands for probabilistic models.
- Overlay actual cleared prices when available.
- Add a model score table.

### Week 5: Production-Like Local Operations

- Add Docker Compose for Postgres, API, worker, scheduler, and frontend.
- Add a scheduler command that runs the full publishing loop locally.
- Add stale-data detection through `created_at_utc`.
- Add run statuses: `running`, `success`, and `failed`.
- Add useful logging around forecast generation, scoring, artifact writes, and
  API startup.
- Practice by starting from a clean local environment, running one full scheduled
  cycle, opening the dashboard, and explaining every moving part.

## Interfaces And Contracts

Prediction rows must include:

```text
forecast_origin_utc
ds_utc
ds_local
area
model_label
y_pred
```

Probabilistic model rows may also include:

```text
q10
q50
q90
```

Rows may include actual cleared prices after they become available:

```text
actual_price
```

Scores should be grouped by:

```text
model_label
area
```

The API should return JSON only. Parquet remains an internal artifact format.
The frontend should never call model code directly; it should only call the
FastAPI backend. The scheduler should write artifacts and metadata, while the
API serves the latest successful data.

## Test Plan

Unit tests:

- Artifact schema validation.
- Score calculation.
- API response shape.
- Database insert/read helpers.

Integration tests:

- Worker writes artifacts and registers a successful run.
- API returns the latest registered successful run.
- Failed worker run records `failed` status without replacing latest successful
  data.

Frontend checks:

- Chart renders with one model.
- Chart renders with multiple models.
- Chart renders when no probabilistic band is available.
- Toggles hide and show models correctly.
- Stale-data warning appears when metadata is old.

End-to-end acceptance:

- `docker compose up` starts the stack.
- One command or scheduled job produces artifacts.
- Dashboard displays latest DK1/DK2 forecasts and recent model scores.

## Assumptions

- End state is a local production-style stack; cloud deployment is deferred.
- Timeline assumes intensive work, roughly 8-15 hours per week.
- Frontend stack is React and Vite.
- Backend stack is FastAPI and Postgres.
- Artifact storage starts as local files and should later be movable to
  S3, R2, GCS, or similar object storage.
- The product remains batch forecast publishing, not per-request live model
  inference.
