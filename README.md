# Danish Electricity Price Forecasts

Daily hourly day-ahead price forecasts for Denmark's DK1 and DK2 bidding zones,
with uncertainty intervals and transparent model diagnostics.

**Hosting status:** The AWS-hosted forecast website has been shut down because
the hosting costs were too high. There is no longer a live public dashboard.
The forecasting code and AWS deployment configuration remain in this repository;
see [Run locally](#run-locally) to build and view a dashboard locally.

The dashboard shows the previous forecast against the newly available official
prices, tomorrow's forecast, and rolling 30-day diagnostics for the production
model and three simple baselines. It is generated as a static HTML page.

## What is being forecast?

Electricity is traded one day ahead, with a separate market price for every
delivery hour. The project predicts those hourly prices in DKK/MWh for both
Danish bidding zones. A normal delivery day has 24 hours; daylight-saving
transitions correctly produce 23 or 25.

The configured production release is a weather-aware Chronos-2 model with a
fixed LoRA adapter. A deterministic weighted median is the only publication
fallback. The publication path does not tune, retrain, rank, or promote models.

## Current model evidence

The exact configured Chronos release was compared with the fixed weighted
median on 1,152 paired rows from 24 historical forecast origins:

| Model | MAE | RMSE | Bias | 80% interval coverage |
|---|---:|---:|---:|---:|
| Chronos-2 LoRA weather | 170.2 | 370.3 | -65.2 | 79.0% |
| Weighted median | 255.1 | 476.6 | -12.9 | — |

Chronos improves MAE by 84.9 DKK/MWh on this period. The 95% moving-block
interval for the paired difference is -177.1 to -12.8 DKK/MWh. See the
[versioned comparison report](docs/model-evidence/chronos-weather-sha256-55814a7fd0d36973/model_comparison.md)
for per-origin, subgroup, calibration, release, and input-hash details.

This is a limited historical evaluation, not a guarantee of future performance
or a long live-production record. Weather availability is approximated from
archived forecast metadata, the price target changes source resolution in
October 2025, and electricity-price regimes can shift quickly.

## How it works

The former AWS deployment used the architecture below. Its configuration is
retained as a reference for anyone who wants to deploy their own instance.

```text
EventBridge Scheduler · 10:00 Europe/Copenhagen
                         │
                         ▼
one short-lived ECS/Fargate task
prices + weather → features → fixed model → immutable forecast run
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
      private S3 artifacts    self-contained index.html
                                      │
                                      ▼
                       S3 HTTPS object endpoint
```

One daily batch is enough because the dashboard changes only when a new
forecast is produced. The architecture uses static HTML with no
always-on application server, load balancer, database, automatic retraining,
or model-promotion service. AWS hosting still incurs costs, which led to the
public website being shut down.

A completed forecast is an immutable directory:

```text
forecast_runs/<run-id>/
├── predictions.parquet
├── diagnostic_predictions.parquet
├── model_scores.parquet
├── forecast_dashboard.json
├── manifest.json
└── COMPLETED.json
```

`COMPLETED.json` is written last. Only then may `latest.json` point at the run.
The manifest records checksums, model release, source revision, information
cutoff, and delivery contract. The dashboard is a derived view, not the source
of truth.

## Run locally

The project uses [uv](https://docs.astral.sh/uv/) for its Python environment
and dependency lockfile. Nothing is installed globally: `uv` creates and
maintains the project-local `.venv` from `uv.lock`.

When dependencies change, run `uv lock` and commit the updated `uv.lock`; the
`--locked` flag below makes setup and CI fail instead of silently changing it.

```bash
uv sync --locked --extra dev
uv run --locked --extra dev python -m pytest
```

The repository pins Python 3.11 in `.python-version`, matching the production
container. If that interpreter is not installed, uv downloads it for the
project environment. You can activate `.venv` if you prefer, but using
`uv run` keeps commands on the locked environment without relying on shell
activation.

Optional dependencies are explicit:

```bash
uv sync --locked --extra dev --extra chronos --extra aws       # production pipeline
uv sync --locked --extra dev --extra notebooks                  # analysis notebooks
uv sync --locked --extra dev --extra catboost --extra tuning   # research only
```

For the common combinations, `make install`, `make install-production`, and
`make install-analysis` run the corresponding uv commands.

Common workflows are thin commands over the same library code:

```bash
make data-prices
make data-weather
make backtest-baseline
make publish
make static-dashboard
make test
```

To inspect a completed run as a static site:

```bash
uv run --locked --extra dev python scripts/build_static_dashboard.py \
  --input artifacts/forecast_runs/<run-id>/forecast_dashboard.json \
  --history results/baseline_v1/predictions.parquet \
  --output build/static-dashboard/index.html

uv run --locked --extra dev python -m http.server 8000 --directory build/static-dashboard
```

Open `http://127.0.0.1:8000/`. This command serves the generated static page
locally.

## Repository guide

```text
config/                fixed production model declaration
docs/                  contracts, evidence, operations, and codebase tour
infra/aws/             compact Terraform stack and deployment procedure
notebooks/             analytical and model-development narratives
scripts/               command-line entry points
src/dkenergy_data/     external data ingestion and normalization
src/dkenergy_forecast/ forecasting, evaluation, publication, and dashboard code
tests/                  unit, contract, workflow, and infrastructure tests
```

Start with the [codebase tour](docs/codebase-tour.md) for a didactic end-to-end
explanation. Then use the narrower references as needed:

- [Forecasting contract](docs/forecasting-contract.md) — cutoff, horizon, time,
  and publication semantics.
- [Data card](docs/data-card.md) — source coverage, target regime, weather proxy,
  and limitations.
- [Chronos model card](docs/model-card-chronos-v1.md) — the exact production
  model and evidence.
- [Model evaluation](docs/evaluation.md) — reproducible descriptive comparison.
- [Forecast improvement plan](docs/forecast-improvement-execution-plan.md) —
  non-blocking experiments, evidence gates, and deferred research.
- [Production operations](docs/operations.md) — health checks and failure
  handling.
- [AWS infrastructure](infra/aws/README.md) — the deployment procedure and
  resource boundary.

Generated `data/`, `results/`, `artifacts/`, `runtime/`, `dashboard/`, and
`build/` files are ignored by Git. The small evidence reports, contracts, tests,
and source-controlled release declaration carry the durable claims.
