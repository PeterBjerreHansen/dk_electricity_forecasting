# Danish Electricity Price Forecasts

Daily hourly day-ahead price forecasts for Denmark's DK1 and DK2 bidding zones,
with uncertainty intervals and transparent model diagnostics.

**[Open the live DK1/DK2 forecast](http://dk-energy-forecasts-site-653044339519.s3-website.eu-central-1.amazonaws.com/)**

The page shows the previous forecast against the newly available official
prices, tomorrow's forecast, and rolling 30-day diagnostics for the production
model and three simple baselines. It is rebuilt each morning before the
day-ahead auction; no Python server remains running after publication.

## What is being forecast?

Electricity is traded one day ahead, with a separate market price for every
delivery hour. The project predicts those hourly prices in DKK/MWh for both
Danish bidding zones. A normal delivery day has 24 hours; daylight-saving
transitions correctly produce 23 or 25.

The configured production release is a weather-aware Chronos-2 model with a
fixed LoRA adapter. A deterministic weighted median is the only publication
fallback. The live path does not tune, retrain, rank, or promote models.

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
                            CloudFront HTTPS page
```

One daily batch is enough because the public product changes only when a new
forecast is produced. Static HTML keeps the deployment cheap and legible: no
always-on application server, load balancer, database, automatic retraining,
or model-promotion service.

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

The core development environment is intentionally ordinary:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest
```

Optional dependencies are explicit:

```bash
python -m pip install -e ".[chronos,aws]"     # production pipeline
python -m pip install -e ".[notebooks]"       # analysis notebooks
python -m pip install -e ".[catboost,tuning]" # research only
```

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
python scripts/build_static_dashboard.py \
  --input artifacts/forecast_runs/<run-id>/forecast_dashboard.json \
  --history results/baseline_v1/predictions.parquet \
  --output build/static-dashboard/index.html

python -m http.server 8000 --directory build/static-dashboard
```

Open `http://127.0.0.1:8000/`. Python is used to build the file, not to serve
the deployed page.

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
- [Production operations](docs/operations.md) — health checks and failure
  handling.
- [AWS infrastructure](infra/aws/README.md) — the deployment procedure and
  resource boundary.

Generated `data/`, `results/`, `artifacts/`, `runtime/`, `dashboard/`, and
`build/` files are ignored by Git. The small evidence reports, contracts, tests,
and source-controlled release declaration carry the durable claims.
