# Danish electricity price forecasts

This project produces daily day-ahead electricity-price forecasts for the DK1
and DK2 bidding zones. The production model is a weather-aware Chronos-2 model
with a fixed LoRA adapter. A deterministic weighted-median model is the only
publication fallback.

The production system is deliberately small:

```text
EventBridge Scheduler (10:00 Europe/Copenhagen)
                    │
                    ▼
one short-lived ECS/Fargate task
  fetch prices and weather → build features → forecast → save artifacts
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
private S3 artifacts     private S3 index.html
                                  │
                                  ▼
                         CloudFront HTTPS page
```

There is no continuously running web server, load balancer, model-promotion
service, database, or automatic retraining. The public dashboard is a
self-contained HTML file rebuilt by the daily task. This keeps the operational
path cheap and understandable while retaining detailed forecasts, uncertainty
intervals, baselines, provenance, and 30-day model diagnostics.

## Start here

Create a virtual environment and install the core development dependencies:

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

The most useful introductions are:

- [Codebase tour](docs/codebase-tour.md) — a didactic end-to-end explanation.
- [Forecasting contract](docs/forecasting-contract.md) — time, cutoff, and
  publication semantics.
- [AWS deployment plan](docs/aws-deployment-plan.md) — the staged production
  setup.
- [Operations](docs/operations.md) — the small production runbook.

## Common local workflows

The `Makefile` is a convenience layer over ordinary Python scripts:

```bash
make data-prices       # fetch and normalize Energi Data Service prices
make data-weather      # fetch and build point-in-time Open-Meteo features
make backtest-baseline # run reproducible baseline evaluation
make publish           # publish one local immutable forecast run
make static-dashboard  # build build/static-dashboard/index.html
make test
```

`make static-dashboard` follows `artifacts/latest.json` by default. You can point
it at any completed run and supply explicit evaluated histories:

```bash
python scripts/build_static_dashboard.py \
  --input artifacts/forecast_runs/<run-id>/forecast_dashboard.json \
  --history results/baseline_v1/predictions.parquet \
  --output build/static-dashboard/index.html \
  --history-output build/static-dashboard/forecast_history.parquet

python -m http.server 8000 --directory build/static-dashboard
```

Then open `http://127.0.0.1:8000/`. The browser receives only one static file;
Python is not involved after the build.

## Production model policy

[`config/production.json`](config/production.json) is intentionally boring. It
names exactly one primary and one fallback:

- `chronos_weather`: Chronos-2 plus the reviewed LoRA adapter and point-in-time
  calendar/weather covariates;
- `weighted_median_v1`: a fixed weekday/weekend recency-weighted median.

The live path does not rank models, promote winners, tune hyperparameters, or
retrain the adapter. If Chronos fails, the fallback is published and the run is
marked `degraded`. Three cheap diagnostic baselines are also recorded for the
dashboard, but they are never publication candidates.

Experimental CatBoost and zero-shot Chronos implementations remain available
through `models/comparison_registry.py` for notebooks and comparisons. Their
location and names make their non-production status explicit.

## Artifact contract

A successful forecast is an immutable directory:

```text
artifacts/forecast_runs/<run-id>/
├── predictions.parquet             # the one published model
├── diagnostic_predictions.parquet  # published model + three baselines
├── model_scores.parquet
├── forecast_dashboard.json
├── manifest.json
└── COMPLETED.json                   # written last
```

Only after the complete directory exists may `artifacts/latest.json` point to
it. Checksums, run identity, information cutoff, decision deadline, model
release, and source commit remain inspectable. The static dashboard is a
derived view; it is not the source of truth.

The private dashboard archive stores registered predictions and fills in
official prices on later runs. The public page receives only the newest 30
evaluated delivery days per model and area. A one-time seed from existing
backtests is acceptable; daily registered forecasts naturally replace it.
The public build requires complete DK1 and DK2 delivery grids and an explicit
model release ID. The outlook joins two days only when the left side is the
immediately preceding Danish delivery day from that same release. If that
registered forecast is missing, the page shows the new forecast alone instead
of compressing a calendar gap or mixing model versions.

## Repository map

```text
config/                       production model declaration
docs/                         contracts, runbook, deployment, codebase tour
infra/aws/                    small Terraform stack
notebooks/                    analysis and model-development narratives
scripts/                      thin command-line entry points
src/dkenergy_data/            external data-source clients
src/dkenergy_forecast/        reusable forecasting and operations code
tests/                        unit, contract, workflow, and infrastructure tests
Dockerfile.pipeline           one production container image
```

Generated `data/`, `results/`, `artifacts/`, `dashboard/`, and `build/` files
are intentionally not source code and are ignored by Git.

## Quality checks

Before a deployable revision:

```bash
python -m ruff check .
python -m pytest
python -m compileall -q src scripts
terraform -chdir=infra/aws fmt -check -recursive
terraform -chdir=infra/aws validate
```

The production container is also built in CI. Its base image, direct runtime
dependencies, non-root user, and Git revision are pinned or verified.

## AWS in one paragraph

Terraform creates private versioned artifact and site buckets, one CloudFront
HTTPS distribution, one ECR repository, one ECS cluster/task definition, a
small public VPC, CloudWatch logs, and—when enabled—one EventBridge schedule.
The task has read access to the private artifact prefix, narrowly scoped write
access to runtime outputs, and permission to replace only the site
`index.html`. See
[infra/aws/README.md](infra/aws/README.md) for exact commands.
