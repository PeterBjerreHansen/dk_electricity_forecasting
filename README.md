# Danish electricity forecasts

A compact, production-oriented project for forecasting Danish day-ahead
electricity prices in DK1 and DK2. Its technical centre is an explicitly
configured Chronos-2 LoRA model with point-in-time weather covariates. A fixed
weighted-median model exists only as a transparent operational fallback and as
an analytical reference.

The repository is deliberately split into two paths:

- **Production** refreshes data, runs one immutable Chronos release, validates
  the forecast, and publishes one atomic pointer to a completed run.
- **Diagnostics** scores forecasts saved earlier and compares model releases.
  Diagnostics can fail without changing or delaying production.

There is no automatic model selection. The production choice lives in
[`config/production.json`](config/production.json) and changes through an
ordinary reviewed commit.

New to the project? Read the [codebase tour](docs/codebase-tour.md). The
[forecasting contract](docs/forecasting-contract.md) defines the time and data
semantics used throughout the code.

## Quick start

Python 3.10 or newer is supported. Python 3.11 is used by the production
containers.

```bash
python -m pip install -e ".[dev,app,chronos]"
cp .env.example .env
python -m pytest
python -m ruff check .
```

Lighter installs are available when Chronos is not needed:

```bash
python -m pip install -e ".[dev]"                 # library and tests
python -m pip install -e ".[app]"                 # dashboard
python -m pip install -e ".[notebooks,tuning]"    # research notebooks
```

Runtime artifacts are file based and ignored by Git. They live under `data/`,
`artifacts/`, `results/`, and `app_data/`.

## What is forecast

The target is the hourly day-ahead area price in DKK/MWh for every hour of the
next Danish local delivery day.

| Delivery time | Source product | Hourly target |
|---|---|---|
| Before 2025-10-01 | Native hourly `Elspotprices` | Source value |
| From 2025-10-01 | Native 15-minute `DayAheadPrices` | Arithmetic mean of four quarters |

The delivery horizon is built between Copenhagen midnights and then converted
to UTC. A daylight-saving transition therefore produces 23, 24, or 25 hours
per area without ambiguous local-time keys.

Every row records its target regime and aggregation. See the
[data card](docs/data-card.md) and
[Energi Data Service processing contract](docs/data-processing/energi_data_service_v1.md).

## Build the data

Fetch immutable raw price responses and build the canonical hourly panel:

```bash
python scripts/fetch_eds_prices.py --areas DK1 DK2
python scripts/build_price_panel.py --allow-incomplete-recent
```

Fetch Open-Meteo Previous Runs weather and build the canonical long weather
table:

```bash
python scripts/fetch_open_meteo_previous_runs.py
python scripts/build_open_meteo_weather_features.py
```

The two canonical model inputs are:

```text
data/model_ready/price_panel_hourly_v1.parquet
data/features/weather_open_meteo_area_hourly_long_open_meteo_previous_runs_v1.parquet
```

Weather is not permanently merged into the price panel. The training,
backtesting, and serving paths all call the same availability-safe join.

For each `(area, valid hour, weather model, parameter)`, that join selects the
newest forecast whose availability timestamp is no later than the forecast's
`information_cutoff_utc`. Lead time and source vintage remain metadata; they
are not encoded in the stable model feature name. Wind direction is excluded
because the source builder's arithmetic area mean is not a valid circular
statistic.

Open-Meteo does not expose an observed publication timestamp for Previous Runs.
The project therefore labels `valid time - requested lead` as a synthetic
availability/reference-time proxy. This limitation is retained in the data and
model manifests rather than hidden. See the
[Open-Meteo contract](docs/data-processing/open_meteo_weather_v1.md).

## Production model

[`config/production.json`](config/production.json) is the sole production model
selection:

```json
{
  "primary": {
    "model": "chronos_weather",
    "artifact_path": "artifacts/models/<configured-artifact>"
  },
  "fallback": {
    "model": "weighted_median_v1"
  },
  "schema_version": 1
}
```

The checked-in file contains the actual artifact path. A trained Chronos
manifest supplies the immutable `release_id` and `artifact_content_sha256`.
Every published row records the logical model name, release ID, artifact hash,
forecast run ID, information cutoff, and delivery date. Changing weights does
not silently reuse an analytical identity.

List the two models permitted on the live path:

```bash
python scripts/run_publish_forecast.py --list-models
```

Chronos publishes `q10`, `q50`, and `q90`, with `y_pred = q50`. Its artifact
manifest is a weights-to-features contract: it records the base model,
covariates, selection and missing-value policies, target definition, training
interval, source hashes, code commit, model-file hashes, and release identity.
Serving rejects incompatible or corrupt artifacts.

Training and serving use identical weather semantics: choose the newest
eligible value, perform no temporal fill, validate required coverage, and only
then replace remaining missing covariate cells with the declared zero value.
Retrain explicitly; production never updates model weights:

```bash
python scripts/train_chronos_lora.py \
  --base-model-revision <immutable-revision> \
  --overwrite
```

Do not overwrite an artifact currently referenced by production. Export and
evaluate a new artifact, then change `config/production.json` in a reviewed
commit. See the [Chronos model card](docs/model-card-chronos-v1.md).

## Forecast-time contract

These timestamps have separate meanings:

| Field | Meaning |
|---|---|
| `information_cutoff_utc` | Latest information the forecast claims to know |
| `decision_deadline_utc` | Latest time a live publication may become durable |
| `generated_at_utc` | When the forecasting process started |
| `committed_at_utc` | When the immutable run was durably completed |
| `delivery_date_local` | Copenhagen calendar day being forecast |

The scheduled task begins at 10:00 `Europe/Copenhagen`; live publication must
complete by local noon. The actual process start is the default information
cutoff. An explicit historical cutoff creates a `replay`, which cannot update
the live pointer or enter saved-forecast scoring.

Price rows are eligible under:

```text
price_available_at_utc < information_cutoff_utc
```

Weather rows are eligible under:

```text
forecast_available_at_utc <= information_cutoff_utc
```

## Publish one forecast

After building the price and weather artifacts and exporting the configured
Chronos artifact:

```bash
python scripts/run_publish_forecast.py --allow-incomplete-panel
```

For a historical reconstruction:

```bash
python scripts/run_publish_forecast.py \
  --run-kind replay \
  --information-cutoff-utc 2026-06-30T08:00:00Z \
  --delivery-date-local 2026-07-01 \
  --allow-incomplete-panel
```

The primary Chronos failure policy is explicit. If artifact loading, contract
validation, weather coverage, or inference fails, the fixed
`weighted_median_v1` model may publish a **degraded** run. Its rows and manifest
record the requested model, the model actually published, and the primary
failure. The substitution is never presented under the Chronos name.

## Publication protocol

Each attempt is an immutable directory:

```text
artifacts/forecast_runs/<run_id>/
  predictions.parquet
  model_scores.parquet
  forecast_dashboard.json
  manifest.json
  COMPLETED.json

artifacts/latest.json
```

Locally, the run is prepared in a hidden sibling directory and atomically
renamed into place. In S3, ordinary artifacts are uploaded first,
`COMPLETED.json` is uploaded last, and only then is the single `latest.json`
pointer replaced. Readers follow the pointer and accept only completed runs, so
they cannot combine files from different attempts.

## Daily production and independent diagnostics

The short production path refreshes data and publishes one forecast:

```bash
python scripts/run_daily_pipeline.py --with-weather --skip-backtest
```

Inspect its subprocess commands without running them:

```bash
python scripts/run_daily_pipeline.py --with-weather --skip-backtest --dry-run
```

Recent rolling-origin diagnostics are separate:

```bash
python scripts/run_recent_diagnostics.py --allow-incomplete-panel
```

Score only completed forecasts that were saved before their deadlines, without
recomputing predictions:

```bash
python scripts/score_published_forecasts.py --allow-incomplete-panel
```

Compare two already-generated model prediction sets descriptively:

```bash
python scripts/run_model_comparison.py \
  --predictions results/example/predictions.parquet \
  --reference-model weighted_median_v1 \
  --comparison-model chronos_weather \
  --start-utc 2026-04-01T00:00:00Z \
  --end-utc 2026-07-01T00:00:00Z \
  --output-dir results/model_comparison/chronos_vs_weighted_median
```

The report includes exact pairing, MAE/RMSE/bias, probabilistic scores,
stratified results, and moving-block bootstrap intervals. It reports evidence;
it never changes production configuration. See the
[evaluation guide](docs/evaluation.md).

## Repository map

```text
src/dkenergy_data/
  sources/       HTTP clients and immutable raw-response writers
  build/         deterministic price and weather builders

src/dkenergy_forecast/
  backtesting/   forecast origins, Danish horizons, rolling execution
  features/      price features and point-in-time weather selection
  models/        baselines, Chronos adapters, production/comparison registries
  evaluation/    metrics, exact pairing, bootstrap CIs, stratification
  operations/    forecast contracts, live publication, diagnostics
  publishing/    immutable runs, checksums, completion receipts, latest pointer

scripts/          thin command-line entry points
app/              read-only Streamlit dashboard
infra/aws/        minimal AWS deployment
notebooks/        exploratory and didactic analysis
```

## Dashboard and containers

Run Streamlit against local artifacts:

```bash
python -m streamlit run app/streamlit_app.py
```

Or build the separate lightweight web and Chronos pipeline images:

```bash
docker compose build
docker compose up dashboard
docker compose --profile jobs run --rm pipeline
```

Both production images run as a non-root user, install constrained production
dependencies without editable mounts, and carry the build Git SHA. The web
container is an artifact reader; the pipeline container owns mutation.

## AWS MVP

[`infra/aws/`](infra/aws/) deploys:

- Private, versioned, encrypted S3 artifact storage.
- An optional near-zero-idle-cost public S3 website for a prebuilt forecast
  dashboard.
- Immutable ECR repositories for separate web and pipeline images.
- A read-only Streamlit task behind ALB and CloudFront HTTPS.
- A write-capable scheduled Chronos pipeline task.
- An independent scheduled saved-forecast scoring task.
- Separate IAM roles and security groups for web and pipeline workloads.
- CloudWatch logs, ECS failure notifications, and a post-deadline publication
  check against `latest.json` and the referenced completion receipt.

Schedules are disabled by default. Upload a compatible Chronos artifact before
enabling production. Deployment details are in the
[AWS guide](infra/aws/README.md).

For the small public portfolio view, build one self-contained HTML file from a
dashboard payload and upload it to the dedicated site bucket:

```bash
make static-dashboard
aws s3 cp build/static-dashboard/index.html \
  "$(terraform -chdir=infra/aws output -raw static_site_s3_uri)/index.html" \
  --content-type 'text/html; charset=utf-8'
```

The static page has no Python server and cannot read private S3 data at request
time. Rebuild and upload it after selecting a newer dashboard payload.

## Quality checks

```bash
python -m pytest
python -m ruff check .
python -m compileall -q src scripts app
terraform -chdir=infra/aws fmt -check -recursive
terraform -chdir=infra/aws init -backend=false
terraform -chdir=infra/aws validate
```

CI runs the Python suite across supported versions, validates Terraform, and
smoke-checks both production images. Production deployment is a separate manual
workflow in [`.github/workflows/production.yml`](.github/workflows/production.yml).
