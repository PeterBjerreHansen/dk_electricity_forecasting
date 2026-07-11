# Danish Energy Forecasts

Production-oriented reference skeleton for Danish day-ahead electricity price
forecasting. The project keeps the core simple: public API ingestion, immutable
raw responses, normalized parquet tables, leakage-safe rolling-origin
backtests, a small production forecast set, notebook-first comparison
experiments, one default Chronos-2 LoRA weather model, and a minimal
artifact-only Streamlit dashboard.

## Setup

```bash
python -m pip install -e ".[dev,app,chronos]"
cp .env.example .env
python -m pytest
```

For a dashboard-only install, `python -m pip install -e ".[app]"` is enough.
Optional modeling extras:

```bash
python -m pip install -e ".[tuning]"
python -m pip install -e ".[notebooks]"
python -m pip install -e ".[catboost]"
```

The default workflow is file based and writes ignored runtime artifacts under
`data/`, `results/`, `artifacts/`, and `app_data/`.

New to the repository? Start with the [codebase tour](docs/codebase-tour.md).
The detailed model-comparison workflow is documented in
[the evaluation guide](docs/evaluation.md).

## Data Sources

Price data comes from Energi Data Service. The ingestion script archives raw
JSON bytes and metadata for DK1/DK2, then builds an hourly model-ready panel:

```bash
python scripts/fetch_eds_prices.py --areas DK1 DK2
python scripts/build_price_panel.py --allow-incomplete-recent
```

Every price row declares its target contract. Before 2025-10-01 the target is a
native hourly price; from that local-market boundary onward it is the arithmetic
mean of four native quarter-hour prices. `market_regime`,
`native_resolution_minutes`, `target_aggregation`, and `target_definition`
travel with panels, horizons, predictions, and evaluation strata.

Open-Meteo Previous Runs provides representative DK1/DK2 weather forecast
features. Raw API payloads are stored before normalized forecasts and the
canonical long area-hour weather feature table are built:

```bash
python scripts/fetch_open_meteo_previous_runs.py
python scripts/build_open_meteo_weather_features.py
```

Weather outputs preserve the production timing/provenance fields
`forecast_reference_time`, `valid_time`, `lead_time_hours`, `model`, `variable`,
and `price_area`, while retaining existing project columns such as
`forecast_available_at_utc`, `ds_utc`, `weather_model`, and `parameter_id`.
Because the Previous Runs endpoint has no observed initialization timestamp,
the reference and availability times are explicitly labeled synthetic proxies;
`weather_vintage_id` is a project-generated identity, not an upstream run ID.

## Architecture

```text
src/dkenergy_data/
  sources/      API clients and raw response writers
  build/        raw-to-normalized and model-ready parquet builders

src/dkenergy_forecast/
  backtesting/  origin and horizon builders plus rolling-origin execution
  features/     price and availability-safe weather feature joins
  models/       baselines, fixed CatBoost adapter, Chronos adapters,
                and the production model registry
  evaluation/   metrics, frozen splits, paired comparisons, bootstrap CIs,
                stratification, and champion-promotion rules
  operations/   separate live publication, recent diagnostics, and daily wiring
  publishing/   transactional immutable runs, checksums, score eligibility,
                and atomic latest dashboard exports

scripts/
  fetch_*.py, build_*.py, run_*_backtest.py, run_publish_forecast.py,
  run_recent_diagnostics.py, score_published_forecasts.py,
  run_evaluation_arena.py, run_daily_pipeline.py

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

Live publication and diagnostics are deliberately separate. Run a live forecast
before its decision cutoff:

```bash
python scripts/run_publish_forecast.py --allow-incomplete-panel
```

The live origin is local market noon on the execution date. A live run started
after that cutoff is rejected. Supplying `--forecast-origin-utc` defaults to a
`replay` run; replay and shadow runs never replace latest. Run rolling-origin
diagnostics independently so their latency or failure cannot block publication:

```bash
python scripts/run_recent_diagnostics.py --allow-incomplete-panel
python scripts/run_baseline_backtest.py --allow-incomplete-panel
```

Score saved forecasts that were actually published earlier, without
recomputing any model predictions:

```bash
python scripts/score_published_forecasts.py --allow-incomplete-panel
```

The default latest-forecast set is deliberately small:
`same_hour_last_week`,
`median_weekday_exp_hl4_floor10_42d__median_weekend_exp_hl28_floor20_56d`,
and `chronos2_lora_calendar_weather_ctx1024_v1`.

The Chronos production model loads a manually exported LoRA artifact from
`artifacts/models/chronos2_lora_calendar_weather_ctx1024_v1/`, consumes the
Open-Meteo long weather feature parquet, and publishes `q10`, `q50`, `q90`,
with `y_pred=q50`. It fails if the required weather artifact, covariate schema,
or default full future-covariate coverage is missing. A zero fallback is
available only when both the artifact and runtime explicitly select it. Daily publishing loads the trained
LoRA artifact and does not update its weights; export a new artifact explicitly
when you want to retrain:

```bash
python scripts/train_chronos_lora.py
```

Chronos LoRA artifact schema v2 records the point-in-time weather selection,
role-specific fill rules, coverage threshold, and fallback policy. Training and
context use forward-only fill; future covariates never borrow from another valid
time. Runtime refuses to serve with a coverage policy that differs from the
artifact. Older artifacts are rejected and must be retrained.
New exports also record the random seed, optional base-model revision,
Chronos/PyTorch versions, training-data hashes, and hashes of every model file.

`rolling_median_hour_weekend_56d`, `rolling_median_local_hour_28d`,
`catboost_price_manual_v1`, and `chronos_zero_shot_v1` live in the comparison
registry for notebooks and smoke diagnostics. They are intentionally not
accepted by the production publish command.

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

These frames are historical modeling/backtest artifacts. The production Chronos
model instead consumes the long Open-Meteo feature parquet directly. Larger
windows are explicit, for
example `--frame-kind custom --days 730 --output-path data/features/weather_experiment_frame_backtest_730d.parquet`.

CatBoost development remains notebook-first. A fixed adapter,
`catboost_price_manual_v1`, remains available through the comparison registry,
but it is not part of production latest-forecast publishing.

## Evaluation

Rolling-origin evaluation uses local market-noon forecast origins and explicit
price availability metadata. Price history is eligible when
`price_available_at_utc < forecast_origin_utc`; weather remains masked by
`forecast_available_at_utc <= forecast_origin_utc`. Horizon builders strip
target columns before model prediction; actuals are joined only after
predictions are produced. Tests cover UTC/local alignment, DST day lengths,
price and weather availability masking, and leakage prevention.

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

Metrics include MAE, RMSE, bias, pinball loss, interval coverage/width/score,
weighted interval score, and calibration error. Candidate promotion uses exact
candidate/champion key pairing, a deterministic moving-block bootstrap over
forecast origins, and guardrails by month, area, local hour, DST, price regime,
extreme/negative prices, and target regime:

```bash
python scripts/run_evaluation_arena.py \
  --predictions results/example/predictions.parquet \
  --candidate candidate_label \
  --champion champion_label \
  --splits-json config/evaluation_splits.example.json \
  --split test \
  --output-dir results/evaluation_arena/candidate_vs_champion
```

Commit the small deterministic JSON/Markdown report after review. Prediction
parquets may remain in durable artifact storage; their SHA-256 is recorded in
the report.

## Daily Job

Run the file-based daily pipeline. By default this refreshes prices, runs the
baseline backtest, and publishes the registry-default forecast artifacts used by
Streamlit. Weather refresh is explicit so data ingestion remains independent of
model-registry choices:

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
python scripts/run_daily_pipeline.py --skip-weather
python scripts/run_daily_pipeline.py --skip-backtest
python scripts/run_daily_pipeline.py --with-diagnostics
python scripts/run_daily_pipeline.py --strict-panel
```

Use `--with-weather` when the Open-Meteo source artifacts should be refreshed
before publishing. Weather-aware models consume the existing long weather feature
artifact and fail rather than silently fetching or falling back if that artifact
or its covariate schema is missing.

In production, the cloud wrapper runs the live path with `--skip-backtest`,
hydrates the small runtime state it needs from S3, writes a transactional
immutable run, and uploads the `latest/` dashboard artifact last. Recent
diagnostics and `score_published_forecasts.py` are separate operator-scheduled
jobs. Existing history files may be synchronized, but the live job does not
recompute either diagnostics or published-history scores.

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
results/published_forecast_history/predictions.parquet
results/published_forecast_history/model_scores.parquet
app_data/forecast_dashboard.json
```

It shows DK1/DK2 actual day-ahead prices, the next available forecast,
point/quantile forecast rows when present, published forecast performance when
available, and recent rolling-origin diagnostics as a fallback.

## Docker

```bash
docker compose build
docker compose up dashboard
docker compose --profile jobs run --rm pipeline
```

Compose mirrors the production artifact layout with separate images:

- `Dockerfile.web` builds the lightweight Streamlit dashboard image.
- `Dockerfile.pipeline` builds the heavier Chronos/PyTorch pipeline image.

Both images install through exact direct-production pins in
`constraints-production.txt`; change those pins deliberately and verify both
container builds.

It uses a file-backed store under ignored `cloud_store/` and a runtime workdir
under ignored `runtime/`. Bootstrap the local store with the trained LoRA
artifact before running the real pipeline:

```bash
mkdir -p cloud_store/models
rsync -a artifacts/models/chronos2_lora_calendar_weather_ctx1024_v1 cloud_store/models/
```

The containerized pipeline defaults to rolling production windows rather than
fetching all historical data on every run. Override `EDS_START` or
`OPEN_METEO_START` only when you intentionally want to rebuild a longer slice.

## AWS MVP

The manual `Production Deploy` GitHub Actions workflow deploys a small AWS MVP
from `infra/aws/`: separate web and pipeline images, private S3 artifacts, ECR,
ECS/Fargate, EventBridge Scheduler, ALB, CloudFront HTTPS, and CloudWatch logs.

Set the backend/deploy values, build the stack, and upload the one-time Chronos
LoRA artifact before enabling scheduled runs:

```bash
export TF_STATE_BUCKET=<terraform-state-bucket>
export AWS_REGION=eu-central-1
make aws-deploy
export AWS_MODEL_ARTIFACT_URI="$(terraform -chdir=infra/aws output -raw model_artifact_uri)"
make aws-bootstrap-model
AWS_ENABLE_PIPELINE_SCHEDULE=true make aws-deploy
```

The scheduled production task runs at 10:00 `Europe/Copenhagen`, leaving two
hours before the noon decision cutoff. It loads the existing LoRA artifact and
refreshes weather data by default; it does not retrain Chronos or run recent
diagnostics on every update.

For a local cloud-layout smoke test:

```bash
python scripts/run_cloud_pipeline.py --artifact-store-uri file:///tmp/dkenergy-store --workdir /tmp/dkenergy-work --dry-run
```

## CI

GitHub Actions runs Ruff, pytest, and compile checks on Python 3.10, 3.11, and
3.12, validates Terraform, and builds/smoke-checks both production containers.
The workflow lives in `.github/workflows/ci.yml`.
Deployments are deliberately separate: start `.github/workflows/production.yml`
manually after configuring the `production` GitHub environment and the
`AWS_DEPLOY_ROLE_ARN` and `TF_STATE_BUCKET` repository secrets. The deployment
reruns the tests before building/pushing images and applying Terraform.
