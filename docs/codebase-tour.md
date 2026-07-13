# Codebase tour: Danish electricity forecasting, from raw bytes to a published forecast

This document is a guided tour of the repository for a developer or data scientist who is new to production forecasting. It explains not only *where* code lives, but also *why* the boundaries exist and which guarantees each layer is responsible for.

The project forecasts Danish day-ahead electricity prices for DK1 and DK2. It is deliberately file-oriented: raw API responses, normalized tables, model-ready panels, model artifacts, predictions, score tables, and dashboard payloads are all explicit files. That makes a forecast inspectable without a database and keeps the operational path small enough to understand end to end.

The central idea is:

> A production forecast is not just a number. It is a number tied to a target definition, an information cutoff, a model artifact, a weather vintage, an execution lifecycle, and an immutable publication record.

If you remember only one thing while reading the code, remember that the timestamp of an observation and the timestamp at which that observation became usable are different facts.

## 1. The system in one picture

```text
Energi Data Service                 Open-Meteo Previous Runs
        |                                      |
        v                                      v
append-only raw JSON + manifest     append-only raw JSON + manifest
        |                                      |
        v                                      v
normalized source parquet           normalized long forecast parquet
        |                                      |
        v                                      v
hourly DK1/DK2 price panel          area-hour weather feature table
        |                                      |
        +------------------+-------------------+
                           |
                           v
              origin + delivery-day horizon
                           |
                  availability filtering
                           |
                           v
          production registry -> model predictions
                           |
                   schema/key validation
                           |
                           v
             immutable forecast run directory
                           |
              +------------+-------------+
              |                          |
              v                          v
      latest live promotion       later actuals arrive
              |                          |
              v                          v
       Streamlit dashboard       published-run scoring
                                         |
                                         v
                             candidate/champion arena
```

There are three intentionally different evaluation loops:

1. **Rolling-origin backtests** recompute models at historical origins and answer, “How would this model have behaved?”
2. **Published-forecast scoring** never recomputes predictions and answers, “How did the forecasts we actually saved perform?”
3. **The evaluation arena** compares an exactly paired candidate and champion under an explicit promotion policy.

Keeping those questions separate is one of the most important architectural choices in the repository.

## 2. The six mental rules

### Rule 1: UTC is the storage and joining clock; Copenhagen time defines the market day

`ds_utc` is the canonical event timestamp used for joins, ordering, and uniqueness. `ds_local` and derived local-calendar columns describe the Danish market meaning of that timestamp.

This is necessary because a Danish delivery day can contain 23, 24, or 25 hourly rows around daylight-saving transitions. “Take the next 24 UTC hours” is therefore not the same as “forecast tomorrow’s Danish delivery day.”

The foundational helpers live in [types.py](../src/dkenergy_forecast/types.py), and delivery-day construction lives in [horizons.py](../src/dkenergy_forecast/backtesting/horizons.py).

### Rule 2: event time is not availability time

A price row has a delivery timestamp, `ds_utc`, and a modeled publication timestamp, `price_available_at_utc`. History is eligible only when:

```text
price_available_at_utc < forecast_origin_utc
```

A weather row likewise has a valid time and a forecast-availability timestamp. Weather is eligible when:

```text
forecast_available_at_utc <= forecast_origin_utc
```

These inequalities are explicit and intentionally differ. Do not replace either with a filter on `ds_utc` alone.

### Rule 3: the future frame contains known structure, never target values

The horizon contains identifiers, timestamps, calendar facts, target-contract metadata, and the forecast origin. Target columns such as `y`, `price_dkk_per_mwh`, and `price_eur_per_mwh` are stripped before a model receives the future frame.

The rolling-origin engine validates that every requested key is predicted exactly once, then joins actuals only after prediction. See [rolling_origin.py](../src/dkenergy_forecast/backtesting/rolling_origin.py).

### Rule 4: a model label is an operational contract

The production registry contains only models that may participate in latest-forecast publishing. The comparison registry contains notebook and smoke-test models. Moving a model into production is therefore a code review and deployment decision, not a command-line accident.

See [registry.py](../src/dkenergy_forecast/models/registry.py) and [comparison_registry.py](../src/dkenergy_forecast/models/comparison_registry.py).

### Rule 5: immutable runs are evidence; `latest` is only a pointer-like convenience

Each published run is written as a complete immutable directory with a manifest and checksums. Mutable exports under `results/latest_forecast/`, `results/recent_scores/`, and `app_data/` exist for consumers such as Streamlit.

When investigating what happened, begin with the immutable run, not with a mutable convenience file.

### Rule 6: replaying history is not live publication

Runs are explicitly classified as `live`, `shadow`, or `replay`. Only a timely `live` run may update latest exports. A timely `shadow` run may later be scored but does not become latest. A `replay` is retrospective and is excluded from published-performance scoring.

The lifecycle is implemented in [publish_forecast.py](../src/dkenergy_forecast/operations/publish_forecast.py) and enforced again when immutable runs are read in [artifacts.py](../src/dkenergy_forecast/publishing/artifacts.py).

## 3. Directory map

### Source packages

| Path | Responsibility |
|---|---|
| [`src/dkenergy_data/sources/`](../src/dkenergy_data/sources/) | HTTP clients, retries, raw response writes, request metadata, and response hashes. |
| [`src/dkenergy_data/build/`](../src/dkenergy_data/build/) | Deterministic transformation from archived raw responses to normalized and model-ready tables. |
| [`src/dkenergy_forecast/types.py`](../src/dkenergy_forecast/types.py) | Shared dataframe schemas, time helpers, target contract, availability helpers, and the `ForecastModel` protocol. |
| [`src/dkenergy_forecast/backtesting/`](../src/dkenergy_forecast/backtesting/) | Forecast-origin selection, horizon construction, and leakage-safe rolling-origin execution. |
| [`src/dkenergy_forecast/features/`](../src/dkenergy_forecast/features/) | Price, calendar, weather, ensemble, and derived feature construction. |
| [`src/dkenergy_forecast/models/`](../src/dkenergy_forecast/models/) | Baselines, CatBoost, Chronos adapters, and the production/comparison registries. |
| [`src/dkenergy_forecast/evaluation/`](../src/dkenergy_forecast/evaluation/) | Point/probabilistic metrics, frozen splits, stratification, paired comparisons, bootstrap intervals, and promotion policy. |
| [`src/dkenergy_forecast/operations/`](../src/dkenergy_forecast/operations/) | User-facing workflows: publish, recent diagnostics, and daily orchestration. |
| [`src/dkenergy_forecast/publishing/`](../src/dkenergy_forecast/publishing/) | Artifact schemas, transactional immutable writes, latest promotion, checksums, dashboard payloads, and published-run scoring. |
| [`src/dkenergy_forecast/storage.py`](../src/dkenergy_forecast/storage.py) | Local, `file://`, and S3 artifact access. |
| [`src/dkenergy_forecast/cloud_pipeline.py`](../src/dkenergy_forecast/cloud_pipeline.py) | Hydration of runtime state, container orchestration, freshness checks, and artifact-store uploads. |

### Entrypoints and product surfaces

| Path | Responsibility |
|---|---|
| [`scripts/`](../scripts/) | Small command-line entrypoints. They parse arguments and call package code. |
| [`app/streamlit_app.py`](../app/streamlit_app.py) | Artifact-only dashboard for forecasts, actual prices, diagnostics, published performance, and run provenance. |
| [`notebooks/`](../notebooks/) | Exploratory data analysis and model development. Notebooks are consumers of library code, not the production scheduler. |
| [`tests/`](../tests/) | Unit, integration, contract, CLI, leakage, DST, artifact, cloud, and infrastructure-wiring tests. |
| [`infra/aws/`](../infra/aws/) | Terraform for S3, ECR, ECS/Fargate, the web path, scheduling, IAM, and logs. |
| [`.github/workflows/`](../.github/workflows/) | CI and deployment workflows. |
| [`Dockerfile.pipeline`](../Dockerfile.pipeline) | Production data/forecast job image. |
| [`Dockerfile.web`](../Dockerfile.web) | Streamlit image. |
| [`docker-compose.yml`](../docker-compose.yml) | Local container wiring. |

### Runtime files

The default layout is centralized in [layout.py](../src/dkenergy_forecast/layout.py). These directories are runtime state and are normally ignored by Git.

| Runtime path | Meaning |
|---|---|
| `data/raw/energi_data_service/` | Archived EDS JSON, metadata, and append-only manifest entries. |
| `data/raw/open_meteo/` | Archived Open-Meteo JSON and append-only manifest entries. |
| `data/normalized/` | Source-shaped normalized parquet tables. |
| `data/model_ready/price_panel_hourly_v1.parquet` | Canonical hourly target panel. |
| `data/model_ready/price_panel_hourly_v1.qa.json` | Price-panel build audit and quality contract. |
| `data/features/weather_open_meteo_area_hourly_long_open_meteo_previous_runs_v1.parquet` | Canonical long weather feature artifact. |
| `artifacts/models/.../` | Exported trained model artifacts, including the Chronos manifest. |
| `artifacts/forecast_runs/<run_id>/` | Immutable forecast publications. |
| `results/latest_forecast/` | Mutable latest live forecast exports. |
| `results/recent_scores/` | Separate rolling-origin diagnostic artifacts. |
| `results/published_forecast_history/` | Scores derived from saved eligible forecast runs. |
| `results/evaluation_arena/` | Default candidate/champion evaluation reports. |
| `app_data/forecast_dashboard.json` | Preassembled dashboard payload. |

## 4. The core dataframe vocabulary

Most of the repository communicates through dataframes. Understanding a few columns makes the call graph much easier to follow.

| Column | Meaning |
|---|---|
| `unique_id` | Time-series identity, normally `day_ahead_price_DK1` or `day_ahead_price_DK2`. |
| `area` / `price_area` | Danish bidding zone, DK1 or DK2. |
| `ds_utc` | Canonical valid/delivery timestamp in UTC. |
| `ds_local` | The same instant represented in `Europe/Copenhagen`. |
| `forecast_origin_utc` | The information cutoff represented by the forecast. |
| `horizon` | One-based ordinal within each origin and series, after sorting by `ds_utc`. |
| `y` | Actual target value in DKK/MWh. |
| `y_pred` | Point forecast. For production Chronos this equals q50. |
| `q10`, `q50`, `q90` | Optional predictive quantiles. They must be all present or all absent on a row, and must not cross. |
| `model_name`, `model_version` | Adapter-level model identity. |
| `model_label` | Registry/publication identity used in score tables and the dashboard. |
| `price_available_at_utc` | When a target row is considered usable as historical information. |
| `forecast_available_at_utc` | When a weather forecast row is considered usable. |
| `forecast_reference_time` | The weather run/reference timestamp, observed or explicitly synthetic. |
| `weather_vintage_id` | Stable identifier for a weather vintage. |
| `dataset_version` | Version of the built panel or feature artifact. |

The minimum panel and prediction schemas are defined in [types.py](../src/dkenergy_forecast/types.py). Publication adds stricter requirements in [artifacts.py](../src/dkenergy_forecast/publishing/artifacts.py).

## 5. Price-data lifecycle

### 5.1 Fetch: preserve the source response

[`fetch_eds_prices.py`](../scripts/fetch_eds_prices.py) chooses the correct EDS dataset on each side of the October 2025 market transition, requests DK1/DK2 in date chunks, and delegates HTTP behavior to [energidataservice.py](../src/dkenergy_data/sources/energidataservice.py).

For every response, the source layer stores:

- The exact response bytes.
- A retrieval timestamp.
- Request URL and parameters.
- HTTP status and record count.
- SHA-256 of the response and saved file.
- A batch identifier and relative raw path in `manifest.jsonl`.

The fetch script skips an existing batch only after checking that it is readable JSON with the expected shape and, when available, the expected manifest hash. `--force` creates another timestamped raw batch; it does not silently rewrite the earlier file.

This layer answers, “What did the external source return?” It should not contain modeling decisions.

### 5.2 Normalize: retain source semantics

[`eds_prices_v1.py`](../src/dkenergy_data/build/eds_prices_v1.py) reads raw batches, validates source fields and local timestamps, converts source timestamps to UTC, records source resolution, and deduplicates repeated retrievals.

Deduplication uses source dataset, source timestamp, and area as the logical key. Conflicting duplicates inside a batch fail; otherwise the latest retrieval is retained in the normalized view. Raw files themselves remain available for audit.

### 5.3 Stitch: build one hourly target contract

The model-ready panel joins two source eras:

| Danish delivery time | Source | Native resolution | Hourly target construction |
|---|---|---:|---|
| Before `2025-10-01 00:00 Europe/Copenhagen` | `Elspotprices` | 60 minutes | Identity. |
| At or after the boundary | `DayAheadPrices` | 15 minutes | Arithmetic mean of exactly four quarter-hours. |

The builder rejects non-hourly legacy timestamps, non-quarter-hour boundaries, missing historical quarter-hours, conflicting overlaps, gaps in each area’s UTC sequence, and unequal DK1/DK2 coverage. With `--allow-incomplete-recent`, only the latest incomplete quarter-hour group per area may be omitted.

The result contains price values plus calendar and provenance columns, and it writes a QA JSON alongside the parquet. [`load_price_panel()`](../src/dkenergy_forecast/io.py) can require that QA status to be `final_historical`; live refresh workflows may opt into an incomplete-recent panel explicitly.

### 5.4 Availability: model what was knowable

Every panel used by forecasting has a deterministic price availability
timestamp: local market noon on the previous Copenhagen calendar day. The
policy is encoded by
[`add_price_availability()`](../src/dkenergy_forecast/types.py), and
[`load_price_panel()`](../src/dkenergy_forecast/io.py) restores the column when
loading an older or compact panel artifact that does not persist it.

For a delivery hour on Tuesday, the assumed availability is Monday at 12:00 Copenhagen time. A forecast at exactly Monday noon does **not** get to use that Tuesday target as history, because price history uses the strict `< forecast_origin_utc` rule.

This timestamp is a project policy, not an observed exchange publication event. If the operational source later supplies observed publication times, the correct change is to preserve those times explicitly—not to bury the change inside a model.

## 6. Target regimes are part of the schema

The panel does not pretend that one unchanging physical product exists across the whole history. [`add_target_contract()`](../src/dkenergy_forecast/types.py) attaches:

| Column | Before the boundary | At/after the boundary |
|---|---|---|
| `market_regime` | `native_hourly` | `native_quarter_hour` |
| `native_resolution_minutes` | `60` | `15` |
| `target_aggregation` | `identity` | `arithmetic_mean_of_4_quarter_hours` |
| `target_definition` | `hourly_day_ahead_area_price_dkk_per_mwh` | Same hourly modeling target. |

These columns travel through horizon metadata, rolling-origin outputs, model artifacts, and evaluation strata. A model still predicts an hourly series in both periods, but evaluation can expose whether performance changed when the native market resolution changed.

When adding a future target regime, update the contract in one place, make the source builder prove that its aggregation matches the declared regime, and add boundary tests. Do not infer target meaning from a filename or date inside a notebook.

## 7. Weather-data lifecycle and provenance

### 7.1 What is fetched

[`fetch_open_meteo_previous_runs.py`](../scripts/fetch_open_meteo_previous_runs.py) requests Open-Meteo Previous Runs data for:

- Three weather models: GFS Global, ICON EU, and MET Norway Nordic.
- Ten representative locations: five in DK1 and five in DK2.
- Seven base variables, including temperature, wind, radiation, cloud, and precipitation.
- One-day and two-day previous-run leads.

Like the price source, [open_meteo.py](../src/dkenergy_data/sources/open_meteo.py) archives raw response bytes and an append-only manifest with request and hash metadata.

### 7.2 The important limitation: reference times are synthetic proxies

The Previous Runs endpoint used here does not expose an observed model initialization or publication timestamp. The builder therefore makes the limitation explicit:

```text
forecast_reference_time = valid_time - requested lead
forecast_available_at_utc = forecast_reference_time
forecast_reference_time_is_observed = false
forecast_reference_time_type = synthetic_valid_time_minus_lead
forecast_availability_time_type = synthetic_reference_time_proxy
```

The synthetic provenance is retained in both normalized and area-feature tables and in the Chronos model manifest. It is leakage-safe under the declared proxy, but it is not proof of the provider’s real publication latency.

This distinction matters scientifically. A future source with observed initialization and availability times should populate the same semantic fields with observed values, allowing downstream code to remain unchanged.

### 7.3 From locations to area-level features

[`open_meteo_weather_v1.py`](../src/dkenergy_data/build/open_meteo_weather_v1.py) normalizes every `(model, lead, location, valid_time, variable)` value, then aggregates locations to an area mean for each `(area, valid_time, model, lead, variable)`.

The long area table is canonical. A feature has a name such as:

```text
weather_gfs_global_lead1d_temperature_2m
```

Each row carries both its value and its evidence:

- Location count and expected location count.
- Per-hour location coverage ratio and pass flag.
- Whole-build-window feature coverage ratio and pass flag.
- Availability and reference timestamps.
- Reference-time type and whether it is observed.
- Weather vintage identifier.

The default coverage threshold is 95%. The optional wide artifact is an inspection convenience; model code starts from the long table so availability and provenance remain row-level facts.

### 7.4 Selecting a coherent eligible vintage

[`join_weather_features()`](../src/dkenergy_forecast/features/weather_features.py) first removes feature groups or valid-hour rows that failed coverage. It joins candidates on area and valid time, then masks any candidate whose availability is later than the row’s forecast origin.

Eligible features are grouped by weather model and lead. Within each group, the newest eligible `forecast_reference_time` is chosen before features are pivoted wide. This prevents a row from silently combining, for example, temperature from one GFS run with wind from an older GFS run when the newer run lacks wind.

The wide output retains per-feature metadata columns such as:

```text
<feature>_available_at_utc
<feature>_reference_time_utc
<feature>_vintage_id
<feature>_reference_time_type
<feature>_reference_time_is_observed
<feature>_availability_time_type
```

Those columns are not model values, but they make a selected feature auditable.

### 7.5 Coverage and fallback are separate decisions

The production Chronos adapter validates future weather coverage *before* filling missing values. Coverage is the fraction of required weather-covariate cells present across the full prediction frame, reported separately for each series.

The default production policy is:

```text
minimum future weather coverage = 1.0
insufficient coverage fallback = error
```

An explicit `zero` fallback is available as a configuration choice. It is not silent: the trained artifact manifest must declare the same minimum and fallback policy as serving, or loading fails.

Fill behavior depends on the role of the frame:

| Frame role | Weather fill policy |
|---|---|
| Training/context | Never carry values across valid times; fill remaining missing covariates with zero. |
| Future | Never carry values across valid times; fill remaining missing covariates with zero only after coverage policy permits it. |

There is no backward fill. In particular, a future hour cannot borrow a weather value from a later valid time.

## 8. Time and availability semantics in depth

Production forecasting has several clocks. Treating them as one timestamp creates plausible but false backtests.

| Timestamp | Question it answers |
|---|---|
| `ds_utc` | When is the electricity delivered or weather valid? |
| `price_available_at_utc` | When may this price become model history? |
| `forecast_reference_time` | Which weather run/vintage produced this feature? |
| `forecast_available_at_utc` | When may this weather feature be used? |
| `forecast_origin_utc` | What information cutoff does the forecast claim? |
| `decision_cutoff_utc` | By when must the operational forecast be published? |
| `generated_at_utc` | When did the process begin generating the forecast? |
| `published_at_utc` | When were the prediction results ready for publication? |

### Local noon without DST drift

[`copenhagen_timestamp()`](../src/dkenergy_forecast/types.py) combines a local calendar date with a wall-clock time and only then localizes it to `Europe/Copenhagen`. As a result, “12:00 local” becomes 10:00 UTC in summer and 11:00 UTC in winter.

Avoid constructing a UTC noon and converting it to Copenhagen. That asks a different question and shifts the market cutoff across DST.

### Delivery horizons

[`make_danish_delivery_day_horizon()`](../src/dkenergy_forecast/backtesting/horizons.py) takes the origin’s Copenhagen calendar date and, by default, builds the next local delivery day. It creates the interval from local midnight to the following local midnight, converts both boundaries to UTC, and enumerates UTC hours in between.

That naturally yields:

- 23 rows per area on the spring-forward day.
- 24 rows per area on an ordinary day.
- 25 rows per area on the fall-back day.

`horizon` is an ordinal within each series, not a promise that an always-24-hour day exists.

### Live origin and execution timing

Without an explicit origin, publication derives the decision date from the process reference time and constructs the configured Copenhagen wall-clock cutoff, normally noon. Supplying `--forecast-origin-utc` defaults the run to `replay` because an arbitrary historical origin is not evidence of a live execution.

A `live` run fails immediately if `generated_at_utc` is after its decision cutoff. If it starts before the cutoff but finishes after it, the immutable run may still be written for audit, but it is marked score-ineligible and is not promoted to latest.

The live workflow also checks that the panel covers the complete current Danish delivery day before it forecasts the next one. This catches a stale or partial price context independently of the model.

## 9. Origins, horizons, and rolling-origin backtests

[`origins.py`](../src/dkenergy_forecast/backtesting/origins.py) selects recent daily origins whose complete target horizons fit inside the panel. It supports a Copenhagen wall-clock origin or a legacy fixed UTC hour, a minimum history period, a holdout gap, and an optional maximum origin count.

For each selected origin, [`rolling_origin_backtest()`](../src/dkenergy_forecast/backtesting/rolling_origin.py) performs this sequence:

```text
1. Normalize panel and origin timestamps to UTC.
2. Keep only price rows with availability strictly before the origin.
3. Construct a fresh model instance.
4. Fit it on eligible history.
5. Build the requested future horizon.
6. Remove all target/leakage columns from that future frame.
7. Ask the model to predict.
8. Require the standard prediction schema.
9. Require an exact one-to-one match with requested future keys.
10. Join actual targets and descriptive metadata after prediction.
```

Creating a fresh model per origin prevents learned state from leaking from a later backtest origin into an earlier one. Exact key validation prevents a model from silently skipping hard rows or predicting extra rows and still receiving a favorable aggregate score.

Actuals can be missing for a live future horizon; they are present for complete historical backtests. Publication validation requires point predictions but does not require future actuals.

## 10. Price and weather feature construction

### Price features

[`price_features.py`](../src/dkenergy_forecast/features/price_features.py) builds features against an explicit origin. Important feature families include:

- Local calendar fields.
- 24-, 48-, and 168-hour lags.
- Origin-safe rolling means and medians.
- Seasonal local-hour and hour/weekend medians.
- The robust weekday/weekend weighted-median baseline.
- Lagged DK1-minus-DK2 spreads.

Feature lookup is anchored to target timestamps but masked by price availability before the origin. Training matrices are built from historical origin/horizon examples whose targets were fully available before the current training origin.

### Weather features

[`weather_features.py`](../src/dkenergy_forecast/features/weather_features.py) supplies three layers:

1. Availability-safe, coherent weather-vintage joins.
2. Across-model ensemble summaries such as mean, min, max, and spread.
3. Optional physical or relational transforms such as wind components, wind shear, cloud/radiation interactions, lead deltas, and DK1-minus-DK2 weather spreads.

[`feature_sets.py`](../src/dkenergy_forecast/features/feature_sets.py) chooses named feature subsets for CatBoost experiments while excluding metadata from numeric model input.

Do not call `merge()` directly on the weather parquet inside a new model. Doing so bypasses the availability mask, coherent-vintage selection, and coverage flags.

## 11. Model interface and registries

Every adapter follows the small `ForecastModel` protocol in [types.py](../src/dkenergy_forecast/types.py):

```python
model.fit(history)
predictions = model.predict(future, history=history)
```

The required prediction columns are:

```text
unique_id, ds_utc, forecast_origin_utc, horizon,
model_name, model_version, y_pred
```

Models may add q10/q50/q90 and target-contract metadata.

### Production registry

[`registry.py`](../src/dkenergy_forecast/models/registry.py) currently enables:

| Label | Family | Forecast |
|---|---|---|
| `chronos_weather` | Chronos | Primary LoRA-adapted probabilistic forecast with calendar and point-in-time weather covariates. |
| `weighted_median_v1` | Baseline | Fixed, explicitly labeled operational fallback after a primary failure. |

The registry records dependency extras, quantile support, and weather
requirements. The production orchestrator requests only the configured Chronos
primary; it constructs the fixed baseline separately if the primary fails.
Dependency checks fail before expensive work begins.

### Comparison registry

[`comparison_registry.py`](../src/dkenergy_forecast/models/comparison_registry.py) contains models intended for notebooks, diagnostics, or comparison:

- Two seasonal rolling medians.
- `catboost_price_manual_v1`.
- `chronos_zero_shot_v1`.

They cannot be selected by `run_publish_forecast.py`. This boundary protects the product surface from exploratory model selection.

### Baselines are first-class models

[`baselines.py`](../src/dkenergy_forecast/models/baselines.py) implements lag-naive, seasonal median, weighted seasonal median, and the split weekday/weekend policy. Each baseline applies the same availability rules and prediction schema as a neural model.

That makes baselines useful in three ways: operational fallback references, features for tabular models, and honest champions that a complex model must beat.

## 12. Chronos: training and serving the same contract

The production adapter is [chronos_production.py](../src/dkenergy_forecast/models/chronos_production.py), and the explicit training/export command is [train_chronos_lora.py](../scripts/train_chronos_lora.py).

### Training

Training performs the following steps:

1. Load and validate the price panel and long weather table.
2. Stop target history before the first evaluation origin.
3. Require regular hourly series and sufficient context plus prediction length.
4. Set each historical target row’s feature origin to its own `price_available_at_utc`.
5. Join availability-safe weather using the same join code used in serving.
6. Select calendar plus configured weather covariates.
7. Require weather signal, apply the declared training fill policy, and fit LoRA weights.
8. Export the LoRA adapter directory and schema-v3 `manifest.json`, with the
   immutable base-model revision written into both contracts.

The manifest records the base model and required immutable revision, random seed,
training settings, covariate names, target contract, price and weather
availability policies, weather vintage semantics, coverage/fallback policy, fill
policy, optional validation evidence, dependency versions, Git commit, hashes of
the price/weather training inputs, hashes of exported artifact files, and one
content identity derived from those artifact hashes.

Daily publication never updates weights. Retraining and artifact export are explicit operations.

### Serving

For one forecast origin, the adapter:

1. Loads the artifact manifest, rejects unsupported schema versions, verifies
   declared model-file hashes, and requires the adapter and manifest to pin the
   same base-model revision.
2. Requires runtime weather coverage/fallback settings to equal the trained manifest.
3. Selects exactly `context_length` eligible target rows per series.
4. Requires both series to share a final context timestamp.
5. Builds the complete hourly bridge from the last context row through the end of the delivery horizon.
6. Adds calendar and weather covariates to context and future frames.
7. Validates future weather coverage and applies role-specific fill policy.
8. Converts UTC timestamps to timezone-naive hourly timestamps for the Chronos library.
9. Calls `predict_df()` for q10, q50, and q90.
10. Maps predictions back to the requested delivery rows and sets `y_pred = q50`.

The bridge matters: the market-origin time and the first delivery hour are not necessarily adjacent, but a regular hourly sequence model must forecast every intervening step even when only the delivery-day subset is published.

### Artifact compatibility is deliberately strict

Serving expects artifact schema version 2. Older artifacts are not silently interpreted under newer weather or fill semantics. Retrain/export instead. A trained model is only compatible when its declared covariates and weather policy agree with runtime configuration.

## 13. Live, shadow, replay, and diagnostic runs

These terms describe different evidence and must not be used interchangeably.

| Kind | Typical use | May update latest? | May enter published scoring? |
|---|---|---:|---:|
| `live` | Scheduled operational forecast for the upcoming delivery day. | Yes, only when timely. | Yes, only when timely. |
| `shadow` | Candidate run executed under real information constraints without changing the product forecast. | No. | Yes, only when timely. |
| `replay` | Retrospective/debug run at an explicitly supplied origin. | No. | No. |
| `diagnostic` | Recomputed rolling-origin model diagnostics. | No. | No; it has its own score namespace. |

[`run_publish_forecast.py`](../scripts/run_publish_forecast.py) defaults to `live` when the origin is implicit and `replay` when `--forecast-origin-utc` is supplied. This makes the dangerous operation—the one that can change latest—require a truthful live execution context.

The forecast manifest records:

- `run_kind`.
- `decision_cutoff_utc`.
- `generated_at_utc`.
- `published_at_utc`.
- `score_eligible` and, when false, a reason.
- Selected registry metadata.
- Input/artifact paths and model metadata.
- The idempotency key and output checksums.

`generated_at_utc` is captured before data/model execution. `published_at_utc` is captured after predictions are ready. Both are needed: starting before a deadline does not prove that the forecast was available before the deadline.

## 14. Publication is a transaction

### Immutable run write

[`write_forecast_run_artifacts()`](../src/dkenergy_forecast/publishing/artifacts.py) validates predictions and scores, then writes all files to a hidden sibling directory. It computes SHA-256 checksums, adds an artifact identity hash to the manifest, and atomically renames the complete directory into place.

Readers therefore see either no run or a complete run—not a directory containing half a parquet file.

A typical run contains:

```text
artifacts/forecast_runs/<run_id>/
  predictions.parquet
  model_scores.parquet
  manifest.json
  forecast_dashboard.json
```

The live publication path intentionally leaves `model_scores.parquet` empty: recent diagnostics and published-history scoring are separate jobs.

### Idempotency

The live run ID is deterministic from the forecast origin, and the idempotency key includes run kind, origin, and selected model labels. An exact retry may reuse the existing complete run. Reusing the same key for different core artifacts fails.

Concurrent retries race through temporary directories, but only one immutable destination wins; the loser validates that the winning artifacts have the same identity.

### Latest promotion

Only a timely live run calls [`update_latest_exports()`](../src/dkenergy_forecast/publishing/artifacts.py). Promotion:

- Acquires an exclusive local file lock.
- Rejects a candidate older than the current latest origin.
- Atomically replaces individual parquet/JSON files.
- Writes the latest manifest last as the reader-visible commit marker.
- Records checksums for the promoted set.

The manifest-last rule gives local readers a consistent commit marker. S3 synchronization is a later storage operation; S3 bucket versioning provides recovery history, but a multi-object upload is not itself one filesystem rename.

## 15. Recent diagnostics and published scoring are different products

### Recent rolling-origin diagnostics

[`run_recent_diagnostics.py`](../scripts/run_recent_diagnostics.py) calls [recent_diagnostics.py](../src/dkenergy_forecast/operations/recent_diagnostics.py) to recompute production models over recent complete historical origins.

It writes:

```text
results/recent_scores/runs/<diagnostic_run_id>/
  predictions.parquet
  model_scores.parquet
  probabilistic_metrics.parquet
  manifest.json

results/recent_scores/
  predictions.parquet              # mutable convenience copy
  model_scores.parquet
  probabilistic_metrics.parquet
  manifest.json
```

The immutable diagnostic run is assembled transactionally, then the convenience files are replaced atomically. Failure here cannot change a live forecast run or latest pointer.

Diagnostics answer whether the current code and model artifact work well on a recent historical window. They do **not** prove what was actually published on those days.

### Published-forecast performance

[`score_published_forecasts.py`](../scripts/score_published_forecasts.py) reads completed immutable runs, verifies checksums when present, filters lifecycle eligibility, and joins actual prices that have since arrived.

Modern runs enter history only when:

```text
status is completed
and run_kind is live or shadow
and score_eligible is true
```

Legacy manifests are supported under an explicit compatibility rule. A directory without a manifest is never treated as complete.

When more than one eligible publication exists for the same origin, target, area, and model, the earliest durable publication is retained. This prevents retries or later corrections from giving that target extra weight or selecting a potentially better-informed revision.

The result lives under `results/published_forecast_history/` and answers the operational question: “How did saved, eligible forecasts perform once actuals became available?”

Published-history scoring is deliberately **not** invoked by live publication
or by the cloud live pipeline. Operators should schedule
[`score_published_forecasts.py`](../scripts/score_published_forecasts.py) as an
independent job after delivery outcomes have arrived. A scoring failure can
then delay performance reporting without delaying or corrupting the next live
forecast.

### How the daily coordinator preserves the separation

[`run_daily_pipeline.py`](../scripts/run_daily_pipeline.py) is a command
coordinator, not another forecasting implementation. Its local default can
refresh prices, run the older baseline-development backtest, and publish. Use
`--skip-backtest` for the short live critical path. Recent production-model
diagnostics are opt-in via `--with-diagnostics` and execute only after
publication.

The cloud wrapper always supplies `--skip-backtest` and does not enable recent
diagnostics or published-history scoring. Those heavier/evidence-producing jobs
have separate entrypoints and should have separate schedules and failure
handling.

## 16. Metrics and the evaluation arena

### Point and probabilistic metrics

[`point_metrics.py`](../src/dkenergy_forecast/evaluation/point_metrics.py) implements MAE, RMSE, and bias. [`probabilistic_metrics.py`](../src/dkenergy_forecast/evaluation/probabilistic_metrics.py) implements:

- Quantile pinball loss.
- q10–q90 coverage and average width.
- Central 80% interval score.
- Weighted interval score using q10/q50/q90.
- Signed quantile calibration error.
- Mean absolute calibration error across q10/q50/q90.

[`summary.py`](../src/dkenergy_forecast/evaluation/summary.py) produces the wide score table used by operations and the dashboard. Metrics exclude rows missing their required values; publication schema validation separately prevents partial quantile rows.

### Frozen intervals

[`splits.py`](../src/dkenergy_forecast/evaluation/splits.py) supports either an explicit half-open UTC interval or a JSON file containing named, non-overlapping splits with `"frozen": true`.

The split file’s SHA-256 is written into the evaluation report. “Frozen” is a declared contract, not filesystem magic: changing the file changes the hash and therefore the report provenance.

### Exactly paired comparison

[`arena.py`](../src/dkenergy_forecast/evaluation/arena.py) refuses an unfair comparison. Candidate and champion must have identical, duplicate-free keys—normally origin, series, target timestamp, and area—and must agree on actual target values.

This prevents a candidate from looking better because it omitted a difficult area, hour, origin, or DST row.

The arena produces:

- Overall point and probabilistic metrics.
- Candidate-minus-champion metrics for each forecast origin.
- Deterministic circular moving-block bootstrap confidence intervals over chronological origins.
- Stratified scores and subgroup guardrails.
- A machine-readable promotion decision.

The block bootstrap samples adjacent origin blocks instead of treating every hourly row as independent. Electricity-price forecast errors are serially related, and the model decision is made at the origin level.

### Stratification

[`stratification.py`](../src/dkenergy_forecast/evaluation/stratification.py) reports performance by:

- Copenhagen month.
- DK1/DK2 area.
- Local delivery hour.
- DST versus standard time.
- Negative versus non-negative price.
- Typical versus extreme absolute price.
- Target market regime.

The default extreme threshold is the selected interval’s 95th percentile of absolute actual prices; the resolved numeric threshold is stored in the report.

### Promotion policy

[`PromotionPolicy`](../src/dkenergy_forecast/evaluation/arena.py) makes the decision criteria explicit. By default it requires at least a 1% overall MAE improvement. Its checks cover:

- Required overall MAE improvement.
- Whether the paired MAE-difference confidence interval supports the candidate.
- Maximum allowed WIS degradation.
- Calibration relative to the champion and an absolute calibration ceiling.
- Maximum allowed MAE degradation within sufficiently large subgroups.
- Whether probabilistic evidence is mandatory.

The default policy expects both models to provide probabilistic forecasts. When
the existing champion is point-only,
`--allow-missing-probabilistic-comparison` skips only the relative WIS and
candidate-versus-champion calibration checks that cannot be computed. The
candidate must still provide q10/q50/q90 and pass the absolute calibration
limit.

[`run_evaluation_arena.py`](../scripts/run_evaluation_arena.py) writes strict JSON plus a human-readable Markdown report. It does **not** edit the production registry or deploy a model. Promotion remains an intentional code/artifact change after reviewing the report. See [the evaluation guide](evaluation.md) and [the frozen-split example](../config/evaluation_splits.example.json) for a complete command and policy reference.

## 17. Dashboard, local containers, and cloud deployment

### Dashboard

[`streamlit_app.py`](../app/streamlit_app.py) is a read-only artifact consumer. It does not train or invoke models. It can read local files, `file://` URIs, or S3 objects materialized through [storage.py](../src/dkenergy_forecast/storage.py).

The four tabs show:

- **Backtests:** evaluated prediction artifacts and metrics.
- **Forecasts:** latest forecast plus published performance, with recent diagnostics as a fallback.
- **Prices:** recent DK1/DK2 actual prices.
- **Run:** artifact paths, existence checks, and manifest provenance.

The dashboard warns when a latest artifact is stale, when the visible run is not live, or when a run is score-ineligible. The JSON payload is a convenient preassembled view; parquet fallbacks keep individual artifacts independently inspectable.

### Containers

[`Dockerfile.web`](../Dockerfile.web) installs the lightweight app/AWS dependencies and runs Streamlit through [container_entrypoint.py](../scripts/container_entrypoint.py). [`Dockerfile.pipeline`](../Dockerfile.pipeline) installs Chronos and AWS dependencies and runs [run_cloud_pipeline.py](../scripts/run_cloud_pipeline.py). Both container builds apply the exact direct-production pins in [`constraints-production.txt`](../constraints-production.txt); `pyproject.toml` remains the more flexible development/package declaration.

[`docker-compose.yml`](../docker-compose.yml) connects both images to local `cloud_store/` and `runtime/` directories so the same storage layout can be exercised without AWS.

### Artifact storage

[`ArtifactStore`](../src/dkenergy_forecast/storage.py) supports local paths, `file://`, and `s3://`. It can download/upload one file or an entire prefix. `materialize_uri()` gives the dashboard a local cache path for a remote object.

The cloud pipeline hydrates recent source/model-ready state and prior forecast
runs into a writable work directory, downloads the trained Chronos artifact
separately, invokes the short daily live workflow, checks weather freshness,
uploads durable state and run namespaces, and then uploads the small `latest/`
consumer artifacts. It may carry forward already-produced recent-score or
published-history files during synchronization, but it does not recompute
either evidence product on the live path.

### AWS shape

[`infra/aws/main.tf`](../infra/aws/main.tf) provisions:

- A private, encrypted, versioned S3 artifact bucket.
- A pipeline ECR repository and an opt-in web repository.
- An opt-in Streamlit ECS/Fargate service behind an ALB and CloudFront.
- A scheduled ECS/Fargate pipeline task.
- IAM roles and CloudWatch log groups.
- An EventBridge Scheduler trigger in the Copenhagen timezone.

The default schedule is 10:00 Copenhagen time, leaving headroom before the noon decision cutoff. The schedule is disabled by default until images and a compatible Chronos artifact have been bootstrapped. See [the AWS guide](../infra/aws/README.md).

CI in [ci.yml](../.github/workflows/ci.yml) runs lint, tests, source compilation, Terraform formatting/validation, both container builds, and a dry-run check of pipeline command wiring.

## 18. Tracing one forecast row end to end

Suppose you are investigating the DK1 forecast for one delivery hour. Use the same identifiers all the way through the system.

### Step 1: identify the published row

Open the immutable run’s `predictions.parquet` and filter by:

```text
run_id
model_label
unique_id = day_ahead_price_DK1
forecast_origin_utc
ds_utc
```

Record `horizon`, `y_pred`, optional quantiles, the target-contract columns, and the run manifest’s checksums and lifecycle timestamps.

### Step 2: verify the requested horizon

Convert `forecast_origin_utc` to Copenhagen time and determine the intended next local delivery date. Rebuild the horizon with [`make_danish_delivery_day_horizon()`](../src/dkenergy_forecast/backtesting/horizons.py).

The row must appear exactly once. Around DST, check the entire local day rather than assuming 24 rows.

### Step 3: inspect the target definition

Find the matching `(unique_id, ds_utc)` row in the loaded
`price_panel_hourly_v1.parquet`. Inspect:

```text
y
source_dataset
source_resolution_minutes
market_regime
native_resolution_minutes
target_aggregation
target_definition
price_available_at_utc
```

If the compact parquet does not physically contain `price_available_at_utc`,
derive it with the same shared loader/helper used by forecasting; do not invent
a different timestamp in an ad hoc query.

For a future live row, `y` may not have existed when the forecast was published. For post-transition history, trace the hourly value back to four normalized `DayAheadPrices` quarter-hours.

### Step 4: reconstruct eligible price history

Filter the panel using [`filter_price_history_available_before()`](../src/dkenergy_forecast/types.py), not `ds_utc < origin`. Confirm every retained row satisfies the strict availability inequality.

For the lag baseline, inspect the exact 168-hour lookup. For the weighted median, inspect candidates with the same local hour/weekend category, lookback window, and weight policy.

### Step 5: reconstruct eligible weather

If the model uses weather, filter the long weather table to the row’s area and `ds_utc`. Inspect all candidate models/leads/variables, coverage flags, availability times, reference times, and vintage IDs.

Run the join through [`join_weather_features()`](../src/dkenergy_forecast/features/weather_features.py). Confirm that selected rows were available by the origin and belong to the latest coherent eligible vintage for their model/lead group.

For Chronos, remember that the model forecasts a full bridge from the context end to the delivery-day end. Weather coverage is validated across that full future frame, not only the one displayed delivery row.

### Step 6: inspect model identity

Use the run manifest to find the registry label, Git commit, artifact path, and Chronos schema/covariate metadata. Then inspect the model artifact’s own `manifest.json`.

The artifact manifest explains which covariates and fill/coverage policies the weights were trained with. The run manifest explains which artifact and inputs were used for this execution.

### Step 7: verify publication integrity

Hash the immutable `predictions.parquet` and compare it with `artifact_sha256.predictions` in the run manifest. If you are reading mutable latest exports, compare their hashes with `latest_artifact_sha256` and ensure the latest manifest origin matches the prediction rows.

### Step 8: trace the eventual score

After the actual arrives, find the same run and prediction key in `results/published_forecast_history/predictions.parquet`. Check that the run was `live` or `shadow`, completed, score-eligible, and the earliest retained publication for that key.

This path—from immutable prediction to later actual—is the trustworthy source for operational performance.

## 19. Common extension recipes

### Add a new baseline or experimental model

1. Implement the `ForecastModel` protocol under [`models/`](../src/dkenergy_forecast/models/).
2. Keep origin/availability logic in shared helpers; do not recalculate eligibility ad hoc.
3. Return exactly the requested keys and required prediction columns.
4. Register it in [`comparison_registry.py`](../src/dkenergy_forecast/models/comparison_registry.py).
5. Add rolling-origin, missing-row, duplicate-key, and leakage tests.
6. Generate exactly paired predictions and compare it in the evaluation arena.

This does not authorize latest publication.

### Promote a model into the production registry

1. Produce frozen-split and shadow/published evidence against the current champion.
2. Define a stable model label and versioned artifact/schema.
3. Make training and serving feature contracts identical and manifest-visible.
4. Add dependency and artifact-loading checks.
5. Add a real-artifact/container smoke test proportional to the model’s runtime risk.
6. Add a `ProductionModelSpec` in [`registry.py`](../src/dkenergy_forecast/models/registry.py).
7. Update dashboard display metadata and documentation.
8. Review and deploy the registry/artifact change intentionally.

The arena’s `promote_candidate` output is evidence for this process; it does not perform these steps automatically.

### Add a weather variable or provider

1. Preserve raw bytes and request/retrieval metadata first.
2. Define observed reference and availability timestamps when the provider exposes them; otherwise label proxies explicitly.
3. Normalize to a long table with valid time, area/location, variable, value, model, lead, availability, reference time, and vintage ID.
4. Add coverage rules before aggregation.
5. Reuse the availability-safe coherent-vintage join.
6. Decide and record future coverage/fallback semantics.
7. Retrain models whose covariate list changes; never inject a new serving-only feature into old weights.
8. Add tests for late availability, mixed vintages, missing cells, and fallback behavior.

### Add a target regime

1. Define the boundary in Copenhagen market time.
2. Preserve native source resolution and source dataset.
3. Make the aggregation rule explicit and validate the required native observations.
4. Extend [`add_target_contract()`](../src/dkenergy_forecast/types.py).
5. Carry the contract through horizons, predictions, manifests, and strata.
6. Report performance on each regime separately.

### Add an evaluation metric

1. Implement a small, independently testable function in [`evaluation/`](../src/dkenergy_forecast/evaluation/).
2. Define missing-value behavior and whether lower or higher is better.
3. Add it to the production-facing score table only if operational consumers can interpret it.
4. Add it to arena policy only with a documented threshold and paired semantics.
5. Extend artifact schema validation and dashboard columns when necessary.

### Add a new dashboard consumer

Read published artifacts; do not import a model and recompute predictions in the UI. Treat the latest manifest as the local commit marker, retain run IDs in links or API responses, and expose lifecycle/provenance alongside numbers.

## 20. Testing and debugging

### Fast verification commands

From the repository root:

```bash
python -m pytest
python -m ruff check .
python -m compileall -q src scripts app
```

Useful focused suites include:

```bash
python -m pytest tests/test_time_alignment_and_leakage.py
python -m pytest tests/test_weather_target_contracts.py
python -m pytest tests/test_operational_hardening.py
python -m pytest tests/test_evaluation_arena.py
python -m pytest tests/test_production_models.py
python -m pytest tests/test_cloud_pipeline.py tests/test_workflows.py
```

Infrastructure checks:

```bash
terraform -chdir=infra/aws fmt -check -recursive
terraform -chdir=infra/aws init -backend=false
terraform -chdir=infra/aws validate
docker compose build
```

### Inspect commands without changing runtime state

```bash
python scripts/run_daily_pipeline.py --dry-run
python scripts/run_publish_forecast.py --list-models
python scripts/run_cloud_pipeline.py \
  --artifact-store-uri file:///tmp/cloud_store \
  --model-artifact-uri file:///tmp/missing-model \
  --workdir /tmp/runtime \
  --dry-run
```

### Failure messages as contract hints

| Failure | Usually means |
|---|---|
| Missing required dataframe columns | A producer/consumer schema boundary changed without updating both sides. |
| Prediction keys do not match future frame | The model skipped, duplicated, shifted, or invented horizon rows. |
| Live generation started after cutoff | The request is not a timely live run; use replay for retrospective work. |
| Live price context incomplete | Price ingestion/build is stale or the current Danish delivery day is partial. |
| Weather coverage below minimum | One or more required future covariate cells are absent after availability and coverage masking. |
| Runtime weather policy differs from artifact | Serving configuration no longer matches the contract under which the weights were trained. |
| Unsupported Chronos artifact schema | The model artifact must be retrained/exported under the current schema. |
| Crossed or partially populated quantiles | Model output is not a valid publication artifact. |
| Artifact checksum mismatch | An immutable file changed after its manifest was written. Treat this as integrity failure. |
| Refusing older latest origin | A stale/concurrent process attempted to regress the consumer view. |
| Candidate and champion keys differ | The evaluation is not exactly paired and would be unfair. |

### Debug from contracts outward

When a forecast fails, debug in this order:

1. Run manifest and lifecycle timestamps.
2. Input artifact existence, QA status, and hashes.
3. Origin and Copenhagen delivery-day horizon.
4. Price-history availability mask.
5. Weather availability, vintage, and coverage.
6. Model artifact manifest compatibility.
7. Prediction key/schema validation.
8. Transactional write and latest-promotion state.

Starting at the model’s numerical output often wastes time when the real problem is an earlier provenance or timing contract.

## 21. Common pitfalls

### Filtering history with `ds_utc < origin`

This confuses delivery time with publication time and can leak day-ahead targets. Use the shared availability helpers.

### Using a fixed UTC origin for a local market cutoff

A fixed UTC hour moves by one local hour across DST. Prefer the Copenhagen wall-clock origin unless you are deliberately testing a legacy policy.

### Assuming every delivery day has 24 rows

Build local-day boundaries and convert them to UTC. Do not hard-code 24 in validation, charting, or model output.

### Treating Open-Meteo proxy times as observed run times

The current feature artifact labels them synthetic. Preserve that limitation in reports and model cards.

### Filling future weather across time before measuring coverage

That can make an incomplete forecast look complete. Validate raw selected coverage first; future temporal fill is intentionally disabled.

### Mixing weather vintages variable by variable

Choosing the latest row for each feature independently can create a physically incoherent pseudo-run. Use the coherent model/lead vintage selection.

### Running expensive diagnostics on the live critical path

Use the separate diagnostics command. Live publication should generate, validate, write, and promote only the requested future forecast.

### Scoring recomputed forecasts as “published performance”

Backtests and recent diagnostics are valuable, but only immutable saved runs show what was actually forecast. Use the published-history scorer for operational claims.

### Letting replay overwrite latest

An explicit historical origin defaults to replay for a reason. Do not bypass lifecycle checks to make a dashboard screenshot.

### Editing an immutable run in place

Checksums will fail, and the audit trail becomes untrustworthy. Publish a new run or model version and preserve the original.

### Changing a covariate without retraining

The Chronos manifest is the weights-to-features contract. A renamed, added, differently filled, or differently available covariate is a model change.

### Reading only aggregate MAE

Inspect uncertainty, paired origin differences, DK1/DK2, local hour, DST, negative/extreme prices, and target regime. Aggregate improvement can hide operationally unacceptable regressions.

## 22. Recommended reading order

For a first pass, follow one forecast from shared semantics into operations:

1. [README.md](../README.md) for setup and supported workflows.
2. [types.py](../src/dkenergy_forecast/types.py) for the vocabulary, availability rules, calendar fields, and target contract.
3. [horizons.py](../src/dkenergy_forecast/backtesting/horizons.py) for Copenhagen delivery days and DST.
4. [rolling_origin.py](../src/dkenergy_forecast/backtesting/rolling_origin.py) for the leakage boundary and model protocol in action.
5. [baselines.py](../src/dkenergy_forecast/models/baselines.py) for the simplest complete model implementations.
6. [registry.py](../src/dkenergy_forecast/models/registry.py) and [comparison_registry.py](../src/dkenergy_forecast/models/comparison_registry.py) for the deployment boundary.
7. [publish_forecast.py](../src/dkenergy_forecast/operations/publish_forecast.py) for the live/shadow/replay lifecycle.
8. [artifacts.py](../src/dkenergy_forecast/publishing/artifacts.py) for immutable evidence, scoring eligibility, and latest promotion.

Then study each data path:

9. [energidataservice.py](../src/dkenergy_data/sources/energidataservice.py) and [eds_prices_v1.py](../src/dkenergy_data/build/eds_prices_v1.py).
10. [the EDS processing guide](data-processing/energi_data_service_v1.md).
11. [open_meteo.py](../src/dkenergy_data/sources/open_meteo.py), [open_meteo_weather_v1.py](../src/dkenergy_data/build/open_meteo_weather_v1.py), and [the Open-Meteo guide](data-processing/open_meteo_weather_v1.md).
12. [weather_features.py](../src/dkenergy_forecast/features/weather_features.py) for availability-safe vintage selection.

Then study the complex model and evidence path:

13. [chronos_production.py](../src/dkenergy_forecast/models/chronos_production.py).
14. [train_chronos_lora.py](../scripts/train_chronos_lora.py).
15. [recent_diagnostics.py](../src/dkenergy_forecast/operations/recent_diagnostics.py) and [score_published_forecasts.py](../scripts/score_published_forecasts.py).
16. [arena.py](../src/dkenergy_forecast/evaluation/arena.py), [splits.py](../src/dkenergy_forecast/evaluation/splits.py), and [stratification.py](../src/dkenergy_forecast/evaluation/stratification.py).
17. [streamlit_app.py](../app/streamlit_app.py), [cloud_pipeline.py](../src/dkenergy_forecast/cloud_pipeline.py), and [the AWS guide](../infra/aws/README.md).

Finally, read the tests as executable specifications. Start with [test_time_alignment_and_leakage.py](../tests/test_time_alignment_and_leakage.py), [test_weather_target_contracts.py](../tests/test_weather_target_contracts.py), [test_operational_hardening.py](../tests/test_operational_hardening.py), and [test_evaluation_arena.py](../tests/test_evaluation_arena.py).

## 23. Closing mental model

The repository is easiest to reason about as a sequence of narrowing contracts:

```text
raw source evidence
    -> normalized source meaning
    -> explicit target and weather provenance
    -> information available at one origin
    -> exactly requested forecast keys
    -> validated immutable publication
    -> eligible published evidence
    -> paired promotion decision
```

Each arrow should remove ambiguity. If a new feature makes it harder to answer “what did this row mean, what was knowable, what produced it, and what was actually published?”, the feature belongs behind a clearer contract before it belongs in production.
