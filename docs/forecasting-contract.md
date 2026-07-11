# Forecasting contract

This document defines the meaning of a production forecast. Code, manifests,
diagnostics, and the dashboard should use these terms literally.

## Target

The system predicts the DK1 and DK2 day-ahead area price in DKK/MWh for every
hour of the next Danish local delivery day.

| Local delivery time | Native source | Hourly target |
|---|---|---|
| Before 2025-10-01 | Hourly `Elspotprices` | Identity |
| From 2025-10-01 | 15-minute `DayAheadPrices` | Arithmetic mean of four quarters |

Every row carries `market_regime`, `native_resolution_minutes`,
`target_aggregation`, and `target_definition`. From the transition onward, the
forecast is for a derived hourly mean rather than a native hourly clearing
product.

The delivery horizon is constructed between Copenhagen local midnights and
then converted to UTC. It therefore contains 23, 24, or 25 hourly rows per area
on daylight-saving transition days.

## The five clocks

Production forecasting has several clocks. Collapsing them into a single
"origin" makes plausible-looking leakage possible.

| Field | Question answered |
|---|---|
| `delivery_date_local` | Which Copenhagen market day is being forecast? |
| `information_cutoff_utc` | What was the latest information this forecast could use? |
| `decision_deadline_utc` | By when must a live forecast become durable? |
| `generated_at_utc` | When did this attempt start? |
| `committed_at_utc` | When did the completed publication become durable? |

`forecast_origin_utc` remains as a compatibility name for
`information_cutoff_utc`; it does not mean the deadline.

The AWS live task is scheduled at 10:00 `Europe/Copenhagen`. When no explicit
cutoff is supplied, its actual process start becomes the information cutoff.
The decision deadline is 12:00 Copenhagen time on the same date. The delivery
date defaults to the following Copenhagen calendar day.

A live request is invalid when:

- Generation starts after the deadline.
- Its delivery date is not after the cutoff's local calendar date.
- The completed run reaches durable publication after the deadline.

Historical work must supply its cutoff explicitly and use `run_kind=replay`.

## Price information set

Every historical price row has `price_available_at_utc`. A row is eligible
only when:

```text
price_available_at_utc < information_cutoff_utc
```

The strict inequality is intentional. The default availability timestamp is a
project convention—local noon on the preceding calendar day—not an observed
publication or revision event.

The model never receives the horizon's actual target during prediction. Horizon
builders strip target columns, models predict, and actuals are joined only for
later evaluation.

## Weather information set

The canonical source is the long Open-Meteo area-hour table. A weather row is
eligible only when:

```text
forecast_available_at_utc <= information_cutoff_utc
```

For each target row and `(weather model, parameter)`, the shared weather join:

1. Removes location-hour and feature-group rows that fail coverage.
2. Removes values unavailable at the information cutoff.
3. Selects the newest eligible forecast reference time.
4. Exposes the value under a stable `weather_<model>_<parameter>` name.
5. Retains the selected lead, source feature, reference time, availability
   type, and vintage as metadata.

Lead-specific source columns are storage provenance, not separate mandatory
model signals. A target hour may use a different eligible lead from its
neighbor when that is the newest information actually available.

The Open-Meteo Previous Runs source does not expose observed model-run
publication timestamps. Its reference and availability times are explicitly
synthetic `valid_time - requested lead` proxies. This is a declared limitation,
not proof of upstream publication latency.

Wind direction is excluded because an arithmetic spatial average is not a
valid circular statistic.

## Missing-weather semantics

Training, replay, and live serving use the same covariate construction:

1. Point-in-time newest-eligible selection.
2. No forward or backward fill across valid timestamps.
3. Required-horizon coverage measurement before imputation.
4. Failure when coverage is below the artifact's declared threshold and the
   policy is `error`.
5. Replacement of remaining missing covariate cells with `0.0` under the
   declared `no_temporal_fill_then_zero` representation.

The production Chronos configuration requires complete future weather
coverage. Runtime rejects an artifact whose selection, exclusion, fill,
coverage, or fallback policy differs from serving configuration.

## Production model and fallback

[`../config/production.json`](../config/production.json) names exactly one
primary and one fixed fallback:

- `chronos_weather` is the normal production model.
- `weighted_median_v1` is the operational fallback.

The fallback is attempted only when the configured Chronos release fails to
produce a contract-valid forecast. A degraded publication records:

```text
forecast_status = degraded
requested_model = chronos_weather
published_model = weighted_median_v1
primary_failure = {type, message}
```

The fallback is never relabeled as Chronos. Diagnostics never switch this
configuration automatically.

## Model release identity

Logical model name and trained release are separate:

- `model_name` identifies the implementation contract.
- `model_release_id` identifies one immutable trained release.
- `model_artifact_sha256` identifies its exact artifact content.

The Chronos manifest supplies its release ID and content hash. The weighted
median uses a stable hash of its fixed parameter contract. Forecast and score
rows retain these identities so metrics from different releases cannot be
silently pooled.

## Run lifecycle

| Kind | Meaning | May update `latest.json` | Saved-forecast scoring |
|---|---|---:|---:|
| `live` | Timely production attempt | Yes, after durable completion | Yes |
| `replay` | Historical reconstruction | No | No |
| Diagnostic run | Independently recomputed rolling evaluation | No | No |

Diagnostics use a separate output namespace rather than pretending to be a
production lifecycle state.

## Prediction artifact

Core row fields are:

```text
run_id
forecast_origin_utc
information_cutoff_utc
delivery_date_local
ds_utc
ds_local
unique_id
area
horizon
model_label
model_name
model_release_id
model_artifact_sha256
requested_model
forecast_status
y_pred
```

Chronos rows contain all of `q10`, `q50`, and `q90`; `y_pred` equals `q50`.
Quantiles must be complete and ordered. A model must emit exactly one row for
every requested `(unique_id, ds_utc, forecast_origin_utc)` key.

## Durable publication

A run is committed by its receipt, not by its manifest alone:

```text
forecast_runs/<run_id>/manifest.json
forecast_runs/<run_id>/predictions.parquet
forecast_runs/<run_id>/COMPLETED.json
latest.json
```

`COMPLETED.json` is written or uploaded only after the run's other artifacts.
`latest.json` is replaced only after that receipt exists. It names one run
prefix and one completion key. Consumers follow the pointer and reject partial
runs.

Only completed live runs committed by their deadline enter saved-forecast
scoring. Scoring reads their stored predictions and joins later actuals; it
never recomputes the forecast.

## Evaluation claim

Model comparison is descriptive. Exact row pairing, frozen or explicit date
intervals, probabilistic metrics, subgroup tables, and block-bootstrap
intervals make the evidence reviewable. No metric threshold changes production
configuration. A new production release is selected manually through a
reviewed change to `config/production.json`.

See the [evaluation guide](evaluation.md),
[operations runbook](operations.md), and [codebase tour](codebase-tour.md).
