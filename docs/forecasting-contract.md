# Forecasting contract

This document defines what a production forecast means. Code and dashboards
should use these terms literally.

## Target

The system predicts the DK1 and DK2 day-ahead area price in DKK/MWh for every
hour of the next Danish local delivery day.

| Local delivery time | Native source | Hourly target |
|---|---|---|
| Before 2025-10-01 | Hourly `Elspotprices` | Identity |
| From 2025-10-01 | 15-minute `DayAheadPrices` | Arithmetic mean of four quarters |

Every row carries `market_regime`, `native_resolution_minutes`,
`target_aggregation`, and `target_definition`. The post-transition target is a
derived hourly product, not a native hourly clearing product.

## Decision and delivery time

- The decision cutoff is 12:00 `Europe/Copenhagen` on the execution date.
- The delivery horizon is the following Copenhagen calendar day.
- Horizons are constructed with local midnights and converted to UTC, producing
  23, 24, or 25 hourly rows per area when DST requires it.
- `generated_at_utc` is when the process starts.
- `published_at_utc` is the publication timestamp recorded for the durable run.
- A live run started after the cutoff is rejected.

The live origin is derived from the execution date, never from the maximum
delivery timestamp already present in the panel. An explicitly supplied origin
defaults to `replay`.

## Information set

Price history is eligible only when:

```text
price_available_at_utc < forecast_origin_utc
```

Weather is eligible only when:

```text
forecast_available_at_utc <= forecast_origin_utc
```

The current Open-Meteo source does not expose observed run initialization or
publication times. Its reference and availability timestamps are explicitly
synthetic proxies (`valid_time - requested lead`). Results based on this source
must retain that limitation.

Future weather is selected from a coherent model/lead vintage. It is never
forward- or backward-filled across valid timestamps. The production default
requires every expected future covariate cell; an explicit zero fallback is
allowed only when the trained artifact and runtime declare the same policy.

## Run lifecycle

| Kind | Meaning | Updates latest | Published scoring |
|---|---|---:|---:|
| `live` | Timely production request | Yes, when eligible | Yes |
| `shadow` | Timely production-like candidate | No | Yes |
| `replay` | Retrospective reconstruction | No | No |
| `diagnostic` | Recomputed rolling-origin health check | No | No |

Published-history scoring reads saved immutable predictions. It never
recomputes them. Replays, late runs, incomplete runs, and manifest-less
directories are excluded.

## Prediction artifact

Required row fields are:

```text
forecast_origin_utc, ds_utc, ds_local, unique_id, area,
model_label, y_pred, horizon
```

Probabilistic rows contain all of `q10`, `q50`, and `q90`, or none. Quantiles
must be ordered. A model must return exactly one row for every requested
`(unique_id, ds_utc, forecast_origin_utc)` key.

## Evaluation claim

A model is a production champion only after an exactly paired frozen-interval
evaluation passes the configured MAE, uncertainty, calibration, confidence
interval, and subgroup guardrails. Seven recent origins are an operational
diagnostic, not champion-selection evidence.

See [evaluation.md](evaluation.md) and [codebase-tour.md](codebase-tour.md).
