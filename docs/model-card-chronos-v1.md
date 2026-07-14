# Model card: Chronos-2 LoRA calendar-weather

## Intended use

`chronos_weather` produces DK1/DK2 hourly point and q10/q50/q90 forecasts for
the next Danish local delivery day. `y_pred` is q50. The model is
batch-published; it is not a request-time API model. The configured production
release is `sha256-55814a7fd0d36973`.

## Inputs and artifact contract

- Up to 1,024 hourly target-context rows per area.
- Copenhagen calendar covariates.
- Availability-masked Open-Meteo weather covariates recorded by the artifact.
- Full required future-weather cell coverage by default.

Artifact schema v3 records the covariate list, role-specific fill policy,
weather coverage/fallback policy, target contract, random seed, required pinned
base-model revision, library versions, training interval, source-data hashes,
model-file hashes, and training code commit. It supports an optional embedded
validation summary; this release leaves that field empty and keeps reviewed
evidence in a separate versioned report. Runtime rejects schema,
base-revision, file-hash, covariate, or weather-policy mismatches.

The current release uses `no_temporal_fill_then_zero` for training, serving
context, and serving future covariates. Weather coverage is measured before
fill. A zero future-weather fallback is opt-in and must match between artifact
and runtime.

## Evaluation

The [reviewed report](model-evidence/chronos-weather-sha256-55814a7fd0d36973/model_comparison.md)
contains 1,152 exactly paired DK1/DK2 rows across 24 historical forecast
origins. On that period, Chronos MAE is 170.2 DKK/MWh versus 255.1 for the fixed
weighted-median reference. The paired MAE difference is -84.9 DKK/MWh; its 95%
moving-block interval is -177.1 to -12.8. The nominal 80% Chronos interval
covers 79.0% of observations.

This is promising historical-origin evidence for the exact configured release,
not a guarantee of future performance or a long live-production record. The
report records its prediction-source hash, releases, interval, pairing keys,
and statistical settings. See [evaluation.md](evaluation.md) for the reusable
descriptive workflow.

## Failure behavior

The adapter fails rather than silently changing model semantics when:

- The artifact or manifest is missing.
- Artifact schema, file hashes, covariates, or weather policy disagree.
- Current context is too short or irregular.
- Required weather coverage is insufficient.
- The model omits horizon rows or returns incomplete/crossed quantiles.

No automatic zero-shot or baseline substitution occurs under the Chronos label.
Operational fallback should be an explicitly published and labeled baseline.

## Limitations

- Weather availability is based on a documented synthetic proxy.
- The target spans a native-product regime change.
- Weather features omit more direct market fundamentals.
- LoRA quality and calibration can drift by season and price regime.
- The public label remains stable, so the manifest content hash is
  required to distinguish actual trained artifacts.

See [forecasting-contract.md](forecasting-contract.md),
[data-card.md](data-card.md), and [operations.md](operations.md).
