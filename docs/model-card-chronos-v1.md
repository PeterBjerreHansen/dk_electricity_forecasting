# Model card: Chronos-2 LoRA calendar-weather v1

## Intended use

`chronos2_lora_calendar_weather_ctx1024_v1` produces DK1/DK2 hourly point and
q10/q50/q90 forecasts for the next Danish local delivery day. `y_pred` is q50.
The model is batch-published; it is not a request-time API model.

## Inputs and artifact contract

- Up to 1,024 hourly target-context rows per area.
- Copenhagen calendar covariates.
- Availability-masked Open-Meteo weather covariates recorded by the artifact.
- Full required future-weather cell coverage by default.

Artifact schema v2 records the covariate list, role-specific fill policy,
weather coverage/fallback policy, target contract, random seed, optional base
revision, library versions, training interval, validation summary, source-data
hashes, model-file hashes, and training code commit. Runtime rejects schema or
weather-policy mismatches and verifies file hashes when present.

Training/context weather may carry earlier values forward within a series and
then zero-fill leading gaps. Future weather never fills across valid times. A
zero future fallback is opt-in and must match between artifact and runtime.

## Evaluation

The repository contains a reusable frozen evaluation arena rather than a
hard-coded claim in this card. Promotion requires exactly paired rows, origin-
block confidence intervals, probabilistic/calibration checks, and subgroup
guardrails. Run and commit a reviewed report using [evaluation.md](evaluation.md)
whenever the weights, data, covariates, or promotion thresholds change.

Local ignored artifacts have shown promising results, but ignored results are
not a durable production claim. The committed report and its input/split hashes
are the evidence of record.

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
- The hard-coded public label remains stable, so the manifest content hash is
  required to distinguish actual trained artifacts.

See [forecasting-contract.md](forecasting-contract.md),
[data-card.md](data-card.md), and [operations.md](operations.md).
