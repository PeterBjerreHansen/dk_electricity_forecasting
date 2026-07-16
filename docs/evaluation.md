# Model evaluation

Evaluation is descriptive. It does not select, promote, or deploy a model.
Production always uses the explicit primary and fallback named in
[`config/production.json`](../config/production.json).

## Comparison contract

`scripts/run_model_comparison.py` compares two model releases on exactly paired
prediction rows. The input parquet must contain:

- `model_label`, `model_release_id`, and `forecast_origin_utc`;
- `information_cutoff_utc`, `ds_utc`, `area`, `y`, and `y_pred`;
- `evidence_id` equal to the audit's `audit_sha256` when
  `--availability-audit-json` is supplied;
- optional `q10`, `q50`, and `q90` interval forecasts.

The command rejects duplicate rows, mismatched forecast grids, missing actuals,
different information cutoffs, and incomplete or crossed quantiles. It records
the input file's SHA-256, selected releases, interval boundaries, pairing keys,
statistical settings, and evidence tier in deterministic JSON.

Point metrics are MAE, RMSE, and bias. Probabilistic diagnostics include
pinball loss, central-80% coverage and width, interval score, weighted interval
score, and calibration error. A circular moving-block bootstrap over forecast
origins preserves short-range dependence better than treating every hour as an
independent observation.

## Evidence contract

Every comparison is labelled with one of three machine-validated tiers:

- `strict_point_in_time`: selected mutable inputs passed the cutoff audit using
  observed availability timestamps and revisions verified as of the cutoff.
- `reconstructed_proxy`: timing or source vintages were reconstructed or
  proxied, or no availability audit exists. When an audit is supplied, it must
  still find no selected value after its cutoff.
- `saved_live_forecast`: a live forecast from an append-only artifact store has
  a timely completion receipt whose manifest and artifact identities were
  verified, and whose
  cutoff, origin, deadline, and durable publication all precede the derived
  price-availability timestamp for every target.

Saved-live verification proves internal consistency against the supplied
receipt and hashes. It assumes the source artifact store preserves completed
run records as append-only or otherwise externally controlled; it cannot
detect an actor rewriting the prediction, manifest, receipt, and all local
copies together. A saved-live comparison must exactly retain every prediction
row and quantile column represented by its selected evidence receipts;
cherry-picked hours or omitted intervals are rejected. Older live receipts
whose publication timestamp was not hash-bound remain
`reconstructed_proxy`, while any integrity hashes they do contain are still
verified.

Without an availability audit, historical comparisons default to
`reconstructed_proxy`, and no cutoff or source-vintage verification claim is
made. When an audit is supplied, missing required audit fields, unknown
availability semantics, a tampered audit hash, or any selected value after the
cutoff fails closed. Saved-live and reconstructed rows cannot be pooled in one
aggregate report.

## Run a comparison

Use either explicit half-open UTC boundaries or a reviewed frozen split file:

```bash
python scripts/run_model_comparison.py \
  --predictions path/to/predictions.parquet \
  --reference-model weighted_median_v1 \
  --comparison-model chronos_weather \
  --reference-release weighted_median_v1 \
  --comparison-release sha256-55814a7fd0d36973 \
  --start-utc 2026-06-14T00:00:00Z \
  --end-utc 2026-07-15T00:00:00Z \
  --availability-audit-json path/to/availability_audit.json \
  --output-dir results/model_comparison
```

The command writes `model_comparison.json` for provenance and
`model_comparison.md` for review. Commit a report when the production weights,
data contract, covariates, or training recipe change.

## Current production evidence

The reviewed report for the configured Chronos release is
[`model-evidence/chronos-weather-sha256-55814a7fd0d36973/model_comparison.md`](model-evidence/chronos-weather-sha256-55814a7fd0d36973/model_comparison.md).
It evaluates historical forecast origins, not a long live-production record.
Its evidence tier is `reconstructed_proxy` because price publication timing and
historical revisions are not observed as-of vintages, and weather availability
is synthetic. Verified registered daily forecasts are labelled separately as
`saved_live_forecast`.

Interpret the report as evidence about a limited historical period, not a
guarantee of future performance. Electricity prices are non-stationary, the
weather-availability model is approximate, and interval calibration can drift.
