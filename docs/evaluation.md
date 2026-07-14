# Model evaluation

Evaluation is descriptive. It does not select, promote, or deploy a model.
Production always uses the explicit primary and fallback named in
[`config/production.json`](../config/production.json).

## Comparison contract

`scripts/run_model_comparison.py` compares two model releases on exactly paired
prediction rows. The input parquet must contain:

- `model_label`, `model_release_id`, and `forecast_origin_utc`;
- `ds_utc`, `area`, `y`, and `y_pred`;
- optional `q10`, `q50`, and `q90` interval forecasts.

The command rejects duplicate rows, mismatched forecast grids, missing actuals,
and incomplete or crossed quantiles. It records the input file's SHA-256, the
selected releases, interval boundaries, pairing keys, and statistical settings
in deterministic JSON.

Point metrics are MAE, RMSE, and bias. Probabilistic diagnostics include
pinball loss, central-80% coverage and width, interval score, weighted interval
score, and calibration error. A circular moving-block bootstrap over forecast
origins preserves short-range dependence better than treating every hour as an
independent observation.

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
  --output-dir results/model_comparison
```

The command writes `model_comparison.json` for provenance and
`model_comparison.md` for review. Commit a report when the production weights,
data contract, covariates, or training recipe change.

## Current production evidence

The reviewed report for the configured Chronos release is
[`model-evidence/chronos-weather-sha256-55814a7fd0d36973/model_comparison.md`](model-evidence/chronos-weather-sha256-55814a7fd0d36973/model_comparison.md).
It evaluates historical forecast origins, not a long live-production record.
The public dashboard will gradually replace this initial evidence with
registered daily forecasts.

Interpret the report as evidence about a limited historical period, not a
guarantee of future performance. Electricity prices are non-stationary, the
weather-availability model is approximate, and interval calibration can drift.
