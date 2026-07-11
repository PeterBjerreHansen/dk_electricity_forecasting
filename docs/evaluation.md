# Model evaluation arena

The evaluation arena compares one candidate with the current champion from an
existing prediction parquet. It does not train or rerun either model. The output
is a deterministic JSON report for automation and a concise Markdown report for
review and version control.

## Required prediction columns

The parquet must contain:

- `model_label`
- `forecast_origin_utc`
- `ds_utc`
- `area`
- `y`
- `y_pred`

`unique_id` is included in the pairing key when present. Probabilistic promotion
checks require complete, ordered `q10`, `q50`, and `q90` values for both selected
models. When the existing champion is point-only, pass
`--allow-missing-probabilistic-comparison`. This skips only the relative WIS and
calibration checks that cannot be computed for the champion. The candidate must
still provide q10/q50/q90 and pass its absolute calibration limit; the flag never
silently converts missing candidate uncertainty into a pass.

Candidate and champion rows must match exactly on the available pairing keys.
The command fails on duplicates, missing actuals, partial quantiles, crossed
quantiles, or hours emitted by only one model. This prevents selective missing
forecasts from biasing the comparison.

## Run with an explicit interval

Evaluation intervals are half-open and use `forecast_origin_utc` by default:

```bash
python scripts/run_evaluation_arena.py \
  --predictions results/notebook_chronos2_experimental_v1/predictions.parquet \
  --candidate chronos2_lora_calendar_weather_ctx1024_300steps \
  --champion chronos2_zs_calendar_weather_ctx1344 \
  --start-utc 2026-04-01T00:00:00Z \
  --end-utc 2026-07-01T00:00:00Z \
  --output-dir results/evaluation_arena/lora_vs_zero_shot
```

Use `--timestamp-column ds_utc` only when a target-time split is intentional.
Do not switch timestamp semantics between model-selection rounds.

## Run with frozen date splits

For promotion decisions, prefer a reviewed split file whose content hash will be
recorded in the report:

```json
{
  "frozen": true,
  "timestamp_column": "forecast_origin_utc",
  "splits": {
    "development": {
      "start_utc": "2025-01-01T00:00:00Z",
      "end_utc": "2026-01-01T00:00:00Z"
    },
    "validation": {
      "start_utc": "2026-01-01T00:00:00Z",
      "end_utc": "2026-04-01T00:00:00Z"
    },
    "test": {
      "start_utc": "2026-04-01T00:00:00Z",
      "end_utc": "2026-07-01T00:00:00Z"
    }
  }
}
```

Splits must not overlap. Run a named split with:

```bash
python scripts/run_evaluation_arena.py \
  --predictions path/to/predictions.parquet \
  --candidate candidate_model_label \
  --champion champion_model_label \
  --splits-json config/evaluation_splits.example.json \
  --split test \
  --output-dir results/evaluation_arena/candidate_vs_champion
```

Changing a supposedly frozen split changes the SHA-256 recorded in
`evaluation_report.json`. Review that hash like a model or data artifact change.

## Metrics

The reports contain:

- MAE, RMSE, and bias.
- Pinball loss at q10, q50, and q90.
- Coverage, width, and interval score for the central 80% interval.
- Weighted interval score (WIS) for q10/q50/q90. Lower is better.
- Mean absolute quantile calibration error across q10/q50/q90. Lower is better.
- One candidate-minus-champion record per forecast origin.
- Deterministic circular moving-block bootstrap intervals over chronological
  forecast origins. The default block length is seven origins.

The production `model_score_table` also carries the pinball, interval-score, WIS,
and calibration columns. Existing required score columns remain unchanged, so
older dashboard and artifact readers continue to work.

## Stratified guardrails

MAE is checked by:

- Delivery month.
- DK area.
- Copenhagen local hour.
- DST versus standard time.
- Negative versus non-negative actual price.
- Extreme versus typical absolute price.
- Native-hourly versus quarter-hour-aggregated target regime.

Unless an absolute `--extreme-threshold` is supplied, extreme prices are defined
using the recorded absolute-target quantile (q95 by default). The market-regime
boundary is 2025-10-01 local time; an explicit `market_regime` or
`source_resolution_minutes` column takes precedence when available.

Groups below `--min-subgroup-rows` are reported but not used as promotion
guardrails. This avoids making a promotion decision from a handful of hours.

## Default promotion policy

The candidate is promoted only when all applicable checks pass:

1. Overall MAE improves by at least 1% by default.
2. The upper confidence bound of the paired origin-level MAE difference is at
   most zero.
3. WIS does not degrade beyond its configured tolerance.
4. Calibration does not worsen by more than 0.02 and is at most 0.10.
5. No eligible subgroup MAE degrades by more than 10%.

All thresholds are command-line options and are serialized into the report.
Changing a threshold therefore creates a visible report diff. The command exits
successfully even when the decision is `retain_champion`; that is an evaluation
outcome, not a pipeline error.

## Version-control workflow

Commit the small report files, not necessarily every prediction row:

```text
evaluation_report.json
evaluation_report.md
```

The JSON is sorted, contains no non-standard `NaN` values, omits wall-clock
generation timestamps, and records the prediction parquet hash and split
provenance. Re-running the same input and settings therefore produces a clean
diff. Keep the source prediction artifact in durable storage and retain its hash
with the reviewed report.
