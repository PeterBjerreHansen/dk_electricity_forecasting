# Forecast model comparison

`chronos_weather` is compared with reference `weighted_median_v1` on `forecast_origin_utc` in `[2026-06-14T00:00:00+00:00, 2026-07-15T00:00:00+00:00)`.

Releases: comparison `sha256-55814a7fd0d36973`; reference `weighted_median_v1`.

Prediction source SHA-256: `55f45dfeae5546a7623963400edb4afe79089f2b277149ebc224ca258221e01e`.

The comparison contains 1152 exactly paired rows across 24 forecast origins. Differences are comparison minus reference; negative differences favor the comparison for error and scoring metrics.

## Overall metrics

| Role | Model | MAE | RMSE | Bias | WIS | Calibration error | 80% coverage |
|---|---|---:|---:|---:|---:|---:|---:|
| reference | weighted_median_v1 | 255.099 | 476.576 | -12.924 | n/a | n/a | n/a |
| comparison | chronos_weather | 170.246 | 370.283 | -65.192 | 113.628 | 0.0622 | 0.7899 |

## Origin-block bootstrap

| Metric difference | Mean | Lower | Upper | Confidence | Block |
|---|---:|---:|---:|---:|---:|
| mae | -84.853 | -177.054 | -12.786 | 0.950 | 7 |
| weighted_interval_score | n/a | n/a | n/a | n/a | n/a |
| calibration_error | n/a | n/a | n/a | n/a | n/a |

## Per-origin differences

| Forecast origin (UTC) | Rows | Reference MAE | Comparison MAE | MAE difference | WIS difference |
|---|---:|---:|---:|---:|---:|
| 2026-06-14T08:00:00+00:00 | 48 | 364.789 | 116.025 | -248.764 | n/a |
| 2026-06-15T08:00:00+00:00 | 48 | 64.722 | 85.268 | 20.546 | n/a |
| 2026-06-16T08:00:00+00:00 | 48 | 81.823 | 88.523 | 6.699 | n/a |
| 2026-06-17T08:00:00+00:00 | 48 | 325.213 | 301.192 | -24.021 | n/a |
| 2026-06-18T08:00:00+00:00 | 48 | 107.468 | 120.974 | 13.506 | n/a |
| 2026-06-19T08:00:00+00:00 | 48 | 93.597 | 97.331 | 3.734 | n/a |
| 2026-06-20T08:00:00+00:00 | 48 | 43.863 | 61.733 | 17.871 | n/a |
| 2026-06-21T08:00:00+00:00 | 48 | 130.040 | 109.007 | -21.033 | n/a |
| 2026-06-22T08:00:00+00:00 | 48 | 486.681 | 408.622 | -78.059 | n/a |
| 2026-06-23T08:00:00+00:00 | 48 | 717.242 | 593.754 | -123.488 | n/a |
| 2026-06-24T08:00:00+00:00 | 48 | 147.194 | 271.622 | 124.428 | n/a |
| 2026-06-25T08:00:00+00:00 | 48 | 116.082 | 179.708 | 63.627 | n/a |
| 2026-06-26T08:00:00+00:00 | 48 | 174.602 | 94.527 | -80.075 | n/a |
| 2026-06-27T08:00:00+00:00 | 48 | 65.488 | 63.061 | -2.427 | n/a |
| 2026-06-28T08:00:00+00:00 | 48 | 170.474 | 117.843 | -52.631 | n/a |
| 2026-06-29T08:00:00+00:00 | 48 | 475.278 | 347.346 | -127.932 | n/a |
| 2026-06-30T08:00:00+00:00 | 48 | 229.759 | 239.086 | 9.327 | n/a |
| 2026-07-01T08:00:00+00:00 | 48 | 505.436 | 114.881 | -390.555 | n/a |
| 2026-07-02T08:00:00+00:00 | 48 | 719.330 | 132.339 | -586.990 | n/a |
| 2026-07-03T08:00:00+00:00 | 48 | 275.397 | 116.959 | -158.439 | n/a |
| 2026-07-04T08:00:00+00:00 | 48 | 261.965 | 59.688 | -202.277 | n/a |
| 2026-07-05T08:00:00+00:00 | 48 | 179.903 | 169.713 | -10.190 | n/a |
| 2026-07-06T08:00:00+00:00 | 48 | 327.907 | 121.478 | -206.429 | n/a |
| 2026-07-13T08:00:00+00:00 | 48 | 58.119 | 75.230 | 17.111 | n/a |

## Stratified differences

| Stratum | Value | Rows | MAE difference | WIS difference | Calibration difference |
|---|---|---:|---:|---:|---:|
| area | DK1 | 576 | -79.673 | n/a | n/a |
| area | DK2 | 576 | -90.032 | n/a | n/a |
| dst | dst | 1152 | -84.853 | n/a | n/a |
| extreme_price | extreme | 58 | -97.046 | n/a | n/a |
| extreme_price | typical | 1094 | -84.206 | n/a | n/a |
| hour | 00 | 48 | -97.027 | n/a | n/a |
| hour | 01 | 48 | -88.598 | n/a | n/a |
| hour | 02 | 48 | -84.222 | n/a | n/a |
| hour | 03 | 48 | -81.898 | n/a | n/a |
| hour | 04 | 48 | -90.920 | n/a | n/a |
| hour | 05 | 48 | -91.515 | n/a | n/a |
| hour | 06 | 48 | -81.374 | n/a | n/a |
| hour | 07 | 48 | -82.338 | n/a | n/a |
| hour | 08 | 48 | -55.178 | n/a | n/a |
| hour | 09 | 48 | -42.553 | n/a | n/a |
| hour | 10 | 48 | -41.397 | n/a | n/a |
| hour | 11 | 48 | -52.903 | n/a | n/a |
| hour | 12 | 48 | -58.483 | n/a | n/a |
| hour | 13 | 48 | -52.801 | n/a | n/a |
| hour | 14 | 48 | -54.175 | n/a | n/a |
| hour | 15 | 48 | -51.292 | n/a | n/a |
| hour | 16 | 48 | -85.230 | n/a | n/a |
| hour | 17 | 48 | -126.764 | n/a | n/a |
| hour | 18 | 48 | -120.336 | n/a | n/a |
| hour | 19 | 48 | -101.709 | n/a | n/a |
| hour | 20 | 48 | -159.924 | n/a | n/a |
| hour | 21 | 48 | -150.364 | n/a | n/a |
| hour | 22 | 48 | -111.057 | n/a | n/a |
| hour | 23 | 48 | -74.403 | n/a | n/a |
| market_regime | quarter_hour_aggregated_to_hourly | 1152 | -84.853 | n/a | n/a |
| month | 2026-06 | 768 | -31.751 | n/a | n/a |
| month | 2026-07 | 384 | -191.055 | n/a | n/a |
| negative_price | negative | 35 | -118.973 | n/a | n/a |
| negative_price | non_negative | 1117 | -83.783 | n/a | n/a |

## Method

- Model rows match exactly on the recorded pairing keys.
- WIS uses q10, q50, and q90; calibration error is the mean absolute quantile calibration error.
- Confidence intervals use a deterministic circular moving-block bootstrap over chronological forecast origins.
- Extreme-price groups use the absolute-target threshold recorded in the JSON report.
- The report is descriptive and does not select or deploy a model.
