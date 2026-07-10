# Open-Meteo Weather Forecast Features v1

This document is the v1 processing contract for the Open-Meteo weather forecast
MVP. It intentionally keeps weather preprocessing separate from Energi Data
Service prices. The canonical price panel remains:

```text
data/model_ready/price_panel_hourly_v1.parquet
```

The canonical price panel remains weather-free. Availability-safe joins are
used both for experiment frames and by the production Chronos adapter directly
from the canonical long weather table.

## Source

Provider:

```text
Open-Meteo
```

Product:

```text
Previous Runs API
https://previous-runs-api.open-meteo.com/v1/forecast
```

Phase 1 model ids:

```text
gfs_global
icon_eu
metno_nordic
```

Phase 3 candidate:

```text
dmi_harmonie_arome_europe
```

The Phase 3 DMI HARMONIE model is a recent-period upgrade only. It is not the
long historical backbone.

## Request Contract

Base variables:

```text
temperature_2m
wind_speed_10m
wind_direction_10m
wind_speed_100m
shortwave_radiation
cloud_cover
precipitation
```

Leads:

```text
previous_day1
previous_day2
```

Request parameters:

```text
timezone=UTC
wind_speed_unit=ms
temperature_unit=celsius
precipitation_unit=mm
```

The source client builds hourly parameter names by combining each base variable
with each lead. Raw HTTP response bytes are preserved under:

```text
data/raw/open_meteo/previous_runs/<weather_model>/<location_id>/fetched_at=<timestamp>/start=<start>_end=<end>.json
```

The raw manifest lives at:

```text
data/raw/open_meteo/manifest.jsonl
```

Each manifest row includes request parameters, retrieval time, HTTP status,
response hash, saved-file hash, row count, model id, location id, and raw path.
Duplicate logical rows inside one immutable raw batch must agree. Across
separate retrieval batches, the latest `retrieved_at_utc` row wins
deterministically so upstream revisions are represented while `raw_batch_id`
preserves provenance.

## Coordinate Basket

DK1:

```text
dk1_aalborg          57.0488,  9.9217
dk1_aarhus           56.1629, 10.2039
dk1_esbjerg          55.4765,  8.4594
dk1_odense           55.4038, 10.4024
dk1_herning          56.1393,  8.9738
```

DK2:

```text
dk2_copenhagen       55.6761, 12.5683
dk2_holbaek          55.7175, 11.7128
dk2_naestved         55.2299, 11.7609
dk2_nykobing_falster 54.7654, 11.8755
dk2_roenne           55.1009, 14.7066
```

V1 uses a simple unweighted mean across available basket points. There is no
interpolation and no forward fill.

## Normalized Long Form

Normalized rows are stored with:

```text
source_provider
source_product
weather_model
lead_time_days
location_id
area
latitude
longitude
valid_time_utc
forecast_available_at_utc
parameter_id
value
unit
raw_batch_id
retrieved_at_utc
```

For Open-Meteo `previous_dayN`, the v1 availability timestamp is:

```text
forecast_available_at_utc = valid_time_utc - N days
```

This is conservative and explicit. A `previous_day1` value for a late delivery
hour can still be unavailable at a 10:00 UTC day-ahead forecast origin, so
availability masking is mandatory at join time.

## Area Features

The area-hour long table is keyed by:

```text
area
ds_utc
weather_model
lead_time_days
parameter_id
```

The feature name pattern is:

```text
weather_<model>_lead<Nd>_<parameter>
```

Each row contains:

```text
value
unit
location_count
expected_location_count
location_coverage_ratio
location_coverage_pass
feature_window_coverage_ratio
feature_group_pass
forecast_available_at_utc
dataset_version
```

`location_coverage_pass` is the per-hour basket-point coverage check.
`feature_window_coverage_ratio` is the share of hours in the built window whose
per-hour coverage passed. `feature_group_pass` is true only when the feature
window coverage is at least 95 percent. Missing values stay missing. V1 does not
impute weather features.

## Availability-Safe Joins

Weather is joined through:

```text
dkenergy_forecast.features.weather_features
```

The join key is:

```text
area
ds_utc
```

The leakage rule is:

```text
forecast_available_at_utc <= forecast_origin_utc
```

Rows failing that rule are left null. Coverage-failing feature groups and
individual low-coverage hours are excluded by default. Materialized joined
experiment frames live under `data/features/` and must not overwrite the
canonical EDS price panel. Production Chronos applies the same join in memory
for each historical context row and future delivery row, then applies its
artifact-versioned per-series fill policy without changing the source weather
artifact.

## First Commands

Fetch raw Open-Meteo batches:

```bash
python scripts/fetch_open_meteo_previous_runs.py --start 2024-07-01 --end YYYY-MM-DD
```

Build normalized and area-hourly weather features:

```bash
python scripts/build_open_meteo_weather_features.py
```

Build availability-masked price plus weather modeling frames only when you are
doing weather model development or backtests:

```bash
python scripts/build_weather_backtest_frame.py --frame-kind recent
python scripts/build_weather_backtest_frame.py --frame-kind backtest
```

The recent frame is a short diagnostic artifact. The backtest frame uses the
default offline comparison window. For a 730-day frame, use `--frame-kind custom
--days 730 --output-path ...`; that larger build is intentionally explicit.
Neither frame is a live forecast publishing artifact.

Explore weather-enhanced CatBoost models from the modeling notebook after
installing the optional tuning dependencies:

```bash
pip install -e ".[tuning,notebooks]"
jupyter notebook notebooks/05_catboost_model_development.ipynb
```

## QA Expectations

The default builder writes:

```text
data/normalized/open_meteo_previous_runs_open_meteo_previous_runs_v1.parquet
data/features/weather_open_meteo_area_hourly_long_open_meteo_previous_runs_v1.parquet
data/features/weather_open_meteo_area_hourly_open_meteo_previous_runs_v1.qa.json
```

The long table is the canonical weather feature artifact. A derived wide table
can be written for ad hoc inspection with `--write-wide`, but it is not part of
the normal source-processing contract.

The QA report includes raw batch count, row counts, UTC range, model ids,
parameters, lead days, null count, usable feature-row counts, and distinct
feature-group coverage pass counts.

## Judgement Calls

1. Open-Meteo is an MVP processed forecast source, not final raw upstream
   weather provenance.
2. The v1 coordinate basket is fixed and versioned in code.
3. Area aggregation is a simple mean because the first question is whether
   forecast weather improves the price model at all.
4. Model-specific maximum-history experiments are allowed, but comparison
   reports should also include common-overlap tables when coverage differs.
5. Weather ablations must skip missing weather groups rather than silently
   relabeling a price-only run.
6. DMI direct archiving remains a separate future provenance track.
