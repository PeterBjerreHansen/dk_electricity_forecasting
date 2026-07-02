# DMI Weather Features v1 Plan

This note is the preliminary data-processing plan for adding Danish weather
features to the electricity-price forecasting project. It is intentionally not
an implementation contract yet. Weather columns should not be joined into the
model-ready price panel until the EDS-only baseline and CatBoost workflows are
stable.

Project source name:

```text
dmi_weather_v1
```

Forecasting dependency:

```text
docs/forecasting/forecasting_library_contract_v1.md
```

## Status

Research and design only for direct DMI ingestion. The implemented forecast
weather MVP currently uses Open-Meteo Previous Runs and is documented separately
in:

```text
docs/data-processing/open_meteo_weather_v1.md
```

DMI remains the planned direct-archive provenance track, especially for
HARMONIE going forward.

## Source Facts Checked

DMI Open Data is accessible through:

```text
https://opendataapi.dmi.dk/
```

As of DMI's authentication documentation, API keys are no longer required for
`opendataapi.dmi.dk` from 2025-12-02 onward. The API remains subject to fair use
and rate limiting.

Relevant official documentation:

```text
https://www.dmi.dk/friedata/dokumentation/authentication
https://www.dmi.dk/friedata/dokumentation/meteorological-observations-data
https://www.dmi.dk/friedata/dokumentation/climate-data
https://www.dmi.dk/friedata/dokumentation/apis/climate-data-api-1
https://www.dmi.dk/friedata/dokumentation/forecast-data
https://www.dmi.dk/friedata/dokumentation/forecast-data-edr-api
https://www.dmi.dk/friedata/dokumentation/download-data
```

Small live probes on 2026-07-01 confirmed:

1. `climateData/collections/stationValue/items` returns GeoJSON features with
   `parameterId`, `qcStatus`, `from`, `to`, `validity`, `value`, `stationId`,
   and point geometry.
2. Recent hourly Climate Data may have `qcStatus="none"`, so it should not be
   silently mixed into finalized historical feature sets.
3. Station discovery by `status=Active` alone is too broad; it can return manual
   snow stations. Station or grid selection must filter by parameter coverage
   and operating/validity dates.

## Forecasting Use Case

Weather features are meant to explain electricity-price drivers such as:

1. temperature-sensitive demand,
2. wind generation potential,
3. solar generation potential,
4. broad weather regimes that correlate with imports, congestion, and price
   volatility.

The first target remains hourly DK1/DK2 day-ahead electricity prices. Weather
must therefore be shaped to the same hourly UTC panel:

```text
unique_id
ds_utc
area
```

## Non-Negotiable Leakage Policy

Do not use future realized observations as predictors for historical backtests
or live forecasts.

Allowed for model features:

1. known-in-advance calendar features,
2. lagged realized weather observations available before `forecast_origin_utc`,
3. archived weather forecasts that were actually available at
   `forecast_origin_utc`,
4. live DMI forecasts for future production forecasts, if the raw forecast run
   is archived before use.

Not allowed for leakage-free forecasting:

1. joining observed DMI Climate Data for the forecast horizon,
2. using DMI observation data whose `from`/`to` interval ends at or after the
   forecast origin,
3. using the latest DMI forecast endpoint later and pretending it was available
   for older backtest origins.

Judgement call: observed weather for future delivery hours may be useful for
oracle EDA or upper-bound studies, but those artifacts must be labeled
`oracle_weather` and must not be compared as normal forecasts.

## Source Options

### Climate Data: `stationValue`

Best first historical source for realized weather observations.

Strengths:

1. quality-controlled DMI climate product,
2. hourly Denmark station values are in UTC,
3. includes `qcStatus`, `validity`, station metadata, and point geometry,
4. better suited to finalized historical datasets than raw observations.

Risks and judgement calls:

1. DMI says Denmark station values are manually quality controlled; erroneous or
   missing values may be filled by interpolation during quality control. This is
   acceptable for a clean historical weather table but must be documented.
2. Daily quality control may lag recent data; yearly quality control status is
   not fully represented by `qcStatus`.
3. Station availability differs by parameter and time. A station basket must be
   versioned and QA-checked.

### Climate Data: `10kmGridValue`

Preferred later v1/v2 source if hourly parameters and coverage are sufficient.

Strengths:

1. easier to aggregate to DK1/DK2 than sparse stations,
2. less exposed to station openings, closures, and parameter-specific holes,
3. more natural for wind/temperature/solar area summaries.

Risks and judgement calls:

1. Requires mapping DMI grid cells to DK1/DK2 geometry.
2. Bulk download is likely more appropriate than paged item requests.
3. Need verify exact hourly parameter availability before choosing this as the
   first implementation.

### Meteorological Observations: `metObs`

Best for near-real-time monitoring and nowcasting, not finalized historical
features.

Strengths:

1. broad parameter set,
2. near-real-time observations,
3. useful fallback for live refresh before climate QC is available.

Risks and judgement calls:

1. DMI documents metObs as raw, not quality controlled.
2. It should be kept separate from finalized Climate Data and labeled as
   `raw_recent_observation` or similar.
3. Backtests should not silently combine QC climate values with raw recent
   values.

### Forecast Data: HARMONIE EDR/STAC

Best source for future weather predictors, but only if archived going forward.

Strengths:

1. HARMONIE includes useful forecast variables such as 2 m temperature, wind,
   radiation, humidity, pressure, and cloud cover.
2. EDR can request point forecast slices in JSON-like formats.
3. STAC/GRIB is better for full model fields once we need area-wide grids.

Critical limitation:

```text
DMI forecast EDR retains only the latest 24 hours of model runs.
```

Judgement call: DMI forecast data cannot support historical point-in-time
backtests unless we start archiving runs now or add a separate historical
forecast/reanalysis source later.

## Recommended v1 Path

Phase 1, research artifact only:

1. document source choices and leakage policy,
2. probe Climate Data parameter availability for weather variables,
3. decide whether station baskets or 10 km grid cells are the first practical
   historical source,
4. do not join weather into the price panel.

Phase 2, first implementation:

1. fetch and archive raw DMI Climate Data for a small bounded window,
2. normalize hourly observed weather into a separate table,
3. produce weather QA reports,
4. create lagged weather feature tables keyed by `(area, ds_utc)`,
5. compare EDS-only models against EDS-plus-lagged-weather models.

Phase 3, forecast features:

1. start archiving HARMONIE forecast runs,
2. normalize forecast runs with `forecast_reference_time_utc` and
   `valid_time_utc`,
3. join only forecast values whose reference time is available at the model
   origin,
4. rerun backtests only for origins covered by archived forecast runs.

## Candidate v1 Feature Groups

Temperature:

```text
area_temp_mean_c
area_temp_min_c
area_temp_max_c
heating_degree_c
cooling_degree_c
```

Wind:

```text
area_wind_speed_mean_ms
area_wind_speed_p90_ms
area_wind_gust_mean_ms
wind_power_proxy
```

Solar and cloud:

```text
global_radiation_mean_wm2
sunshine_minutes
cloud_cover_mean_pct
solar_proxy
```

Precipitation:

```text
precip_mm
precip_trace_flag
precip_wet_hour_flag
```

Lower-priority weather regime features:

```text
humidity_mean_pct
pressure_mean_hpa
wind_direction_sin
wind_direction_cos
```

## Spatial Strategy

Start with one of two explicit modes.

Station basket mode:

1. Choose stable stations for DK1 and DK2 by parameter coverage, not by active
   status alone.
2. Save a station manifest with station id, name, coordinates, operating
   period, parameters, and assigned price area.
3. Aggregate by area using simple means for v1 unless a better weighting scheme
   is justified.

Grid mode:

1. Use DMI 10 km grid cells if hourly parameter availability is sufficient.
2. Assign cells to DK1/DK2 using geometry.
3. Aggregate by simple cell average first; area/intersection weighting can come
   later if it materially changes features.

Preferred direction: grid mode if the parameter coverage is good; station
basket mode if grid availability or geometry setup slows down the first clean
weather pass.

## Cleaning Rules

Time:

1. Use UTC as canonical.
2. Treat Climate Data `from` as the interval start and `to` as interval end.
3. Use `from` as `ds_utc` for hourly features, after verifying that `to - from`
   is exactly one hour.
4. Danish local time is derived downstream from UTC, not used as a key.

Quality:

1. For finalized historical features, prefer `validity=true`.
2. Prefer manually quality-controlled Climate Data where available.
3. Recent `qcStatus="none"` values are allowed only in an explicitly marked
   live/incomplete artifact.
4. Do not interpolate missing weather values silently in v1. If DMI Climate Data
   has already filled values during QC, keep the value but document the source
   behavior.

Special codes:

1. Precipitation `-0.1` means trace precipitation. Preserve raw value, create a
   `precip_trace_flag`, and set cleaned `precip_mm` to `0.0` for first-pass
   model features.
2. Wind direction `0` means calm, not north. Do not encode it as 0 degrees in
   sine/cosine features unless wind speed is positive and direction is valid.
3. Cloud cover `112` means sky obscured. Preserve raw code, create an obscured
   flag, and do not treat it as 112 percent cloud cover.
4. DMI sunshine parameters changed algorithm in week 11 of 2026; any long-run
   sunshine feature needs a source-change flag or should be lower priority.

## Raw and Normalized Layout

Proposed raw layout:

```text
data/raw/dmi_weather/
  climateData/
    station/
    stationValue/
    10kmGridValue/
  metObs/
  forecastData/
```

Proposed normalized outputs:

```text
data/normalized/dmi_station_values_hourly_v1.parquet
data/normalized/dmi_grid_values_hourly_v1.parquet
data/normalized/dmi_forecast_harmonie_point_v1.parquet
```

Proposed feature outputs:

```text
data/features/weather_lagged_hourly_v1.parquet
data/features/weather_forecast_hourly_v1.parquet
```

Weather features should remain separate from
`data/model_ready/price_panel_hourly_v1.parquet` until the join contract is
reviewed.

## QA Report

Each weather artifact should report:

1. source endpoint and query params,
2. retrieval time and raw response hash,
3. parameter ids and units,
4. station ids or grid cell ids,
5. area assignment manifest hash,
6. UTC range,
7. row count by area, parameter, and source,
8. missing-hour counts,
9. duplicate key counts,
10. `qcStatus` counts,
11. `validity` counts,
12. special-code counts,
13. incomplete/live status.

## Open Questions

1. Should first historical weather features use station baskets or 10 km grids?
2. Which weather parameters have sufficiently complete hourly coverage for
   2024 onward?
3. Do we want true day-ahead weather forecasts for backtesting? If yes, we need
   to start archiving DMI forecast runs immediately or find a historical
   forecast/reanalysis source.
4. Should weather features live in a separate feature store table, or should a
   joined EDS-plus-weather panel be materialized for each experiment?
5. What exact forecast-origin time should weather forecasts be aligned to once
   day-ahead price publication timing is modeled more precisely?
