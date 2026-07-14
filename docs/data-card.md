# Data card

This card summarizes the datasets used by the production forecast and the most
important qualifications on any result derived from them. The detailed,
executable contracts live in the source-specific processing guides.

## Electricity prices

Energi Data Service supplies DK1 and DK2 day-ahead prices. The ingestion layer
archives raw response bytes and request metadata before deterministic
normalization. The builder checks duplicate conflicts, missing UTC intervals,
area symmetry, incomplete quarter-hour groups, the source-product transition,
and 23/24/25-hour Danish delivery days.

There is a target-regime break at 2025-10-01 Copenhagen time:

- Earlier rows are native hourly `Elspotprices` values.
- Later rows are hourly arithmetic means of four native 15-minute
  `DayAheadPrices` values.

Every panel, horizon, prediction, and evaluation row carries
`market_regime`, `native_resolution_minutes`, `target_aggregation`, and
`target_definition`. The post-transition target must not be described as a
native hourly market product.

`price_available_at_utc` uses a deterministic project convention: local noon
on the calendar day before delivery. It models the information set; it is not
an observed source revision or publication timestamp. Historical source
revisions are not reconstructed as-of each old forecast origin.

See the [Energi Data Service processing contract](data-processing/energi_data_service_v1.md).

## Forecast weather

Open-Meteo Previous Runs supplies GFS Global, ICON-EU, and MET Norway Nordic
fields over five representative coordinates in each price area. The builder
stores immutable raw responses, normalized location-level values, and a
canonical long area-hour table. Area values are simple coordinate means with
explicit coverage gates; they are not capacity-, population-, land-area-, or
offshore-weighted fundamentals.

The modeled variables are:

- 2 m temperature.
- 10 m and 100 m wind speed.
- Shortwave radiation.
- Cloud cover.
- Precipitation.

Wind direction is deliberately excluded from model covariates. The current
source table aggregates it arithmetically across coordinates, which is invalid
for a circular quantity. It should only return after vector or sine/cosine
aggregation is implemented and an ablation demonstrates value.

Open-Meteo Previous Runs does not expose an observed forecast-run
initialization or publication timestamp. The project uses
`valid_time - requested lead` as an explicitly synthetic reference and
availability proxy. `weather_vintage_id` is project-generated provenance, not
an upstream numerical-weather-prediction run ID.

At each training, replay, or live row, the canonical weather join selects the
newest eligible value for each `(area, valid hour, model, parameter)` under:

```text
forecast_available_at_utc <= information_cutoff_utc
```

The selected lead, source feature, reference time, availability type, and
vintage stay attached as metadata. The stable model feature name does not
encode lead. Training and serving apply the same selection and missing-value
semantics: no forward or backward fill across valid hours, coverage validation
before missing values are replaced with the artifact-declared zero value.

See the [Open-Meteo processing contract](data-processing/open_meteo_weather_v1.md).

## Known limitations

- The hourly post-transition target hides intrahour price structure.
- Historical price revisions are represented by the latest retrieved raw
  version, not by a complete as-of revision store.
- Price availability uses a project convention rather than observed
  publication events.
- Weather availability uses a synthetic proxy that may differ from actual
  provider latency.
- The weather geography is coarse and not generation-capacity weighted.
- Direct market fundamentals—load and generation forecasts, outages,
  interconnector capacity, neighboring prices, fuels, and carbon—are absent.
- Runtime Parquet files and model weights are ignored by Git; manifests,
  hashes, tests, and reviewed comparison reports carry reproducibility claims.

Any new source should preserve raw bytes, retrieval time, observed
availability/revision time when available, source identity, units, geography,
and a deterministic dataset-version contract.
