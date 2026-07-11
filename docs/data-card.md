# Data card

## Prices

Energi Data Service supplies DK1/DK2 day-ahead prices. Raw JSON and request
metadata are archived before deterministic normalization. Duplicate conflicts,
missing UTC hours, mismatched area coverage, incomplete quarter-hour groups,
transition boundaries, and DST delivery-day lengths are checked during build.

There is a target-regime break at 2025-10-01 local time:

- Earlier rows are native hourly prices.
- Later rows are hourly arithmetic means of four native quarter-hour products.

This distinction is encoded in every row and must be used as an evaluation
stratum. Details are in
[data-processing/energi_data_service_v1.md](data-processing/energi_data_service_v1.md).

Price publication time is a deterministic project convention: noon Copenhagen
on the calendar day before delivery. It is an information-set model, not a
source revision timestamp.

## Weather

Open-Meteo Previous Runs supplies GFS Global, ICON-EU, and MET Norway Nordic
fields over five representative coordinates per price area. Area features are
simple coordinate averages with explicit coverage thresholds; they are not
capacity-, population-, or offshore-weighted fundamentals.

The source does not expose observed model initialization/publication timestamps
for these fields. The project labels `valid_time - lead` as a synthetic
reference and availability proxy and generates a coherent `weather_vintage_id`.
Do not describe that ID as an upstream NWP run.

Details are in
[data-processing/open_meteo_weather_v1.md](data-processing/open_meteo_weather_v1.md).

## Known limitations

- The hourly post-transition target hides intrahour price structure.
- Historical price revision times are not modeled separately.
- Weather geography is coarse and not generation-capacity weighted.
- Synthetic weather availability may differ from real provider publication.
- Direct market fundamentals such as load/generation forecasts, outages,
  interconnector capacity, neighboring prices, fuel, and carbon are absent.
- Runtime artifacts are ignored by Git; reviewed evaluation reports and hashes
  are the source-controlled evidence layer.

Any new source should preserve raw bytes, observed availability/revision times,
source identity, units, geography, and a deterministic build/version contract.
