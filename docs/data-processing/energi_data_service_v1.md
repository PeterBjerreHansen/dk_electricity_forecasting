# Energi Data Service Price Dataset v1

This is the data-processing explainer for the first project dataset built from
Energi Data Service (EDS). It defines how to access, save, normalize, stitch,
clean, and validate the Danish DK1/DK2 day-ahead price data before any
forecasting code consumes it.

Project dataset name: `eds_day_ahead_prices_v1`

Primary model-ready output:

```text
data/model_ready/price_panel_hourly_v1.parquet
```

The model-ready table should contain:

```text
unique_id, ds_utc, ds_local, y, area
```

plus basic calendar features and audit columns.

## Source Facts Checked

EDS access is through unauthenticated HTTP GET requests.

Official API guide:

```text
https://www.energidataservice.dk/guides/api-guides
```

Metadata endpoints:

```text
https://api.energidataservice.dk/meta/dataset/Elspotprices
https://api.energidataservice.dk/meta/dataset/DayAheadPrices
```

Data endpoints:

```text
https://api.energidataservice.dk/dataset/Elspotprices
https://api.energidataservice.dk/dataset/DayAheadPrices
```

The old `Elspotprices` dataset is discontinued. Its own description says it is
no longer updated and points users to `DayAheadPrices` for data after
2025-09-30. For this project, the stitch is therefore part of the v1 data
contract, not an implementation detail.

Relevant source schemas:

| Source dataset | Time column | Local time column | Area column | DKK column | EUR column | Native resolution |
| --- | --- | --- | --- | --- | --- | --- |
| `Elspotprices` | `HourUTC` | `HourDK` | `PriceArea` | `SpotPriceDKK` | `SpotPriceEUR` | hourly |
| `DayAheadPrices` | `TimeUTC` | `TimeDK` | `PriceArea` | `DayAheadPriceDKK` | `DayAheadPriceEUR` | quarter-hourly from the current source data |

Only `DK1` and `DK2` are in scope for v1. Do not include `SYSTEM` or neighboring
price areas in this first model-ready panel.

## Access Plan

Use explicit API parameters for all requests:

```text
start=YYYY-MM-DD
end=YYYY-MM-DD
filter={"PriceArea":["DK1","DK2"]}
columns=<required columns only>
sort=<time column>,PriceArea
limit=0
```

Important EDS API behavior:

1. `start` is inclusive and `end` is exclusive.
2. EDS interprets `start` and `end` in Danish local time.
3. If `sort` is omitted, the API sorts descending by the source key. Always set
   ascending sort explicitly.
4. `limit=0` requests all rows in the selected window.
5. Respect EDS rate limits. Backfills should use date chunks and retry 429
   responses only after the server-provided wait.
6. Retry transient request exceptions and selected retryable 5xx responses. Long
   historical backfills should not fail permanently on a single connection reset
   or temporary server error.

Recommended chunking:

| Source | Fetch window |
| --- | --- |
| `Elspotprices` | monthly or quarterly chunks from first desired history through `2025-10-01` local-exclusive |
| `DayAheadPrices` | monthly chunks from `2025-10-01` local through the current available horizon |

For the first full historical build, fetch the complete DK1/DK2 history exposed
by EDS and let modeling configs choose a later training start if needed. The
raw data volume is small enough that preserving full source history is more
valuable than prematurely dropping older market regimes.

## Raw Save Plan

Raw API responses are immutable audit artifacts. Save the exact HTTP response
bytes before normalization.

Recommended layout:

```text
data/raw/energi_data_service/
  metadata/
    Elspotprices_<retrieved_at_utc>.json
    DayAheadPrices_<retrieved_at_utc>.json
  Elspotprices/
    fetched_at=<YYYYMMDDTHHMMSSZ>/
      start=<YYYY-MM-DD>_end=<YYYY-MM-DD>.json
  DayAheadPrices/
    fetched_at=<YYYYMMDDTHHMMSSZ>/
      start=<YYYY-MM-DD>_end=<YYYY-MM-DD>.json
```

Each raw batch should have a sidecar manifest row with:

```text
batch_id
source_dataset
request_url
request_params
retrieved_at_utc
http_status
record_count
response_sha256
saved_json_sha256
raw_path
```

Do not re-serialize raw JSON to fix schema, ordering, duplicates, or bad
values. The saved raw file should be byte-for-byte identical to the HTTP
response body. All changes happen downstream and must be reproducible from the
raw files plus the manifest.

Builds that use a manifest must verify the saved raw file hash before
normalization. A raw file whose bytes no longer match `saved_json_sha256` /
`response_sha256` is treated as corrupted input and fails the build.

## Normalized Tables

Create one normalized parquet table per source before stitching:

```text
data/normalized/eds_elspotprices_v1.parquet
data/normalized/eds_day_ahead_prices_15min_v1.parquet
```

Normalize both sources into this common schema:

```text
source_dataset
source_time_utc
source_time_local_text
area
price_dkk_per_mwh
price_eur_per_mwh
source_resolution_minutes
raw_batch_id
retrieved_at_utc
```

Judgement calls:

1. `source_time_utc` is the canonical timestamp. Parse source UTC timestamps as
   timezone-aware UTC even though the API string does not include a `Z`.
2. The source local timestamp is audit data only. Keep it as text or a naive
   timestamp, but do not use it as a key because DST fall-back hours repeat.
   Validate it against `source_time_utc` converted to `Europe/Copenhagen`, and
   fail normalization if they disagree.
3. Derive modeling local time from `source_time_utc` using
   `Europe/Copenhagen`.
4. Keep both DKK and EUR prices in normalized data. The model-ready `y` uses
   DKK/MWh for v1.
5. Preserve `source_dataset` and `source_resolution_minutes` for audit and QA.
   Do not use them as model features in the first baseline.

## Stitching Rule

The stitched v1 hourly target is defined by Danish local delivery time:

```text
Elspotprices      where HourDK <  2025-10-01T00:00:00
DayAheadPrices    where TimeDK >= 2025-10-01T00:00:00
```

This creates a clean boundary:

```text
last old local day: 2025-09-30
first new local day: 2025-10-01
```

Do not average the two sources together on overlapping records. If future EDS
backfills create overlap around the boundary, source selection still follows
the rule above.

## Quarter-Hour To Hourly Rule

`DayAheadPrices` is quarter-hourly in the current source data. The project target
is hourly, so aggregate four quarter-hour intervals into one hourly observation
per price area.

Hourly aggregation:

```text
hourly_price = arithmetic_mean(4 quarter-hour prices in the same UTC hour)
```

This is equivalent to a time-weighted average because all included intervals are
15 minutes. Do not sum prices. Do not volume-weight prices because the source
does not provide traded volume per interval.

Completeness requirement:

```text
exactly 4 quarter-hour rows per area per UTC hour
```

If an hour has fewer or more than four quarter-hour rows after deduplication,
mark that area-hour invalid and fail the build unless the run is explicitly
configured to allow incomplete recent data. Never forward-fill prices.

Implementation note: `--allow-incomplete-recent` only drops incomplete
`DayAheadPrices` groups when they are the latest hour for that area. Earlier
incomplete groups still fail the build.

## Model-Ready Panel

Output path:

```text
data/model_ready/price_panel_hourly_v1.parquet
```

Schema:

```text
unique_id
ds_utc
ds_local
local_date
local_hour
local_day_of_week
local_month
is_weekend
is_dst
utc_offset_hours
area
y
market_regime
native_resolution_minutes
target_aggregation
target_definition
price_dkk_per_mwh
price_eur_per_mwh
source_dataset
source_resolution_minutes
dataset_version
price_available_at_utc
```

Column decisions:

1. `unique_id` should be stable and area-specific, e.g.
   `day_ahead_price_DK1` and `day_ahead_price_DK2`.
2. `ds_utc` is the primary time key and must be unique with `area`.
3. `ds_local` is derived from `ds_utc` in `Europe/Copenhagen` and retained as a
   feature/display timestamp.
4. `y = price_dkk_per_mwh` for v1. DKK is the project target because the first
   use case is Danish price forecasting. EUR is retained for audit and later
   experiments.
5. Calendar features are derived from local Danish time, not UTC, because
   demand and market behavior follow local hour, weekday, weekends, and DST.
6. `price_available_at_utc` is a deterministic v1 market-availability timestamp:
   every local delivery day is treated as published at 12:00
   `Europe/Copenhagen` on the previous local calendar day.
7. The target contract is explicit on every row and every future horizon:
   `market_regime` is `native_hourly` before the stitch boundary and
   `native_quarter_hour` from the boundary onward;
   `native_resolution_minutes` is 60 or 15; `target_aggregation` is `identity`
   or `arithmetic_mean_of_4_quarter_hours`; and `target_definition` is
   `hourly_day_ahead_area_price_dkk_per_mwh`.

Primary key:

```text
(area, ds_utc)
```

Expected cadence:

```text
one row per UTC hour per area
```

The number of local rows per Danish local date can be 23, 24, or 25 around DST.
That is valid. The UTC panel should remain hourly and gap-free.

Official v1 builds require exactly `DK1` and `DK2` and require both areas to
share the same UTC timestamp coverage. Single-area panels are allowed only for
explicit experiments by overriding the required areas in the build command.

## Cleaning Rules

Duplicate handling:

1. Raw duplicates are preserved.
2. Duplicate logical rows inside one immutable raw batch must agree; conflicting
   values fail the build.
3. Overlapping batches may contain source revisions. The latest
   `retrieved_at_utc` row wins deterministically while `raw_batch_id` preserves
   its provenance.
4. Conflicting stitched rows for the same `(ds_utc, area)` fail the build.

Missing values:

1. Missing timestamps, areas, or prices fail normalization.
2. Missing hourly observations fail the model-ready build unless the build is
   explicitly marked as an incomplete live refresh.
3. Missing recent future delivery intervals are allowed only in raw/latest
   ingestion, not in a finalized historical dataset.

Price values:

1. Negative prices are valid and must not be clipped.
2. Extreme positive or negative spikes are valid unless EDS later corrects them.
   Flag them in QA, but do not winsorize the target.
3. Store prices as `float64`.
4. Do not recompute DKK from EUR in v1. Use the EDS DKK value as published, and
   keep EUR alongside it. EDS notes that DKK prices are calculated from EUR and
   Danmarks Nationalbank exchange rates in the newer source.

Areas:

1. Allow only `DK1` and `DK2`.
2. Fail on unknown areas in a v1 build, rather than silently dropping them after
   the source query.

Time:

1. Use UTC internally.
2. Local time is a derived feature and display field.
3. Validate the source local timestamp against the UTC-derived
   `Europe/Copenhagen` timestamp, but do not key on source local time.
4. DST repeated local hours are expected. Keep `utc_offset_hours` so repeated
   wall-clock hours can be distinguished.

## Leakage And Backtesting Implications

The data pipeline produces final target values. Forecasting code must not call
EDS directly.

For backtesting:

1. Train only on target rows with `price_available_at_utc` before the forecast
   origin.
2. Build lag and rolling features only from rows that would have been known at
   that forecast origin.
3. Day-ahead rows for delivery hours after the origin clock time may be used
   when their `price_available_at_utc` is before the origin.
4. Store `dataset_version` with every result artifact.

The source does not provide row-level historical publication timestamps in this
v1 plan. The project therefore uses the deterministic Nord Pool day-ahead
publication policy above rather than treating delivery time as availability
time.

## Validation Report

Every build should write a lightweight QA report next to the parquet outputs:

```text
data/model_ready/price_panel_hourly_v1.qa.json
```

Minimum checks:

```text
dataset_version
created_at_utc
source_datasets
source_metadata_sha256
raw_source_audit
artifact_status
allow_incomplete_recent
build_scope
requested_start_local
requested_end_local
observed_start_local
observed_end_local
bounded_local_range_check
row_count
min_ds_utc
max_ds_utc
areas
duplicate_key_count
missing_hour_count
invalid_quarter_hour_group_count
invalid_quarter_hour_group_sample
negative_price_count
null_price_count
min_price_dkk_per_mwh
max_price_dkk_per_mwh
transition_boundary_check
shared_utc_coverage_check
dst_day_checks
```

Artifact status values:

1. `final_historical` means the panel passed structural QA and any requested
   bounded local range is fully covered by observed hourly rows.
2. `incomplete_live_refresh` means the build intentionally dropped only the
   latest incomplete quarter-hour group for a live refresh.
3. `incomplete_bounded_range` means a requested local `--start` / `--end` range
   was not fully covered by the available raw source rows.

Transition boundary check:

1. Last `Elspotprices` model-ready local date should be `2025-09-30`.
2. First `DayAheadPrices` model-ready local date should be `2025-10-01`.
3. There should be no `(area, ds_utc)` gap between the final old hourly row and
   first new hourly row.

Shared UTC coverage check:

1. Official v1 builds require observed areas to be exactly `DK1` and `DK2`.
2. DK1 and DK2 must have identical `ds_utc` timestamp sets.
3. Builds that intentionally use one area must pass an explicit one-area
   `--required-areas` value.

## Implementation map

Implemented files:

```text
src/dkenergy_data/sources/energidataservice.py
src/dkenergy_data/build/eds_prices_v1.py
scripts/fetch_eds_prices.py
scripts/build_price_panel.py
tests/test_eds_prices_v1.py
```

The implementation provides:

1. Add a small EDS client that can fetch metadata and data with explicit
   parameters, retries, and 429 handling.
2. Add raw JSON writers and a manifest writer.
3. Add normalizers for `Elspotprices` and `DayAheadPrices`.
4. Add a stitcher that applies the source boundary rule and quarter-hour hourly
   aggregation.
5. Add model-ready calendar feature generation from `Europe/Copenhagen` local
   time.
6. Add QA report generation and make the build fail on duplicate keys, missing
   historical hours, and incomplete quarter-hour groups.
7. Verify manifest raw hashes before normalization.
8. Make incomplete recent drops visible in QA with dropped group samples.
9. Thin command scripts for the two operations:

```text
scripts/fetch_eds_prices.py
scripts/build_price_panel.py
```

Current commands:

```text
python scripts/fetch_eds_prices.py --areas DK1 DK2
python scripts/build_price_panel.py --dataset-version v1
```

For a small refresh or smoke run, pass explicit local delivery dates:

```text
python scripts/fetch_eds_prices.py --start 2025-09-29 --end 2025-10-03 --areas DK1 DK2
python scripts/build_price_panel.py --start 2025-09-29 --end 2025-10-03
```

For an explicit one-area experiment:

```text
python scripts/fetch_eds_prices.py --start 2025-10-01 --end 2025-10-03 --areas DK1
python scripts/build_price_panel.py --start 2025-10-01 --end 2025-10-03 --required-areas DK1
```
