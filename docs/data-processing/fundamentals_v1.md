# Point-in-Time Fundamentals v1

`fundamentals_v1` is the offline data contract for ex-ante electricity-market
fundamentals. It deliberately contains no API client or live-pipeline wiring.
Its first purpose is to make future ENTSO-E, Energinet, and TSO ingestion safe
to evaluate before those sources are used by a model.

## Scope

The v1 parameter allowlist is intentionally narrow:

```text
total_load_forecast
wind_generation_forecast
solar_generation_forecast
available_transfer_capacity
planned_transmission_outage_flag
```

The first three are `bidding_zone` series. Transfer capacity and transmission
outage flags are `directed_border` series. Capacity and forecast values
use `MW`; outage flags use `boolean` values `0` or `1`.

## Native Schema

Native source intervals use this canonical order:

```text
source_provider
source_dataset
source_document_id
source_series_id
source_revision
fundamentals_vintage_id
parameter_id
entity_type
area
from_area
to_area
valid_time_start_utc
valid_time_end_utc
native_resolution_minutes
value
unit
forecast_reference_time_utc
forecast_reference_time_type
published_at_utc
available_at_utc
availability_time_type
raw_batch_id
retrieved_at_utc
```

For `bidding_zone` rows, `area` is required and `from_area`/`to_area` are null.
For `directed_border` rows, `area` is null and distinct `from_area` and
`to_area` values are required. This avoids representing one physical direction
twice from different area perspectives. All intervals are UTC, remain inside
one UTC hour, and have a duration equal to `native_resolution_minutes`. The
resolution must divide 60 exactly.

Every source revision and `fundamentals_vintage_id` is preserved. Duplicate
rows within one revision/vintage are rejected; the builder never applies a
global "keep latest" deduplication that would erase historical revisions.

## Availability Contract

`available_at_utc` is the timestamp used by point-in-time joins. Its provenance
must be one of:

```text
observed_source_publication
observed_retrieval
provider_schedule_proxy
synthetic_proxy
```

For `observed_source_publication`, `available_at_utc` equals
`published_at_utc`. For `observed_retrieval`, it equals `retrieved_at_utc`.
Proxy timestamps remain explicitly labelled and must not be presented as
strict observed publication history.

The builder also requires:

```text
forecast_reference_time_utc <= available_at_utc <= retrieved_at_utc
```

When a publication timestamp exists, it may not be after retrieval.

## Exact Hourly Build

`build_hourly_fundamentals` groups by the complete source lineage, including
revision and vintage, before aggregating. It therefore cannot combine quarter
hours from different revisions to manufacture a complete hour.

Power-like series use a time-weighted mean. Outage flags use the maximum. Each
hour records:

```text
hourly_aggregation
interval_count
expected_interval_count
interval_coverage_ratio
interval_coverage_pass
```

An incomplete or non-contiguous vintage-hour is retained for auditability, but
its `value` is null and `interval_coverage_pass` is false. It is never eligible
for a model join.

Canonical feature names are:

```text
fundamental_<provider>_<zone>_<parameter_id>_<unit>
fundamental_<provider>_<from_area>_to_<to_area>_<parameter_id>_<unit>
```

Provider, geography, parameter, and unit are all part of feature identity.
Neighbouring zones and alternate providers therefore cannot overwrite a DK
series merely because they publish the same parameter. If normalization of
punctuation would still make two distinct identities produce the same column
name, the build fails instead of silently merging them.

The complete hourly schema is exported as `HOURLY_COLUMNS` from
`dkenergy_data.build.fundamentals_v1`.

## Point-in-Time Join

`join_fundamentals_features` joins on:

```text
ds_utc
feature_name
```

The source geography is encoded in `feature_name` and retained in lineage.
Every target row therefore receives all eligible zone and directed-border
features for its delivery hour, including neighbouring-zone features; target
`area` is not used to discard them.

For every forecast row and feature, the join keeps only complete source hours
with:

```text
available_at_utc <= information_cutoff_utc
```

If `information_cutoff_utc` is absent or null,
`forecast_origin_utc` is used. Among eligible revisions, selection prefers the
latest forecast reference time and then the latest availability timestamp.
Opaque revision or vintage strings are never used as temporal ordering keys. If
multiple distinct revision/vintage identities tie at that newest temporal
point, the join fails closed. Rows with the same revision/vintage identity may
be collapsed across retrieval batches only when their values and semantic
source lineage are equivalent; conflicting repeats also fail closed. Values
unavailable at the cutoff stay null; they are not forward filled or replaced
with a later revision.

Each selected value gets feature-specific lineage columns for its provider,
dataset, source document and series, revision, vintage, reference/publication/
availability/retrieval timestamps, availability type, native resolution,
coverage, and border direction.

## Derived Features

The feature module can add:

```text
fundamental_<provider>_<zone>_renewable_generation_forecast_<unit>
fundamental_<provider>_<zone>_residual_load_forecast_<unit>
fundamental_<provider>_<zone>_total_available_import_capacity_<unit>
fundamental_<provider>_<zone>_total_available_export_capacity_<unit>
```

Residual load is only computed when one provider supplies load, wind, and solar
for the same zone and unit. Capacity totals require an explicit provider
topology such as:

```python
expected_directed_borders = {
    "entsoe": [
        ("NO2", "DK1"),
        ("SE3", "DK1"),
        ("DK1", "DE_LU"),
    ]
}
```

Every observed transfer-capacity feature must match that declaration, and every
declared directed border must exist as a feature series. A missing or unexpected
series fails the build instead of changing the meaning of a total silently. A
total remains null for a row when any declared constituent value is missing.
Derived neighbouring-zone values remain available to every target area.
Missing inputs stay missing.
