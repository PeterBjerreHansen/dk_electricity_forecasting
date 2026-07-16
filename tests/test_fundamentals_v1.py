from __future__ import annotations

import pandas as pd
import pytest

from dkenergy_data.build.fundamentals_v1 import (
    HOURLY_COLUMNS,
    NATIVE_COLUMNS,
    build_hourly_fundamentals,
    normalize_native_fundamentals,
)


def test_normalize_native_fundamentals_preserves_source_revisions() -> None:
    native = pd.concat(
        [
            _native_quarters(
                values=[10.0, 20.0, 30.0, 40.0],
                source_revision="1",
                vintage_id="load-revision-1",
            ),
            _native_quarters(
                values=[11.0, 21.0, 31.0, 41.0],
                source_revision="2",
                vintage_id="load-revision-2",
                available_at="2025-01-01T11:00:00Z",
                forecast_reference_time="2025-01-01T09:00:00Z",
            ),
        ],
        ignore_index=True,
    )

    normalized = normalize_native_fundamentals(native)

    assert list(normalized.columns) == NATIVE_COLUMNS
    assert len(normalized) == 8
    assert set(normalized["source_revision"]) == {"1", "2"}
    assert set(normalized["fundamentals_vintage_id"]) == {
        "load-revision-1",
        "load-revision-2",
    }


def test_normalize_native_fundamentals_rejects_unknown_parameter() -> None:
    native = _native_quarters(values=[1.0, 1.0, 1.0, 1.0])
    native["parameter_id"] = "actual_load"

    with pytest.raises(ValueError, match="Unsupported fundamentals parameter_id"):
        normalize_native_fundamentals(native)


def test_normalize_native_fundamentals_validates_availability_semantics() -> None:
    native = _native_quarters(values=[1.0, 1.0, 1.0, 1.0])
    native["available_at_utc"] = "2025-01-01T09:30:00Z"

    with pytest.raises(
        ValueError,
        match="observed_source_publication requires available_at_utc",
    ):
        normalize_native_fundamentals(native)


def test_build_hourly_fundamentals_aggregates_within_each_vintage() -> None:
    native = pd.concat(
        [
            _native_quarters(
                values=[10.0, 20.0, 30.0, 40.0],
                source_revision="1",
                vintage_id="load-revision-1",
            ),
            _native_quarters(
                values=[100.0, 200.0, 300.0, 400.0],
                source_revision="2",
                vintage_id="load-revision-2",
                available_at="2025-01-01T11:00:00Z",
                forecast_reference_time="2025-01-01T09:00:00Z",
            ),
        ],
        ignore_index=True,
    )

    hourly = build_hourly_fundamentals(native)

    assert list(hourly.columns) == HOURLY_COLUMNS
    assert len(hourly) == 2
    by_revision = hourly.set_index("source_revision")
    assert by_revision.loc["1", "value"] == pytest.approx(25.0)
    assert by_revision.loc["2", "value"] == pytest.approx(250.0)
    assert by_revision["interval_coverage_pass"].all()
    assert by_revision["interval_coverage_ratio"].tolist() == pytest.approx(
        [1.0, 1.0]
    )
    assert set(hourly["feature_name"]) == {
        "fundamental_test_provider_dk1_total_load_forecast_mw"
    }


def test_build_hourly_fundamentals_never_combines_partial_vintages() -> None:
    revision_1 = _native_quarters(
        values=[10.0, 20.0],
        source_revision="1",
        vintage_id="load-revision-1",
        start_interval=0,
    )
    revision_2 = _native_quarters(
        values=[30.0, 40.0],
        source_revision="2",
        vintage_id="load-revision-2",
        start_interval=2,
        available_at="2025-01-01T11:00:00Z",
        forecast_reference_time="2025-01-01T09:00:00Z",
    )

    hourly = build_hourly_fundamentals(
        pd.concat([revision_1, revision_2], ignore_index=True)
    )

    assert len(hourly) == 2
    assert hourly["value"].isna().all()
    assert not hourly["interval_coverage_pass"].any()
    assert hourly["interval_coverage_ratio"].tolist() == pytest.approx([0.5, 0.5])


def test_build_hourly_fundamentals_uses_maximum_for_outage_flag() -> None:
    native = _native_quarters(
        values=[0.0, 1.0, 0.0, 0.0],
        parameter_id="planned_transmission_outage_flag",
        entity_type="directed_border",
        area=None,
        from_area="DK1",
        to_area="DE_LU",
        unit="boolean",
    )

    hourly = build_hourly_fundamentals(native)

    assert hourly.loc[0, "value"] == pytest.approx(1.0)
    assert hourly.loc[0, "hourly_aggregation"] == "maximum"
    assert hourly.loc[0, "feature_name"] == (
        "fundamental_test_provider_dk1_to_de_lu_"
        "planned_transmission_outage_flag_boolean"
    )


def test_directed_border_requires_null_area_and_distinct_endpoints() -> None:
    with_area = _native_quarters(
        values=[1.0, 1.0, 1.0, 1.0],
        parameter_id="available_transfer_capacity",
        entity_type="directed_border",
        area="DK1",
        from_area="DK1",
        to_area="DE_LU",
    )
    same_endpoint = with_area.copy()
    same_endpoint["area"] = None
    same_endpoint["to_area"] = "DK1"

    with pytest.raises(ValueError, match="require null area"):
        normalize_native_fundamentals(with_area)
    with pytest.raises(ValueError, match="must differ"):
        normalize_native_fundamentals(same_endpoint)


def test_feature_names_separate_providers_and_bidding_zones() -> None:
    native = pd.concat(
        [
            _native_quarters(
                values=[1.0, 1.0, 1.0, 1.0],
                source_provider="entsoe",
                area="DK1",
            ),
            _native_quarters(
                values=[2.0, 2.0, 2.0, 2.0],
                source_provider="energinet",
                area="DK1",
                source_revision="2",
                vintage_id="energinet-dk1",
            ),
            _native_quarters(
                values=[3.0, 3.0, 3.0, 3.0],
                source_provider="entsoe",
                area="DE_LU",
                source_revision="3",
                vintage_id="entsoe-de-lu",
            ),
        ],
        ignore_index=True,
    )

    hourly = build_hourly_fundamentals(native)

    assert set(hourly["feature_name"]) == {
        "fundamental_entsoe_dk1_total_load_forecast_mw",
        "fundamental_energinet_dk1_total_load_forecast_mw",
        "fundamental_entsoe_de_lu_total_load_forecast_mw",
    }


def test_feature_name_normalization_rejects_provider_collision() -> None:
    native = pd.concat(
        [
            _native_quarters(
                values=[1.0, 1.0, 1.0, 1.0],
                source_provider="test-provider",
            ),
            _native_quarters(
                values=[2.0, 2.0, 2.0, 2.0],
                source_provider="test provider",
                source_revision="2",
                vintage_id="provider-collision",
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="feature names collide"):
        build_hourly_fundamentals(native)


def _native_quarters(
    *,
    values: list[float],
    source_revision: str = "1",
    vintage_id: str = "load-revision-1",
    available_at: str = "2025-01-01T10:00:00Z",
    forecast_reference_time: str = "2025-01-01T08:00:00Z",
    start_interval: int = 0,
    parameter_id: str = "total_load_forecast",
    entity_type: str = "bidding_zone",
    area: str | None = "DK1",
    from_area: str | None = None,
    to_area: str | None = None,
    unit: str = "MW",
    source_provider: str = "test_provider",
) -> pd.DataFrame:
    hour = pd.Timestamp("2025-01-02T00:00:00Z")
    rows = []
    for offset, value in enumerate(values, start=start_interval):
        start = hour + pd.Timedelta(minutes=15 * offset)
        rows.append(
            {
                "source_provider": source_provider,
                "source_dataset": "day_ahead_fundamentals",
                "source_document_id": f"document-{source_revision}",
                "source_series_id": "series-load-dk1",
                "source_revision": source_revision,
                "fundamentals_vintage_id": vintage_id,
                "parameter_id": parameter_id,
                "entity_type": entity_type,
                "area": area,
                "from_area": from_area,
                "to_area": to_area,
                "valid_time_start_utc": start,
                "valid_time_end_utc": start + pd.Timedelta(minutes=15),
                "native_resolution_minutes": 15,
                "value": value,
                "unit": unit,
                "forecast_reference_time_utc": forecast_reference_time,
                "forecast_reference_time_type": "observed_source_reference",
                "published_at_utc": available_at,
                "available_at_utc": available_at,
                "availability_time_type": "observed_source_publication",
                "raw_batch_id": f"batch-{source_revision}",
                "retrieved_at_utc": "2025-01-01T12:00:00Z",
            }
        )
    return pd.DataFrame(rows, columns=NATIVE_COLUMNS)
