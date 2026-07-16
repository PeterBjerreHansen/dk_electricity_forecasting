from __future__ import annotations

import pandas as pd
import pytest

from dkenergy_data.build.fundamentals_v1 import (
    NATIVE_COLUMNS,
    build_hourly_fundamentals,
)
from dkenergy_forecast.features.fundamentals_features import (
    add_fundamentals_derived_features,
    fundamentals_value_columns,
    join_fundamentals_features,
)

LOAD_FEATURE = "fundamental_test_provider_dk1_total_load_forecast_mw"


def test_join_selects_newest_revision_eligible_at_information_cutoff() -> None:
    hourly = pd.concat(
        [
            _hourly_revision(
                value=100.0,
                source_revision="1",
                vintage_id="load-revision-1",
                available_at="2025-01-01T10:00:00Z",
                forecast_reference_time="2025-01-01T08:00:00Z",
            ),
            _hourly_revision(
                value=200.0,
                source_revision="2",
                vintage_id="load-revision-2",
                available_at="2025-01-01T11:00:00Z",
                forecast_reference_time="2025-01-01T09:00:00Z",
            ),
        ],
        ignore_index=True,
    )
    frame = pd.DataFrame(
        {
            "area": ["DK1", "DK1", "DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")] * 3,
            "information_cutoff_utc": [
                pd.Timestamp("2025-01-01T09:00:00Z"),
                pd.Timestamp("2025-01-01T10:00:00Z"),
                pd.Timestamp("2025-01-01T11:00:00Z"),
            ],
            "forecast_origin_utc": [
                pd.Timestamp("2025-01-01T12:00:00Z"),
            ]
            * 3,
        }
    )

    joined = join_fundamentals_features(frame, hourly)

    feature = LOAD_FEATURE
    assert pd.isna(joined.loc[0, feature])
    assert joined.loc[1, feature] == pytest.approx(100.0)
    assert joined.loc[2, feature] == pytest.approx(200.0)
    assert pd.isna(joined.loc[0, f"{feature}_source_revision"])
    assert joined.loc[1, f"{feature}_source_revision"] == "1"
    assert joined.loc[2, f"{feature}_source_revision"] == "2"
    assert joined.loc[2, f"{feature}_vintage_id"] == "load-revision-2"
    assert joined.loc[1, f"{feature}_available_at_utc"] == pd.Timestamp(
        "2025-01-01T10:00:00Z"
    )


def test_join_uses_temporal_order_not_opaque_revision_order() -> None:
    hourly = pd.concat(
        [
            _hourly_revision(
                value=100.0,
                source_revision="zzz-older",
                vintage_id="load-older",
                available_at="2025-01-01T10:00:00Z",
                forecast_reference_time="2025-01-01T08:00:00Z",
            ),
            _hourly_revision(
                value=200.0,
                source_revision="aaa-newer",
                vintage_id="load-newer",
                available_at="2025-01-01T11:00:00Z",
                forecast_reference_time="2025-01-01T09:00:00Z",
            ),
        ],
        ignore_index=True,
    )
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T12:00:00Z")],
        }
    )

    joined = join_fundamentals_features(frame, hourly)

    assert joined.loc[0, LOAD_FEATURE] == pytest.approx(200.0)
    assert joined.loc[0, f"{LOAD_FEATURE}_source_revision"] == "aaa-newer"


def test_join_fails_when_distinct_revision_identities_tie_at_newest_time() -> None:
    hourly = pd.concat(
        [
            _hourly_revision(
                value=100.0,
                source_revision="2",
                vintage_id="load-revision-2",
                available_at="2025-01-01T10:00:00Z",
                forecast_reference_time="2025-01-01T08:00:00Z",
            ),
            _hourly_revision(
                value=200.0,
                source_revision="10",
                vintage_id="load-revision-10",
                available_at="2025-01-01T10:00:00Z",
                forecast_reference_time="2025-01-01T08:00:00Z",
            ),
        ],
        ignore_index=True,
    )
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T12:00:00Z")],
        }
    )

    with pytest.raises(
        ValueError,
        match="Multiple distinct source revision/vintage identities",
    ):
        join_fundamentals_features(frame, hourly)


@pytest.mark.parametrize(
    ("column", "replacement"),
    [
        ("value", 200.0),
        ("source_series_id", "different-source-series"),
    ],
)
def test_join_fails_when_same_revision_identity_has_conflicting_semantics(
    column: str,
    replacement: object,
) -> None:
    original = _hourly_revision(
        value=100.0,
        source_revision="same-revision",
        vintage_id="same-vintage",
        available_at="2025-01-01T10:00:00Z",
        forecast_reference_time="2025-01-01T08:00:00Z",
    )
    conflicting = original.copy()
    conflicting[column] = replacement
    conflicting["raw_batch_id"] = "different-retrieval-batch"
    hourly = pd.concat([original, conflicting], ignore_index=True)
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T12:00:00Z")],
        }
    )

    with pytest.raises(
        ValueError,
        match="disagree on value or semantic source lineage",
    ):
        join_fundamentals_features(frame, hourly)


def test_join_collapses_equivalent_repeat_retrievals_deterministically() -> None:
    original = _hourly_revision(
        value=100.0,
        source_revision="same-revision",
        vintage_id="same-vintage",
        available_at="2025-01-01T10:00:00Z",
        forecast_reference_time="2025-01-01T08:00:00Z",
    )
    repeated = original.copy()
    repeated["raw_batch_id"] = "later-retrieval-batch"
    repeated["retrieved_at_utc"] = pd.Timestamp("2025-01-01T13:00:00Z")
    hourly = pd.concat([original, repeated], ignore_index=True)
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T14:00:00Z")],
        }
    )

    joined = join_fundamentals_features(frame, hourly)

    assert joined.loc[0, LOAD_FEATURE] == pytest.approx(100.0)
    assert (
        joined.loc[0, f"{LOAD_FEATURE}_raw_batch_id"]
        == "later-retrieval-batch"
    )


def test_join_falls_back_to_origin_and_uses_inclusive_availability() -> None:
    hourly = _hourly_revision(
        value=200.0,
        source_revision="2",
        vintage_id="load-revision-2",
        available_at="2025-01-01T11:00:00Z",
        forecast_reference_time="2025-01-01T09:00:00Z",
    )
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T11:00:00Z")],
        }
    )

    joined = join_fundamentals_features(frame, hourly)

    assert joined.loc[0, LOAD_FEATURE] == pytest.approx(200.0)


def test_join_rejects_information_cutoff_after_origin() -> None:
    hourly = _hourly_revision(
        value=200.0,
        source_revision="2",
        vintage_id="load-revision-2",
        available_at="2025-01-01T11:00:00Z",
        forecast_reference_time="2025-01-01T09:00:00Z",
    )
    origin = pd.Timestamp("2025-01-01T10:00:00Z")
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "forecast_origin_utc": [origin],
            "information_cutoff_utc": [origin + pd.Timedelta(minutes=1)],
        }
    )

    with pytest.raises(
        ValueError,
        match="information_cutoff_utc must not be after forecast_origin_utc",
    ):
        join_fundamentals_features(frame, hourly)


def test_join_excludes_incomplete_hourly_vintage_but_keeps_null_column() -> None:
    native = _native_intervals(
        value=100.0,
        source_revision="1",
        vintage_id="load-revision-1",
        available_at="2025-01-01T10:00:00Z",
        forecast_reference_time="2025-01-01T08:00:00Z",
        interval_count=3,
    )
    hourly = build_hourly_fundamentals(native)
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T12:00:00Z")],
        }
    )

    joined = join_fundamentals_features(frame, hourly)

    assert LOAD_FEATURE in joined
    assert pd.isna(joined.loc[0, LOAD_FEATURE])


def test_join_rejects_non_boolean_hourly_coverage_flag() -> None:
    hourly = _hourly_revision(
        value=100.0,
        source_revision="1",
        vintage_id="load-revision-1",
        available_at="2025-01-01T10:00:00Z",
        forecast_reference_time="2025-01-01T08:00:00Z",
    )
    hourly["interval_coverage_pass"] = "False"
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T12:00:00Z")],
        }
    )

    with pytest.raises(ValueError, match="must contain booleans"):
        join_fundamentals_features(frame, hourly)


def test_join_revalidates_observed_publication_timestamps() -> None:
    hourly = _hourly_revision(
        value=100.0,
        source_revision="1",
        vintage_id="load-revision-1",
        available_at="2025-01-01T10:00:00Z",
        forecast_reference_time="2025-01-01T08:00:00Z",
    )
    hourly["published_at_utc"] = pd.Timestamp("2025-01-01T11:00:00Z")
    frame = pd.DataFrame(
        {
            "area": ["DK1"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T12:00:00Z")],
        }
    )

    with pytest.raises(
        ValueError,
        match="observed_source_publication requires",
    ):
        join_fundamentals_features(frame, hourly)


def test_join_exposes_all_source_geographies_to_each_target_row() -> None:
    hourly = pd.concat(
        [
            _hourly_revision(
                value=100.0,
                source_revision="1",
                vintage_id="load-dk1",
                available_at="2025-01-01T10:00:00Z",
                forecast_reference_time="2025-01-01T08:00:00Z",
                area="DK1",
            ),
            _hourly_revision(
                value=300.0,
                source_revision="1",
                vintage_id="load-de-lu",
                available_at="2025-01-01T10:00:00Z",
                forecast_reference_time="2025-01-01T08:00:00Z",
                area="DE_LU",
            ),
        ],
        ignore_index=True,
    )
    frame = pd.DataFrame(
        {
            "area": ["DK2"],
            "ds_utc": [pd.Timestamp("2025-01-02T00:00:00Z")],
            "forecast_origin_utc": [pd.Timestamp("2025-01-01T12:00:00Z")],
        }
    )

    joined = join_fundamentals_features(frame, hourly)

    assert joined.loc[
        0, "fundamental_test_provider_dk1_total_load_forecast_mw"
    ] == pytest.approx(100.0)
    assert joined.loc[
        0, "fundamental_test_provider_de_lu_total_load_forecast_mw"
    ] == pytest.approx(300.0)


def test_derived_features_add_residual_load_and_directional_capacity() -> None:
    load = "fundamental_entsoe_dk1_total_load_forecast_mw"
    wind = "fundamental_entsoe_dk1_wind_generation_forecast_mw"
    solar = "fundamental_entsoe_dk1_solar_generation_forecast_mw"
    inbound = "fundamental_entsoe_no2_to_dk1_available_transfer_capacity_mw"
    outbound = "fundamental_entsoe_dk1_to_de_lu_available_transfer_capacity_mw"
    frame = pd.DataFrame(
        {
            load: [100.0, 90.0],
            wind: [30.0, 20.0],
            solar: [10.0, 10.0],
            inbound: [50.0, 60.0],
            outbound: [25.0, 30.0],
        }
    )
    _add_identity(
        frame,
        load,
        parameter_id="total_load_forecast",
        entity_type="bidding_zone",
        area="DK1",
    )
    _add_identity(
        frame,
        wind,
        parameter_id="wind_generation_forecast",
        entity_type="bidding_zone",
        area="DK1",
    )
    _add_identity(
        frame,
        solar,
        parameter_id="solar_generation_forecast",
        entity_type="bidding_zone",
        area="DK1",
    )
    _add_identity(
        frame,
        inbound,
        parameter_id="available_transfer_capacity",
        entity_type="directed_border",
        from_area="NO2",
        to_area="DK1",
    )
    _add_identity(
        frame,
        outbound,
        parameter_id="available_transfer_capacity",
        entity_type="directed_border",
        from_area="DK1",
        to_area="DE_LU",
    )

    output = add_fundamentals_derived_features(
        frame,
        expected_directed_borders={
            "entsoe": [
                ("NO2", "DK1"),
                ("DK1", "DE_LU"),
            ]
        },
    )

    renewable = "fundamental_entsoe_dk1_renewable_generation_forecast_mw"
    residual = "fundamental_entsoe_dk1_residual_load_forecast_mw"
    import_total = (
        "fundamental_entsoe_dk1_total_available_import_capacity_mw"
    )
    export_total = (
        "fundamental_entsoe_dk1_total_available_export_capacity_mw"
    )
    assert output[renewable].tolist() == [
        40.0,
        30.0,
    ]
    assert output[residual].tolist() == [60.0, 60.0]
    assert output[import_total].tolist() == pytest.approx([50.0, 60.0])
    assert output[export_total].tolist() == pytest.approx([25.0, 30.0])
    assert inbound in fundamentals_value_columns(output)
    assert f"{inbound}_from_area" not in fundamentals_value_columns(output)


def test_capacity_totals_fail_when_declared_border_series_is_absent() -> None:
    inbound = "fundamental_entsoe_no2_to_dk1_available_transfer_capacity_mw"
    frame = pd.DataFrame({inbound: [50.0]})
    _add_identity(
        frame,
        inbound,
        parameter_id="available_transfer_capacity",
        entity_type="directed_border",
        from_area="NO2",
        to_area="DK1",
    )

    with pytest.raises(ValueError, match="missing=.*se3_to_dk1"):
        add_fundamentals_derived_features(
            frame,
            expected_directed_borders={
                "entsoe": [
                    ("NO2", "DK1"),
                    ("SE3", "DK1"),
                ]
            },
        )


def test_capacity_total_is_null_when_any_declared_constituent_is_null() -> None:
    no2 = "fundamental_entsoe_no2_to_dk1_available_transfer_capacity_mw"
    se3 = "fundamental_entsoe_se3_to_dk1_available_transfer_capacity_mw"
    frame = pd.DataFrame(
        {
            no2: [50.0, 60.0],
            se3: [20.0, float("nan")],
        }
    )
    for feature_name, from_area in ((no2, "NO2"), (se3, "SE3")):
        _add_identity(
            frame,
            feature_name,
            parameter_id="available_transfer_capacity",
            entity_type="directed_border",
            from_area=from_area,
            to_area="DK1",
        )

    output = add_fundamentals_derived_features(
        frame,
        expected_directed_borders={
            "entsoe": [
                ("NO2", "DK1"),
                ("SE3", "DK1"),
            ]
        },
    )

    total = "fundamental_entsoe_dk1_total_available_import_capacity_mw"
    assert output[total].iloc[0] == pytest.approx(70.0)
    assert pd.isna(output[total].iloc[1])


def _hourly_revision(
    *,
    value: float,
    source_revision: str,
    vintage_id: str,
    available_at: str,
    forecast_reference_time: str,
    area: str = "DK1",
) -> pd.DataFrame:
    return build_hourly_fundamentals(
        _native_intervals(
            value=value,
            source_revision=source_revision,
            vintage_id=vintage_id,
            available_at=available_at,
            forecast_reference_time=forecast_reference_time,
            interval_count=4,
            area=area,
        )
    )


def _native_intervals(
    *,
    value: float,
    source_revision: str,
    vintage_id: str,
    available_at: str,
    forecast_reference_time: str,
    interval_count: int,
    area: str = "DK1",
) -> pd.DataFrame:
    hour = pd.Timestamp("2025-01-02T00:00:00Z")
    rows = []
    for interval in range(interval_count):
        start = hour + pd.Timedelta(minutes=15 * interval)
        rows.append(
            {
                "source_provider": "test_provider",
                "source_dataset": "day_ahead_fundamentals",
                "source_document_id": f"document-{source_revision}",
                "source_series_id": "series-load-dk1",
                "source_revision": source_revision,
                "fundamentals_vintage_id": vintage_id,
                "parameter_id": "total_load_forecast",
                "entity_type": "bidding_zone",
                "area": area,
                "from_area": None,
                "to_area": None,
                "valid_time_start_utc": start,
                "valid_time_end_utc": start + pd.Timedelta(minutes=15),
                "native_resolution_minutes": 15,
                "value": value,
                "unit": "MW",
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


def _add_identity(
    frame: pd.DataFrame,
    feature_name: str,
    *,
    parameter_id: str,
    entity_type: str,
    area: str | None = None,
    from_area: str | None = None,
    to_area: str | None = None,
) -> None:
    frame[f"{feature_name}_parameter_id"] = parameter_id
    frame[f"{feature_name}_entity_type"] = entity_type
    frame[f"{feature_name}_source_provider"] = "entsoe"
    frame[f"{feature_name}_unit"] = "MW"
    frame[f"{feature_name}_source_area"] = area
    frame[f"{feature_name}_from_area"] = from_area
    frame[f"{feature_name}_to_area"] = to_area
