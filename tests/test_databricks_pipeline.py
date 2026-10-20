from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from dkenergy_data.build.eds_prices_v1 import SOURCE_SPECS, build_price_panel_from_raw
from dkenergy_data.sources.energidataservice import ApiResponse
from dkenergy_databricks.inference import (
    assemble_forecasts,
    baseline_predictions,
    predict_champion,
    prepare_forecast_inputs,
)
from dkenergy_databricks.ingest import bronze_records_frame, default_fetch_range, land_eds_prices
from dkenergy_databricks.monitoring import (
    degradation_check,
    population_stability_index,
    price_drift_checks,
    score_forecasts,
)
from dkenergy_databricks.silver import (
    build_silver_price_panel,
    panel_from_delta_frame,
    panel_to_delta_frame,
)
from dkenergy_databricks.training import (
    CANDIDATE_MODEL_LABEL,
    DATABRICKS_CATBOOST_CONFIG,
    backtest_candidate,
    backtest_models,
    baseline_model_factories,
    build_training_pool,
    choose_backtest_origins,
    forecast_origin_for,
    promotion_decision,
    summarize_backtest,
    train_candidate,
    training_rows_for_origin,
)
from dkenergy_forecast.features.price_features import build_training_matrix
from dkenergy_forecast.models.catboost_experimental import (
    ExperimentalCatBoostDayAhead,
    predict_catboost_regressor,
)
from dkenergy_forecast.models.registry import WEIGHTED_MEDIAN_MODEL_LABEL
from dkenergy_forecast.types import filter_price_history_available_before


START = date(2025, 9, 1)
END = date(2026, 1, 20)
SMALL_CONFIG = replace(
    DATABRICKS_CATBOOST_CONFIG,
    training_origin_days=21,
    params={**DATABRICKS_CATBOOST_CONFIG.params, "iterations": 40},
)


class FakeEdsClient:
    """Serves deterministic prices in each EDS dataset's own format for any window."""

    def __init__(self, *, fail: bool = False, price_shift: float = 0.0) -> None:
        self.fail = fail
        self.price_shift = price_shift
        self.dataset_calls: list[tuple[str, str, str]] = []

    def fetch_metadata(self, dataset: str) -> ApiResponse:
        if self.fail:
            raise requests.ConnectionError("blocked by egress policy")
        return _response({"name": dataset}, {})

    def fetch_dataset(self, dataset: str, params: dict) -> ApiResponse:
        self.dataset_calls.append((dataset, params["start"], params["end"]))
        spec = SOURCE_SPECS[dataset]
        start = pd.Timestamp(params["start"]).tz_localize("Europe/Copenhagen")
        end = pd.Timestamp(params["end"]).tz_localize("Europe/Copenhagen")
        times = pd.date_range(start, end, freq=f"{spec.source_resolution_minutes}min", inclusive="left")
        records = []
        for area, offset in [("DK1", 0.0), ("DK2", 40.0)]:
            for ts in times:
                price = _synthetic_price(ts, offset) + self.price_shift
                records.append(
                    {
                        spec.time_column: ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%S"),
                        spec.local_time_column: ts.strftime("%Y-%m-%dT%H:%M:%S"),
                        spec.area_column: area,
                        spec.price_dkk_column: price,
                        spec.price_eur_column: price / 7.46,
                    }
                )
        return _response({"dataset": dataset, "records": records}, params)


def _synthetic_price(ts: pd.Timestamp, offset: float) -> float:
    hour_shape = 150 * np.sin((ts.hour - 6) / 24 * 2 * np.pi)
    weekend = -80 if ts.dayofweek >= 5 else 0
    noise = (hash((ts.value, offset)) % 1000) / 1000 * 60 - 30
    return round(500 + offset + hour_shape + weekend + noise, 2)


def _response(payload: dict, params: dict) -> ApiResponse:
    content = json.dumps(payload).encode()
    return ApiResponse(
        url="https://api.energidataservice.dk/test",
        params=dict(params),
        status_code=200,
        content=content,
        payload=payload,
        retrieved_at_utc=datetime.now(timezone.utc),
        response_sha256=hashlib.sha256(content).hexdigest(),
    )


@pytest.fixture(scope="module")
def raw_root(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("landing") / "energi_data_service"
    result = land_eds_prices(root, start=START, end=END, client=FakeEdsClient())
    assert result.status == "fetched"
    return root


@pytest.fixture(scope="module")
def panel(raw_root: Path) -> pd.DataFrame:
    silver, _ = build_silver_price_panel(bronze_records_frame(raw_root))
    return silver


def test_default_fetch_range_starts_on_a_month_boundary_and_includes_today() -> None:
    start, end = default_fetch_range(date(2026, 10, 20), history_days=450)
    assert start == date(2025, 7, 1)
    assert end == date(2026, 10, 21)


def test_landing_skips_windows_that_are_already_landed(raw_root: Path) -> None:
    client = FakeEdsClient()
    result = land_eds_prices(raw_root, start=START, end=END, client=client)
    assert result.status == "up_to_date"
    assert client.dataset_calls == []

    longer = land_eds_prices(raw_root, start=START, end=END + timedelta(days=1), client=client)
    assert longer.status == "fetched"
    assert client.dataset_calls == [("DayAheadPrices", "2026-01-01", "2026-01-21")]


def test_landing_reports_unreachable_api_instead_of_raising(tmp_path: Path) -> None:
    result = land_eds_prices(tmp_path, start=START, end=END, client=FakeEdsClient(fail=True))
    assert result.status == "api_unreachable"
    assert "egress" in (result.error or "")


def test_bronze_skips_known_batches(raw_root: Path) -> None:
    bronze = bronze_records_frame(raw_root)
    assert bronze["batch_id"].nunique() >= 4
    first_batch = bronze["batch_id"].iloc[0]
    again = bronze_records_frame(raw_root, known_batch_ids=[first_batch])
    assert first_batch not in set(again["batch_id"])
    assert len(again) < len(bronze)


def test_silver_from_bronze_matches_the_file_based_builder(raw_root: Path, panel: pd.DataFrame, tmp_path: Path) -> None:
    reference = build_price_panel_from_raw(
        raw_root=raw_root,
        normalized_dir=tmp_path / "normalized",
        model_ready_dir=tmp_path / "model_ready",
        allow_incomplete_recent=True,
    )
    expected = pd.read_parquet(reference.panel_path).sort_values(["unique_id", "ds_utc"]).reset_index(drop=True)
    assert len(panel) == len(expected)
    np.testing.assert_allclose(panel["y"].to_numpy(), expected["y"].to_numpy())
    assert set(panel["area"]) == {"DK1", "DK2"}


def test_silver_keeps_latest_revision_when_bronze_has_overlapping_batches(raw_root: Path, tmp_path: Path) -> None:
    revised_root = tmp_path / "revised"
    land_eds_prices(revised_root, start=START, end=END, client=FakeEdsClient())
    land_eds_prices(revised_root, start=END - timedelta(days=5), end=END, client=FakeEdsClient(price_shift=100.0))
    revised, _ = build_silver_price_panel(bronze_records_frame(revised_root))
    last_day = revised[revised["local_date"] == (END - timedelta(days=1)).isoformat()]
    first_day = revised[revised["local_date"] == START.isoformat()]
    original, _ = build_silver_price_panel(bronze_records_frame(raw_root))
    original_last = original[original["local_date"] == (END - timedelta(days=1)).isoformat()]
    original_first = original[original["local_date"] == START.isoformat()]
    np.testing.assert_allclose(last_day["y"].to_numpy(), original_last["y"].to_numpy() + 100.0)
    np.testing.assert_allclose(first_day["y"].to_numpy(), original_first["y"].to_numpy())


def test_panel_survives_a_delta_round_trip(panel: pd.DataFrame) -> None:
    stored = panel_to_delta_frame(panel)
    assert "ds_local" not in stored
    # Spark with a UTC session hands timestamps back as naive UTC values.
    as_read = stored.copy()
    for column in ["ds_utc", "price_available_at_utc"]:
        as_read[column] = as_read[column].dt.tz_localize(None)
    restored = panel_from_delta_frame(as_read)
    pd.testing.assert_series_equal(restored["ds_utc"], panel["ds_utc"])
    pd.testing.assert_series_equal(restored["ds_local"], panel["ds_local"])
    np.testing.assert_allclose(restored["y"], panel["y"])


def test_forecast_origin_is_ten_local_across_dst() -> None:
    assert forecast_origin_for(date(2026, 1, 15)) == pd.Timestamp("2026-01-15T09:00:00Z")
    assert forecast_origin_for(date(2026, 7, 15)) == pd.Timestamp("2026-07-15T08:00:00Z")


def test_backtest_and_promotion_gate(panel: pd.DataFrame) -> None:
    pytest.importorskip("catboost")
    origins = choose_backtest_origins(panel, days=5, max_origins=3)
    pool = build_training_pool(panel, origins["forecast_origin_utc"].min(), origins["forecast_origin_utc"].max(), SMALL_CONFIG)
    predictions = pd.concat(
        [
            backtest_candidate(panel, origins, pool, SMALL_CONFIG),
            backtest_models(panel, baseline_model_factories(), origins),
        ],
        ignore_index=True,
    )
    summary = summarize_backtest(predictions)
    assert set(summary["model_label"]) == {CANDIDATE_MODEL_LABEL, *baseline_model_factories()}
    assert (summary["origins"] == 3).all()

    decision = promotion_decision(summary)
    by_label = summary.set_index("model_label")["mae"]
    assert decision.promote == (by_label[CANDIDATE_MODEL_LABEL] < by_label[WEIGHTED_MEDIAN_MODEL_LABEL])


def test_pooled_training_rows_match_a_fresh_build_per_origin(panel: pd.DataFrame) -> None:
    origins = choose_backtest_origins(panel, days=5, max_origins=3)["forecast_origin_utc"]
    pool = build_training_pool(panel, origins.min(), origins.max(), SMALL_CONFIG)
    for origin in origins:
        expected = build_training_matrix(
            filter_price_history_available_before(panel, origin),
            origin,
            training_origin_days=SMALL_CONFIG.training_origin_days,
            forecast_local_time=SMALL_CONFIG.forecast_local_time,
            config=SMALL_CONFIG.price_feature_config,
        )
        actual = training_rows_for_origin(pool, panel, origin, SMALL_CONFIG)
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


def test_pooled_backtest_matches_the_research_model(panel: pd.DataFrame) -> None:
    pytest.importorskip("catboost")
    origins = choose_backtest_origins(panel, days=3, max_origins=2)
    pool = build_training_pool(panel, origins["forecast_origin_utc"].min(), origins["forecast_origin_utc"].max(), SMALL_CONFIG)
    pooled = backtest_candidate(panel, origins, pool, SMALL_CONFIG)
    research = backtest_models(panel, {"research": lambda: ExperimentalCatBoostDayAhead(SMALL_CONFIG)}, origins)
    np.testing.assert_allclose(pooled["y_pred"].to_numpy(), research["y_pred"].to_numpy())


def test_promotion_gate_requires_the_configured_improvement() -> None:
    summary = pd.DataFrame(
        {"model_label": [CANDIDATE_MODEL_LABEL, WEIGHTED_MEDIAN_MODEL_LABEL], "mae": [95.0, 100.0]}
    )
    assert promotion_decision(summary).promote
    assert not promotion_decision(summary, min_relative_improvement=0.10).promote


def test_logged_mlflow_model_reproduces_the_fitted_model(panel: pd.DataFrame, tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("catboost")
    mlflow = pytest.importorskip("mlflow")
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    from dkenergy_databricks.mlflow_model import log_day_ahead_model

    origin = forecast_origin_for(END - timedelta(days=1))
    fitted, training = train_candidate(panel, origin, SMALL_CONFIG)
    inputs = prepare_forecast_inputs(panel, origin)

    mlflow.set_tracking_uri((tmp_path / "mlruns").as_uri())
    with mlflow.start_run():
        info = log_day_ahead_model(fitted, input_example=training)
    loaded = mlflow.pyfunc.load_model(info.model_uri)

    expected = predict_catboost_regressor(fitted, inputs.features).to_numpy()
    actual = loaded.predict(inputs.features)["y_pred"].to_numpy()
    np.testing.assert_allclose(actual, expected)
    assert len(actual) == 48


def test_forecast_publishes_champion_or_falls_back(panel: pd.DataFrame) -> None:
    origin = forecast_origin_for(END - timedelta(days=1))
    inputs = prepare_forecast_inputs(panel, origin)
    baselines = baseline_predictions(inputs)

    assert "y" not in inputs.future
    assert inputs.history["price_available_at_utc"].max() < origin

    fallback = assemble_forecasts(inputs, None, baselines, run_id="r1")
    published = fallback[fallback["is_published"]]
    assert set(published["model_name"]) == {WEIGHTED_MEDIAN_MODEL_LABEL}
    assert published.groupby("area").size().to_dict() == {"DK1": 24, "DK2": 24}

    class Constant:
        def predict(self, frame: pd.DataFrame) -> pd.DataFrame:
            return pd.DataFrame({"y_pred": np.full(len(frame), 123.0)})

    champion = predict_champion(lambda: Constant(), inputs.features, model_name="catboost_price_v1", model_version="3")
    with_champion = assemble_forecasts(inputs, champion, baselines, run_id="r2")
    published = with_champion[with_champion["is_published"]]
    assert set(published["model_name"]) == {"catboost_price_v1"}
    assert (published["y_pred"] == 123.0).all()
    assert with_champion["area"].notna().all()


def test_broken_champion_is_reported_not_raised(panel: pd.DataFrame) -> None:
    inputs = prepare_forecast_inputs(panel, forecast_origin_for(END - timedelta(days=1)))

    def broken() -> object:
        raise RuntimeError("model registry unavailable")

    result = predict_champion(broken, inputs.features, model_name="m", model_version="1")
    assert result.y_pred is None
    assert "registry unavailable" in (result.error or "")
    assert assemble_forecasts(inputs, result, baseline_predictions(inputs), run_id="r")[
        lambda f: f["is_published"]
    ]["model_name"].eq(WEIGHTED_MEDIAN_MODEL_LABEL).all()


def test_scores_and_degradation_check() -> None:
    origins = pd.date_range("2026-01-01T09:00Z", periods=8, freq="D")
    rows = []
    for origin in origins:
        for hour in range(24):
            ds = origin + pd.Timedelta(hours=15 + hour)
            for model, error, published in [("catboost_price_v1", 30.0, True), (WEIGHTED_MEDIAN_MODEL_LABEL, 20.0, False)]:
                rows.append(
                    {
                        "forecast_origin_utc": origin,
                        "unique_id": "day_ahead_price_DK1",
                        "area": "DK1",
                        "ds_utc": ds,
                        "model_name": model,
                        "model_version": "1",
                        "is_published": published,
                        "y_pred": 100.0 + error,
                    }
                )
    forecasts = pd.DataFrame(rows)
    actuals = forecasts[["unique_id", "ds_utc"]].drop_duplicates().assign(y=100.0)

    scores = score_forecasts(forecasts, actuals)
    assert len(scores) == 16
    assert scores.loc[scores["model_name"] == "catboost_price_v1", "mae"].eq(30.0).all()

    check = degradation_check(scores, reference_model=WEIGHTED_MEDIAN_MODEL_LABEL, window_origins=7)
    assert check.status == "alert"
    assert check.value == pytest.approx(1.5)

    short = degradation_check(scores, reference_model=WEIGHTED_MEDIAN_MODEL_LABEL, window_origins=10)
    assert short.status == "insufficient_data"


def test_population_stability_index_flags_shifts() -> None:
    rng = np.random.default_rng(0)
    reference = pd.Series(rng.normal(500, 100, 5000))
    assert population_stability_index(reference, pd.Series(rng.normal(500, 100, 1000))) < 0.05
    assert population_stability_index(reference, pd.Series(rng.normal(800, 100, 1000))) > 0.25


def test_price_drift_checks_run_per_area(panel: pd.DataFrame) -> None:
    checks = price_drift_checks(panel, as_of=panel["ds_utc"].max())
    assert [check.subject for check in checks] == ["DK1", "DK2"]
    assert all(check.status == "ok" for check in checks)
