from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from dkenergy_forecast.backtesting.horizons import (
    make_daily_origins,
    make_danish_delivery_day_horizon,
    make_next_utc_hours_horizon,
)
from dkenergy_forecast.backtesting.rolling_origin import rolling_origin_backtest
from dkenergy_forecast.evaluation.point_metrics import bias, mae, rmse
from dkenergy_forecast.evaluation.probabilistic_metrics import (
    average_interval_width,
    interval_coverage,
    pinball_loss,
)
from dkenergy_forecast.io import load_price_panel
from dkenergy_forecast.models.baselines import (
    LagNaive,
    SeasonalRollingMedian,
    WeekdayWeekendWeightedMedian,
    WeightedSeasonalMedian,
)
from dkenergy_forecast.types import add_copenhagen_calendar
from dkenergy_forecast.types import add_price_availability


def test_load_price_panel_normalizes_utc_and_validates_qa(tmp_path) -> None:
    panel = _panel(periods=2)
    panel["ds_utc"] = panel["ds_utc"].dt.tz_localize(None)
    panel_path = tmp_path / "panel.parquet"
    qa_path = tmp_path / "panel.qa.json"
    panel.to_parquet(panel_path, index=False)
    qa_path.write_text(json.dumps({"artifact_status": "final_historical"}), encoding="utf-8")

    loaded = load_price_panel(panel_path, qa_path)

    assert str(loaded["ds_utc"].dtype) == "datetime64[ns, UTC]"
    assert loaded["unique_id"].tolist() == ["day_ahead_price_DK1", "day_ahead_price_DK1"]


def test_load_price_panel_rejects_missing_columns_duplicates_and_incomplete_qa(tmp_path) -> None:
    panel = _panel(periods=2)
    missing_path = tmp_path / "missing.parquet"
    duplicate_path = tmp_path / "duplicate.parquet"
    qa_path = tmp_path / "panel.qa.json"

    panel.drop(columns=["area"]).to_parquet(missing_path, index=False)
    pd.concat([panel, panel.iloc[[0]]], ignore_index=True).to_parquet(duplicate_path, index=False)
    qa_path.write_text(json.dumps({"artifact_status": "incomplete_live_refresh"}), encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_price_panel(missing_path)
    with pytest.raises(ValueError, match="duplicate"):
        load_price_panel(duplicate_path)
    with pytest.raises(ValueError, match="not final_historical"):
        load_price_panel(_write_panel(tmp_path, panel), qa_path)


def test_lag_naive_uses_utc_lag_for_24_and_168_hours() -> None:
    panel = _panel(start="2024-01-01T00:00:00Z", periods=200)
    origin = pd.Timestamp("2024-01-08T00:00:00Z")
    future = make_next_utc_hours_horizon(panel, origin, hours=1)
    history = panel[panel["ds_utc"] < origin]

    pred_24 = LagNaive(lag_hours=24).fit(history).predict(future)
    pred_168 = LagNaive(lag_hours=168).fit(history).predict(future)

    assert pred_24["y_pred"].iloc[0] == 145.0
    assert pred_168["y_pred"].iloc[0] == 1.0


def test_lag_naive_missing_lag_and_last_available_fallback_use_published_prices() -> None:
    panel = _panel(start="2024-01-01T00:00:00Z", periods=4)
    origin = pd.Timestamp("2024-01-01T02:00:00Z")
    future = make_next_utc_hours_horizon(panel, origin, hours=1)
    history_with_future = panel.copy()
    history_with_future.loc[history_with_future["ds_utc"] >= origin, "y"] = 999.0

    missing = LagNaive(lag_hours=24).predict(future, history=history_with_future)
    fallback = LagNaive(lag_hours=24, fallback="last_available").predict(
        future,
        history=history_with_future,
    )

    assert pd.isna(missing["y_pred"].iloc[0])
    assert fallback["y_pred"].iloc[0] == 999.0


def test_price_availability_uses_previous_local_day_noon_across_dst() -> None:
    frame = pd.DataFrame(
        {
            "ds_utc": pd.to_datetime(
                [
                    "2024-01-10T00:00:00Z",
                    "2024-03-31T00:00:00Z",
                    "2024-10-27T00:00:00Z",
                ],
                utc=True,
            )
        }
    )

    output = add_price_availability(frame)

    assert output["price_available_at_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S%z").tolist() == [
        "2024-01-09T11:00:00+0000",
        "2024-03-30T11:00:00+0000",
        "2024-10-26T10:00:00+0000",
    ]


def test_seasonal_rolling_median_respects_origin_window_keys_and_min_periods() -> None:
    panel = _panel(start="2024-01-01T00:00:00Z", periods=240)
    origin = pd.Timestamp("2024-01-10T00:00:00Z")
    future = make_next_utc_hours_horizon(panel, origin, hours=1)
    future_hour = future["local_hour"].iloc[0]
    history = panel[panel["ds_utc"] < origin].copy()
    history.loc[:, "y"] = 1000.0

    values = {
        pd.Timestamp("2024-01-07T01:00:00Z"): 10.0,
        pd.Timestamp("2024-01-08T01:00:00Z"): 20.0,
        pd.Timestamp("2024-01-09T01:00:00Z"): 30.0,
        pd.Timestamp("2024-01-06T01:00:00Z"): -999.0,
        pd.Timestamp("2024-01-09T02:00:00Z"): -999.0,
    }
    for timestamp, value in values.items():
        history.loc[history["ds_utc"] == timestamp, "y"] = value
    assert history.loc[history["ds_utc"] == pd.Timestamp("2024-01-07T01:00:00Z"), "local_hour"].iloc[0] == future_hour

    model = SeasonalRollingMedian(
        lookback_days=3,
        seasonal_keys=("local_hour",),
        min_periods=3,
    )
    prediction = model.predict(future, history=history)
    insufficient = SeasonalRollingMedian(
        lookback_days=3,
        seasonal_keys=("local_hour",),
        min_periods=4,
    ).predict(future, history=history)

    assert prediction["y_pred"].iloc[0] == 20.0
    assert pd.isna(insufficient["y_pred"].iloc[0])


def test_weighted_seasonal_median_supports_equal_linear_floor_and_exponential_weights() -> None:
    origin = pd.Timestamp("2024-01-10T00:00:00Z")
    future = pd.DataFrame(
        {
            "unique_id": ["x"],
            "ds_utc": [pd.Timestamp("2024-01-10T01:00:00Z")],
            "forecast_origin_utc": [origin],
            "horizon": [1],
            "local_hour": [1],
        }
    )
    history = pd.DataFrame(
        {
            "unique_id": ["x", "x", "x", "x"],
            "ds_utc": [
                pd.Timestamp("2024-01-07T00:00:00Z"),
                pd.Timestamp("2024-01-08T00:00:00Z"),
                pd.Timestamp("2024-01-09T00:00:00Z"),
                pd.Timestamp("2024-01-09T01:00:00Z"),
            ],
            "local_hour": [1, 1, 1, 2],
            "y": [100.0, 50.0, 0.0, -999.0],
        }
    )

    common = {
        "lookback_days": 4,
        "seasonal_keys": ("local_hour",),
        "min_periods": 3,
    }
    equal = WeightedSeasonalMedian(weight_family="equal", **common).predict(
        future,
        history=history,
    )
    linear = WeightedSeasonalMedian(weight_family="linear", **common).predict(
        future,
        history=history,
    )
    linear_floor = WeightedSeasonalMedian(
        weight_family="linear_floor",
        floor=0.2,
        **common,
    ).predict(future, history=history)
    exponential = WeightedSeasonalMedian(
        weight_family="exponential",
        half_life_days=1,
        **common,
    ).predict(future, history=history)
    exponential_floor = WeightedSeasonalMedian(
        weight_family="exponential",
        half_life_days=1,
        floor=1.0,
        **common,
    ).predict(future, history=history)

    assert equal["y_pred"].iloc[0] == 50.0
    assert linear["y_pred"].iloc[0] == 0.0
    assert linear_floor["y_pred"].iloc[0] == 50.0
    assert exponential["y_pred"].iloc[0] == 0.0
    assert exponential_floor["y_pred"].iloc[0] == 50.0


def test_weekday_weekend_weighted_median_uses_separate_parameter_sets() -> None:
    origin = pd.Timestamp("2024-01-15T00:00:00Z")
    future = pd.DataFrame(
        {
            "unique_id": ["x", "x"],
            "ds_utc": [
                pd.Timestamp("2024-01-15T01:00:00Z"),
                pd.Timestamp("2024-01-20T01:00:00Z"),
            ],
            "forecast_origin_utc": [
                origin,
                pd.Timestamp("2024-01-20T00:00:00Z"),
            ],
            "horizon": [1, 2],
            "local_hour": [1, 1],
            "is_weekend": [False, True],
        }
    )
    history = pd.DataFrame(
        {
            "unique_id": ["x"] * 6,
            "ds_utc": [
                pd.Timestamp("2024-01-12T01:00:00Z"),
                pd.Timestamp("2024-01-13T01:00:00Z"),
                pd.Timestamp("2024-01-14T01:00:00Z"),
                pd.Timestamp("2024-01-17T01:00:00Z"),
                pd.Timestamp("2024-01-18T01:00:00Z"),
                pd.Timestamp("2024-01-19T01:00:00Z"),
            ],
            "local_hour": [1, 1, 1, 1, 1, 1],
            "is_weekend": [False, False, False, True, True, True],
            "y": [100.0, 50.0, 0.0, 1000.0, 500.0, 0.0],
        }
    )

    prediction = WeekdayWeekendWeightedMedian(
        weekday_lookback_days=4,
        weekday_half_life_days=1,
        weekday_floor=None,
        weekend_lookback_days=4,
        weekend_half_life_days=10,
        weekend_floor=1.0,
        min_periods=3,
    ).predict(future, history=history)

    assert prediction["y_pred"].tolist() == [0.0, 500.0]


def test_weighted_seasonal_median_linear_boundary_weight_does_not_count_for_min_periods() -> None:
    origin = pd.Timestamp("2024-01-10T00:00:00Z")
    future = pd.DataFrame(
        {
            "unique_id": ["x"],
            "ds_utc": [pd.Timestamp("2024-01-10T01:00:00Z")],
            "forecast_origin_utc": [origin],
            "horizon": [1],
            "local_hour": [1],
        }
    )
    history = pd.DataFrame(
        {
            "unique_id": ["x", "x", "x"],
            "ds_utc": [
                pd.Timestamp("2024-01-07T00:00:00Z"),
                pd.Timestamp("2024-01-08T00:00:00Z"),
                pd.Timestamp("2024-01-09T00:00:00Z"),
            ],
            "local_hour": [1, 1, 1],
            "y": [100.0, 50.0, 0.0],
        }
    )

    prediction = WeightedSeasonalMedian(
        lookback_days=3,
        seasonal_keys=("local_hour",),
        min_periods=3,
        weight_family="linear",
    ).predict(future, history=history)

    assert pd.isna(prediction["y_pred"].iloc[0])


def test_next_utc_horizon_is_target_free_and_daily_origins_are_utc_bounded() -> None:
    panel = _panel(periods=48)
    origins = make_daily_origins(panel, "2024-01-01", "2024-01-03", at_hour_utc=10)
    horizon = make_next_utc_hours_horizon(panel, origins["forecast_origin_utc"].iloc[0], hours=2)

    assert origins["forecast_origin_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S%z").tolist() == [
        "2024-01-01T10:00:00+0000",
        "2024-01-02T10:00:00+0000",
    ]
    assert horizon["ds_utc"].dt.strftime("%Y-%m-%dT%H:%M:%S%z").tolist() == [
        "2024-01-01T11:00:00+0000",
        "2024-01-01T12:00:00+0000",
    ]
    assert "y" not in horizon.columns
    assert "price_dkk_per_mwh" not in horizon.columns


def test_danish_delivery_day_horizon_preserves_dst_day_lengths() -> None:
    panel = _panel(periods=1)
    spring = make_danish_delivery_day_horizon(
        panel,
        "2024-03-30T10:00:00Z",
        delivery_date_local="2024-03-31",
    )
    autumn = make_danish_delivery_day_horizon(
        panel,
        "2024-10-26T10:00:00Z",
        delivery_date_local="2024-10-27",
    )

    assert len(spring) == 23
    assert len(autumn) == 25
    assert spring["local_date"].nunique() == 1
    assert autumn["local_date"].nunique() == 1


def test_rolling_origin_backtest_is_leakage_safe_and_joins_actuals_after_prediction() -> None:
    panel = _panel(start="2024-01-01T00:00:00Z", periods=50)
    origin = pd.Timestamp("2024-01-02T00:00:00Z")
    panel.loc[panel["ds_utc"] == origin + pd.Timedelta(hours=1), "y"] = 999.0
    origins = pd.DataFrame({"forecast_origin_utc": [origin]})

    predictions = rolling_origin_backtest(
        model_factory=lambda: LagNaive(lag_hours=24),
        panel=panel,
        origins=origins,
        horizon_builder=lambda panel_arg, origin_arg: make_next_utc_hours_horizon(
            panel_arg,
            origin_arg,
            hours=1,
        ),
    )

    assert predictions["y_pred"].iloc[0] == 1.0
    assert predictions["y"].iloc[0] == 999.0
    assert predictions["area"].iloc[0] == "DK1"
    assert predictions["dataset_version"].iloc[0] == "v1"


def test_rolling_origin_backtest_rejects_insufficient_training_rows() -> None:
    panel = _panel(periods=5)
    origins = pd.DataFrame({"forecast_origin_utc": [pd.Timestamp("2024-01-01T01:00:00Z")]})

    with pytest.raises(ValueError, match="Not enough training rows"):
        rolling_origin_backtest(
            model_factory=lambda: LagNaive(lag_hours=24),
            panel=panel,
            origins=origins,
            horizon_builder=lambda panel_arg, origin_arg: make_next_utc_hours_horizon(
                panel_arg,
                origin_arg,
                hours=1,
            ),
            min_train_rows=6,
        )


def test_rolling_origin_backtest_rejects_prediction_key_mismatch() -> None:
    class DroppingModel:
        model_name = "dropping_model"
        model_version = "test"

        def fit(self, history: pd.DataFrame) -> "DroppingModel":
            return self

        def predict(self, future: pd.DataFrame, history: pd.DataFrame | None = None) -> pd.DataFrame:
            output = future.iloc[[0]][
                ["unique_id", "ds_utc", "forecast_origin_utc", "horizon"]
            ].copy()
            output["model_name"] = self.model_name
            output["model_version"] = self.model_version
            output["y_pred"] = 0.0
            return output

    panel = _panel(start="2024-01-01T00:00:00Z", periods=50)
    origin = pd.Timestamp("2024-01-02T00:00:00Z")

    with pytest.raises(ValueError, match="do not match the requested future frame"):
        rolling_origin_backtest(
            model_factory=DroppingModel,
            panel=panel,
            origins=pd.DataFrame({"forecast_origin_utc": [origin]}),
            horizon_builder=lambda panel_arg, origin_arg: make_next_utc_hours_horizon(
                panel_arg,
                origin_arg,
                hours=2,
            ),
        )


def test_point_and_probabilistic_metrics_are_hand_computed() -> None:
    predictions = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "y_pred": [1.0, 3.0, 1.0],
            "q50": [1.0, 1.0, 5.0],
            "q10": [0.0, 2.0, 4.0],
            "q90": [2.0, 3.0, 5.0],
        }
    )

    assert mae(predictions) == 1.0
    assert rmse(predictions) == pytest.approx(math.sqrt(5 / 3))
    assert bias(predictions) == pytest.approx(-1 / 3)
    assert pinball_loss(predictions, quantile=0.5) == 0.5
    assert interval_coverage(predictions) == pytest.approx(2 / 3)
    assert average_interval_width(predictions) == pytest.approx(4 / 3)


def _panel(
    *,
    start: str = "2024-01-01T00:00:00Z",
    periods: int,
    area: str = "DK1",
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "unique_id": [f"day_ahead_price_{area}"] * periods,
            "ds_utc": pd.date_range(start, periods=periods, freq="h"),
            "area": [area] * periods,
            "y": [float(value) for value in range(periods)],
            "dataset_version": ["v1"] * periods,
        }
    )
    frame = add_copenhagen_calendar(frame)
    frame["price_dkk_per_mwh"] = frame["y"]
    frame["price_eur_per_mwh"] = frame["y"] / 7.45
    return frame


def _write_panel(tmp_path, panel: pd.DataFrame):
    path = tmp_path / "panel.parquet"
    panel.to_parquet(path, index=False)
    return path
