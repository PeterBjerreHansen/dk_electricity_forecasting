from __future__ import annotations

import json

import pandas as pd

from dkenergy_forecast.tuning.catboost_validation import (
    CatBoostCandidateSpec,
    CatBoostValidationConfig,
    CatBoostValidationResult,
    drop_unusable_target_rows,
    make_retune_month_blocks,
    per_origin_delta_table,
    prepare_nested_frame,
    run_catboost_validation,
    scheduled_retrain_origins,
    target_values,
    training_rows_for_origin,
    write_catboost_validation_artifacts,
)


def test_prepare_nested_frame_adds_outer_month_labels() -> None:
    frame = pd.DataFrame(
        {
            "unique_id": ["x", "x"],
            "area": ["DK1", "DK1"],
            "forecast_origin_utc": [
                "2024-03-31T10:00:00Z",
                "2024-04-01T10:00:00Z",
            ],
            "ds_utc": [
                "2024-04-01T00:00:00Z",
                "2024-04-02T00:00:00Z",
            ],
            "y": [10.0, 20.0],
        }
    )

    prepared = prepare_nested_frame(frame)

    assert prepared["outer_month"].tolist() == ["2024-03", "2024-04"]
    assert str(prepared["forecast_origin_utc"].dtype) == "datetime64[ns, UTC]"


def test_catboost_artifact_writer_uses_compact_levels_and_jsonl_trials(tmp_path) -> None:
    origin = pd.Timestamp("2026-06-01T10:00:00Z")
    trials = pd.DataFrame(
        [
            {
                "number": 0,
                "state": "COMPLETE",
                "value": 12.5,
                "param_depth": 3,
                "param_learning_rate": 0.05,
                "user_train_rows": 1000,
            }
        ]
    )
    candidate_scores = pd.DataFrame(
        [
            {
                "family": "price",
                "policy_label": "smoke_policy",
                "validation_block": "2026-06",
                "retune_at_utc": origin,
                "candidate_label": "price_baseline_calendar__residual_baseline",
                "feature_set": "price_baseline_calendar",
                "target_mode": "residual_baseline",
                "search_profile": "conservative",
                "recency_label": "exp_hl180",
                "sample_weight_half_life_days": 180.0,
                "sample_weight_floor": None,
                "feature_count": 8,
                "status": "ok",
                "tuning_mae": 12.5,
                "trials": trials,
            }
        ]
    )
    predictions = pd.DataFrame(
        [
            {
                "forecast_origin_utc": origin,
                "ds_utc": pd.Timestamp("2026-06-02T00:00:00Z"),
                "area": "DK1",
                "y": 20.0,
                "y_pred": 21.0,
                "model_label": "catboost__candidate",
                "selected_by_tuning": True,
            }
        ]
    )
    scores = pd.DataFrame({"model_label": ["catboost__candidate"], "area": ["ALL"], "mae": [1.0]})
    result = CatBoostValidationResult(
        candidate_tuning_scores=candidate_scores,
        selected_validation_configs=candidate_scores.copy(),
        catboost_predictions=predictions,
        catboost_replay_metadata=pd.DataFrame({"forecast_origin_utc": [origin]}),
        feature_importance=pd.DataFrame({"feature": ["x"], "importance": [1.0]}),
        combined_model_scores=scores,
        outer_month_model_scores=scores.assign(outer_month="2026-06"),
        per_origin_model_scores=scores.assign(forecast_origin_utc=origin),
        per_origin_deltas=pd.DataFrame({"forecast_origin_utc": [origin], "catboost_minus_best_baseline_mae": [0.0]}),
    )

    summary_dir = tmp_path / "summary"
    write_catboost_validation_artifacts(summary_dir, result, artifact_level="summary")

    assert (summary_dir / "candidate_tuning_scores.parquet").exists()
    assert (summary_dir / "selected_configs.parquet").exists()
    assert (summary_dir / "model_scores.parquet").exists()
    assert not (summary_dir / "selected_predictions.parquet").exists()
    assert not (summary_dir / "tuning_trials.jsonl").exists()

    diagnostic_dir = tmp_path / "diagnostic"
    write_catboost_validation_artifacts(diagnostic_dir, result, artifact_level="diagnostic")

    assert (diagnostic_dir / "selected_predictions.parquet").exists()
    assert (diagnostic_dir / "tuning_trials.jsonl").exists()
    assert not (diagnostic_dir / "tuning").exists()
    assert not (diagnostic_dir / "all_predictions.parquet").exists()
    assert not (diagnostic_dir / "replay_metadata.parquet").exists()
    trial_record = json.loads((diagnostic_dir / "tuning_trials.jsonl").read_text(encoding="utf-8").strip())
    assert trial_record["policy_label"] == "smoke_policy"
    assert trial_record["params"] == {"depth": 3, "learning_rate": 0.05}
    assert trial_record["user_attrs"] == {"train_rows": 1000}

    audit_dir = tmp_path / "audit"
    write_catboost_validation_artifacts(audit_dir, result, artifact_level="audit")

    assert (audit_dir / "all_predictions.parquet").exists()
    assert (audit_dir / "replay_metadata.parquet").exists()


def test_catboost_validation_retune_blocks_default_to_six_month_steps() -> None:
    blocks = make_retune_month_blocks(
        (
            "2026-08",
            "2026-01",
            "2026-02",
            "2026-03",
            "2026-04",
            "2026-05",
            "2026-06",
            "2026-07",
        ),
        retune_every_months=6,
    )

    assert blocks == [
        ("2026-01", "2026-02", "2026-03", "2026-04", "2026-05", "2026-06"),
        ("2026-07", "2026-08"),
    ]


def test_scheduled_retrain_origins_use_time_cadence() -> None:
    origins = list(pd.date_range("2026-01-01T10:00:00Z", periods=10, freq="D"))

    retrain_origins = scheduled_retrain_origins(origins, retrain_every_days=4)

    assert retrain_origins == [
        pd.Timestamp("2026-01-01T10:00:00Z"),
        pd.Timestamp("2026-01-05T10:00:00Z"),
        pd.Timestamp("2026-01-09T10:00:00Z"),
    ]


def test_training_rows_for_origin_can_use_all_prior_history() -> None:
    frame = pd.DataFrame(
        {
            "forecast_origin_utc": pd.to_datetime(
                [
                    "2025-01-01T10:00:00Z",
                    "2025-06-01T10:00:00Z",
                    "2025-12-31T10:00:00Z",
                    "2026-01-02T10:00:00Z",
                ],
                utc=True,
            ),
            "ds_utc": pd.to_datetime(
                [
                    "2025-01-02T00:00:00Z",
                    "2025-06-02T00:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-03T00:00:00Z",
                ],
                utc=True,
            ),
            "y": [1.0, 2.0, 3.0, 4.0],
        }
    )

    all_history = training_rows_for_origin(
        frame,
        origin=pd.Timestamp("2026-01-01T10:00:00Z"),
        training_origin_days=None,
    )
    hard_window = training_rows_for_origin(
        frame,
        origin=pd.Timestamp("2026-01-01T10:00:00Z"),
        training_origin_days=30,
    )

    assert all_history["y"].tolist() == [1.0, 2.0, 3.0]
    assert hard_window["y"].tolist() == [3.0]


def test_catboost_validation_reuses_completed_blocks_as_future_history(monkeypatch) -> None:
    frame = _catboost_validation_test_frame()
    candidate = CatBoostCandidateSpec(
        feature_set="price_full_engineered",
        target_mode="direct",
    )
    tune_calls: list[dict[str, object]] = []

    def fake_tune_candidates_for_retune(
        frame_arg: pd.DataFrame,
        *,
        candidates: list[CatBoostCandidateSpec],
        config: CatBoostValidationConfig,
        retune_at: pd.Timestamp,
        validation_block: str,
        validation_months: tuple[str, ...],
        progress=None,
    ) -> list[dict[str, object]]:
        tune_calls.append(
            {
                "validation_block": validation_block,
                "retune_at": retune_at,
                "max_origin": frame_arg["forecast_origin_utc"].max(),
            }
        )
        return [
            {
                "retune_at_utc": retune_at,
                "validation_block": validation_block,
                "validation_months": validation_months,
                "candidate_label": candidates[0].label,
                "feature_set": candidates[0].feature_set,
                "target_mode": candidates[0].target_mode,
                "search_profile": candidates[0].search_profile,
                "recency_label": candidates[0].recency.label,
                "sample_weight_half_life_days": None,
                "sample_weight_floor": None,
                "feature_count": 1,
                "status": "ok",
                "tuning_mae": 10.0,
                "best_trial_number": 0,
                "trial_count": 1,
                "tuning_validation_origin_count": 1,
                "feature_columns": ["feature"],
                "catboost_params": {"depth": 3},
                "trials": pd.DataFrame({"value": [10.0]}),
            }
        ]

    def fake_replay_candidate_on_schedule(
        frame_arg: pd.DataFrame,
        *,
        validation_origins: pd.DataFrame,
        candidate: dict[str, object],
        selected_candidate_label: str,
        config: CatBoostValidationConfig,
    ) -> tuple[pd.DataFrame, list[dict[str, object]], pd.DataFrame]:
        predict = frame_arg[
            frame_arg["forecast_origin_utc"].isin(validation_origins["forecast_origin_utc"])
        ].copy()
        predict["y_pred"] = predict["y"]
        predict["model_label"] = f"{config.model_prefix}__{candidate['candidate_label']}"
        predict["candidate_label"] = candidate["candidate_label"]
        predict["feature_set"] = candidate["feature_set"]
        predict["target_mode"] = candidate["target_mode"]
        predict["search_profile"] = candidate["search_profile"]
        predict["recency_label"] = candidate["recency_label"]
        predict["selected_by_tuning"] = candidate["candidate_label"] == selected_candidate_label
        predict["validation_block"] = candidate["validation_block"]
        meta_rows = [
            {
                "forecast_origin_utc": origin,
                "outer_month": pd.Timestamp(origin).strftime("%Y-%m"),
                "retune_at_utc": candidate["retune_at_utc"],
                "retrain_origin_utc": origin,
                "validation_block": candidate["validation_block"],
                "validation_months": candidate["validation_months"],
                "model_label": f"{config.model_prefix}__{candidate['candidate_label']}",
                "candidate_label": candidate["candidate_label"],
                "selected_by_tuning": candidate["candidate_label"] == selected_candidate_label,
                "feature_set": candidate["feature_set"],
                "target_mode": candidate["target_mode"],
                "search_profile": candidate["search_profile"],
                "recency_label": candidate["recency_label"],
                "tuning_validation_mae": candidate["tuning_mae"],
                "train_rows": 100,
                "train_fit_rows": 80,
                "train_eval_rows": 20,
                "train_fit_mae": 8.0,
                "train_eval_mae": 9.0,
                "predict_rows": 1,
                "best_iteration": 7,
            }
            for origin in validation_origins["forecast_origin_utc"]
        ]
        return predict, meta_rows, pd.DataFrame()

    monkeypatch.setattr(
        "dkenergy_forecast.tuning.catboost_validation.tune_candidates_for_retune",
        fake_tune_candidates_for_retune,
    )
    monkeypatch.setattr(
        "dkenergy_forecast.tuning.catboost_validation.replay_candidate_on_schedule",
        fake_replay_candidate_on_schedule,
    )

    result = run_catboost_validation(
        frame,
        candidates=[candidate],
        config=CatBoostValidationConfig(
            validation_months=(
                "2026-01",
                "2026-02",
                "2026-03",
                "2026-04",
                "2026-05",
                "2026-06",
                "2026-07",
            ),
            retune_every_months=6,
            retrain_every_days=7,
            replay_all_candidates=False,
        ),
        progress=None,
    )

    assert [call["validation_block"] for call in tune_calls] == [
        "2026-01..2026-06",
        "2026-07",
    ]
    assert [call["retune_at"] for call in tune_calls] == [
        pd.Timestamp("2026-01-01T00:00:00Z"),
        pd.Timestamp("2026-07-01T00:00:00Z"),
    ]
    assert tune_calls[0]["max_origin"] == pd.Timestamp("2025-12-01T10:00:00Z")
    assert tune_calls[1]["max_origin"] == pd.Timestamp("2026-06-01T10:00:00Z")
    assert result.selected_validation_configs["validation_block"].tolist() == [
        "2026-01..2026-06",
        "2026-07",
    ]
    assert sorted(result.catboost_predictions["outer_month"].unique().tolist()) == [
        "2026-01",
        "2026-02",
        "2026-03",
        "2026-04",
        "2026-05",
        "2026-06",
        "2026-07",
    ]


def test_nested_validation_per_origin_delta_table_tracks_train_eval_and_baseline_deltas() -> None:
    origin = pd.Timestamp("2024-05-01T10:00:00Z")
    per_origin_scores = pd.DataFrame(
        [
            {
                "forecast_origin_utc": origin,
                "outer_month": "2024-05",
                "model_label": "catboost__price_baseline_calendar__residual_baseline__conservative__unweighted",
                "rows": 48,
                "evaluated_rows": 48,
                "mae": 120.0,
                "rmse": 150.0,
                "bias": -10.0,
            },
            {
                "forecast_origin_utc": origin,
                "outer_month": "2024-05",
                "model_label": "same_hour_last_week",
                "rows": 48,
                "evaluated_rows": 48,
                "mae": 130.0,
                "rmse": 160.0,
                "bias": 15.0,
            },
            {
                "forecast_origin_utc": origin,
                "outer_month": "2024-05",
                "model_label": "rolling_median_hour_weekend_56d",
                "rows": 48,
                "evaluated_rows": 48,
                "mae": 110.0,
                "rmse": 140.0,
                "bias": 5.0,
            },
        ]
    )
    replay_metadata = pd.DataFrame(
        [
            {
                "forecast_origin_utc": origin,
                "model_label": "catboost__price_baseline_calendar__residual_baseline__conservative__unweighted",
                "candidate_label": "price_baseline_calendar__residual_baseline__conservative__unweighted",
                "selected_by_tuning": True,
                "feature_set": "price_baseline_calendar",
                "target_mode": "residual_baseline",
                "search_profile": "conservative",
                "recency_label": "unweighted",
                "tuning_validation_mae": 100.0,
                "train_rows": 1000,
                "train_fit_rows": 800,
                "train_eval_rows": 200,
                "train_fit_mae": 60.0,
                "train_eval_mae": 90.0,
                "predict_rows": 48,
                "best_iteration": 37,
            }
        ]
    )

    result = per_origin_delta_table(
        per_origin_scores,
        replay_metadata=replay_metadata,
        catboost_model_prefix="catboost",
    )

    assert result.loc[0, "best_baseline_label"] == "rolling_median_hour_weekend_56d"
    assert result.loc[0, "catboost_minus_best_baseline_mae"] == 10.0
    assert result.loc[0, "validation_minus_tuning_mae"] == 20.0
    assert result.loc[0, "validation_minus_train_fit_mae"] == 60.0
    assert result.loc[0, "validation_minus_train_eval_mae"] == 30.0
    assert result.loc[0, "train_rows"] == 1000


def test_nested_validation_residual_helpers_use_configured_baseline_column() -> None:
    frame = pd.DataFrame({"y": [10.0, 20.0], "custom_baseline": [7.0, 15.0]})

    target = target_values(
        frame,
        target_mode="residual_baseline",
        residual_baseline_column="custom_baseline",
    )
    kept = drop_unusable_target_rows(
        frame,
        target_mode="residual_baseline",
        residual_baseline_column="custom_baseline",
    )

    assert target.tolist() == [3.0, 5.0]
    assert len(kept) == 2


def _catboost_validation_test_frame() -> pd.DataFrame:
    origins = pd.date_range(
        "2025-12-01T10:00:00Z",
        "2026-07-01T10:00:00Z",
        freq="MS",
    )
    return pd.DataFrame(
        {
            "unique_id": ["DK1"] * len(origins),
            "area": ["DK1"] * len(origins),
            "forecast_origin_utc": origins,
            "ds_utc": origins + pd.Timedelta(days=1),
            "ds_local": origins + pd.Timedelta(days=1),
            "local_date": (origins + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
            "horizon": [1] * len(origins),
            "y": list(range(len(origins))),
            "feature": list(range(len(origins))),
        }
    )
