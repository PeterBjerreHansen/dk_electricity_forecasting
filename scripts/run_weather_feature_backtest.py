#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.evaluation.summary import (  # noqa: E402
    cheapest_k_table,
    model_score_table,
    probabilistic_metric_table,
)
from dkenergy_forecast.features.weather_features import weather_value_columns  # noqa: E402
from dkenergy_forecast.publishing import git_commit, json_safe  # noqa: E402


QUANTILES = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
METADATA_COLUMNS = {
    "unique_id",
    "ds_utc",
    "ds_local",
    "local_date",
    "forecast_origin_utc",
    "horizon",
    "y",
    "dataset_version",
    "price_dkk_per_mwh",
    "price_eur_per_mwh",
    "source_dataset",
    "source_resolution_minutes",
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.experiment_frame_path)
    frame["ds_utc"] = pd.to_datetime(frame["ds_utc"], utc=True)
    frame["forecast_origin_utc"] = pd.to_datetime(frame["forecast_origin_utc"], utc=True)

    try:
        CatBoostRegressor, Pool = load_catboost()
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    predictions = []
    feature_sets = args.feature_sets
    for feature_set in feature_sets:
        feature_columns = feature_columns_for_set(frame, feature_set)
        if not feature_columns:
            print(f"Skipping {feature_set}: no feature columns selected", flush=True)
            continue
        feature_set_predictions = run_feature_set(
            frame,
            feature_set=feature_set,
            feature_columns=feature_columns,
            catboost_regressor=CatBoostRegressor,
            pool_class=Pool,
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            random_seed=args.random_seed,
            min_train_rows=args.min_train_rows,
        )
        if feature_set_predictions.empty:
            print(
                f"Skipping {feature_set}: no origins reached the minimum training rows",
                flush=True,
            )
            continue
        predictions.append(feature_set_predictions)

    if not predictions:
        raise SystemExit("No weather feature backtest predictions were produced.")

    predictions_frame = pd.concat(predictions, ignore_index=True)
    metrics = model_score_table(predictions_frame)
    probabilistic_metrics = probabilistic_metric_table(predictions_frame)
    value_metrics = cheapest_k_table(predictions_frame, k=args.cheapest_k)
    manifest = make_manifest(args=args, predictions=predictions_frame, metrics=metrics)

    predictions_frame.to_parquet(output_dir / "predictions.parquet", index=False)
    metrics.to_parquet(output_dir / "model_scores.parquet", index=False)
    metrics.to_parquet(output_dir / "metrics.parquet", index=False)
    probabilistic_metrics.to_parquet(output_dir / "probabilistic_metrics.parquet", index=False)
    value_metrics.to_parquet(output_dir / "value_metrics.parquet", index=False)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote predictions: {output_dir / 'predictions.parquet'}")
    print(f"Wrote model scores: {output_dir / 'model_scores.parquet'}")
    print(f"Wrote metrics: {output_dir / 'metrics.parquet'}")
    print(f"Wrote probabilistic metrics: {output_dir / 'probabilistic_metrics.parquet'}")
    print(f"Wrote value metrics: {output_dir / 'value_metrics.parquet'}")
    print(f"Wrote manifest: {output_dir / 'run_manifest.json'}")
    print(metrics.sort_values(["area", "mae", "model_label"]).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CatBoost backtests over a price + weather experiment frame.")
    parser.add_argument(
        "--experiment-frame-path",
        default=str(ROOT / "data" / "features" / "weather_experiment_frame_v1.parquet"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "weather_catboost_v1"))
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        default=["price_only", "gfs_global", "icon_eu", "metno_nordic", "all_weather", "ensemble"],
    )
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--min-train-rows", type=int, default=1000)
    parser.add_argument("--cheapest-k", type=int, default=6)
    return parser.parse_args()


def load_catboost():
    try:
        from catboost import CatBoostRegressor, Pool
    except ImportError as exc:
        raise ImportError(
            "Weather feature backtests require CatBoost. Install it with "
            '`pip install -e ".[catboost]"` or `pip install catboost>=1.2`.'
        ) from exc
    return CatBoostRegressor, Pool


def run_feature_set(
    frame: pd.DataFrame,
    *,
    feature_set: str,
    feature_columns: list[str],
    catboost_regressor,
    pool_class,
    iterations: int,
    depth: int,
    learning_rate: float,
    random_seed: int,
    min_train_rows: int,
) -> pd.DataFrame:
    outputs = []
    origins = frame["forecast_origin_utc"].sort_values().drop_duplicates()
    cat_features = [column for column in ["area"] if column in feature_columns]

    for origin in origins:
        train = frame[
            (frame["forecast_origin_utc"] < origin)
            & (frame["ds_utc"] < origin)
            & frame["y"].notna()
        ].copy()
        predict = frame[frame["forecast_origin_utc"] == origin].copy()
        if len(train) < min_train_rows:
            continue

        train_pool = pool_class(train[feature_columns], label=train["y"], cat_features=cat_features)
        predict_pool = pool_class(predict[feature_columns], cat_features=cat_features)
        origin_predictions = predict[
            [
                column
                for column in [
                    "unique_id",
                    "area",
                    "ds_utc",
                    "ds_local",
                    "local_date",
                    "forecast_origin_utc",
                    "horizon",
                    "y",
                    "dataset_version",
                ]
                if column in predict.columns
            ]
        ].copy()

        for quantile_name, alpha in QUANTILES.items():
            model = catboost_regressor(
                loss_function=f"Quantile:alpha={alpha}",
                iterations=iterations,
                depth=depth,
                learning_rate=learning_rate,
                random_seed=random_seed,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(train_pool)
            origin_predictions[quantile_name] = model.predict(predict_pool)

        raw_crossing_rate = (
            (origin_predictions["q10"] > origin_predictions["q50"])
            | (origin_predictions["q50"] > origin_predictions["q90"])
        ).mean()
        sorted_quantiles = np.sort(origin_predictions[["q10", "q50", "q90"]].to_numpy(), axis=1)
        origin_predictions[["q10", "q50", "q90"]] = sorted_quantiles
        origin_predictions["raw_quantile_crossing_rate"] = float(raw_crossing_rate)
        origin_predictions["y_pred"] = origin_predictions["q50"]
        origin_predictions["model_label"] = f"catboost_{feature_set}"
        origin_predictions["feature_set"] = feature_set
        origin_predictions["feature_column_count"] = len(feature_columns)
        outputs.append(origin_predictions)

    if not outputs:
        return pd.DataFrame()
    predictions = pd.concat(outputs, ignore_index=True)
    predictions["error"] = predictions["y_pred"] - predictions["y"]
    predictions["abs_error"] = predictions["error"].abs()
    return predictions.reset_index(drop=True)


def feature_columns_for_set(frame: pd.DataFrame, feature_set: str) -> list[str]:
    price_columns = [
        column
        for column in frame.columns
        if column not in METADATA_COLUMNS
        and not column.startswith("weather_")
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    if "area" in frame.columns:
        price_columns = ["area"] + price_columns

    weather_columns = weather_value_columns(frame)
    if feature_set == "price_only":
        return price_columns
    if feature_set == "all_weather":
        selected_weather = [
            column for column in weather_columns if not column.startswith("weather_ensemble_")
        ]
        return price_columns + selected_weather if selected_weather else []
    if feature_set == "ensemble":
        selected_weather = [
            column for column in weather_columns if column.startswith("weather_ensemble_")
        ]
        return price_columns + selected_weather if selected_weather else []
    model_prefix = f"weather_{feature_set}_"
    selected_weather = [column for column in weather_columns if column.startswith(model_prefix)]
    return price_columns + selected_weather if selected_weather else []


def make_manifest(
    *,
    args: argparse.Namespace,
    predictions: pd.DataFrame,
    metrics: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "run_id": "weather_catboost_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_frame_path": str(args.experiment_frame_path),
        "feature_sets": args.feature_sets,
        "prediction_row_count": int(len(predictions)),
        "metric_row_count": int(len(metrics)),
        "iterations": int(args.iterations),
        "depth": int(args.depth),
        "learning_rate": float(args.learning_rate),
        "random_seed": int(args.random_seed),
        "min_train_rows": int(args.min_train_rows),
        "git_commit": git_commit(ROOT),
    }


if __name__ == "__main__":
    main()
