#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.backtesting.horizons import (  # noqa: E402
    make_daily_origins,
    make_danish_delivery_day_horizon,
)
from dkenergy_forecast.backtesting.rolling_origin import rolling_origin_backtest  # noqa: E402
from dkenergy_forecast.evaluation.point_metrics import bias, mae, rmse  # noqa: E402
from dkenergy_forecast.evaluation.probabilistic_metrics import (  # noqa: E402
    average_interval_width,
    interval_coverage,
    pinball_loss,
)
from dkenergy_forecast.evaluation.value_metrics import cheapest_k_hit_rate  # noqa: E402
from dkenergy_forecast.io import load_price_panel  # noqa: E402
from dkenergy_forecast.models.catboost_quantile import CatBoostQuantileModel  # noqa: E402


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel_path = Path(args.panel_path)
    qa_path = Path(args.qa_path) if args.qa_path else None
    panel = load_price_panel(
        panel_path,
        qa_path if qa_path and qa_path.exists() else None,
        require_final_historical=not args.allow_incomplete_panel,
    )
    origins = choose_origins(
        panel,
        days=args.days,
        at_hour_utc=args.at_hour_utc,
        max_origins=args.max_origins,
    )
    fitted_models: list[CatBoostQuantileModel] = []

    def model_factory() -> CatBoostQuantileModel:
        model = CatBoostQuantileModel(
            training_origin_days=args.training_origin_days,
            at_hour_utc=args.at_hour_utc,
            iterations=args.iterations,
            depth=args.depth,
            learning_rate=args.learning_rate,
            random_seed=args.random_seed,
        )
        fitted_models.append(model)
        return model

    try:
        predictions = rolling_origin_backtest(
            model_factory=model_factory,
            panel=panel,
            origins=origins,
            horizon_builder=lambda panel_arg, origin_arg: make_danish_delivery_day_horizon(
                panel_arg,
                origin_arg,
                days_ahead=1,
            ),
            min_train_rows=args.min_train_days * 24 * panel["area"].nunique(),
        )
    except ImportError as exc:
        raise SystemExit(str(exc)) from exc

    predictions["model_label"] = "catboost_quantile"
    predictions["error"] = predictions["y_pred"] - predictions["y"]
    predictions["abs_error"] = predictions["error"].abs()
    predictions["squared_error"] = predictions["error"] ** 2

    metrics = metric_table(predictions)
    value_metrics = value_metric_table(predictions, k=args.cheapest_k)
    probabilistic_metrics = probabilistic_metric_table(predictions)
    feature_importance = feature_importance_table(fitted_models)
    manifest = make_manifest(
        args=args,
        panel=panel,
        origins=origins,
        predictions=predictions,
        feature_importance=feature_importance,
        panel_path=panel_path,
        qa_path=qa_path,
    )

    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    metrics.to_parquet(output_dir / "metrics.parquet", index=False)
    probabilistic_metrics.to_parquet(output_dir / "probabilistic_metrics.parquet", index=False)
    value_metrics.to_parquet(output_dir / "value_metrics.parquet", index=False)
    feature_importance.to_parquet(output_dir / "feature_importance.parquet", index=False)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"Wrote predictions: {output_dir / 'predictions.parquet'}")
    print(f"Wrote metrics: {output_dir / 'metrics.parquet'}")
    print(f"Wrote probabilistic metrics: {output_dir / 'probabilistic_metrics.parquet'}")
    print(f"Wrote value metrics: {output_dir / 'value_metrics.parquet'}")
    print(f"Wrote feature importance: {output_dir / 'feature_importance.parquet'}")
    print(f"Wrote manifest: {output_dir / 'run_manifest.json'}")
    print(metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the optional CatBoost quantile backtest.")
    parser.add_argument(
        "--panel-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.parquet"),
    )
    parser.add_argument(
        "--qa-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.qa.json"),
    )
    parser.add_argument("--output-dir", default=str(ROOT / "results" / "catboost_v1"))
    parser.add_argument("--days", type=int, default=21)
    parser.add_argument("--max-origins", type=int, default=8)
    parser.add_argument("--at-hour-utc", type=int, default=10)
    parser.add_argument("--min-train-days", type=int, default=90)
    parser.add_argument("--training-origin-days", type=int, default=70)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--cheapest-k", type=int, default=6)
    parser.add_argument(
        "--allow-incomplete-panel",
        action="store_true",
        help="Allow QA artifacts that are not marked final_historical.",
    )
    return parser.parse_args()


def choose_origins(
    panel: pd.DataFrame,
    *,
    days: int,
    at_hour_utc: int,
    max_origins: int,
) -> pd.DataFrame:
    min_origin = (panel["ds_utc"].min() + pd.Timedelta(days=90)).normalize()
    max_origin = (panel["ds_utc"].max() - pd.Timedelta(days=2)).normalize()
    start = max(min_origin, max_origin - pd.Timedelta(days=days))
    end = max_origin + pd.Timedelta(days=1)
    origins = make_daily_origins(panel, start=start, end=end, at_hour_utc=at_hour_utc)

    valid_origins = []
    for origin in origins["forecast_origin_utc"]:
        horizon = make_danish_delivery_day_horizon(panel, origin, days_ahead=1)
        if horizon["ds_utc"].min() >= panel["ds_utc"].min() and horizon["ds_utc"].max() <= panel["ds_utc"].max():
            valid_origins.append(origin)
    if not valid_origins:
        raise ValueError("No valid forecast origins fit inside the panel range.")
    selected = valid_origins[-max_origins:] if max_origins > 0 else valid_origins
    return pd.DataFrame({"forecast_origin_utc": selected})


def metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    groups = [(("ALL",), predictions)]
    groups.extend([((area,), frame) for area, frame in predictions.groupby("area")])

    for (area,), frame in groups:
        rows.append(
            {
                "model_label": "catboost_quantile",
                "area": area,
                "rows": int(len(frame)),
                "evaluated_rows": int(frame["y_pred"].notna().sum()),
                "mae": mae(frame),
                "rmse": rmse(frame),
                "bias": bias(frame),
                "missing_rate": float(frame["y_pred"].isna().mean()),
                "raw_quantile_crossing_rate": float(frame["raw_quantile_crossing_rate"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["area"]).reset_index(drop=True)


def probabilistic_metric_table(predictions: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"metric": "pinball_q10", "value": pinball_loss(predictions, quantile=0.10)},
            {"metric": "pinball_q50", "value": pinball_loss(predictions, quantile=0.50)},
            {"metric": "pinball_q90", "value": pinball_loss(predictions, quantile=0.90)},
            {"metric": "p10_p90_coverage", "value": interval_coverage(predictions)},
            {"metric": "p10_p90_avg_width", "value": average_interval_width(predictions)},
        ]
    )


def value_metric_table(predictions: pd.DataFrame, *, k: int) -> pd.DataFrame:
    values = cheapest_k_hit_rate(predictions, k=k)
    values["model_label"] = "catboost_quantile"
    return values


def feature_importance_table(models: list[CatBoostQuantileModel]) -> pd.DataFrame:
    frames = [model.feature_importance_frame() for model in models]
    frames = [frame for frame in frames if not frame.empty]
    columns = ["forecast_origin_utc", "quantile", "feature", "importance"]
    if not frames:
        return pd.DataFrame(columns=columns)

    output = pd.concat(frames, ignore_index=True)
    output["model_label"] = "catboost_quantile"
    return output[
        ["model_label", "forecast_origin_utc", "quantile", "feature", "importance"]
    ].sort_values(
        ["forecast_origin_utc", "quantile", "importance", "feature"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)


def make_manifest(
    *,
    args: argparse.Namespace,
    panel: pd.DataFrame,
    origins: pd.DataFrame,
    predictions: pd.DataFrame,
    feature_importance: pd.DataFrame,
    panel_path: Path,
    qa_path: Path | None,
) -> dict[str, Any]:
    return {
        "run_id": "catboost_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_path": str(panel_path),
        "qa_path": str(qa_path) if qa_path else None,
        "dataset_version": sorted(panel["dataset_version"].dropna().unique().tolist()),
        "forecast_origin_min_utc": origins["forecast_origin_utc"].min(),
        "forecast_origin_max_utc": origins["forecast_origin_utc"].max(),
        "forecast_origin_count": int(len(origins)),
        "prediction_row_count": int(len(predictions)),
        "feature_importance_row_count": int(len(feature_importance)),
        "days": int(args.days),
        "max_origins": int(args.max_origins),
        "at_hour_utc": int(args.at_hour_utc),
        "min_train_days": int(args.min_train_days),
        "training_origin_days": int(args.training_origin_days),
        "iterations": int(args.iterations),
        "depth": int(args.depth),
        "learning_rate": float(args.learning_rate),
        "random_seed": int(args.random_seed),
        "cheapest_k": int(args.cheapest_k),
        "git_commit": git_commit(),
    }


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


if __name__ == "__main__":
    main()
