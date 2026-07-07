#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.io import load_price_panel  # noqa: E402
from dkenergy_forecast.models.chronos_production import (  # noqa: E402
    CALENDAR_COVARIATES,
    CHRONOS_LORA_ARTIFACT_SCHEMA_VERSION,
    DEFAULT_CHRONOS_LORA_ARTIFACT_PATH,
    DEFAULT_WEATHER_FEATURES_LONG_PATH,
    selected_weather_columns,
    to_chronos_timestamp,
)
from dkenergy_forecast.features.weather_features import add_weather_ensemble_features  # noqa: E402
from dkenergy_forecast.publishing import git_commit, json_safe  # noqa: E402
from dkenergy_forecast.types import (  # noqa: E402
    DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
    PRICE_AVAILABILITY_COLUMN,
    COPENHAGEN_TZ,
    ensure_price_availability,
    filter_price_history_available_before,
    normalize_utc_column,
    to_utc_timestamp,
)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"Output directory already exists: {output_dir}; pass --overwrite to replace it.")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_price_panel(
        Path(args.panel_path),
        Path(args.qa_path) if args.qa_path and Path(args.qa_path).exists() else None,
        require_final_historical=not args.allow_incomplete_panel,
    )
    weather = pd.read_parquet(args.weather_features_long_path)
    train_df, covariates, diagnostics = make_lora_training_frame(
        panel,
        weather,
        first_eval_origin=to_utc_timestamp(args.first_eval_origin_utc),
        context_length=args.context_length,
        prediction_length=args.prediction_length,
        weather_covariate_mode=args.weather_covariate_mode,
    )

    from chronos import BaseChronosPipeline
    from chronos.chronos2 import preprocess

    train_inputs = preprocess.from_data_frame(
        train_df,
        target_columns=["target"],
        prediction_length=args.prediction_length,
        id_column="item_id",
        timestamp_column="timestamp",
        known_covariates_names=covariates,
    )
    pipeline = BaseChronosPipeline.from_pretrained(args.base_model_id, device_map=args.device_map)
    trainer_dir = output_dir / "_trainer"
    lora_pipeline = pipeline.fit(
        inputs=train_inputs,
        prediction_length=args.prediction_length,
        context_length=args.context_length,
        min_past=args.context_length,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        logging_steps=args.logging_steps,
        finetune_mode="lora",
        output_dir=str(trainer_dir),
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        save_strategy="no",
        report_to="none",
    )
    lora_pipeline.save_pretrained(output_dir)
    shutil.rmtree(trainer_dir, ignore_errors=True)

    manifest = make_manifest(
        args=args,
        panel=panel,
        weather=weather,
        covariates=covariates,
        diagnostics=diagnostics,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote Chronos LoRA artifact: {output_dir}")
    print(f"Training rows: {diagnostics['training_rows']}; covariates: {len(covariates)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and export the production Chronos-2 LoRA weather artifact.")
    parser.add_argument(
        "--panel-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.parquet"),
    )
    parser.add_argument(
        "--qa-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.qa.json"),
    )
    parser.add_argument("--weather-features-long-path", default=str(DEFAULT_WEATHER_FEATURES_LONG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_CHRONOS_LORA_ARTIFACT_PATH))
    parser.add_argument("--base-model-id", default="amazon/chronos-2")
    parser.add_argument("--first-eval-origin-utc", default="2026-04-01T10:00:00Z")
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--prediction-length", type=int, default=36)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--weather-covariate-mode", choices=["all", "raw", "ensemble", "ensemble_mean"], default="all")
    parser.add_argument("--device-map", default="cpu")
    parser.add_argument("--allow-incomplete-panel", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def make_lora_training_frame(
    panel: pd.DataFrame,
    weather: pd.DataFrame,
    *,
    first_eval_origin: pd.Timestamp,
    context_length: int,
    prediction_length: int,
    weather_covariate_mode: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    panel_utc = ensure_price_availability(
        normalize_utc_column(panel, "ds_utc")
    ).sort_values(["unique_id", "ds_utc"])
    weather_covariates = weather_valid_time_covariate_frame(weather)
    weather_columns = selected_weather_columns(weather_covariates, mode=weather_covariate_mode)
    if not weather_columns:
        raise ValueError("No weather covariates are available for Chronos LoRA training.")

    start = max(panel_utc["ds_utc"].min(), weather_covariates["ds_utc"].min())
    available = filter_price_history_available_before(panel_utc, first_eval_origin)
    train = available[available["ds_utc"] >= start].copy()
    if train.empty:
        raise ValueError(f"No Chronos LoRA training rows before {first_eval_origin.isoformat()}")
    assert_regular_hourly_training_series(train)
    train = train.merge(
        weather_covariates[["area", "ds_utc", *weather_columns]],
        on=["area", "ds_utc"],
        how="left",
    )
    covariates = [column for column in CALENDAR_COVARIATES if column in train.columns] + weather_columns
    covariates = list(dict.fromkeys(covariates))
    train = fill_lora_covariates(train, covariates)
    if train["y"].isna().any():
        raise ValueError(f"Chronos LoRA training target has missing rows: {int(train['y'].isna().sum())}")

    lengths = train.groupby("unique_id")["ds_utc"].size()
    min_required = context_length + prediction_length
    too_short = lengths[lengths < min_required]
    if not too_short.empty:
        raise ValueError("Chronos LoRA training series are too short:\n" + too_short.to_string())

    train["item_id"] = train["unique_id"]
    train["timestamp"] = to_chronos_timestamp(train["ds_utc"])
    train_df = train[["item_id", "timestamp", "y", *covariates]].rename(columns={"y": "target"})
    diagnostics = {
        "training_mode": "long_series",
        "first_eval_origin_utc": first_eval_origin,
        "training_start_utc": train["ds_utc"].min(),
        "training_end_utc": train["ds_utc"].max(),
        "training_rows": int(len(train_df)),
        "training_items": int(train_df["item_id"].nunique()),
        "min_series_length": int(lengths.min()),
        "max_series_length": int(lengths.max()),
        "available_random_windows_lower_bound": int((lengths - min_required + 1).clip(lower=0).sum()),
        "covariate_count": int(len(covariates)),
        "weather_covariate_count": int(sum(column.startswith("weather_") for column in covariates)),
        "price_availability_policy": {
            "column": PRICE_AVAILABILITY_COLUMN,
            "publication_local_time": DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
            "timezone": COPENHAGEN_TZ,
            "eligibility_operator": "< forecast_origin_utc",
        },
    }
    return train_df.sort_values(["item_id", "timestamp"]).reset_index(drop=True), covariates, diagnostics


def weather_valid_time_covariate_frame(weather: pd.DataFrame) -> pd.DataFrame:
    required = {
        "area",
        "ds_utc",
        "feature_name",
        "value",
        "location_coverage_pass",
        "feature_group_pass",
    }
    missing = required - set(weather.columns)
    if missing:
        raise ValueError(f"weather table is missing required columns: {sorted(missing)}")
    filtered = normalize_utc_column(weather, "ds_utc")
    filtered = filtered[filtered["feature_group_pass"] & filtered["location_coverage_pass"]].copy()
    wide = (
        filtered.pivot_table(
            index=["area", "ds_utc"],
            columns="feature_name",
            values="value",
            aggfunc="last",
        )
        .reset_index()
        .sort_values(["area", "ds_utc"])
    )
    wide.columns.name = None
    return add_weather_ensemble_features(wide).reset_index(drop=True)


def fill_lora_covariates(frame: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    output = frame.sort_values(["unique_id", "ds_utc"]).copy()
    missing_columns = [column for column in covariates if column not in output.columns]
    if missing_columns:
        output = output.assign(**{column: np.nan for column in missing_columns})
    for column in covariates:
        if pd.api.types.is_bool_dtype(output[column]):
            output[column] = output[column].astype("int8")
    filled = (
        output.groupby("unique_id", observed=True)[covariates]
        .transform(lambda values: values.ffill().bfill())
        .fillna(0.0)
    )
    return pd.concat([output.drop(columns=covariates), filled], axis=1)


def assert_regular_hourly_training_series(train: pd.DataFrame) -> None:
    gap_rows = []
    for item_id, timestamps in train.groupby("unique_id", observed=True)["ds_utc"]:
        deltas = timestamps.sort_values().diff().dropna()
        bad = deltas[deltas.ne(pd.Timedelta(hours=1))]
        if not bad.empty:
            gap_rows.append(
                {
                    "unique_id": item_id,
                    "bad_gap_count": int(len(bad)),
                    "min_bad_gap": bad.min(),
                    "max_bad_gap": bad.max(),
                }
            )
    if gap_rows:
        raise ValueError(
            "Chronos LoRA training frame must be regular hourly data:\n"
            + pd.DataFrame(gap_rows).to_string(index=False)
        )


def make_manifest(
    *,
    args: argparse.Namespace,
    panel: pd.DataFrame,
    weather: pd.DataFrame,
    covariates: list[str],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    try:
        import chronos

        chronos_version = getattr(chronos, "__version__", None)
    except ImportError:
        chronos_version = None
    scores = read_validation_scores()
    return {
        "artifact_schema_version": CHRONOS_LORA_ARTIFACT_SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc),
        "base_model_id": args.base_model_id,
        "model_family": "chronos2_lora_calendar_weather",
        "context_length": int(args.context_length),
        "prediction_length": int(args.prediction_length),
        "num_steps": int(args.num_steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "weather_covariate_mode": args.weather_covariate_mode,
        "covariates": list(covariates),
        "calendar_covariates": [column for column in covariates if column in CALENDAR_COVARIATES],
        "weather_covariates": [column for column in covariates if column.startswith("weather_")],
        "price_availability_policy": {
            "column": PRICE_AVAILABILITY_COLUMN,
            "publication_local_time": DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
            "timezone": COPENHAGEN_TZ,
            "eligibility_operator": "< forecast_origin_utc",
        },
        "forecast_origin_policy": {
            "timezone": COPENHAGEN_TZ,
            "local_time": DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
        },
        "diagnostics": diagnostics,
        "validation_scores": scores,
        "panel_path": args.panel_path,
        "weather_features_long_path": args.weather_features_long_path,
        "panel_dataset_version": sorted(panel["dataset_version"].dropna().unique().tolist()),
        "weather_min_ds_utc": pd.to_datetime(weather["ds_utc"], utc=True).min() if "ds_utc" in weather else None,
        "weather_max_ds_utc": pd.to_datetime(weather["ds_utc"], utc=True).max() if "ds_utc" in weather else None,
        "chronos_version": chronos_version,
        "git_commit": git_commit(ROOT),
    }


def read_validation_scores() -> dict[str, Any] | None:
    score_path = ROOT / "results" / "notebook_chronos2_experimental_v1" / "model_scores.parquet"
    if not score_path.exists():
        return None
    scores = pd.read_parquet(score_path)
    model = "chronos2_lora_calendar_weather_ctx1024_300steps"
    row = scores[(scores["model_label"].eq(model)) & (scores["area"].eq("ALL"))].head(1)
    if row.empty:
        return None
    values = row.iloc[0].to_dict()
    return {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in values.items()
    }


if __name__ == "__main__":
    main()
