#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.io import load_price_panel  # noqa: E402
from dkenergy_forecast.layout import PROJECT_ROOT, runtime_layout  # noqa: E402
from dkenergy_forecast.models.chronos_production import (  # noqa: E402
    CALENDAR_COVARIATES,
    CHRONOS_LORA_ARTIFACT_SCHEMA_VERSION,
    CONTEXT_WEATHER_FILL_POLICY,
    DEFAULT_INFORMATION_CUTOFF_LOCAL_TIME,
    EXCLUDED_WEATHER_PARAMETERS,
    FUTURE_WEATHER_FILL_POLICY,
    WEATHER_CUTOFF_COLUMNS,
    WEATHER_HORIZON_COVERAGE_UNIT,
    WEATHER_MISSING_VALUE_POLICY,
    WEATHER_SELECTION_POLICY,
    Chronos2LoRAWeatherConfig,
    DEFAULT_CHRONOS_LORA_ARTIFACT_PATH,
    DEFAULT_WEATHER_FEATURES_LONG_PATH,
    add_weather_covariates,
    fill_lora_covariates,
    require_lora_weather_signal,
    selected_weather_columns,
    to_chronos_timestamp,
)
from dkenergy_forecast.publishing import git_commit, json_safe  # noqa: E402
from dkenergy_forecast.types import (  # noqa: E402
    DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
    PRICE_AVAILABILITY_COLUMN,
    TARGET_CONTRACT_COLUMNS,
    COPENHAGEN_TZ,
    add_price_availability,
    ensure_price_availability,
    filter_price_history_available_before,
    normalize_utc_column,
    require_columns,
    to_utc_timestamp,
)


DEFAULT_LAYOUT = runtime_layout(PROJECT_ROOT)


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
        information_cutoff_local_time=args.information_cutoff_local_time,
    )

    from chronos import BaseChronosPipeline
    from chronos.chronos2 import preprocess
    from transformers import set_seed

    set_seed(args.random_seed)

    train_inputs = preprocess.from_data_frame(
        train_df,
        target_columns=["target"],
        prediction_length=args.prediction_length,
        id_column="item_id",
        timestamp_column="timestamp",
        known_covariates_names=covariates,
    )
    load_kwargs: dict[str, Any] = {"device_map": args.device_map}
    if args.base_model_revision:
        load_kwargs["revision"] = args.base_model_revision
    pipeline = BaseChronosPipeline.from_pretrained(args.base_model_id, **load_kwargs)
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
        default=str(DEFAULT_LAYOUT.price_panel),
    )
    parser.add_argument(
        "--qa-path",
        default=str(DEFAULT_LAYOUT.price_panel_qa),
    )
    parser.add_argument("--weather-features-long-path", default=str(DEFAULT_WEATHER_FEATURES_LONG_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_CHRONOS_LORA_ARTIFACT_PATH))
    parser.add_argument("--base-model-id", default="amazon/chronos-2")
    parser.add_argument(
        "--base-model-revision",
        help="Optional immutable Hugging Face model revision (recommended for production exports).",
    )
    parser.add_argument("--first-eval-origin-utc", default="2026-04-01T10:00:00Z")
    parser.add_argument(
        "--information-cutoff-local-time",
        default=DEFAULT_INFORMATION_CUTOFF_LOCAL_TIME,
        help="Historical delivery-day cutoff in Europe/Copenhagen wall-clock time.",
    )
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--prediction-length", type=int, default=36)
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--random-seed", type=int, default=2026)
    parser.add_argument(
        "--weather-covariate-mode",
        choices=["all", "raw", "ensemble", "ensemble_mean"],
        default="ensemble_mean",
    )
    parser.add_argument("--weather-horizon-min-coverage", type=float, default=1.0)
    parser.add_argument(
        "--weather-future-fallback-policy",
        choices=["error", "zero"],
        default="error",
    )
    parser.add_argument(
        "--validation-scores-path",
        help="Optional model_scores.parquet path to summarize into the exported artifact manifest.",
    )
    parser.add_argument(
        "--validation-score-model-label",
        help="Optional model_label to select from --validation-scores-path.",
    )
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
    information_cutoff_local_time: str = DEFAULT_INFORMATION_CUTOFF_LOCAL_TIME,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    panel_utc = ensure_price_availability(
        normalize_utc_column(panel, "ds_utc")
    ).sort_values(["unique_id", "ds_utc"])
    require_columns(
        weather,
        [
            "area",
            "ds_utc",
            "feature_name",
            "value",
            "location_coverage_ratio",
            "location_coverage_pass",
            "feature_group_pass",
            "forecast_available_at_utc",
        ],
        "weather table",
    )
    if weather.empty:
        raise ValueError("Weather table is empty; cannot train Chronos LoRA weather covariates.")
    weather_utc = normalize_utc_column(weather, "ds_utc")
    start = max(panel_utc["ds_utc"].min(), weather_utc["ds_utc"].min())
    available = filter_price_history_available_before(panel_utc, first_eval_origin)
    train = available[available["ds_utc"] >= start].copy()
    if train.empty:
        raise ValueError(f"No Chronos LoRA training rows before {first_eval_origin.isoformat()}")
    assert_regular_hourly_training_series(train)
    train["forecast_origin_utc"] = train[PRICE_AVAILABILITY_COLUMN]
    train = add_price_availability(
        train,
        publication_local_time=information_cutoff_local_time,
        column="information_cutoff_utc",
    )
    train = add_weather_covariates(
        train,
        weather_utc,
        config=Chronos2LoRAWeatherConfig(weather_covariate_mode=weather_covariate_mode),
    )
    weather_columns = selected_weather_columns(train, mode=weather_covariate_mode)
    if not weather_columns:
        raise ValueError("No availability-safe weather covariates are available for Chronos LoRA training.")
    covariates = [column for column in CALENDAR_COVARIATES if column in train.columns] + weather_columns
    covariates = list(dict.fromkeys(covariates))
    require_lora_weather_signal(train, covariates, "Chronos LoRA training frame")
    train = fill_lora_covariates(train, covariates, role="training")
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
        "weather_origin_policy": (
            f"delivery-day-minus-one at {information_cutoff_local_time} Europe/Copenhagen"
        ),
        "weather_selection_policy": WEATHER_SELECTION_POLICY,
        "information_cutoff_local_time": information_cutoff_local_time,
        "covariate_fill_policy": WEATHER_MISSING_VALUE_POLICY,
        "training_frame_sha256": _dataframe_sha256(train_df),
        "price_availability_policy": {
            "column": PRICE_AVAILABILITY_COLUMN,
            "publication_local_time": DEFAULT_PRICE_PUBLICATION_LOCAL_TIME,
            "timezone": COPENHAGEN_TZ,
            "eligibility_operator": "< forecast_origin_utc",
        },
    }
    return train_df.sort_values(["item_id", "timestamp"]).reset_index(drop=True), covariates, diagnostics


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
    try:
        import torch

        torch_version = torch.__version__
    except ImportError:
        torch_version = None
    output_dir = getattr(args, "output_dir", None)
    artifact_files_sha256 = (
        _artifact_file_hashes(Path(output_dir)) if output_dir else {}
    )
    scores = read_validation_scores(
        args.validation_scores_path,
        model_label=args.validation_score_model_label,
    )
    artifact_content_sha256 = _hash_mapping(artifact_files_sha256)
    information_cutoff_local_time = getattr(
        args,
        "information_cutoff_local_time",
        DEFAULT_INFORMATION_CUTOFF_LOCAL_TIME,
    )
    return {
        "artifact_schema_version": CHRONOS_LORA_ARTIFACT_SCHEMA_VERSION,
        "model_name": "chronos_weather",
        "release_id": f"sha256-{artifact_content_sha256[:16]}",
        "created_at_utc": datetime.now(timezone.utc),
        "base_model_id": args.base_model_id,
        "base_model_revision": getattr(args, "base_model_revision", None),
        "model_family": "chronos2_lora_calendar_weather",
        "context_length": int(args.context_length),
        "prediction_length": int(args.prediction_length),
        "num_steps": int(args.num_steps),
        "batch_size": int(args.batch_size),
        "learning_rate": float(args.learning_rate),
        "random_seed": int(getattr(args, "random_seed", 2026)),
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
        "weather_availability_policy": {
            "training_cutoff": (
                "delivery-day-minus-one at "
                f"{information_cutoff_local_time} Europe/Copenhagen"
            ),
            "serving_context_cutoff": (
                "delivery-day-minus-one at "
                f"{information_cutoff_local_time} Europe/Copenhagen"
            ),
            "serving_future_cutoff": "information_cutoff_utc",
            "eligibility_operator": "forecast_available_at_utc <= information_cutoff_utc",
        },
        "weather_selection_policy": {
            "name": WEATHER_SELECTION_POLICY,
            "cutoff_priority": list(WEATHER_CUTOFF_COLUMNS),
            "stable_feature_names": True,
            "excluded_parameters": list(EXCLUDED_WEATHER_PARAMETERS),
            "historical_cutoff_local_time": information_cutoff_local_time,
        },
        "weather_vintage_policy": {
            "vintage_id_column": "weather_vintage_id",
            "reference_time_column": "forecast_reference_time",
            "reference_time_types": _unique_values(weather, "forecast_reference_time_type"),
            "reference_time_is_observed": _unique_values(
                weather,
                "forecast_reference_time_is_observed",
            ),
            "availability_time_column": "forecast_available_at_utc",
            "availability_time_types": _unique_values(
                weather,
                "forecast_availability_time_type",
            ),
        },
        "weather_horizon_coverage_policy": {
            "unit": WEATHER_HORIZON_COVERAGE_UNIT,
            "minimum": float(args.weather_horizon_min_coverage),
            "insufficient_coverage_fallback": args.weather_future_fallback_policy,
        },
        "covariate_fill_policy": {
            "training": CONTEXT_WEATHER_FILL_POLICY,
            "serving_context": CONTEXT_WEATHER_FILL_POLICY,
            "serving_future": FUTURE_WEATHER_FILL_POLICY,
            "temporal_fill": False,
            "missing_value": 0.0,
        },
        "target_contract": {
            "columns": TARGET_CONTRACT_COLUMNS,
            "market_regimes": _unique_values(panel, "market_regime"),
            "native_resolution_minutes": _unique_values(
                panel,
                "native_resolution_minutes",
            ),
            "target_aggregations": _unique_values(panel, "target_aggregation"),
            "target_definitions": _unique_values(panel, "target_definition"),
        },
        "diagnostics": diagnostics,
        "validation_scores": scores,
        "panel_path": args.panel_path,
        "weather_features_long_path": args.weather_features_long_path,
        "training_data_sha256": {
            "price_panel": _file_sha256_if_present(Path(args.panel_path)),
            "weather_features": _file_sha256_if_present(
                Path(args.weather_features_long_path)
            ),
        },
        "artifact_files_sha256": artifact_files_sha256,
        "artifact_content_sha256": artifact_content_sha256,
        "panel_dataset_version": sorted(panel["dataset_version"].dropna().unique().tolist()),
        "weather_min_ds_utc": pd.to_datetime(weather["ds_utc"], utc=True).min() if "ds_utc" in weather else None,
        "weather_max_ds_utc": pd.to_datetime(weather["ds_utc"], utc=True).max() if "ds_utc" in weather else None,
        "chronos_version": chronos_version,
        "torch_version": torch_version,
        "git_commit": git_commit(ROOT),
    }


def _unique_values(frame: pd.DataFrame, column: str) -> list[Any]:
    if column not in frame.columns:
        return []
    return sorted(frame[column].dropna().unique().tolist())


def _artifact_file_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json" and "_trainer" not in path.parts
    }


def _file_sha256_if_present(path: Path) -> str | None:
    return _file_sha256(path) if path.exists() else None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_mapping(values: dict[str, str]) -> str:
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _dataframe_sha256(frame: pd.DataFrame) -> str:
    """Hash the materialized training frame, including schema and row order."""

    digest = hashlib.sha256()
    schema = [(column, str(frame[column].dtype)) for column in frame.columns]
    digest.update(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def read_validation_scores(
    score_path: str | Path | None,
    *,
    model_label: str | None = None,
) -> dict[str, Any] | None:
    if not score_path:
        return None
    path = Path(score_path)
    if not path.exists():
        raise FileNotFoundError(f"Validation scores file does not exist: {path}")

    scores = pd.read_parquet(path)
    row = scores
    if model_label is not None and "model_label" in row.columns:
        row = row[row["model_label"].eq(model_label)]
    if "area" in row.columns and row["area"].eq("ALL").any():
        row = row[row["area"].eq("ALL")]
    row = row.head(1)
    if row.empty:
        return None
    values = row.iloc[0].to_dict()
    output = {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in values.items()
    }
    output["source_path"] = str(path)
    return output


if __name__ == "__main__":
    main()
