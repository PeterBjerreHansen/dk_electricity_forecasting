#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkenergy_forecast.backtesting.horizons import make_danish_delivery_day_horizon  # noqa: E402
from dkenergy_forecast.backtesting.origins import choose_recent_complete_daily_origins  # noqa: E402
from dkenergy_forecast.backtesting.rolling_origin import rolling_origin_backtest  # noqa: E402
from dkenergy_forecast.evaluation.summary import (  # noqa: E402
    add_prediction_diagnostics,
    model_score_table,
)
from dkenergy_forecast.io import load_price_panel  # noqa: E402
from dkenergy_forecast.models.registry import baseline_model_factories  # noqa: E402
from dkenergy_forecast.publishing import git_commit, json_safe  # noqa: E402


BASELINE_FACTORIES = baseline_model_factories()
NS_PER_DAY = 86_400_000_000_000

DayType = Literal["weekday", "weekend"]
WeightFamily = Literal["equal", "exponential"]
HistoryGroup = dict[tuple[str, int, bool], tuple[list[int], list[float]]]
ScheduleSpec = tuple[str, WeightFamily, float | None, float | None]


@dataclass(frozen=True)
class WeightedMedianSpec:
    label: str
    lookback_days: int
    weight_family: WeightFamily
    half_life_days: float | None = None
    floor: float | None = None
    day_type: DayType | None = None


@dataclass
class ScoreAccumulator:
    rows: int = 0
    evaluated_rows: int = 0
    error_sum: float = 0.0
    abs_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    missing_rows: int = 0

    def update(self, y: float, y_pred: float | None) -> None:
        self.rows += 1
        if y_pred is None or pd.isna(y_pred):
            self.missing_rows += 1
            return
        error = float(y_pred) - float(y)
        self.evaluated_rows += 1
        self.error_sum += error
        self.abs_error_sum += abs(error)
        self.squared_error_sum += error**2

    def as_metrics(self) -> dict[str, float | int]:
        if self.evaluated_rows == 0:
            mae = rmse = bias = math.nan
        else:
            mae = self.abs_error_sum / self.evaluated_rows
            rmse = (self.squared_error_sum / self.evaluated_rows) ** 0.5
            bias = self.error_sum / self.evaluated_rows
        return {
            "rows": self.rows,
            "evaluated_rows": self.evaluated_rows,
            "mae": mae,
            "rmse": rmse,
            "bias": bias,
            "missing_rate": self.missing_rows / self.rows if self.rows else math.nan,
        }


def main() -> None:
    args = parse_args()
    panel_path = Path(args.panel_path)
    qa_path = Path(args.qa_path) if args.qa_path else None
    output_dir = resolve_output_dir(args)
    output_dir.mkdir(parents=True, exist_ok=True)

    panel = load_price_panel(
        panel_path,
        qa_path if qa_path and qa_path.exists() else None,
        require_final_historical=not args.allow_incomplete_panel,
    )
    panel["ds_utc"] = pd.to_datetime(panel["ds_utc"], utc=True)

    if args.weighted_median_grid == "none":
        run_official_baseline_backtest(
            args=args,
            panel=panel,
            output_dir=output_dir,
            panel_path=panel_path,
            qa_path=qa_path,
        )
    elif args.weighted_median_grid == "common":
        run_common_weighted_median_backtest(
            args=args,
            panel=panel,
            output_dir=output_dir,
            panel_path=panel_path,
            qa_path=qa_path,
        )
    elif args.weighted_median_grid == "weekday-weekend":
        run_weekday_weekend_weighted_median_backtest(
            args=args,
            panel=panel,
            output_dir=output_dir,
            panel_path=panel_path,
            qa_path=qa_path,
        )
    else:
        raise ValueError(f"Unsupported weighted_median_grid: {args.weighted_median_grid}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run baseline backtests. By default this runs the compact official "
            "baseline set; optional weighted-median modes run heavier tuning grids."
        ),
    )
    parser.add_argument(
        "--panel-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.parquet"),
    )
    parser.add_argument(
        "--qa-path",
        default=str(ROOT / "data" / "model_ready" / "price_panel_hourly_v1.qa.json"),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults depend on the selected baseline mode.",
    )
    parser.add_argument("--days", type=int, default=45)
    parser.add_argument("--at-hour-utc", type=int, default=10)
    parser.add_argument("--min-train-days", type=int, default=60)
    parser.add_argument(
        "--weighted-median-grid",
        choices=("none", "common", "weekday-weekend"),
        default="none",
        help=(
            "Run no weighted tuning grid, a common-window weighted median grid, "
            "or an independently tuned weekday/weekend weighted median grid."
        ),
    )
    parser.add_argument(
        "--weighted-candidate-grid",
        choices=("diagnostic", "broad"),
        default="diagnostic",
        help="Candidate grid for --weighted-median-grid weekday-weekend.",
    )
    parser.add_argument("--weighted-lookback-days", type=int, default=56)
    parser.add_argument("--weighted-min-periods", type=int, default=4)
    parser.add_argument("--weighted-min-history-days", type=int, default=56)
    parser.add_argument(
        "--weighted-elspot-origin-stride-days",
        type=int,
        default=None,
        help=(
            "Use every Nth valid Elspot origin for weighted tuning. Defaults to "
            "7 for common grids and 1 for weekday/weekend grids."
        ),
    )
    parser.add_argument(
        "--weekday-lookback-days",
        nargs="+",
        type=int,
        default=[7, 14, 21, 28, 42, 56, 84],
    )
    parser.add_argument(
        "--weekend-lookback-days",
        nargs="+",
        type=int,
        default=[14, 21, 28, 35, 42, 56, 84],
    )
    parser.add_argument("--max-elspot-origins", type=int, default=0)
    parser.add_argument("--max-dayahead-origins", type=int, default=0)
    parser.add_argument("--max-selection-missing-rate", type=float, default=0.0)
    parser.add_argument(
        "--allow-incomplete-panel",
        action="store_true",
        help="Allow QA artifacts that are not marked final_historical.",
    )
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    defaults = {
        "none": ROOT / "results" / "baseline_v1",
        "common": ROOT / "results" / "baseline_weighted_median_common_v1",
        "weekday-weekend": ROOT / "results" / "baseline_weighted_median_weekday_weekend_v1",
    }
    return defaults[args.weighted_median_grid]


def run_official_baseline_backtest(
    *,
    args: argparse.Namespace,
    panel: pd.DataFrame,
    output_dir: Path,
    panel_path: Path,
    qa_path: Path | None,
) -> None:
    origins = choose_recent_complete_daily_origins(
        panel,
        days=args.days,
        at_hour_utc=args.at_hour_utc,
        min_history_days=args.min_train_days,
    )

    prediction_frames = []
    for model_label, factory in BASELINE_FACTORIES.items():
        predictions = rolling_origin_backtest(
            model_factory=factory,
            panel=panel,
            origins=origins,
            horizon_builder=delivery_day_horizon_builder,
            min_train_rows=args.min_train_days * 24 * panel["area"].nunique(),
        )
        predictions["model_label"] = model_label
        prediction_frames.append(predictions)

    predictions = add_prediction_diagnostics(pd.concat(prediction_frames, ignore_index=True))
    metrics = model_score_table(predictions)
    manifest = make_official_manifest(
        args=args,
        panel=panel,
        origins=origins,
        predictions=predictions,
        panel_path=panel_path,
        qa_path=qa_path,
    )

    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    metrics.to_parquet(output_dir / "model_scores.parquet", index=False)
    metrics.to_parquet(output_dir / "metrics.parquet", index=False)
    write_manifest(output_dir, manifest)

    print(f"Wrote predictions: {output_dir / 'predictions.parquet'}")
    print(f"Wrote model scores: {output_dir / 'model_scores.parquet'}")
    print(f"Wrote metrics: {output_dir / 'metrics.parquet'}")
    print(f"Wrote manifest: {output_dir / 'run_manifest.json'}")
    print(metrics.loc[metrics["area"] == "ALL"].sort_values("mae").to_string(index=False))


def run_common_weighted_median_backtest(
    *,
    args: argparse.Namespace,
    panel: pd.DataFrame,
    output_dir: Path,
    panel_path: Path,
    qa_path: Path | None,
) -> None:
    require_source_dataset(panel)
    history_groups = make_history_groups(panel)
    candidates = common_weighted_median_candidates(args.weighted_lookback_days)
    elspot_origins = select_source_horizon_origins(
        panel,
        source_dataset="Elspotprices",
        at_hour_utc=args.at_hour_utc,
        min_history_days=args.weighted_min_history_days,
        stride_days=weighted_elspot_stride_days(args),
        max_origins=args.max_elspot_origins,
    )
    dayahead_origins = select_source_horizon_origins(
        panel,
        source_dataset="DayAheadPrices",
        at_hour_utc=args.at_hour_utc,
        min_history_days=args.weighted_min_history_days,
        stride_days=1,
        max_origins=args.max_dayahead_origins,
    )
    min_train_rows = args.weighted_min_history_days * 24 * panel["area"].nunique()
    elspot_base = make_prediction_base(panel, origins=elspot_origins, min_train_rows=min_train_rows)
    dayahead_base = make_prediction_base(panel, origins=dayahead_origins, min_train_rows=min_train_rows)

    print(
        f"Scoring {len(candidates)} common weighted-median candidates over "
        f"{len(elspot_origins)} Elspot origins and {len(dayahead_origins)} DayAhead origins.",
        flush=True,
    )
    elspot_predictions = make_weighted_candidate_predictions(
        elspot_base,
        history_groups,
        candidates=candidates,
        min_periods=args.weighted_min_periods,
        model_name="weighted_seasonal_median",
        model_version="v1",
    )
    dayahead_predictions = make_weighted_candidate_predictions(
        dayahead_base,
        history_groups,
        candidates=candidates,
        min_periods=args.weighted_min_periods,
        model_name="weighted_seasonal_median",
        model_version="v1",
    )
    elspot_metrics = model_score_table(elspot_predictions)
    dayahead_metrics = model_score_table(dayahead_predictions)
    selection = common_weighted_model_selection_table(elspot_metrics, dayahead_metrics)
    manifest = make_common_weighted_manifest(
        args=args,
        panel=panel,
        panel_path=panel_path,
        qa_path=qa_path,
        candidates=candidates,
        elspot_origins=elspot_origins,
        dayahead_origins=dayahead_origins,
        elspot_predictions=elspot_predictions,
        dayahead_predictions=dayahead_predictions,
    )

    elspot_predictions.to_parquet(output_dir / "elspot_validation_predictions.parquet", index=False)
    dayahead_predictions.to_parquet(output_dir / "dayahead_holdout_predictions.parquet", index=False)
    elspot_metrics.to_parquet(output_dir / "elspot_validation_model_scores.parquet", index=False)
    dayahead_metrics.to_parquet(output_dir / "dayahead_holdout_model_scores.parquet", index=False)
    selection.to_parquet(output_dir / "model_selection.parquet", index=False)
    write_manifest(output_dir, manifest)

    print(f"Wrote common weighted-median artifacts to {output_dir}")
    print("\nElspot tuning scores, ALL area:")
    print(elspot_metrics.loc[elspot_metrics["area"] == "ALL"].sort_values("mae").to_string(index=False))
    print("\nDayAhead holdout scores, ALL area:")
    print(dayahead_metrics.loc[dayahead_metrics["area"] == "ALL"].sort_values("mae").to_string(index=False))


def run_weekday_weekend_weighted_median_backtest(
    *,
    args: argparse.Namespace,
    panel: pd.DataFrame,
    output_dir: Path,
    panel_path: Path,
    qa_path: Path | None,
) -> None:
    require_source_dataset(panel)
    history_groups = make_history_groups(panel)
    candidates = weekday_weekend_weighted_median_candidates(
        weekday_lookbacks=args.weekday_lookback_days,
        weekend_lookbacks=args.weekend_lookback_days,
        candidate_grid=args.weighted_candidate_grid,
    )
    elspot_origins = select_source_horizon_origins(
        panel,
        source_dataset="Elspotprices",
        at_hour_utc=args.at_hour_utc,
        min_history_days=args.weighted_min_history_days,
        stride_days=weighted_elspot_stride_days(args),
        max_origins=args.max_elspot_origins,
    )
    dayahead_origins = select_source_horizon_origins(
        panel,
        source_dataset="DayAheadPrices",
        at_hour_utc=args.at_hour_utc,
        min_history_days=args.weighted_min_history_days,
        stride_days=1,
        max_origins=args.max_dayahead_origins,
    )
    min_train_rows = args.weighted_min_history_days * 24 * panel["area"].nunique()
    elspot_base = make_prediction_base(panel, origins=elspot_origins, min_train_rows=min_train_rows)
    dayahead_base = make_prediction_base(panel, origins=dayahead_origins, min_train_rows=min_train_rows)

    print(
        f"Scoring {len(candidates)} weekday/weekend weighted-median candidates over "
        f"{len(elspot_origins)} Elspot origins and {len(dayahead_origins)} DayAhead origins.",
        flush=True,
    )
    elspot_scores = score_marginal_candidates(
        elspot_base,
        history_groups,
        candidates=candidates,
        source_dataset="Elspotprices",
        min_periods=args.weighted_min_periods,
    )
    dayahead_scores = score_marginal_candidates(
        dayahead_base,
        history_groups,
        candidates=candidates,
        source_dataset="DayAheadPrices",
        min_periods=args.weighted_min_periods,
    )
    selection = select_marginal_models(
        elspot_scores,
        dayahead_scores,
        max_missing_rate=args.max_selection_missing_rate,
    )
    selected_predictions = predict_selected_composite(
        dayahead_base,
        history_groups,
        selection,
        min_periods=args.weighted_min_periods,
    )
    selected_scores = model_score_table(selected_predictions)
    manifest = make_weekday_weekend_weighted_manifest(
        args=args,
        panel=panel,
        panel_path=panel_path,
        qa_path=qa_path,
        candidates=candidates,
        elspot_origins=elspot_origins,
        dayahead_origins=dayahead_origins,
        elspot_base=elspot_base,
        dayahead_base=dayahead_base,
    )

    elspot_scores.to_parquet(output_dir / "elspot_marginal_scores.parquet", index=False)
    dayahead_scores.to_parquet(output_dir / "dayahead_marginal_scores.parquet", index=False)
    selection.to_parquet(output_dir / "marginal_model_selection.parquet", index=False)
    selected_predictions.to_parquet(
        output_dir / "selected_composite_dayahead_predictions.parquet",
        index=False,
    )
    selected_scores.to_parquet(
        output_dir / "selected_composite_dayahead_scores.parquet",
        index=False,
    )
    selected_predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    selected_scores.to_parquet(output_dir / "model_scores.parquet", index=False)
    selected_scores.to_parquet(output_dir / "metrics.parquet", index=False)
    write_manifest(output_dir, manifest)

    print(f"Wrote weekday/weekend weighted-median artifacts to {output_dir}")
    print("\nSelected marginal models:")
    print(selection.to_string(index=False))
    print("\nSelected composite DayAhead scores:")
    print(selected_scores.sort_values(["area", "mae"]).to_string(index=False))


def delivery_day_horizon_builder(panel: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    return make_danish_delivery_day_horizon(panel, origin, days_ahead=1)


def require_source_dataset(panel: pd.DataFrame) -> None:
    if "source_dataset" not in panel.columns:
        raise ValueError("panel must contain source_dataset for Elspot/DayAhead splitting")


def weighted_elspot_stride_days(args: argparse.Namespace) -> int:
    if args.weighted_elspot_origin_stride_days is not None:
        return args.weighted_elspot_origin_stride_days
    if args.weighted_median_grid == "common":
        return 7
    return 1


def common_weighted_median_candidates(lookback_days: int) -> dict[str, WeightedMedianSpec]:
    candidates = {
        f"median_equal_{lookback_days}d_hour_weekend": WeightedMedianSpec(
            label=f"median_equal_{lookback_days}d_hour_weekend",
            weight_family="equal",
            lookback_days=lookback_days,
        )
    }
    for half_life_days in (7.0, 14.0, 28.0):
        for floor in (None, 0.05, 0.10, 0.20):
            label = common_weighted_label(lookback_days, half_life_days, floor)
            candidates[label] = WeightedMedianSpec(
                label=label,
                weight_family="exponential",
                lookback_days=lookback_days,
                half_life_days=half_life_days,
                floor=floor,
            )
    return candidates


def common_weighted_label(
    lookback_days: int,
    half_life_days: float,
    floor: float | None,
) -> str:
    label = f"median_exp_hl{int(half_life_days)}"
    if floor is not None:
        label = f"{label}_floor{int(round(floor * 100)):02d}"
    return f"{label}_{lookback_days}d_hour_weekend"


def weekday_weekend_weighted_median_candidates(
    *,
    weekday_lookbacks: list[int],
    weekend_lookbacks: list[int],
    candidate_grid: str,
) -> dict[str, WeightedMedianSpec]:
    lookbacks_by_day_type = {
        "weekday": weekday_lookbacks,
        "weekend": weekend_lookbacks,
    }
    schedules_by_day_type = {
        "weekday": broad_weight_schedules(),
        "weekend": broad_weight_schedules(),
    }
    if candidate_grid == "diagnostic":
        lookbacks_by_day_type = {
            "weekday": [28, 42, 56],
            "weekend": [35, 42, 56, 84],
        }
        schedules_by_day_type = diagnostic_weight_schedules()
    elif candidate_grid != "broad":
        raise ValueError(f"Unsupported weighted candidate grid: {candidate_grid}")

    candidates: dict[str, WeightedMedianSpec] = {}
    for day_type, lookbacks in lookbacks_by_day_type.items():
        for lookback_days in lookbacks:
            for label_part, family, half_life_days, floor in schedules_by_day_type[day_type]:
                label = f"median_{day_type}_{label_part}_{lookback_days}d"
                candidates[label] = WeightedMedianSpec(
                    label=label,
                    day_type=day_type,  # type: ignore[arg-type]
                    lookback_days=lookback_days,
                    weight_family=family,
                    half_life_days=half_life_days,
                    floor=floor,
                )
    return candidates


def diagnostic_weight_schedules() -> dict[str, list[ScheduleSpec]]:
    return {
        "weekday": exponential_schedules(
            half_lives=(4.0, 5.0, 6.0, 7.0, 10.0, 14.0),
            floors=(None, 0.05, 0.10),
        ),
        "weekend": exponential_schedules(
            half_lives=(10.0, 14.0, 21.0, 28.0, 35.0, 42.0),
            floors=(None, 0.10, 0.20),
        ),
    }


def broad_weight_schedules() -> list[ScheduleSpec]:
    return exponential_schedules(
        half_lives=(4.0, 5.0, 6.0, 7.0, 10.0, 14.0, 21.0, 28.0, 35.0, 42.0),
        floors=(None, 0.05, 0.10, 0.20),
    )


def exponential_schedules(
    *,
    half_lives: tuple[float, ...],
    floors: tuple[float | None, ...],
) -> list[ScheduleSpec]:
    schedules: list[ScheduleSpec] = [("equal", "equal", None, None)]
    for half_life_days in half_lives:
        for floor in floors:
            schedules.append(
                (
                    schedule_label(half_life_days, floor),
                    "exponential",
                    half_life_days,
                    floor,
                )
            )
    return schedules


def schedule_label(half_life_days: float, floor: float | None) -> str:
    label = f"exp_hl{int(half_life_days)}"
    if floor is None:
        return label
    return f"{label}_floor{int(round(floor * 100)):02d}"


def select_source_horizon_origins(
    panel: pd.DataFrame,
    *,
    source_dataset: str,
    at_hour_utc: int,
    min_history_days: int,
    stride_days: int,
    max_origins: int,
) -> pd.DataFrame:
    if stride_days <= 0:
        raise ValueError("stride_days must be positive")
    if max_origins < 0:
        raise ValueError("max_origins must be non-negative")

    min_origin = panel["ds_utc"].min() + pd.Timedelta(days=min_history_days)
    source_days = (
        panel.groupby("local_date")
        .agg(
            source_count=("source_dataset", "nunique"),
            source_name=("source_dataset", "first"),
            area_count=("area", "nunique"),
        )
        .reset_index()
    )
    delivery_dates = source_days.loc[
        source_days["source_count"].eq(1)
        & source_days["source_name"].eq(source_dataset)
        & source_days["area_count"].eq(panel["area"].nunique()),
        "local_date",
    ].sort_values()

    origins = pd.to_datetime(delivery_dates) - pd.Timedelta(days=1)
    origins = pd.DatetimeIndex(origins).tz_localize("UTC") + pd.Timedelta(hours=at_hour_utc)
    origins = origins[origins >= min_origin]
    selected = list(origins[::stride_days])
    if max_origins:
        selected = selected[-max_origins:]
    if not selected:
        raise ValueError(f"No valid {source_dataset} origins selected")
    return pd.DataFrame({"forecast_origin_utc": pd.Series(selected, dtype="datetime64[ns, UTC]")})


def make_prediction_base(
    panel: pd.DataFrame,
    *,
    origins: pd.DataFrame,
    min_train_rows: int,
) -> pd.DataFrame:
    metadata_columns = [
        "unique_id",
        "ds_utc",
        "y",
        "area",
        "ds_local",
        "local_date",
        "local_hour",
        "local_day_of_week",
        "local_month",
        "is_weekend",
        "is_dst",
        "utc_offset_hours",
        "dataset_version",
        "source_dataset",
    ]
    frames = []
    panel_utc = panel.sort_values(["unique_id", "ds_utc"]).reset_index(drop=True)
    for origin in origins["forecast_origin_utc"].sort_values().drop_duplicates():
        train_rows = int((panel_utc["ds_utc"] < origin).sum())
        if train_rows < min_train_rows:
            raise ValueError(
                "Not enough training rows before forecast origin "
                f"{origin.isoformat()}: {train_rows} < {min_train_rows}"
            )
        delivery_date = (origin.tz_convert("Europe/Copenhagen").date() + timedelta(days=1)).isoformat()
        frame = panel_utc.loc[
            panel_utc["local_date"].eq(delivery_date),
            metadata_columns,
        ].copy()
        if frame.empty:
            raise ValueError(f"No delivery-day rows found for {delivery_date}")
        frame["forecast_origin_utc"] = origin
        frame = frame.sort_values(["unique_id", "ds_utc"]).reset_index(drop=True)
        frame["horizon"] = frame.groupby("unique_id").cumcount() + 1
        frames.append(frame)
    columns = [
        "unique_id",
        "ds_utc",
        "forecast_origin_utc",
        "horizon",
        *[column for column in metadata_columns if column not in {"unique_id", "ds_utc"}],
    ]
    return pd.concat(frames, ignore_index=True)[columns]


def make_history_groups(panel: pd.DataFrame) -> HistoryGroup:
    groups: HistoryGroup = {}
    for key, frame in panel.groupby(["unique_id", "local_hour", "is_weekend"], sort=False):
        group_key = (str(key[0]), int(key[1]), bool(key[2]))
        sorted_frame = frame.sort_values("ds_utc")
        groups[group_key] = (
            sorted_frame["ds_utc"].astype("int64").tolist(),
            sorted_frame["y"].astype(float).tolist(),
        )
    return groups


def make_weighted_candidate_predictions(
    base: pd.DataFrame,
    history_groups: HistoryGroup,
    *,
    candidates: dict[str, WeightedMedianSpec],
    min_periods: int,
    model_name: str,
    model_version: str,
) -> pd.DataFrame:
    prediction_frames = []
    base_with_origin_ns = base.copy()
    base_with_origin_ns["origin_ns"] = base_with_origin_ns["forecast_origin_utc"].astype("int64")
    for label, spec in candidates.items():
        print(f"Running {label} over {base['forecast_origin_utc'].nunique()} origins...", flush=True)
        predictions = base_with_origin_ns.copy()
        predictions["model_name"] = model_name
        predictions["model_version"] = model_version
        predictions["model_label"] = label
        predictions["y_pred"] = [
            predict_weighted_median(row, history_groups, spec=spec, min_periods=min_periods)
            for row in predictions.itertuples(index=False)
        ]
        prediction_frames.append(predictions.drop(columns=["origin_ns"]))
    return add_prediction_diagnostics(pd.concat(prediction_frames, ignore_index=True))


def score_marginal_candidates(
    base: pd.DataFrame,
    history_groups: HistoryGroup,
    *,
    candidates: dict[str, WeightedMedianSpec],
    source_dataset: str,
    min_periods: int,
) -> pd.DataFrame:
    work = base.copy()
    work["origin_ns"] = work["forecast_origin_utc"].astype("int64")
    rows = list(work.itertuples(index=False))
    rows_by_day_type = {
        "weekday": [row for row in rows if not bool(row.is_weekend)],
        "weekend": [row for row in rows if bool(row.is_weekend)],
    }
    score_rows: list[dict[str, object]] = []
    for index, spec in enumerate(candidates.values(), start=1):
        if index == 1 or index % 50 == 0 or index == len(candidates):
            print(f"Scoring {source_dataset}: {index}/{len(candidates)} {spec.label}", flush=True)
        if spec.day_type is None:
            relevant_rows = rows
        else:
            relevant_rows = rows_by_day_type[spec.day_type]
        if not relevant_rows:
            continue
        stats = {"ALL": ScoreAccumulator()}
        for row in relevant_rows:
            prediction = predict_weighted_median(row, history_groups, spec=spec, min_periods=min_periods)
            stats["ALL"].update(row.y, prediction)
            area = str(row.area)
            stats.setdefault(area, ScoreAccumulator()).update(row.y, prediction)
        for area, accumulator in stats.items():
            output = {
                "source_dataset": source_dataset,
                "day_type": spec.day_type,
                "model_label": spec.label,
                "area": area,
                **accumulator.as_metrics(),
                **candidate_metadata(spec),
            }
            score_rows.append(output)
    return pd.DataFrame(score_rows).sort_values(
        ["source_dataset", "day_type", "area", "mae", "model_label"],
        na_position="first",
    ).reset_index(drop=True)


def predict_weighted_median(
    row: object,
    history_groups: HistoryGroup,
    *,
    spec: WeightedMedianSpec,
    min_periods: int,
) -> float | None:
    key = (str(row.unique_id), int(row.local_hour), bool(row.is_weekend))
    group = history_groups.get(key)
    if group is None:
        return None
    timestamps, values = group
    origin_ns = int(row.origin_ns)
    start = origin_ns - spec.lookback_days * NS_PER_DAY
    left = bisect_left(timestamps, start)
    right = bisect_left(timestamps, origin_ns)
    if right <= left:
        return None
    window_timestamps = timestamps[left:right]
    window_values = values[left:right]
    if len(window_values) < min_periods:
        return None
    if spec.weight_family == "equal":
        return median(window_values)

    weights = candidate_weights(origin_ns, window_timestamps, spec)
    positive = [(value, weight) for value, weight in zip(window_values, weights) if weight > 0]
    if len(positive) < min_periods:
        return None
    return weighted_median_from_pairs(positive)


def candidate_weights(
    origin_ns: int,
    timestamps: list[int],
    spec: WeightedMedianSpec,
) -> list[float]:
    ages = [(origin_ns - timestamp) / NS_PER_DAY for timestamp in timestamps]
    if spec.weight_family == "exponential":
        half_life_days = float(spec.half_life_days)
        weights = [0.5 ** (float(age) / half_life_days) for age in ages]
    else:
        raise ValueError(f"Unsupported weighted family for weights: {spec.weight_family}")

    if spec.floor is not None:
        floor = float(spec.floor)
        weights = [floor + (1 - floor) * weight for weight in weights]
    return weights


def median(values: list[float]) -> float:
    sorted_values = sorted(values)
    count = len(sorted_values)
    midpoint = count // 2
    if count % 2:
        return float(sorted_values[midpoint])
    return float((sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2)


def weighted_median_from_pairs(pairs: list[tuple[float, float]]) -> float | None:
    sorted_pairs = sorted(pairs, key=lambda pair: pair[0])
    total_weight = sum(weight for _, weight in sorted_pairs)
    if total_weight <= 0:
        return None
    cutoff = total_weight / 2
    cumulative = 0.0
    for value, weight in sorted_pairs:
        cumulative += weight
        if cumulative >= cutoff:
            return float(value)
    return float(sorted_pairs[-1][0])


def candidate_metadata(spec: WeightedMedianSpec) -> dict[str, object]:
    return {
        "lookback_days": spec.lookback_days,
        "weight_family": spec.weight_family,
        "half_life": spec.half_life_days,
        "floor": spec.floor,
    }


def common_weighted_model_selection_table(
    elspot_metrics: pd.DataFrame,
    dayahead_metrics: pd.DataFrame,
) -> pd.DataFrame:
    elspot_all = elspot_metrics[elspot_metrics["area"] == "ALL"].copy()
    dayahead_all = dayahead_metrics[dayahead_metrics["area"] == "ALL"].copy()
    columns = ["model_label", "mae", "rmse", "bias", "missing_rate"]
    selection = elspot_all[columns].rename(
        columns={
            "mae": "elspot_mae",
            "rmse": "elspot_rmse",
            "bias": "elspot_bias",
            "missing_rate": "elspot_missing_rate",
        }
    ).merge(
        dayahead_all[columns].rename(
            columns={
                "mae": "dayahead_mae",
                "rmse": "dayahead_rmse",
                "bias": "dayahead_bias",
                "missing_rate": "dayahead_missing_rate",
            }
        ),
        on="model_label",
        how="left",
    )
    selection["elspot_rank_mae"] = selection["elspot_mae"].rank(method="min")
    selection["dayahead_rank_mae"] = selection["dayahead_mae"].rank(method="min")
    return selection.sort_values(["elspot_mae", "model_label"]).reset_index(drop=True)


def select_marginal_models(
    elspot_scores: pd.DataFrame,
    dayahead_scores: pd.DataFrame,
    *,
    max_missing_rate: float,
) -> pd.DataFrame:
    selections = []
    for day_type in ["weekday", "weekend"]:
        eligible = elspot_scores[
            elspot_scores["day_type"].eq(day_type)
            & elspot_scores["area"].eq("ALL")
            & elspot_scores["evaluated_rows"].gt(0)
            & elspot_scores["missing_rate"].le(max_missing_rate)
        ].sort_values(["mae", "rmse", "model_label"])
        if eligible.empty:
            eligible = elspot_scores[
                elspot_scores["day_type"].eq(day_type)
                & elspot_scores["area"].eq("ALL")
                & elspot_scores["evaluated_rows"].gt(0)
            ].sort_values(["missing_rate", "mae", "rmse", "model_label"])
        if eligible.empty:
            raise ValueError(f"No evaluated Elspot rows available for {day_type} selection")
        selected = eligible.iloc[0]
        holdout = dayahead_scores[
            dayahead_scores["day_type"].eq(day_type)
            & dayahead_scores["area"].eq("ALL")
            & dayahead_scores["model_label"].eq(selected["model_label"])
        ].iloc[0]
        selections.append(
            {
                "day_type": day_type,
                "model_label": selected["model_label"],
                "lookback_days": selected["lookback_days"],
                "weight_family": selected["weight_family"],
                "half_life": selected["half_life"],
                "floor": selected["floor"],
                "elspot_mae": selected["mae"],
                "elspot_rmse": selected["rmse"],
                "elspot_bias": selected["bias"],
                "elspot_missing_rate": selected["missing_rate"],
                "dayahead_mae": holdout["mae"],
                "dayahead_rmse": holdout["rmse"],
                "dayahead_bias": holdout["bias"],
                "dayahead_missing_rate": holdout["missing_rate"],
            }
        )
    return pd.DataFrame(selections)


def predict_selected_composite(
    base: pd.DataFrame,
    history_groups: HistoryGroup,
    selection: pd.DataFrame,
    *,
    min_periods: int,
) -> pd.DataFrame:
    specs = {
        row.day_type: WeightedMedianSpec(
            label=str(row.model_label),
            day_type=row.day_type,
            lookback_days=int(row.lookback_days),
            weight_family=row.weight_family,
            half_life_days=None if pd.isna(row.half_life) else float(row.half_life),
            floor=None if pd.isna(row.floor) else float(row.floor),
        )
        for row in selection.itertuples(index=False)
    }
    output = base.copy()
    output["origin_ns"] = output["forecast_origin_utc"].astype("int64")
    output["model_name"] = "weekday_weekend_weighted_seasonal_median"
    output["model_version"] = "v1"
    output["model_label"] = "weekday_weekend_weighted_median_composite"
    output["selected_component_label"] = [
        specs["weekend" if bool(row.is_weekend) else "weekday"].label
        for row in output.itertuples(index=False)
    ]
    output["y_pred"] = [
        predict_weighted_median(
            row,
            history_groups,
            spec=specs["weekend" if bool(row.is_weekend) else "weekday"],
            min_periods=min_periods,
        )
        for row in output.itertuples(index=False)
    ]
    output["error"] = output["y_pred"] - output["y"]
    output["abs_error"] = output["error"].abs()
    output["squared_error"] = output["error"] ** 2
    return output.drop(columns=["origin_ns"])


def make_official_manifest(
    *,
    args: argparse.Namespace,
    panel: pd.DataFrame,
    origins: pd.DataFrame,
    predictions: pd.DataFrame,
    panel_path: Path,
    qa_path: Path | None,
) -> dict[str, Any]:
    return {
        "run_id": "baseline_v1",
        "baseline_mode": "official",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_path": str(panel_path),
        "qa_path": str(qa_path) if qa_path else None,
        "dataset_version": sorted(panel["dataset_version"].dropna().unique().tolist()),
        "model_labels": sorted(BASELINE_FACTORIES),
        "forecast_origin_min_utc": origins["forecast_origin_utc"].min(),
        "forecast_origin_max_utc": origins["forecast_origin_utc"].max(),
        "forecast_origin_count": int(len(origins)),
        "prediction_row_count": int(len(predictions)),
        "days": int(args.days),
        "at_hour_utc": int(args.at_hour_utc),
        "min_train_days": int(args.min_train_days),
        "git_commit": git_commit(ROOT),
    }


def make_common_weighted_manifest(
    *,
    args: argparse.Namespace,
    panel: pd.DataFrame,
    panel_path: Path,
    qa_path: Path | None,
    candidates: dict[str, WeightedMedianSpec],
    elspot_origins: pd.DataFrame,
    dayahead_origins: pd.DataFrame,
    elspot_predictions: pd.DataFrame,
    dayahead_predictions: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "run_id": "baseline_weighted_median_common_v1",
        "baseline_mode": "weighted_median_common",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_path": str(panel_path),
        "qa_path": str(qa_path) if qa_path else None,
        "panel_min_utc": panel["ds_utc"].min(),
        "panel_max_utc": panel["ds_utc"].max(),
        "source_datasets": sorted(panel["source_dataset"].dropna().unique().tolist()),
        "candidate_labels": list(candidates),
        "candidate_specs": [
            candidate_metadata(spec) | {"label": spec.label, "day_type": spec.day_type}
            for spec in candidates.values()
        ],
        "seasonal_keys": ["local_hour", "is_weekend"],
        "lookback_days": int(args.weighted_lookback_days),
        "min_periods": int(args.weighted_min_periods),
        "min_history_days": int(args.weighted_min_history_days),
        "at_hour_utc": int(args.at_hour_utc),
        "elspot_origin_stride_days": int(weighted_elspot_stride_days(args)),
        "elspot_origin_count": int(len(elspot_origins)),
        "dayahead_origin_count": int(len(dayahead_origins)),
        "elspot_prediction_rows": int(len(elspot_predictions)),
        "dayahead_prediction_rows": int(len(dayahead_predictions)),
        "git_commit": git_commit(ROOT),
    }


def make_weekday_weekend_weighted_manifest(
    *,
    args: argparse.Namespace,
    panel: pd.DataFrame,
    panel_path: Path,
    qa_path: Path | None,
    candidates: dict[str, WeightedMedianSpec],
    elspot_origins: pd.DataFrame,
    dayahead_origins: pd.DataFrame,
    elspot_base: pd.DataFrame,
    dayahead_base: pd.DataFrame,
) -> dict[str, Any]:
    return {
        "run_id": "baseline_weighted_median_weekday_weekend_v1",
        "baseline_mode": "weighted_median_weekday_weekend",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_path": str(panel_path),
        "qa_path": str(qa_path) if qa_path else None,
        "panel_min_utc": panel["ds_utc"].min(),
        "panel_max_utc": panel["ds_utc"].max(),
        "source_datasets": sorted(panel["source_dataset"].dropna().unique().tolist()),
        "candidate_grid": args.weighted_candidate_grid,
        "candidate_count": len(candidates),
        "candidate_specs": [
            candidate_metadata(spec) | {"label": spec.label, "day_type": spec.day_type}
            for spec in candidates.values()
        ],
        "weekday_lookback_days": sorted(
            {spec.lookback_days for spec in candidates.values() if spec.day_type == "weekday"}
        ),
        "weekend_lookback_days": sorted(
            {spec.lookback_days for spec in candidates.values() if spec.day_type == "weekend"}
        ),
        "weight_schedules": weight_schedule_manifest(args.weighted_candidate_grid),
        "seasonal_keys": ["local_hour", "is_weekend"],
        "min_periods": int(args.weighted_min_periods),
        "min_history_days": int(args.weighted_min_history_days),
        "at_hour_utc": int(args.at_hour_utc),
        "elspot_origin_stride_days": int(weighted_elspot_stride_days(args)),
        "elspot_origin_count": int(len(elspot_origins)),
        "dayahead_origin_count": int(len(dayahead_origins)),
        "elspot_prediction_base_rows": int(len(elspot_base)),
        "dayahead_prediction_base_rows": int(len(dayahead_base)),
        "selection_protocol": (
            "Weekday and weekend candidates are selected independently using "
            "only same-day-type Elspot ALL-area MAE, then evaluated on the "
            "corresponding DayAhead holdout rows."
        ),
        "git_commit": git_commit(ROOT),
    }


def weight_schedule_manifest(candidate_grid: str) -> dict[str, list[str]]:
    if candidate_grid == "diagnostic":
        schedules = diagnostic_weight_schedules()
        return {
            day_type: [schedule[0] for schedule in day_type_schedules]
            for day_type, day_type_schedules in schedules.items()
        }
    labels = [schedule[0] for schedule in broad_weight_schedules()]
    return {"weekday": labels, "weekend": labels}


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "run_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
