from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from dkenergy_forecast.evaluation.summary import add_prediction_diagnostics, model_score_table
from dkenergy_forecast.layout import CHRONOS_LORA_WEATHER_MODEL_LABEL, PROJECT_ROOT, runtime_layout
from dkenergy_forecast.storage import materialize_uri, resource_exists


ROOT = PROJECT_ROOT
DEFAULT_LAYOUT = runtime_layout(ROOT)
DEFAULT_PANEL_PATH = DEFAULT_LAYOUT.price_panel
DEFAULT_DASHBOARD_JSON = DEFAULT_LAYOUT.dashboard_json
DEFAULT_LATEST_PREDICTIONS = DEFAULT_LAYOUT.latest_forecast / "predictions.parquet"
DEFAULT_RECENT_PREDICTIONS = DEFAULT_LAYOUT.recent_scores / "predictions.parquet"
DEFAULT_RECENT_SCORES = DEFAULT_LAYOUT.recent_scores / "model_scores.parquet"
DEFAULT_PUBLISHED_HISTORY_PREDICTIONS = DEFAULT_LAYOUT.published_history / "predictions.parquet"
DEFAULT_PUBLISHED_HISTORY_SCORES = DEFAULT_LAYOUT.published_history / "model_scores.parquet"
DEFAULT_BACKTEST_DIRS = [
    ROOT / "results" / "notebook_chronos2_experimental_v1",
    ROOT / "results" / "baseline_v1",
]
CACHE_TTL_SECONDS = 300

MODEL_DISPLAY_NAMES = {
    CHRONOS_LORA_WEATHER_MODEL_LABEL: "Chronos 2 LoRA Weather",
    "median_weekday_exp_hl4_floor10_42d__median_weekend_exp_hl28_floor20_56d": "Weighted Rolling Median",
    "rolling_median_hour_weekend_56d": "Rolling Median",
    "rolling_median_local_hour_28d": "Rolling Median",
    "same_hour_last_week": "Last Week Baseline",
    "catboost_ensemble": "Weather Boosting Ensemble",
    "catboost_gfs_global": "Weather Boosting GFS",
}


def _cache_data(ttl: int):
    if hasattr(st, "cache_data"):
        return st.cache_data(ttl=ttl)
    return lambda func: func


def main() -> None:
    st.set_page_config(page_title="Danish Electricity Forecasts", layout="wide")

    panel_path = _resource_from_env("DKENERGY_PANEL_PATH", DEFAULT_PANEL_PATH)
    dashboard_json = _resource_from_env("DKENERGY_DASHBOARD_JSON", DEFAULT_DASHBOARD_JSON)
    latest_predictions_path = _resource_from_env(
        "DKENERGY_LATEST_PREDICTIONS_PATH",
        DEFAULT_LATEST_PREDICTIONS,
    )
    recent_predictions_path = _resource_from_env(
        "DKENERGY_RECENT_PREDICTIONS_PATH",
        DEFAULT_RECENT_PREDICTIONS,
    )
    recent_scores_path = _resource_from_env("DKENERGY_RECENT_SCORES_PATH", DEFAULT_RECENT_SCORES)
    published_history_predictions_path = _resource_from_env(
        "DKENERGY_PUBLISHED_HISTORY_PREDICTIONS_PATH",
        DEFAULT_PUBLISHED_HISTORY_PREDICTIONS,
    )
    published_history_scores_path = _resource_from_env(
        "DKENERGY_PUBLISHED_HISTORY_SCORES_PATH",
        DEFAULT_PUBLISHED_HISTORY_SCORES,
    )
    enable_legacy_backtests = _env_bool(
        "DKENERGY_ENABLE_LEGACY_BACKTESTS",
        default=False,
    )
    backtest_dirs = _paths_from_env(
        "DKENERGY_BACKTEST_DIRS",
        DEFAULT_BACKTEST_DIRS if enable_legacy_backtests else [],
    )

    payload = _load_dashboard_payload(dashboard_json)
    panel = _load_parquet(panel_path)
    predictions = _load_predictions(payload, latest_predictions_path)
    scores = _load_scores(payload, recent_scores_path)
    recent_predictions = _load_recent_predictions(payload, recent_predictions_path, [])
    published_history_predictions = _load_published_history_predictions(
        payload,
        published_history_predictions_path,
    )
    published_history_scores = _load_published_history_scores(
        payload,
        published_history_scores_path,
    )
    backtest_predictions = _load_backtest_tab_predictions(
        payload,
        recent_predictions_path,
        backtest_dirs,
    )
    run = payload.get("run", {}) if payload else {}

    st.title("Danish Electricity Forecasts")
    _render_run_summary(run, predictions, published_history_scores if not published_history_scores.empty else scores)

    backtests_tab, forecast_tab, prices_tab, run_tab = st.tabs(
        ["Backtests", "Forecasts", "Prices", "Run"]
    )

    with backtests_tab:
        _render_backtests(backtest_predictions, scores)

    with forecast_tab:
        _render_forecasts(
            predictions,
            published_history_predictions=published_history_predictions,
            published_history_scores=published_history_scores,
            diagnostic_predictions=recent_predictions,
            diagnostic_scores=scores,
        )

    with prices_tab:
        _render_actual_prices(panel)

    with run_tab:
        _render_run_details(
            run,
            panel_path,
            dashboard_json,
            latest_predictions_path,
            recent_predictions_path,
            recent_scores_path,
            published_history_predictions_path,
            published_history_scores_path,
            backtest_dirs,
        )


def _render_run_summary(
    run: dict[str, Any],
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
) -> None:
    forecast_origin = run.get("forecast_origin_utc")
    generated_at = run.get("created_at_utc") or run.get("generated_at_utc")
    model_count = predictions["model_label"].nunique() if "model_label" in predictions else 0
    score_count = len(scores) if not scores.empty else 0

    cols = st.columns(4)
    cols[0].metric("Forecast origin", _format_optional_timestamp(forecast_origin))
    cols[1].metric("Models", model_count)
    cols[2].metric("Score rows", score_count)
    cols[3].metric("Generated", _format_optional_timestamp(generated_at))


def _render_actual_prices(panel: pd.DataFrame) -> None:
    if panel.empty:
        st.warning("No price panel parquet found.")
        return

    frame = panel.copy()
    frame["ds_utc"] = pd.to_datetime(frame["ds_utc"], utc=True)
    value_column = "y" if "y" in frame.columns else "price_dkk_per_mwh"
    days = st.slider("Visible days", min_value=3, max_value=30, value=7)
    cutoff = frame["ds_utc"].max() - pd.Timedelta(days=days)
    recent = frame.loc[frame["ds_utc"] >= cutoff, ["ds_utc", "area", value_column]].copy()

    chart = recent.pivot_table(
        index="ds_utc",
        columns="area",
        values=value_column,
        aggfunc="last",
    ).sort_index()
    st.line_chart(chart, height=360)

    latest = (
        recent.sort_values(["area", "ds_utc"])
        .groupby("area", as_index=False)
        .tail(1)
        .sort_values("area")
    )
    st.dataframe(
        latest.rename(
            columns={
                "ds_utc": "valid_time_utc",
                value_column: "actual_price_dkk_per_mwh",
            }
        ),
        width="stretch",
        hide_index=True,
    )


def _render_forecasts(
    predictions: pd.DataFrame,
    *,
    published_history_predictions: pd.DataFrame,
    published_history_scores: pd.DataFrame,
    diagnostic_predictions: pd.DataFrame,
    diagnostic_scores: pd.DataFrame,
) -> None:
    st.subheader("Latest Published Forecast")
    if predictions.empty:
        st.warning("No latest forecast artifact found.")
    else:
        _render_latest_forecast(predictions)

    st.divider()
    if published_history_predictions.empty:
        st.subheader("Rolling-Origin Diagnostic")
        _render_evaluated_prediction_performance(
            diagnostic_predictions,
            diagnostic_scores,
            key_prefix="forecast_diagnostic",
            visible_days_label="Diagnostic visible days",
        )
    else:
        st.subheader("Published Forecast Performance")
        _render_evaluated_prediction_performance(
            published_history_predictions,
            published_history_scores,
            key_prefix="published_history",
            visible_days_label="Published performance visible days",
        )


def _render_latest_forecast(predictions: pd.DataFrame) -> None:
    frame = predictions.copy()
    frame["ds_utc"] = pd.to_datetime(frame["ds_utc"], utc=True)
    frame["forecast_origin_utc"] = pd.to_datetime(frame["forecast_origin_utc"], utc=True)
    frame = _with_model_display_names(frame)
    labels = sorted(frame["model"].dropna().astype(str).unique().tolist())
    selected = st.multiselect("Models", labels, default=labels, key="latest_forecast_models")
    if selected:
        frame = frame[frame["model"].isin(selected)]

    area_columns = st.columns(2)
    for index, area in enumerate(["DK1", "DK2"]):
        area_frame = frame[frame["area"] == area].copy()
        with area_columns[index]:
            st.subheader(area)
            if area_frame.empty:
                st.info("No forecast rows.")
                continue
            chart = area_frame.pivot_table(
                index="ds_utc",
                columns="model",
                values="y_pred",
                aggfunc="last",
            ).sort_index()
            st.line_chart(chart, height=300)

    st.subheader("Forecast Table")
    table = frame.sort_values(["area", "model", "ds_utc"])[
        [
            column
            for column in [
                "area",
                "forecast_origin_utc",
                "ds_utc",
                "model",
                "y_pred",
                "q10",
                "q50",
                "q90",
            ]
            if column in frame.columns
        ]
    ]
    st.dataframe(table, width="stretch", hide_index=True)

    cols = st.columns(2)
    point_rows = int(frame["y_pred"].notna().sum()) if "y_pred" in frame else 0
    cols[0].metric("Point forecast rows", point_rows)
    cols[1].metric("Quantile forecast rows", _complete_quantile_row_count(frame))


def _render_backtests(predictions: pd.DataFrame, scores: pd.DataFrame) -> None:
    st.subheader("Actual vs Forecast")

    if predictions.empty:
        st.warning("No evaluated prediction artifacts found.")
        score_frame = scores
    else:
        frame = _prepare_backtest_predictions(predictions)
        runs = sorted(frame["run_id"].dropna().astype(str).unique().tolist())
        default_run = runs[-1] if runs else None
        run_id = st.selectbox("Run", runs, index=runs.index(default_run) if default_run in runs else 0)
        run_frame = frame[frame["run_id"] == run_id].copy()

        visible_days = st.slider(
            "Backtest visible days",
            min_value=1,
            max_value=90,
            value=30,
            key=f"backtest_visible_days_{run_id}",
        )
        visible_frame = _filter_recent_backtest_rows(run_frame, visible_days)
        st.caption(_backtest_window_caption(visible_frame, total_rows=len(run_frame)))

        _render_predicted_vs_actual(visible_frame, key_prefix=f"backtest_{run_id}")
        score_frame = _score_table_for_predictions(visible_frame)

    st.divider()
    st.subheader("Metrics")
    _render_scores(score_frame)


def _render_evaluated_prediction_performance(
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    key_prefix: str,
    visible_days_label: str,
) -> None:
    if predictions.empty:
        if scores.empty:
            st.warning("No evaluated prediction artifacts found.")
        else:
            _render_scores(scores)
        return

    frame = _prepare_backtest_predictions(predictions)
    visible_days = st.slider(
        visible_days_label,
        min_value=1,
        max_value=90,
        value=30,
        key=f"{key_prefix}_visible_days",
    )
    visible_frame = _filter_recent_backtest_rows(frame, visible_days)
    st.caption(_backtest_window_caption(visible_frame, total_rows=len(frame)))
    _render_predicted_vs_actual(visible_frame, key_prefix=key_prefix)

    st.divider()
    st.subheader("Metrics")
    score_frame = _score_table_for_predictions(visible_frame)
    _render_scores(score_frame if not score_frame.empty else scores)


def _render_predicted_vs_actual(predictions: pd.DataFrame, *, key_prefix: str) -> None:
    if predictions.empty:
        st.warning("No evaluated prediction artifacts found.")
        return

    frame = predictions.copy()
    frame["ds_utc"] = pd.to_datetime(frame["ds_utc"], utc=True)
    frame["forecast_origin_utc"] = pd.to_datetime(frame["forecast_origin_utc"], utc=True)
    frame = add_prediction_diagnostics(frame)
    frame = _with_model_display_names(frame)

    areas = sorted(frame["area"].dropna().astype(str).unique().tolist())
    area = st.selectbox("Price area", areas, index=0, key=f"{key_prefix}_price_area")
    area_frame = frame[frame["area"] == area].copy()

    models = sorted(area_frame["model"].dropna().astype(str).unique().tolist())
    model = st.selectbox("Model", models, index=0, key=f"{key_prefix}_model")
    selected = area_frame[area_frame["model"] == model].sort_values("ds_utc").copy()

    cols = st.columns(4)
    cols[0].metric("Rows", len(selected))
    cols[1].metric("MAE", f"{selected['abs_error'].mean():.2f}")
    cols[2].metric("RMSE", f"{((selected['error'] ** 2).mean() ** 0.5):.2f}")
    cols[3].metric("Bias", f"{selected['error'].mean():.2f}")

    line_frame = selected.set_index("ds_utc")[["y", "y_pred"]].rename(
        columns={"y": "actual", "y_pred": "predicted"}
    )
    st.line_chart(line_frame, height=360)


def _render_scores(scores: pd.DataFrame) -> None:
    if scores.empty:
        st.warning("No recent model score artifact found.")
        return

    frame = _with_model_display_names(scores.copy())
    preferred = [
        column
        for column in [
            "model",
            "area",
            "rows",
            "evaluated_rows",
            "mae",
            "rmse",
            "bias",
            "coverage",
            "interval_width",
        ]
        if column in frame.columns
    ]
    st.dataframe(frame[preferred].sort_values(["area", "mae"]), width="stretch", hide_index=True)


def _prepare_backtest_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["ds_utc"] = pd.to_datetime(frame["ds_utc"], utc=True)
    frame["forecast_origin_utc"] = pd.to_datetime(frame["forecast_origin_utc"], utc=True)
    if "run_id" not in frame.columns:
        frame["run_id"] = "backtest"
    frame["run_id"] = frame["run_id"].fillna("backtest").astype(str)
    return frame


def _filter_recent_backtest_rows(predictions: pd.DataFrame, visible_days: int) -> pd.DataFrame:
    if predictions.empty:
        return predictions.copy()
    cutoff = predictions["ds_utc"].max() - pd.Timedelta(days=visible_days)
    return predictions[predictions["ds_utc"] >= cutoff].copy()


def _backtest_window_caption(predictions: pd.DataFrame, *, total_rows: int) -> str:
    if predictions.empty:
        return f"Showing 0 of {total_rows} rows."
    start = predictions["ds_utc"].min().strftime("%Y-%m-%d %H:%M UTC")
    end = predictions["ds_utc"].max().strftime("%Y-%m-%d %H:%M UTC")
    return f"Showing {len(predictions):,} of {total_rows:,} rows from {start} to {end}."


def _score_table_for_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    return model_score_table(add_prediction_diagnostics(predictions))


def _render_run_details(
    run: dict[str, Any],
    panel_path: str,
    dashboard_json: str,
    latest_predictions_path: str,
    recent_predictions_path: str,
    recent_scores_path: str,
    published_history_predictions_path: str,
    published_history_scores_path: str,
    backtest_dirs: list[str],
) -> None:
    paths = pd.DataFrame(
        [
            {"artifact": "price_panel", "path": str(panel_path), "exists": _resource_exists(panel_path)},
            {"artifact": "dashboard_json", "path": str(dashboard_json), "exists": _resource_exists(dashboard_json)},
            {
                "artifact": "latest_predictions",
                "path": str(latest_predictions_path),
                "exists": _resource_exists(latest_predictions_path),
            },
            {
                "artifact": "recent_predictions",
                "path": str(recent_predictions_path),
                "exists": _resource_exists(recent_predictions_path),
            },
            {"artifact": "recent_scores", "path": str(recent_scores_path), "exists": _resource_exists(recent_scores_path)},
            {
                "artifact": "published_history_predictions",
                "path": str(published_history_predictions_path),
                "exists": _resource_exists(published_history_predictions_path),
            },
            {
                "artifact": "published_history_scores",
                "path": str(published_history_scores_path),
                "exists": _resource_exists(published_history_scores_path),
            },
            *[
                {
                    "artifact": f"backtest_predictions:{Path(path).name}",
                    "path": str(Path(path) / "predictions.parquet"),
                    "exists": (Path(path) / "predictions.parquet").exists(),
                }
                for path in backtest_dirs
            ],
        ]
    )
    st.dataframe(paths, width="stretch", hide_index=True)
    st.json(_json_display_safe(run))


def _load_dashboard_payload(path: str) -> dict[str, Any]:
    local_path = _materialize_resource(path, required=False)
    if not local_path.exists():
        return {}
    return json.loads(local_path.read_text(encoding="utf-8"))


def _load_predictions(payload: dict[str, Any], fallback_path: str) -> pd.DataFrame:
    if payload.get("predictions"):
        frame = pd.DataFrame(payload["predictions"])
    else:
        frame = _load_parquet(fallback_path)
    return _parse_time_columns(frame, ["forecast_origin_utc", "ds_utc", "ds_local"])


def _load_scores(payload: dict[str, Any], fallback_path: str) -> pd.DataFrame:
    if payload.get("model_scores"):
        return pd.DataFrame(payload["model_scores"])
    return _load_parquet(fallback_path)


def _load_published_history_predictions(payload: dict[str, Any], fallback_path: str) -> pd.DataFrame:
    if payload.get("published_history_predictions"):
        frame = pd.DataFrame(payload["published_history_predictions"])
    else:
        frame = _load_parquet(fallback_path)
    return _parse_time_columns(
        frame,
        [
            "forecast_origin_utc",
            "ds_utc",
            "ds_local",
            "published_at_utc",
            "published_forecast_origin_utc",
        ],
    )


def _load_published_history_scores(payload: dict[str, Any], fallback_path: str) -> pd.DataFrame:
    if payload.get("published_history_scores"):
        return pd.DataFrame(payload["published_history_scores"])
    return _load_parquet(fallback_path)


def _load_recent_predictions(
    payload: dict[str, Any],
    fallback_path: str,
    legacy_backtest_dirs: list[str],
) -> pd.DataFrame:
    if payload.get("recent_predictions"):
        frame = pd.DataFrame(payload["recent_predictions"])
    else:
        frame = _load_parquet(fallback_path)
    if _is_evaluated_prediction_frame(frame):
        return _parse_time_columns(frame, ["forecast_origin_utc", "ds_utc", "ds_local"])
    return _load_backtest_predictions(legacy_backtest_dirs)


def _load_backtest_tab_predictions(
    payload: dict[str, Any],
    recent_predictions_path: str,
    backtest_dirs: list[str],
) -> pd.DataFrame:
    backtests = _load_backtest_predictions(backtest_dirs)
    if not backtests.empty:
        return backtests
    return _load_recent_predictions(payload, recent_predictions_path, backtest_dirs)


def _load_backtest_predictions(backtest_dirs: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in backtest_dirs:
        path = Path(run_dir) / "predictions.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        if "model_label" not in frame.columns and "model_name" in frame.columns:
            frame["model_label"] = frame["model_name"]
        if not {"ds_utc", "forecast_origin_utc", "area", "model_label", "y", "y_pred"}.issubset(frame.columns):
            continue
        frame["run_id"] = Path(run_dir).name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()
    output = pd.concat(frames, ignore_index=True)
    return _parse_time_columns(output, ["forecast_origin_utc", "ds_utc", "ds_local"])


def _is_evaluated_prediction_frame(frame: pd.DataFrame) -> bool:
    return not frame.empty and {
        "ds_utc",
        "forecast_origin_utc",
        "area",
        "model_label",
        "y",
        "y_pred",
    }.issubset(frame.columns)


def _load_parquet(path: str) -> pd.DataFrame:
    local_path = _materialize_resource(path, required=False)
    if not local_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(local_path)


def _parse_time_columns(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column in output.columns:
            output[column] = pd.to_datetime(output[column], utc=True)
    return output


def _resource_from_env(name: str, default: Path) -> str:
    return os.environ.get(name, str(default))


def _paths_from_env(name: str, default: list[Path]) -> list[str]:
    value = os.environ.get(name)
    if value is None:
        return [str(path) for path in default]
    if value.strip() == "":
        return []
    return [str(Path(item).expanduser()) for item in value.split(os.pathsep) if item]


def _env_bool(name: str, *, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resource_exists(resource: str) -> bool:
    return resource_exists(resource)


@_cache_data(ttl=CACHE_TTL_SECONDS)
def _materialize_resource(resource: str, *, required: bool = False) -> Path:
    cache_dir = Path(os.environ.get("DKENERGY_CACHE_DIR", "/tmp/dkenergy-dashboard-cache"))
    return materialize_uri(resource, cache_dir=cache_dir, required=required)


def _complete_quantile_row_count(frame: pd.DataFrame) -> int:
    quantile_columns = ["q10", "q50", "q90"]
    if not all(column in frame.columns for column in quantile_columns):
        return 0
    return int(frame[quantile_columns].notna().all(axis=1).sum())


def _with_model_display_names(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "model_label" in output.columns:
        output["model"] = output["model_label"].map(_model_display_name)
    return output


def _model_display_name(label: object) -> str:
    if pd.isna(label):
        return ""
    key = str(label)
    if key in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[key]
    if key.startswith("weather_catboost_"):
        suffix = key.removeprefix("weather_catboost_").replace("_", " ").title()
        return f"Weather Boosting {suffix}"
    if key.startswith("catboost_"):
        return key.removeprefix("catboost_").replace("_", " ").title()
    return key.replace("_", " ").title()


def _format_optional_timestamp(value: Any) -> str:
    if value in (None, ""):
        return "-"
    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M UTC")
    except (TypeError, ValueError):
        return str(value)


def _json_display_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_display_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_display_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


if __name__ == "__main__":
    main()
