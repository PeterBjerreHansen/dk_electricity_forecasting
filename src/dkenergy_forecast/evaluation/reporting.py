from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_model_comparison(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write deterministic, diff-friendly JSON and Markdown reports."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": destination / "model_comparison.json",
        "markdown": destination / "model_comparison.md",
    }
    safe_report = json_safe(report)
    paths["json"].write_text(
        json.dumps(safe_report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(
        render_model_comparison_markdown(safe_report),
        encoding="utf-8",
    )
    return paths


def render_model_comparison_markdown(report: dict[str, Any]) -> str:
    reference = str(report["reference_model"])
    comparison = str(report["comparison_model"])
    interval = report["evaluation_interval"]
    pairing = report["pairing"]
    overall = report["overall"]
    bootstrap = report["bootstrap_confidence_intervals"]

    lines = [
        "# Forecast model comparison",
        "",
        (
            f"`{_escape(comparison)}` is compared with reference "
            f"`{_escape(reference)}` on `{interval['timestamp_column']}` in "
            f"`[{interval['start_utc']}, {interval['end_utc']})`."
        ),
        "",
        (
            f"The comparison contains {pairing['paired_rows']} exactly paired rows "
            f"across {pairing['origin_count']} forecast origins. Differences are "
            "comparison minus reference; negative differences favor the comparison "
            "for error and scoring metrics."
        ),
        "",
        "## Overall metrics",
        "",
        "| Role | Model | MAE | RMSE | Bias | WIS | Calibration error | 80% coverage |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for role in ("reference", "comparison"):
        model = overall[role]
        metrics = model["metrics"]
        lines.append(
            "| {role} | {model} | {mae} | {rmse} | {bias} | {wis} | {calibration} | {coverage} |".format(
                role=role,
                model=_escape(model["model_label"]),
                mae=_number(metrics["mae"]),
                rmse=_number(metrics["rmse"]),
                bias=_number(metrics["bias"]),
                wis=_number(metrics["weighted_interval_score"]),
                calibration=_number(metrics["calibration_error"], digits=4),
                coverage=_number(metrics["coverage_80"], digits=4),
            )
        )

    lines.extend(
        [
            "",
            "## Origin-block bootstrap",
            "",
            "| Metric difference | Mean | Lower | Upper | Confidence | Block |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for metric, result in bootstrap.items():
        if result is None:
            lines.append(f"| {_escape(metric)} | n/a | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            "| {metric} | {mean} | {lower} | {upper} | {confidence} | {block} |".format(
                metric=_escape(metric),
                mean=_number(result["mean"]),
                lower=_number(result["lower"]),
                upper=_number(result["upper"]),
                confidence=_number(result["confidence"], digits=3),
                block=result["block_length"],
            )
        )

    lines.extend(
        [
            "",
            "## Per-origin differences",
            "",
            "| Forecast origin (UTC) | Rows | Reference MAE | Comparison MAE | MAE difference | WIS difference |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["per_origin_differences"]:
        lines.append(
            "| {origin} | {rows} | {reference_mae} | {comparison_mae} | {mae_difference} | {wis_difference} |".format(
                origin=_escape(row["forecast_origin_utc"]),
                rows=row["rows"],
                reference_mae=_number(row["reference_mae"]),
                comparison_mae=_number(row["comparison_mae"]),
                mae_difference=_number(row["mae_difference"]),
                wis_difference=_number(row["weighted_interval_score_difference"]),
            )
        )

    lines.extend(
        [
            "",
            "## Stratified differences",
            "",
            "| Stratum | Value | Rows | MAE difference | WIS difference | Calibration difference |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in report["stratification"]["differences"]:
        lines.append(
            "| {stratum} | {value} | {rows} | {mae} | {wis} | {calibration} |".format(
                stratum=_escape(row["stratum"]),
                value=_escape(row["stratum_value"]),
                rows=row["rows"],
                mae=_number(row["mae_difference"]),
                wis=_number(row["weighted_interval_score_difference"]),
                calibration=_number(row["calibration_error_difference"], digits=4),
            )
        )

    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Model rows match exactly on the recorded pairing keys.",
            "- WIS uses q10, q50, and q90; calibration error is the mean absolute quantile calibration error.",
            "- Confidence intervals use a deterministic circular moving-block bootstrap over chronological forecast origins.",
            "- Extreme-price groups use the absolute-target threshold recorded in the JSON report.",
            "- The report is descriptive and does not select or deploy a model.",
            "",
        ]
    )
    return "\n".join(lines)


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if hasattr(value, "item"):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _number(value: object, *, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
