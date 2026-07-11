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


def write_evaluation_report(
    report: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write deterministic, diff-friendly JSON and Markdown reports."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": destination / "evaluation_report.json",
        "markdown": destination / "evaluation_report.md",
    }
    safe_report = json_safe(report)
    paths["json"].write_text(
        json.dumps(safe_report, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(
        render_evaluation_markdown(safe_report),
        encoding="utf-8",
    )
    return paths


def render_evaluation_markdown(report: dict[str, Any]) -> str:
    candidate = str(report["candidate_label"])
    champion = str(report["champion_label"])
    promotion = report["promotion"]
    interval = report["evaluation_interval"]
    pairing = report["pairing"]
    overall = report["overall"]
    bootstrap = report["bootstrap_confidence_intervals"]

    lines = [
        "# Forecast model evaluation",
        "",
        f"**Decision:** `{promotion['decision']}`",
        "",
        (
            f"Candidate `{_escape(candidate)}` was compared with champion "
            f"`{_escape(champion)}` on `{interval['timestamp_column']}` in "
            f"`[{interval['start_utc']}, {interval['end_utc']})`."
        ),
        "",
        (
            f"The comparison contains {pairing['paired_rows']} exactly paired rows "
            f"across {pairing['origin_count']} forecast origins."
        ),
        "",
        "## Overall metrics",
        "",
        "| Model | MAE | RMSE | Bias | WIS | Calibration error | 80% coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label in (candidate, champion):
        metrics = overall[label]
        lines.append(
            "| {label} | {mae} | {rmse} | {bias} | {wis} | {calibration} | {coverage} |".format(
                label=_escape(label),
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
            "Negative differences favor the candidate.",
            "",
            "| Metric (candidate − champion) | Mean | Lower | Upper | Confidence | Block |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for name, result in bootstrap.items():
        if result is None:
            lines.append(f"| {_escape(name)} | n/a | n/a | n/a | n/a | n/a |")
        else:
            lines.append(
                "| {name} | {mean} | {lower} | {upper} | {confidence} | {block} |".format(
                    name=_escape(name),
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
            "## Promotion checks",
            "",
            "| Check | Result |",
            "|---|---|",
        ]
    )
    for check in promotion["checks"]:
        result = "skipped" if check["passed"] is None else (
            "pass" if check["passed"] else "fail"
        )
        lines.append(f"| {_escape(check['name'])} | {result} |")

    lines.extend(
        [
            "",
            "## Per-origin paired comparison",
            "",
            "| Forecast origin (UTC) | Rows | Candidate MAE | Champion MAE | Difference | Winner |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["paired_origin_comparisons"]:
        lines.append(
            "| {origin} | {rows} | {candidate_mae} | {champion_mae} | {difference} | {winner} |".format(
                origin=_escape(row["forecast_origin_utc"]),
                rows=row["rows"],
                candidate_mae=_number(row["candidate_mae"]),
                champion_mae=_number(row["champion_mae"]),
                difference=_number(row["mae_difference"]),
                winner=_escape(row["mae_winner"]),
            )
        )

    lines.extend(
        [
            "",
            "## Subgroup MAE guardrails",
            "",
            "Groups below the policy's minimum row count are shown as skipped.",
            "",
            "| Stratum | Value | Rows | Candidate MAE | Champion MAE | Relative change | Result |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
    )
    for row in report["stratification"]["guardrails"]:
        result = "skipped" if row["passed"] is None else (
            "pass" if row["passed"] else "fail"
        )
        lines.append(
            "| {stratum} | {value} | {rows} | {candidate_mae} | {champion_mae} | {change} | {result} |".format(
                stratum=_escape(row["stratum"]),
                value=_escape(row["stratum_value"]),
                rows=row["rows"],
                candidate_mae=_number(row["candidate_mae"]),
                champion_mae=_number(row["champion_mae"]),
                change=_percentage(row["mae_relative_change"]),
                result=result,
            )
        )

    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Model rows must match exactly on the recorded pairing keys.",
            "- WIS uses q10, q50, and q90; calibration error is the mean absolute quantile calibration error.",
            "- Confidence intervals use a deterministic circular moving-block bootstrap over chronological forecast origins.",
            "- Extreme-price groups use the absolute-target threshold recorded in the JSON report.",
            "- Lower MAE, WIS, and calibration error are better.",
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


def _percentage(value: object) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(numeric):
        return "n/a"
    return f"{numeric:.1%}"


def _escape(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
