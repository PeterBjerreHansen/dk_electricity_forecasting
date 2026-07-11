from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from dkenergy_forecast.types import require_columns, to_utc_timestamp


@dataclass(frozen=True)
class EvaluationInterval:
    """A named half-open evaluation interval: ``start <= timestamp < end``."""

    name: str
    start_utc: pd.Timestamp
    end_utc: pd.Timestamp
    timestamp_column: str = "forecast_origin_utc"

    def __post_init__(self) -> None:
        start = to_utc_timestamp(self.start_utc)
        end = to_utc_timestamp(self.end_utc)
        if not self.name.strip():
            raise ValueError("Evaluation interval name must not be empty")
        if not self.timestamp_column.strip():
            raise ValueError("Evaluation interval timestamp_column must not be empty")
        if start >= end:
            raise ValueError("Evaluation interval start_utc must be before end_utc")
        object.__setattr__(self, "start_utc", start)
        object.__setattr__(self, "end_utc", end)

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "start_utc": self.start_utc.isoformat(),
            "end_utc": self.end_utc.isoformat(),
            "timestamp_column": self.timestamp_column,
            "boundary": "[start_utc, end_utc)",
        }


@dataclass(frozen=True)
class FrozenDateSplits:
    intervals: dict[str, EvaluationInterval]
    sha256: str
    source_path: Path

    def select(self, name: str) -> EvaluationInterval:
        try:
            return self.intervals[name]
        except KeyError as error:
            available = ", ".join(sorted(self.intervals))
            raise ValueError(f"Unknown frozen split {name!r}; available splits: {available}") from error


def explicit_evaluation_interval(
    *,
    start_utc: object,
    end_utc: object,
    timestamp_column: str = "forecast_origin_utc",
) -> EvaluationInterval:
    return EvaluationInterval(
        name="explicit_evaluation_interval",
        start_utc=to_utc_timestamp(start_utc),
        end_utc=to_utc_timestamp(end_utc),
        timestamp_column=timestamp_column,
    )


def load_frozen_date_splits(path: str | Path) -> FrozenDateSplits:
    """Load and validate an immutable-by-contract date-split declaration.

    The JSON file must explicitly contain ``"frozen": true``. Its SHA-256 is
    returned for inclusion in evaluation reports, making later edits visible.
    """

    source_path = Path(path)
    raw = source_path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("Frozen split file must contain a JSON object")
    if payload.get("frozen") is not True:
        raise ValueError('Frozen split file must declare "frozen": true')
    raw_splits = payload.get("splits")
    if not isinstance(raw_splits, dict) or not raw_splits:
        raise ValueError("Frozen split file must contain a non-empty splits object")

    default_timestamp_column = str(payload.get("timestamp_column", "forecast_origin_utc"))
    intervals: dict[str, EvaluationInterval] = {}
    for name, values in raw_splits.items():
        if not isinstance(values, dict):
            raise ValueError(f"Frozen split {name!r} must be a JSON object")
        _require_split_fields(name, values)
        intervals[str(name)] = EvaluationInterval(
            name=str(name),
            start_utc=to_utc_timestamp(values["start_utc"]),
            end_utc=to_utc_timestamp(values["end_utc"]),
            timestamp_column=str(values.get("timestamp_column", default_timestamp_column)),
        )

    _validate_non_overlapping(intervals)
    return FrozenDateSplits(
        intervals=intervals,
        sha256=hashlib.sha256(raw).hexdigest(),
        source_path=source_path,
    )


def filter_evaluation_interval(
    predictions: pd.DataFrame,
    interval: EvaluationInterval,
) -> pd.DataFrame:
    require_columns(predictions, [interval.timestamp_column], "predictions")
    output = predictions.copy()
    timestamps = pd.to_datetime(output[interval.timestamp_column], utc=True)
    mask = timestamps.ge(interval.start_utc) & timestamps.lt(interval.end_utc)
    selected = output.loc[mask].copy()
    selected[interval.timestamp_column] = timestamps.loc[mask]
    if selected.empty:
        raise ValueError(
            f"No predictions fall inside evaluation interval {interval.name!r} "
            f"[{interval.start_utc.isoformat()}, {interval.end_utc.isoformat()})"
        )
    return selected.reset_index(drop=True)


def _require_split_fields(name: object, values: dict[str, Any]) -> None:
    missing = [key for key in ("start_utc", "end_utc") if key not in values]
    if missing:
        raise ValueError(f"Frozen split {name!r} is missing required fields: {missing}")


def _validate_non_overlapping(intervals: dict[str, EvaluationInterval]) -> None:
    timestamp_columns = {interval.timestamp_column for interval in intervals.values()}
    if len(timestamp_columns) != 1:
        raise ValueError("All frozen splits must use the same timestamp_column")

    ordered = sorted(intervals.values(), key=lambda interval: interval.start_utc)
    for previous, current in zip(ordered, ordered[1:]):
        if current.start_utc < previous.end_utc:
            raise ValueError(
                f"Frozen splits {previous.name!r} and {current.name!r} overlap"
            )
