from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dkenergy_data.build.eds_prices_v1 import (
    SOURCE_SPECS,
    RawBatch,
    build_price_panel_from_batches,
)
from dkenergy_forecast.io import validate_price_panel
from dkenergy_forecast.types import (
    COPENHAGEN_TZ,
    PRICE_AVAILABILITY_COLUMN,
    normalize_utc_column,
)


# Delta stores instants without a zone, so the Copenhagen-local timestamp is
# dropped on write and rebuilt from ds_utc on read.
LOCAL_ONLY_COLUMNS = ["ds_local"]


def raw_batches_from_bronze(bronze: pd.DataFrame) -> dict[str, list[RawBatch]]:
    """Regroup bronze record rows into the raw batches the panel builder expects."""

    batches: dict[str, list[RawBatch]] = {dataset: [] for dataset in SOURCE_SPECS}
    ordered = bronze.sort_values(["batch_id", "record_index"])
    for (batch_id, dataset, retrieved_at, raw_path), group in ordered.groupby(
        ["batch_id", "source_dataset", "retrieved_at_utc", "raw_path"], sort=True
    ):
        batches.setdefault(dataset, []).append(
            RawBatch(
                source_dataset=dataset,
                batch_id=batch_id,
                retrieved_at_utc=retrieved_at,
                raw_path=Path(raw_path),
                records=[json.loads(record) for record in group["record_json"]],
            )
        )
    return batches


def build_silver_price_panel(bronze: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild the canonical hourly panel from bronze records.

    This reuses the local pipeline's builder, including its stitching of the
    hourly and quarter-hourly sources, revision handling and QA report.
    """

    if bronze.empty:
        raise ValueError("Bronze table has no EDS records")
    batches = raw_batches_from_bronze(bronze)
    built = build_price_panel_from_batches(
        batches.get("Elspotprices", []),
        batches.get("DayAheadPrices", []),
        allow_incomplete_recent=True,
    )
    return validate_price_panel(built.panel), built.qa


def panel_to_delta_frame(panel: pd.DataFrame) -> pd.DataFrame:
    output = panel.drop(columns=[c for c in LOCAL_ONLY_COLUMNS if c in panel.columns])
    for column in ["ds_utc", PRICE_AVAILABILITY_COLUMN]:
        output = normalize_utc_column(output, column)
    return output.reset_index(drop=True)


def panel_from_delta_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Restore the in-memory panel contract from a Delta table read."""

    output = normalize_utc_column(frame, "ds_utc")
    output = normalize_utc_column(output, PRICE_AVAILABILITY_COLUMN)
    output["ds_local"] = output["ds_utc"].dt.tz_convert(COPENHAGEN_TZ)
    return validate_price_panel(output)


def qa_record(qa: dict[str, Any], *, built_at_utc: datetime | None = None) -> pd.DataFrame:
    built_at = built_at_utc or datetime.now(timezone.utc)
    return pd.DataFrame(
        [
            {
                "built_at_utc": pd.Timestamp(built_at).tz_convert("UTC"),
                "row_count": int(qa.get("row_count", 0)),
                "min_ds_utc": str(qa.get("min_ds_utc")),
                "max_ds_utc": str(qa.get("max_ds_utc")),
                "artifact_status": str(qa.get("artifact_status")),
                "qa_json": json.dumps(qa, default=str, sort_keys=True),
            }
        ]
    )
