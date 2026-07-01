from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dkenergy_forecast.types import PANEL_REQUIRED_COLUMNS, normalize_utc_column, require_columns


def load_price_panel(
    path: str | Path,
    qa_path: str | Path | None = None,
    *,
    require_final_historical: bool = True,
) -> pd.DataFrame:
    """Load and validate a model-ready hourly price panel."""

    panel = pd.read_parquet(path)
    require_columns(panel, PANEL_REQUIRED_COLUMNS, "price panel")
    panel = normalize_utc_column(panel, "ds_utc")

    duplicate_count = int(panel.duplicated(["unique_id", "ds_utc"]).sum())
    if duplicate_count:
        raise ValueError(
            "Price panel contains duplicate (unique_id, ds_utc) rows: "
            f"{duplicate_count}"
        )

    if qa_path is not None:
        qa = json.loads(Path(qa_path).read_text(encoding="utf-8"))
        artifact_status = qa.get("artifact_status")
        if require_final_historical and artifact_status != "final_historical":
            raise ValueError(
                "Price panel QA artifact_status is not final_historical: "
                f"{artifact_status!r}"
            )

    return panel.sort_values(["unique_id", "ds_utc"]).reset_index(drop=True)
