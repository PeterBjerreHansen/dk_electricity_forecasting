from __future__ import annotations

import pandas as pd

from dkenergy_forecast.types import require_columns


def cheapest_k_hit_rate(
    predictions: pd.DataFrame,
    *,
    k: int = 6,
    y_col: str = "y",
    pred_col: str = "y_pred",
) -> pd.DataFrame:
    if k <= 0:
        raise ValueError("k must be positive")
    group_cols = ["forecast_origin_utc", "area", "local_date"]
    require_columns(predictions, group_cols + ["ds_utc", y_col, pred_col], "predictions")

    rows: list[dict[str, object]] = []
    for key, group in predictions.dropna(subset=[y_col, pred_col]).groupby(group_cols, dropna=False):
        candidate_count = len(group)
        selected_count = min(k, candidate_count)
        if selected_count == 0:
            continue
        actual_cheapest = set(
            group.sort_values([y_col, "ds_utc"]).head(selected_count)["ds_utc"].tolist()
        )
        predicted_cheapest = set(
            group.sort_values([pred_col, "ds_utc"]).head(selected_count)["ds_utc"].tolist()
        )
        hit_count = len(actual_cheapest & predicted_cheapest)
        rows.append(
            {
                "forecast_origin_utc": key[0],
                "area": key[1],
                "local_date": key[2],
                "k": k,
                "candidate_count": candidate_count,
                "selected_count": selected_count,
                "available_count": selected_count,
                "hit_count": hit_count,
                "hit_rate": hit_count / selected_count,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "forecast_origin_utc",
            "area",
            "local_date",
            "k",
            "candidate_count",
            "selected_count",
            "available_count",
            "hit_count",
            "hit_rate",
        ],
    )
