from __future__ import annotations

import pandas as pd


def weighted_median(values: pd.Series, weights: pd.Series) -> float | None:
    """Return the weighted median after dropping nulls and non-positive weights."""

    frame = pd.DataFrame({"value": values, "weight": weights}).dropna()
    frame = frame[frame["weight"] > 0].sort_values("value").reset_index(drop=True)
    if frame.empty:
        return None

    total_weight = frame["weight"].sum()
    if total_weight <= 0:
        return None

    cutoff = total_weight / 2
    index = frame["weight"].cumsum().ge(cutoff).idxmax()
    return float(frame.loc[index, "value"])
