from __future__ import annotations

from datetime import date

from dkenergy_data.build.eds_prices_v1 import STITCH_BOUNDARY_DATE
from dkenergy_data.sources.energidataservice import iter_date_chunks


def iter_price_source_windows(
    start: date,
    end: date,
    chunk_months: int,
) -> list[tuple[str, date, date]]:
    """Split a local date range into EDS request windows per source dataset.

    Hourly ``Elspotprices`` cover delivery before the October 2025 stitch
    boundary; quarter-hourly ``DayAheadPrices`` cover delivery from it.
    """

    windows: list[tuple[str, date, date]] = []

    if start < STITCH_BOUNDARY_DATE:
        elspot_end = min(end, STITCH_BOUNDARY_DATE)
        for window_start, window_end in iter_date_chunks(start, elspot_end, chunk_months):
            windows.append(("Elspotprices", window_start, window_end))

    if end > STITCH_BOUNDARY_DATE:
        day_ahead_start = max(start, STITCH_BOUNDARY_DATE)
        for window_start, window_end in iter_date_chunks(day_ahead_start, end, chunk_months):
            windows.append(("DayAheadPrices", window_start, window_end))

    return windows
