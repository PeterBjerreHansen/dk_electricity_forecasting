from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BRONZE_EDS_RECORDS = "bronze_eds_price_records"
SILVER_PRICE_PANEL = "silver_price_panel_hourly"
SILVER_PRICE_PANEL_QA = "silver_price_panel_qa"
GOLD_FORECASTS = "gold_forecasts"
GOLD_BACKTEST_PREDICTIONS = "gold_backtest_predictions"
GOLD_FORECAST_SCORES = "gold_forecast_scores"
GOLD_MONITORING_CHECKS = "gold_monitoring_checks"
REGISTERED_MODEL = "dk_day_ahead_price_catboost"
CHAMPION_ALIAS = "champion"


@dataclass(frozen=True)
class Lakehouse:
    """Unity Catalog names for one deployment (one catalog and schema)."""

    catalog: str
    schema: str
    landing_volume: str = "landing"

    def table(self, name: str) -> str:
        return f"{self.catalog}.{self.schema}.{name}"

    @property
    def landing_root(self) -> Path:
        return Path("/Volumes") / self.catalog / self.schema / self.landing_volume

    @property
    def eds_raw_root(self) -> Path:
        return self.landing_root / "energi_data_service"

    @property
    def registered_model_name(self) -> str:
        return self.table(REGISTERED_MODEL)

    @property
    def champion_uri(self) -> str:
        return f"models:/{self.registered_model_name}@{CHAMPION_ALIAS}"
