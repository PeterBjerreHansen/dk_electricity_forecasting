from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

from dkenergy_forecast.layout import PROJECT_ROOT
from dkenergy_forecast.types import COPENHAGEN_TZ, to_utc_timestamp


RunKind = Literal["live", "replay"]


@dataclass(frozen=True)
class ForecastRequest:
    """The complete time contract for one delivery-day forecast."""

    delivery_date_local: date
    information_cutoff_utc: pd.Timestamp
    decision_deadline_utc: pd.Timestamp
    generated_at_utc: pd.Timestamp
    run_kind: RunKind = "live"

    def __post_init__(self) -> None:
        delivery_date = pd.Timestamp(self.delivery_date_local).date()
        information_cutoff = to_utc_timestamp(self.information_cutoff_utc)
        decision_deadline = to_utc_timestamp(self.decision_deadline_utc)
        generated_at = to_utc_timestamp(self.generated_at_utc)
        if self.run_kind not in {"live", "replay"}:
            raise ValueError(f"Unsupported run_kind: {self.run_kind!r}")
        if information_cutoff > generated_at:
            raise ValueError("information_cutoff_utc cannot be after generated_at_utc")
        if self.run_kind == "live" and generated_at > decision_deadline:
            raise ValueError(
                "Live forecast generation started after its decision deadline: "
                f"generated_at_utc={generated_at.isoformat()}, "
                f"decision_deadline_utc={decision_deadline.isoformat()}"
            )

        cutoff_delivery_date = information_cutoff.tz_convert(COPENHAGEN_TZ).date()
        if delivery_date <= cutoff_delivery_date:
            raise ValueError(
                "delivery_date_local must be after the information-cutoff date: "
                f"delivery_date_local={delivery_date}, cutoff_date_local={cutoff_delivery_date}"
            )

        object.__setattr__(self, "delivery_date_local", delivery_date)
        object.__setattr__(self, "information_cutoff_utc", information_cutoff)
        object.__setattr__(self, "decision_deadline_utc", decision_deadline)
        object.__setattr__(self, "generated_at_utc", generated_at)

    @property
    def forecast_origin_utc(self) -> pd.Timestamp:
        """Compatibility name: the forecast origin is the information cutoff."""

        return self.information_cutoff_utc

    def origin_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "forecast_origin_utc": [self.forecast_origin_utc],
                "information_cutoff_utc": [self.information_cutoff_utc],
                "delivery_date_local": [self.delivery_date_local.isoformat()],
            }
        )


@dataclass(frozen=True)
class ProductionConfig:
    """One explicit production model release and one fixed emergency fallback."""

    primary_model: str
    primary_artifact_path: Path
    fallback_model: str
    schema_version: int = 1


def load_production_config(
    path: str | Path = PROJECT_ROOT / "config" / "production.json",
    *,
    runtime_root: str | Path = PROJECT_ROOT,
    artifact_path_override: str | Path | None = None,
) -> ProductionConfig:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(
            "Unsupported production configuration schema: "
            f"{payload.get('schema_version')!r}"
        )
    primary = payload.get("primary")
    fallback = payload.get("fallback")
    if not isinstance(primary, dict) or not isinstance(fallback, dict):
        raise ValueError("Production configuration requires primary and fallback objects")
    primary_model = _required_text(primary, "model")
    fallback_model = _required_text(fallback, "model")
    artifact_value = artifact_path_override or _required_text(primary, "artifact_path")
    artifact_path = Path(artifact_value)
    if not artifact_path.is_absolute():
        artifact_path = Path(runtime_root) / artifact_path
    if primary_model == fallback_model:
        raise ValueError("Primary and fallback production models must be different")
    return ProductionConfig(
        schema_version=1,
        primary_model=primary_model,
        primary_artifact_path=artifact_path,
        fallback_model=fallback_model,
    )


def _required_text(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Production configuration field {key!r} must be non-empty text")
    return value.strip()


def parse_delivery_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    return pd.Timestamp(value).date()
