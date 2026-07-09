from __future__ import annotations

from dkenergy_forecast.layout import (
    CHRONOS_LORA_ARTIFACT_ID,
    CHRONOS_LORA_WEATHER_MODEL_LABEL,
    WEATHER_FEATURES_LONG_FILENAME,
    runtime_layout,
)


def test_runtime_layout_centralizes_runtime_paths(tmp_path) -> None:
    layout = runtime_layout(tmp_path)

    assert CHRONOS_LORA_ARTIFACT_ID == CHRONOS_LORA_WEATHER_MODEL_LABEL
    assert layout.price_panel == tmp_path / "data" / "model_ready" / "price_panel_hourly_v1.parquet"
    assert layout.price_panel_qa == tmp_path / "data" / "model_ready" / "price_panel_hourly_v1.qa.json"
    assert layout.weather_features_long == tmp_path / "data" / "features" / WEATHER_FEATURES_LONG_FILENAME
    assert layout.chronos_model_artifact == tmp_path / "artifacts" / "models" / CHRONOS_LORA_ARTIFACT_ID
    assert layout.published_history == tmp_path / "results" / "published_forecast_history"
