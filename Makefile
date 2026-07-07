PYTHON ?= python

-include .env

TOMORROW_COPENHAGEN := $(shell $(PYTHON) -c 'from datetime import datetime, timedelta; from zoneinfo import ZoneInfo; print((datetime.now(ZoneInfo("Europe/Copenhagen")).date() + timedelta(days=1)).isoformat())')
EDS_START ?= 2024-07-01
OPEN_METEO_START ?= 2024-07-01
OPEN_METEO_END ?= $(TOMORROW_COPENHAGEN)
FORECAST_AT_HOUR_UTC ?= 10
MIN_TRAIN_DAYS ?= 60
SCORE_DAYS ?= 14
SCORE_MAX_ORIGINS ?= 7
SCORE_HOLDOUT_DAYS ?= 2
STREAMLIT_PORT ?= 8501

EDS_END_ARG := $(if $(EDS_END),--end $(EDS_END),)
PUBLISH_MODELS_ARG := $(if $(PUBLISH_MODELS),--models $(PUBLISH_MODELS),)

.PHONY: install install-app install-production test data-prices data-weather backtest-baseline publish daily daily-weather dashboard docker-build docker-dashboard docker-pipeline dry-run dry-run-weather clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-app:
	$(PYTHON) -m pip install -e ".[dev,app]"

install-production:
	$(PYTHON) -m pip install -e ".[dev,app,chronos]"

test:
	$(PYTHON) -m pytest

data-prices:
	$(PYTHON) scripts/fetch_eds_prices.py --start $(EDS_START) $(EDS_END_ARG)
	$(PYTHON) scripts/build_price_panel.py --allow-incomplete-recent

data-weather:
	$(PYTHON) scripts/fetch_open_meteo_previous_runs.py --start $(OPEN_METEO_START) --end $(OPEN_METEO_END)
	$(PYTHON) scripts/build_open_meteo_weather_features.py

backtest-baseline:
	$(PYTHON) scripts/run_baseline_backtest.py --allow-incomplete-panel --at-hour-utc $(FORECAST_AT_HOUR_UTC) --min-train-days $(MIN_TRAIN_DAYS)

publish:
	$(PYTHON) scripts/run_publish_forecast.py --allow-incomplete-panel --at-hour-utc $(FORECAST_AT_HOUR_UTC) --min-train-days $(MIN_TRAIN_DAYS) --score-days $(SCORE_DAYS) --score-max-origins $(SCORE_MAX_ORIGINS) --score-holdout-days $(SCORE_HOLDOUT_DAYS) $(PUBLISH_MODELS_ARG)

daily:
	$(PYTHON) scripts/run_daily_pipeline.py

daily-weather:
	$(PYTHON) scripts/run_daily_pipeline.py --with-weather

dry-run:
	$(PYTHON) scripts/run_daily_pipeline.py --dry-run

dry-run-weather:
	$(PYTHON) scripts/run_daily_pipeline.py --dry-run --with-weather

dashboard:
	$(PYTHON) -m streamlit run app/streamlit_app.py --server.port $(STREAMLIT_PORT)

docker-build:
	docker compose build

docker-dashboard:
	docker compose up dashboard

docker-pipeline:
	docker compose --profile jobs run --rm pipeline

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
