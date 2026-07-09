PYTHON ?= python

-include .env

TOMORROW_COPENHAGEN := $(shell $(PYTHON) -c 'from datetime import datetime, timedelta; from zoneinfo import ZoneInfo; print((datetime.now(ZoneInfo("Europe/Copenhagen")).date() + timedelta(days=1)).isoformat())')
PRICE_START_COPENHAGEN := $(shell $(PYTHON) -c 'from datetime import datetime, timedelta; from zoneinfo import ZoneInfo; d=datetime.now(ZoneInfo("Europe/Copenhagen")).date() - timedelta(days=450); print(d.replace(day=1).isoformat())')
WEATHER_START_COPENHAGEN := $(shell $(PYTHON) -c 'from datetime import datetime, timedelta; from zoneinfo import ZoneInfo; d=datetime.now(ZoneInfo("Europe/Copenhagen")).date() - timedelta(days=90); print(d.replace(day=1).isoformat())')
EDS_START ?= $(PRICE_START_COPENHAGEN)
OPEN_METEO_START ?= $(WEATHER_START_COPENHAGEN)
OPEN_METEO_END ?= $(TOMORROW_COPENHAGEN)
FORECAST_AT_HOUR_UTC ?=
FORECAST_LOCAL_TIME ?= 12:00
MIN_TRAIN_DAYS ?= 60
SCORE_DAYS ?= 14
SCORE_MAX_ORIGINS ?= 7
SCORE_HOLDOUT_DAYS ?= 2
STREAMLIT_PORT ?= 8501
AWS_REGION ?= eu-central-1
AWS_WEB_ECR_REPOSITORY ?= dk-energy-forecasts-web
AWS_PIPELINE_ECR_REPOSITORY ?= dk-energy-forecasts-pipeline
AWS_IMAGE_TAG ?= $(shell git rev-parse --short HEAD)
AWS_ACCOUNT_ID ?= $(shell aws sts get-caller-identity --query Account --output text 2>/dev/null)
AWS_ECR_REGISTRY ?= $(AWS_ACCOUNT_ID).dkr.ecr.$(AWS_REGION).amazonaws.com
AWS_WEB_IMAGE_URI ?= $(AWS_ECR_REGISTRY)/$(AWS_WEB_ECR_REPOSITORY):$(AWS_IMAGE_TAG)
AWS_PIPELINE_IMAGE_URI ?= $(AWS_ECR_REGISTRY)/$(AWS_PIPELINE_ECR_REPOSITORY):$(AWS_IMAGE_TAG)
AWS_ARTIFACT_STORE_URI ?=
AWS_MODEL_ARTIFACT_URI ?=
AWS_ENABLE_PIPELINE_SCHEDULE ?= false
TF_STATE_BUCKET ?=
TF_STATE_KEY ?= dk-energy-forecasts/production.tfstate
TF_STATE_REGION ?= $(AWS_REGION)

EDS_END_ARG := $(if $(EDS_END),--end $(EDS_END),)
PUBLISH_MODELS_ARG := $(if $(PUBLISH_MODELS),--models $(PUBLISH_MODELS),)
FORECAST_AT_HOUR_UTC_ARG := $(if $(FORECAST_AT_HOUR_UTC),--at-hour-utc $(FORECAST_AT_HOUR_UTC),)

.PHONY: install install-app install-production test data-prices data-weather backtest-baseline publish daily daily-weather dashboard docker-build docker-dashboard docker-pipeline dry-run dry-run-weather aws-terraform-init aws-ensure-ecr aws-ecr-login aws-image aws-push aws-bootstrap-model aws-deploy clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-app:
	$(PYTHON) -m pip install -e ".[dev,app]"

install-production:
	$(PYTHON) -m pip install -e ".[dev,app,chronos,aws]"

test:
	$(PYTHON) -m pytest

data-prices:
	$(PYTHON) scripts/fetch_eds_prices.py --start $(EDS_START) $(EDS_END_ARG)
	$(PYTHON) scripts/build_price_panel.py --allow-incomplete-recent

data-weather:
	$(PYTHON) scripts/fetch_open_meteo_previous_runs.py --start $(OPEN_METEO_START) --end $(OPEN_METEO_END)
	$(PYTHON) scripts/build_open_meteo_weather_features.py --start $(OPEN_METEO_START) --end $(OPEN_METEO_END)

backtest-baseline:
	$(PYTHON) scripts/run_baseline_backtest.py --allow-incomplete-panel --forecast-local-time $(FORECAST_LOCAL_TIME) $(FORECAST_AT_HOUR_UTC_ARG) --min-train-days $(MIN_TRAIN_DAYS)

publish:
	$(PYTHON) scripts/run_publish_forecast.py --allow-incomplete-panel --forecast-local-time $(FORECAST_LOCAL_TIME) $(FORECAST_AT_HOUR_UTC_ARG) --min-train-days $(MIN_TRAIN_DAYS) --score-days $(SCORE_DAYS) --score-max-origins $(SCORE_MAX_ORIGINS) --score-holdout-days $(SCORE_HOLDOUT_DAYS) $(PUBLISH_MODELS_ARG)

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

aws-ecr-login:
	@test -n "$(AWS_ACCOUNT_ID)" || (echo "AWS_ACCOUNT_ID is required or AWS CLI must be authenticated" && exit 1)
	aws ecr get-login-password --region $(AWS_REGION) | docker login --username AWS --password-stdin $(AWS_ECR_REGISTRY)

aws-terraform-init:
	@test -n "$(TF_STATE_BUCKET)" || (echo "TF_STATE_BUCKET is required for the Terraform S3 backend" && exit 1)
	terraform -chdir=infra/aws init -backend-config="bucket=$(TF_STATE_BUCKET)" -backend-config="key=$(TF_STATE_KEY)" -backend-config="region=$(TF_STATE_REGION)"

aws-ensure-ecr: aws-terraform-init
	terraform -chdir=infra/aws apply -auto-approve -target=aws_ecr_repository.web -target=aws_ecr_repository.pipeline -var "aws_region=$(AWS_REGION)"

aws-image:
	docker build -f Dockerfile.web -t $(AWS_WEB_IMAGE_URI) .
	docker build -f Dockerfile.pipeline -t $(AWS_PIPELINE_IMAGE_URI) .

aws-push: aws-ecr-login aws-image
	docker push $(AWS_WEB_IMAGE_URI)
	docker push $(AWS_PIPELINE_IMAGE_URI)

aws-bootstrap-model:
	@test -n "$(AWS_MODEL_ARTIFACT_URI)" || (echo "AWS_MODEL_ARTIFACT_URI=s3://... is required" && exit 1)
	aws s3 sync artifacts/models/chronos2_lora_calendar_weather_ctx1024_v1/ $(AWS_MODEL_ARTIFACT_URI) --region $(AWS_REGION)

aws-deploy: aws-ensure-ecr aws-push
	terraform -chdir=infra/aws apply -var "aws_region=$(AWS_REGION)" -var "web_image_uri=$(AWS_WEB_IMAGE_URI)" -var "pipeline_image_uri=$(AWS_PIPELINE_IMAGE_URI)" -var "enable_pipeline_schedule=$(AWS_ENABLE_PIPELINE_SCHEDULE)"

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache runtime cloud_store
