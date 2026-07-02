# Forecasting Library Contract v1

This document defines the first implementation contract for the forecasting
side of the project. It is intentionally practical and notebook-friendly: the
library should be easy to import in exploratory work, while still enforcing the
core habits that matter for forecasting correctness.

Project package name:

```text
dkenergy_forecast
```

Primary input dataset:

```text
data/model_ready/price_panel_hourly_v1.parquet
```

Related data contract:

```text
docs/data-processing/energi_data_service_v1.md
```

## Status

This is a v1 contract, not a frozen public API.

The aim is to agree on data shapes, model protocols, leakage rules, and module
boundaries before the first baseline and backtest implementation. The contract
should support notebook-based work first, then evolve into scripts and
experiment configs once the modeling surface has settled.

## Design Principles

1. Forecasting code consumes model-ready tables only. It must not call external
   data APIs.
2. Inputs and outputs are pandas DataFrames. A notebook user should be able to
   inspect every intermediate table.
3. Simple statistical baselines are implemented locally with pandas/numpy.
4. External forecasting libraries are not part of the core implementation.
   Avoid dependencies such as Prophet, NeuralProphet, sktime, darts, and
   statsforecast for the first milestones.
5. LightGBM and CatBoost are acceptable optional model backends behind thin
   adapters.
6. Time handling must be explicit. UTC is the primary key; Danish local time is
   a feature and display timestamp.
7. Backtesting owns leakage control. Models should receive only the history and
   future frame that would have been available at the forecast origin.
8. The core library should not require YAML configs. Config files can wrap the
   same Python objects later.

## Forecasting Problem

The first target is hourly DK1/DK2 day-ahead electricity prices.

Each forecast run is defined by:

```text
target                 price_dkk_per_mwh / y
forecast unit          unique_id
forecast origin        forecast_origin_utc
forecast horizon       explicit future timestamps per unique_id
prediction granularity hourly
first quantiles        q10, q50, q90
```

The library should not hard-code "next 24 rows" as the only horizon shape.
Daylight-saving transitions mean local delivery days may contain 23, 24, or 25
hours. A backtest helper may offer a convenient next-day Danish-delivery
horizon, but the lower-level model API should accept an explicit future frame.

## Input Panel Contract

All model and backtest functions should accept a model-ready panel with at
least these columns:

```text
unique_id
ds_utc
ds_local
area
y
dataset_version
```

Recommended additional columns from the data-processing contract:

```text
local_date
local_hour
local_day_of_week
local_month
is_weekend
is_dst
utc_offset_hours
source_dataset
source_resolution_minutes
```

Input requirements:

1. `(unique_id, ds_utc)` is unique.
2. `ds_utc` is timezone-aware UTC or is normalized as UTC at load time.
3. Rows are hourly in UTC for each `unique_id`.
4. `y` may be negative and must not be clipped.
5. Missing historical target values fail backtests unless the caller explicitly
   selects an incomplete-data mode.

## Future Frame Contract

Models should predict for an explicit future frame with at least:

```text
unique_id
ds_utc
forecast_origin_utc
```

The future frame may also contain known-in-advance features:

```text
ds_local
local_date
local_hour
local_day_of_week
local_month
is_weekend
is_dst
utc_offset_hours
area
```

It must not contain future target values as model features. Backtesting code can
join `y` after predictions have been generated for evaluation.

## Tabular Feature Contract

The first ML feature builder is EDS-only and lives in:

```text
dkenergy_forecast.features.price_features
```

It may use:

1. calendar fields already known for the future horizon,
2. lagged price values whose source `ds_utc` is strictly before
   `forecast_origin_utc`,
3. rolling price summaries computed only from history before
   `forecast_origin_utc`,
4. seasonal medians computed only from history before `forecast_origin_utc`,
5. DK1-DK2 spread lags whose underlying price timestamps are strictly before
   `forecast_origin_utc`.

It must not use:

1. `y`, `price_dkk_per_mwh`, or `price_eur_per_mwh` from the forecast horizon as
   features,
2. lagged values whose source timestamp is equal to or after the forecast
   origin,
3. training examples whose target delivery timestamp is equal to or after the
   current forecast origin,
4. future realized weather observations.

Training matrices for rolling-origin ML models should be built from historic
forecast origins and should require complete training horizons before the
current origin by default. This is stricter than what may be available in real
day-ahead publication workflows, but it keeps the v1 backtest contract simple
and audit-friendly.

## Forecast Weather Feature Contract

Forecast weather features are allowed only as separate experiment artifacts.
They must not be written into the canonical price panel.

Current v1 source contract:

```text
docs/data-processing/open_meteo_weather_v1.md
```

Weather feature tables should be joined by:

```text
area
ds_utc
```

Every forecast-weather feature must carry an availability timestamp:

```text
forecast_available_at_utc
```

The join rule is:

```text
forecast_available_at_utc <= forecast_origin_utc
```

Any value that fails this rule must be null in the experiment frame. This is
especially important for Open-Meteo `previous_day1`: for a 10:00 UTC day-ahead
forecast origin, late delivery-hour values may still have
`forecast_available_at_utc` after the model origin.

The first weather feature builder lives in:

```text
dkenergy_forecast.features.weather_features
```

Rules:

1. Keep price/calendar/lag features separate from source-specific weather
   preprocessing.
2. Coverage-failing weather feature groups are excluded by default.
3. Individual weather rows with insufficient basket-point coverage are excluded
   even when the broader feature group passes.
4. Missing weather values stay null in v1; no forward fill or interpolation.
5. Weather ablations must skip model groups with no usable feature columns
   rather than silently running a price-only model under a weather label.
6. Ensemble weather summaries are derived only from availability-masked model
   feature columns.

## Forecast Output Contract

Prediction functions should return one row per forecasted `(unique_id, ds_utc,
forecast_origin_utc)`.

Required columns:

```text
unique_id
ds_utc
forecast_origin_utc
horizon
model_name
model_version
y_pred
```

Optional probabilistic columns:

```text
q10
q50
q90
```

Recommended audit columns for saved artifacts:

```text
run_id
dataset_version
created_at_utc
training_start_utc
training_end_utc
```

Rules:

1. `horizon` is the integer step within each origin and unique_id, ordered by
   `ds_utc`.
2. Deterministic baselines may output only `y_pred`.
3. If a deterministic model outputs quantiles, the quantile construction must
   be explicit, for example residual-based calibration. Do not silently copy
   `y_pred` into q10/q90 and present it as an interval.
4. For quantile models, `y_pred` should default to `q50`.

## Model Protocol

The first Python implementation should use a small, duck-typed protocol rather
than a heavy class hierarchy.

Recommended shape:

```python
class ForecastModel:
    model_name: str
    model_version: str

    def fit(self, history: pd.DataFrame) -> "ForecastModel":
        ...

    def predict(
        self,
        future: pd.DataFrame,
        history: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        ...
```

Guidance:

1. `fit` may be a no-op for simple baselines.
2. Baselines may use `history` during `predict` so notebook users can run them
   without creating fitted state.
3. ML models should store fitted backend models on the instance returned by
   `fit`.
4. The protocol should remain friendly to plain functions. A helper can wrap a
   function into this model shape if notebook experiments produce useful
   one-off baselines.

## Optional CatBoost Quantile Model

The first optional CatBoost implementation lives behind:

```text
dkenergy_forecast.models.CatBoostQuantileModel
scripts/run_catboost_backtest.py
```

CatBoost is not a core dependency. The package must remain importable and the
test suite must remain runnable without CatBoost installed. The CatBoost adapter
should raise a clear install message when the optional dependency is missing.

The v1 CatBoost model trains separate quantile regressors for q10, q50, and
q90. Rules:

1. `y_pred` is q50 by default.
2. Raw quantile crossing is measured.
3. Custom quantile dictionaries are canonicalized by quantile value, not by
   insertion order.
4. q10/q50/q90 may be sorted after prediction for metric stability, but the raw
   crossing rate must be kept as an audit column.
5. The adapter must consume the same explicit future-frame contract as the local
   baselines.

## Local Baseline Models

Implement the first statistical baselines locally. They should live in the
project package, use pandas/numpy, and have clear fallback behavior when
required history is unavailable.

### LagNaive

Purpose:

```text
same hour yesterday
same hour last week
```

Parameters:

```text
lag_hours=24 | 168
fallback=None | "last_available" | another ForecastModel
```

Prediction rule:

```text
y_pred(unique_id, ds_utc) = y(unique_id, ds_utc - lag_hours)
```

Notes:

1. The lookup is UTC-based to match the primary panel key.
2. Around DST transitions, this is still an exact hourly lag. A separate
   local-hour baseline can be added if experiments show it is useful.
3. Missing lagged observations should produce null predictions unless a
   fallback is explicitly configured.

### SeasonalRollingMedian

Purpose:

```text
rolling median for similar historical hours
```

Suggested parameters:

```text
lookback_days=28
seasonal_keys=("local_hour",)
min_periods=7
```

Prediction rule:

```text
For each future row, filter history to rows where:
  same unique_id
  ds_utc < forecast_origin_utc
  ds_utc >= forecast_origin_utc - lookback_days
  seasonal keys match the future row

y_pred = median(y)
```

Possible seasonal keys:

```text
("local_hour",)
("local_hour", "is_weekend")
("local_hour", "local_day_of_week")
```

The baseline should never include future rows in the rolling window, even if
the model-ready panel already contains actual values for the forecast horizon.

### EnsembleMean

Purpose:

```text
simple average of baseline predictions
```

This is useful as a notebook experiment once yesterday, last-week, and rolling
median predictions exist. It should combine already generated forecast columns
or child model outputs. It should not become a general stacking framework in
v1.

## Optional Gradient Boosting Models

LightGBM and CatBoost may be added behind thin adapters after the local
baselines and rolling backtest work.

Suggested model names:

```text
LightGBMQuantileModel
CatBoostQuantileModel
```

Contract:

1. Optional dependencies are imported only inside the adapter modules.
2. The core package can be imported without LightGBM or CatBoost installed.
3. Feature generation is separate from backend training.
4. Quantile training should produce q10, q50, and q90 for the first milestone.
5. Each adapter should expose feature importance when the backend supports it.
6. Backend-specific parameters are accepted through a plain dict so notebooks
   can experiment without changing the contract.

The first implementation can train one model per quantile. A later
implementation may use native multi-quantile support where available, as long
as the forecast output contract stays the same.

## Feature Contract

Feature builders should be plain, inspectable functions that return DataFrames.

Recommended categories:

```text
calendar features      known from ds_utc/ds_local
lag features           built from historical y only
rolling features       built from historical y only
spread features        built from DK1/DK2 historical values only
```

Rules:

1. Calendar features may be built for both history and future.
2. Lag and rolling features must be computed as-of the forecast origin.
3. Cross-area spread features must respect availability. For a future DK1 row,
   do not use future DK2 actuals.
4. Feature builders should make their required input columns explicit.
5. Do not bury important feature decisions inside model classes.

Suggested function shapes:

```python
def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    ...

def build_lag_features(
    history: pd.DataFrame,
    future: pd.DataFrame,
    lags: list[int],
) -> pd.DataFrame:
    ...

def build_rolling_features(
    history: pd.DataFrame,
    future: pd.DataFrame,
    windows: list[int],
    group_keys: list[str],
) -> pd.DataFrame:
    ...
```

## Backtesting Contract

Rolling-origin backtesting should be the first orchestration layer. It should
be usable from a notebook without requiring a run directory.

Recommended function shape:

```python
def rolling_origin_backtest(
    model_factory,
    panel: pd.DataFrame,
    origins: pd.DataFrame,
    horizon_builder,
    min_train_rows: int | None = None,
) -> pd.DataFrame:
    ...
```

Where:

```text
model_factory     callable returning a fresh model for each origin
origins           DataFrame with forecast_origin_utc and optional metadata
horizon_builder   callable that returns the future frame for one origin
```

Per-origin rules:

1. `history = panel[ds_utc < forecast_origin_utc]`.
2. Fit only on `history`.
3. Build the future frame without target values.
4. Generate predictions.
5. Join actual `y` after prediction for evaluation.
6. Keep `dataset_version` in prediction artifacts.

The first implementation may be single-process. Parallel execution can be
added after correctness tests exist.

## Origin And Horizon Helpers

The library should provide helper functions, not mandatory orchestration.

Useful helpers:

```text
make_daily_origins
make_next_utc_hours_horizon
make_danish_delivery_day_horizon
```

For day-ahead price work, `make_danish_delivery_day_horizon` should define the
delivery day by Danish local calendar date and then convert the resulting
timestamps back to UTC. This keeps DST behavior visible instead of pretending
every local day always has 24 hours.

## Evaluation Contract

Implement metrics locally.

Point metrics:

```text
MAE
RMSE
bias / mean error
```

Probabilistic metrics:

```text
pinball loss per quantile
p10-p90 interval coverage
p10-p90 average interval width
```

Simple value metrics:

```text
cheapest-k hour hit rate
mean realized price of selected cheapest-k hours
rank correlation within delivery day
```

Metric functions should accept prediction DataFrames that already contain
actual `y`.

Recommended output columns:

```text
metric_name
metric_value
model_name
dataset_version
group
group_value
```

Grouping should be optional and support at least:

```text
overall
area
unique_id
local_month
local_hour
```

## Artifact Contract

Notebook experiments do not need to save every artifact. Scripted backtests
should save enough to reproduce and compare runs.

Recommended layout:

```text
results/
  <run_id>/
    config.json
    predictions.parquet
    metrics.parquet
    feature_importance.parquet
    run_manifest.json
```

Minimum manifest fields:

```text
run_id
created_at_utc
dataset_path
dataset_version
model_name
model_version
forecast_origin_min_utc
forecast_origin_max_utc
prediction_row_count
git_commit
```

If the repo is not a Git worktree, `git_commit` may be null.

## Suggested Module Boundaries

Initial package shape:

```text
src/
  dkenergy_forecast/
    __init__.py
    types.py
    io.py
    models/
      __init__.py
      baselines.py
      lightgbm_quantile.py
      catboost_quantile.py
    features/
      __init__.py
      calendar.py
      lags.py
      rolling.py
    backtesting/
      __init__.py
      horizons.py
      rolling_origin.py
    evaluation/
      __init__.py
      point_metrics.py
      probabilistic_metrics.py
      value_metrics.py
```

Keep the first implementation small. It is acceptable for `baselines.py` and
`rolling_origin.py` to carry most of the early behavior while the experiments
are still fluid.

## Notebook-First Usage Sketch

The first implementation should make this style possible:

```python
import pandas as pd

from dkenergy_forecast.backtesting.horizons import (
    make_daily_origins,
    make_next_utc_hours_horizon,
)
from dkenergy_forecast.backtesting.rolling_origin import rolling_origin_backtest
from dkenergy_forecast.evaluation.point_metrics import mae, rmse
from dkenergy_forecast.models.baselines import LagNaive, SeasonalRollingMedian

panel = pd.read_parquet("data/model_ready/price_panel_hourly_v1.parquet")

origins = make_daily_origins(
    panel,
    start="2024-01-01",
    end="2024-03-01",
    at_hour_utc=10,
)

preds_week = rolling_origin_backtest(
    model_factory=lambda: LagNaive(lag_hours=168),
    panel=panel,
    origins=origins,
    horizon_builder=lambda panel, origin: make_next_utc_hours_horizon(
        panel=panel,
        forecast_origin_utc=origin,
        hours=24,
    ),
)

preds_median = rolling_origin_backtest(
    model_factory=lambda: SeasonalRollingMedian(
        lookback_days=28,
        seasonal_keys=("local_hour", "is_weekend"),
    ),
    panel=panel,
    origins=origins,
    horizon_builder=lambda panel, origin: make_next_utc_hours_horizon(
        panel=panel,
        forecast_origin_utc=origin,
        hours=24,
    ),
)

mae(preds_week)
rmse(preds_week)
```

This example is intentionally direct: the same objects can later be wrapped by
scripts or experiment configs without changing the underlying API.

## First Implementation Order

1. Add the `dkenergy_forecast` package skeleton.
2. Implement `LagNaive` for 24-hour and 168-hour baselines.
3. Implement `SeasonalRollingMedian`.
4. Implement basic horizon helpers and rolling-origin backtesting.
5. Implement MAE, RMSE, pinball loss, interval coverage, and interval width.
6. Add focused tests for leakage behavior, missing lag behavior, DST horizon
   construction, and metric calculations.
7. Add optional LightGBM/CatBoost quantile adapters only after the local
   baseline backtest is working.

## Explicit Non-Goals For v1

1. No automated experiment registry beyond simple result files.
2. No neural models.
3. No reconciliation for prices.
4. No production scheduler.
5. No external forecasting framework dependency.
6. No hidden feature generation inside model adapters.
