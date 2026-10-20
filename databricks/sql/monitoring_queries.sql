-- Queries for a Databricks SQL dashboard on the forecasting tables.
-- Replace workspace.dk_energy with your catalog and schema
-- (dev deploys use a prefixed schema such as workspace.dev_<user>_dk_energy).

-- 1. Tomorrow's published forecast next to the baselines
SELECT
  from_utc_timestamp(ds_utc, 'Europe/Copenhagen') AS delivery_hour_local,
  area,
  model_name,
  is_published,
  round(y_pred, 1) AS forecast_dkk_per_mwh
FROM workspace.dk_energy.gold_forecasts
WHERE forecast_origin_utc = (SELECT max(forecast_origin_utc) FROM workspace.dk_energy.gold_forecasts)
ORDER BY area, model_name, delivery_hour_local;

-- 2. Published forecast against the actual price, last 14 delivery days
SELECT
  from_utc_timestamp(f.ds_utc, 'Europe/Copenhagen') AS delivery_hour_local,
  f.area,
  f.model_name,
  round(f.y_pred, 1) AS forecast,
  round(p.y, 1) AS actual,
  round(f.y_pred - p.y, 1) AS error
FROM workspace.dk_energy.gold_forecasts AS f
JOIN workspace.dk_energy.silver_price_panel_hourly AS p
  ON f.unique_id = p.unique_id AND f.ds_utc = p.ds_utc
WHERE f.is_published
  AND f.ds_utc >= current_timestamp() - INTERVAL 14 DAYS
ORDER BY delivery_hour_local, f.area;

-- 3. Rolling MAE per model over scored origins (live, not backtest)
SELECT
  model_name,
  count(DISTINCT forecast_origin_utc) AS origins,
  round(sum(mae * rows) / sum(rows), 1) AS mae,
  round(sum(bias * rows) / sum(rows), 1) AS bias
FROM workspace.dk_energy.gold_forecast_scores
WHERE forecast_origin_utc >= current_timestamp() - INTERVAL 30 DAYS
GROUP BY model_name
ORDER BY mae;

-- 4. Latest monitoring checks
SELECT checked_at_utc, check, subject, status, round(value, 3) AS value, threshold, detail
FROM workspace.dk_energy.gold_monitoring_checks
WHERE checked_at_utc = (SELECT max(checked_at_utc) FROM workspace.dk_energy.gold_monitoring_checks)
ORDER BY check, subject;

-- 5. Backtest results behind each training run
SELECT
  mlflow_run_id,
  model_label,
  count(DISTINCT forecast_origin_utc) AS origins,
  round(avg(abs(y_pred - y)), 1) AS mae
FROM workspace.dk_energy.gold_backtest_predictions
GROUP BY mlflow_run_id, model_label
ORDER BY mlflow_run_id DESC, mae;
