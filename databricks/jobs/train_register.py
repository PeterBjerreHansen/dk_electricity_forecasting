"""Weekly: backtest the CatBoost candidate against the baselines, train it, register it in
Unity Catalog, and move the champion alias only if it beats the fallback."""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from dkenergy_databricks.cli import lakehouse, parser, run_date

from dkenergy_databricks.lakehouse import (
    CHAMPION_ALIAS,
    GOLD_BACKTEST_PREDICTIONS,
    SILVER_PRICE_PANEL,
)
from dkenergy_databricks.mlflow_model import log_day_ahead_model
from dkenergy_databricks.silver import panel_from_delta_frame
from dkenergy_databricks.spark_io import read_table, spark_session, write_table
from dkenergy_databricks.training import (
    CANDIDATE_MODEL_LABEL,
    DATABRICKS_CATBOOST_CONFIG,
    backtest_candidate,
    backtest_models,
    baseline_model_factories,
    build_training_pool,
    choose_backtest_origins,
    forecast_origin_for,
    promotion_decision,
    summarize_backtest,
    train_candidate,
)

BACKTEST_COLUMNS = [
    "mlflow_run_id",
    "model_label",
    "unique_id",
    "ds_utc",
    "forecast_origin_utc",
    "horizon",
    "y",
    "y_pred",
]


def main() -> None:
    cli = parser(__doc__)
    cli.add_argument("--experiment", default="", help="MLflow experiment path")
    cli.add_argument("--backtest-days", type=int, default=28)
    cli.add_argument("--backtest-origins", type=int, default=14)
    cli.add_argument("--min-relative-improvement", type=float, default=0.0)
    args = cli.parse_args()
    lake = lakehouse(args)
    spark = spark_session()

    mlflow.set_registry_uri("databricks-uc")
    if args.experiment:
        mlflow.set_experiment(args.experiment)

    panel = panel_from_delta_frame(read_table(spark, lake.table(SILVER_PRICE_PANEL)))
    origin = forecast_origin_for(run_date(args))
    config = DATABRICKS_CATBOOST_CONFIG

    with mlflow.start_run(run_name=f"catboost-{origin:%Y%m%d}") as run:
        mlflow.log_params(
            {
                "forecast_origin_utc": origin.isoformat(),
                "feature_set": config.feature_set,
                "target_mode": config.target_mode,
                "training_origin_days": config.training_origin_days,
                "recency_half_life_days": config.recency_half_life_days,
                **{f"catboost_{key}": value for key, value in config.params.items()},
            }
        )
        mlflow.set_tags(
            {
                "panel_max_ds_utc": str(panel["ds_utc"].max()),
                "panel_rows": str(len(panel)),
            }
        )

        origins = choose_backtest_origins(
            panel,
            days=args.backtest_days,
            max_origins=args.backtest_origins,
        )
        # One feature build covers every backtest origin and the final fit.
        pool = build_training_pool(panel, origins["forecast_origin_utc"].min(), origin, config)
        backtest = pd.concat(
            [
                backtest_candidate(panel, origins, pool, config),
                backtest_models(panel, baseline_model_factories(), origins),
            ],
            ignore_index=True,
        )
        summary = summarize_backtest(backtest)
        for row in summary.itertuples(index=False):
            mlflow.log_metrics(
                {
                    f"backtest_mae_{row.model_label}": row.mae,
                    f"backtest_rmse_{row.model_label}": row.rmse,
                    f"backtest_bias_{row.model_label}": row.bias,
                }
            )
        with tempfile.TemporaryDirectory() as tmp:
            summary_path = Path(tmp) / "backtest_summary.csv"
            summary.to_csv(summary_path, index=False)
            mlflow.log_artifact(str(summary_path))
        backtest["mlflow_run_id"] = run.info.run_id
        write_table(
            spark,
            backtest[BACKTEST_COLUMNS],
            lake.table(GOLD_BACKTEST_PREDICTIONS),
            mode="append",
        )
        print(summary.to_string(index=False))

        decision = promotion_decision(summary, min_relative_improvement=args.min_relative_improvement)
        mlflow.log_metric("promoted", int(decision.promote))
        mlflow.set_tag("promotion_reason", decision.reason)

        fitted, training = train_candidate(panel, origin, config, pool=pool)
        mlflow.log_metric("training_rows", fitted.training_rows)
        model_info = log_day_ahead_model(
            fitted,
            input_example=training,
            registered_model_name=lake.registered_model_name,
        )

    client = MlflowClient()
    version = str(model_info.registered_model_version)
    client.set_model_version_tag(lake.registered_model_name, version, "backtest_mae", f"{decision.candidate_mae:.2f}")
    client.set_model_version_tag(lake.registered_model_name, version, "fallback_backtest_mae", f"{decision.reference_mae:.2f}")
    client.set_model_version_tag(lake.registered_model_name, version, "model_label", CANDIDATE_MODEL_LABEL)

    try:
        current = client.get_model_version_by_alias(lake.registered_model_name, CHAMPION_ALIAS).version
    except MlflowException:
        current = None

    if decision.promote:
        client.set_registered_model_alias(lake.registered_model_name, CHAMPION_ALIAS, version)
        print(f"Promoted version {version} to @{CHAMPION_ALIAS} (was {current}): {decision.reason}")
    else:
        print(f"Registered version {version}; @{CHAMPION_ALIAS} stays at {current}: {decision.reason}")


if __name__ == "__main__":
    main()
