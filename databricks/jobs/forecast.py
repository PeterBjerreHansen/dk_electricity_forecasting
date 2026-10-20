"""Task 4: forecast tomorrow's hourly prices with the champion, falling back to the weighted median."""

from __future__ import annotations

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from dkenergy_databricks.cli import lakehouse, parser, run_date

from dkenergy_databricks.inference import (
    assemble_forecasts,
    baseline_predictions,
    predict_champion,
    prepare_forecast_inputs,
)
from dkenergy_databricks.lakehouse import CHAMPION_ALIAS, GOLD_FORECASTS, SILVER_PRICE_PANEL
from dkenergy_databricks.silver import panel_from_delta_frame
from dkenergy_databricks.spark_io import read_table, spark_session, write_table
from dkenergy_databricks.training import CANDIDATE_MODEL_LABEL, forecast_origin_for


def main() -> None:
    cli = parser(__doc__)
    cli.add_argument("--run-id", default="manual", help="Job run id, for lineage")
    args = cli.parse_args()
    lake = lakehouse(args)
    spark = spark_session()
    mlflow.set_registry_uri("databricks-uc")

    panel = panel_from_delta_frame(read_table(spark, lake.table(SILVER_PRICE_PANEL)))
    origin = forecast_origin_for(run_date(args))
    inputs = prepare_forecast_inputs(panel, origin)

    champion = None
    try:
        version = MlflowClient().get_model_version_by_alias(lake.registered_model_name, CHAMPION_ALIAS).version
    except MlflowException as exc:
        print(f"No @{CHAMPION_ALIAS} model registered yet ({exc}); publishing the fallback.")
    else:
        champion = predict_champion(
            lambda: mlflow.pyfunc.load_model(lake.champion_uri),
            inputs.features,
            model_name=CANDIDATE_MODEL_LABEL,
            model_version=str(version),
        )
        if champion.error:
            print(f"WARNING: champion v{version} failed, publishing the fallback: {champion.error}")

    forecasts = assemble_forecasts(inputs, champion, baseline_predictions(inputs), run_id=args.run_id)
    write_table(
        spark,
        forecasts,
        lake.table(GOLD_FORECASTS),
        mode="append",
        replace_where=f"forecast_origin_utc = '{origin.strftime('%Y-%m-%d %H:%M:%S')}'",
    )
    published = forecasts[forecasts["is_published"]]
    print(
        f"Origin {origin.isoformat()}: published {published['model_name'].iloc[0]} "
        f"for {published['ds_utc'].min()} -> {published['ds_utc'].max()} ({len(published)} rows)"
    )


if __name__ == "__main__":
    main()
