"""Task 5: score past forecasts against actual prices and run degradation and drift checks."""

from __future__ import annotations

import pandas as pd

from dkenergy_databricks.cli import lakehouse, parser

from dkenergy_databricks.lakehouse import (
    GOLD_FORECAST_SCORES,
    GOLD_FORECASTS,
    GOLD_MONITORING_CHECKS,
    SILVER_PRICE_PANEL,
)
from dkenergy_databricks.monitoring import (
    checks_frame,
    degradation_check,
    price_drift_checks,
    score_forecasts,
)
from dkenergy_databricks.silver import panel_from_delta_frame
from dkenergy_databricks.spark_io import read_table, spark_session, table_exists, write_table
from dkenergy_forecast.models.registry import WEIGHTED_MEDIAN_MODEL_LABEL


def main() -> None:
    cli = parser(__doc__)
    cli.add_argument(
        "--fail-on-alert",
        action="store_true",
        help="Fail the task on an alert so the job's failure notification fires.",
    )
    args = cli.parse_args()
    lake = lakehouse(args)
    spark = spark_session()

    panel = panel_from_delta_frame(read_table(spark, lake.table(SILVER_PRICE_PANEL)))
    now = pd.Timestamp.now(tz="UTC")

    checks = price_drift_checks(panel, as_of=panel["ds_utc"].max() + pd.Timedelta(hours=1))
    if table_exists(spark, lake.table(GOLD_FORECASTS)):
        scores = score_forecasts(read_table(spark, lake.table(GOLD_FORECASTS)), panel)
        write_table(spark, scores, lake.table(GOLD_FORECAST_SCORES), mode="overwrite")
        checks.append(degradation_check(scores, reference_model=WEIGHTED_MEDIAN_MODEL_LABEL))

    frame = checks_frame(checks, checked_at_utc=now)
    write_table(spark, frame, lake.table(GOLD_MONITORING_CHECKS), mode="append")
    print(frame.to_string(index=False))

    alerts = frame[frame["status"] == "alert"]
    if args.fail_on_alert and not alerts.empty:
        raise SystemExit(f"{len(alerts)} monitoring alert(s): " + "; ".join(alerts["detail"]))


if __name__ == "__main__":
    main()
