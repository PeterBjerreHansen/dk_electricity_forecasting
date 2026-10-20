from __future__ import annotations

from typing import Any

import pandas as pd


def spark_session() -> Any:
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()
    # Timestamps are stored and read back as UTC regardless of workspace zone.
    spark.conf.set("spark.sql.session.timeZone", "UTC")
    return spark


def table_exists(spark: Any, name: str) -> bool:
    return bool(spark.catalog.tableExists(name))


def read_table(spark: Any, name: str) -> pd.DataFrame:
    return spark.table(name).toPandas()


def write_table(
    spark: Any,
    frame: pd.DataFrame,
    name: str,
    *,
    mode: str = "append",
    replace_where: str | None = None,
) -> None:
    """Write a pandas frame to a Delta table, creating it on first write."""

    if frame.empty:
        return
    writer = spark.createDataFrame(frame).write.format("delta")
    if replace_where is not None and table_exists(spark, name):
        writer = writer.mode("overwrite").option("replaceWhere", replace_where)
    elif mode == "overwrite":
        writer = writer.mode("overwrite").option("overwriteSchema", "true")
    else:
        writer = writer.mode(mode)
    writer.saveAsTable(name)
