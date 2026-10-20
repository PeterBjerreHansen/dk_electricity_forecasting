"""Task 3: rebuild the hourly DK1/DK2 price panel from the newest revision of each bronze record."""

from __future__ import annotations

from dkenergy_databricks.cli import lakehouse, parser

from dkenergy_databricks.lakehouse import (
    BRONZE_EDS_RECORDS,
    SILVER_PRICE_PANEL,
    SILVER_PRICE_PANEL_QA,
)
from dkenergy_databricks.silver import build_silver_price_panel, panel_to_delta_frame, qa_record
from dkenergy_databricks.spark_io import spark_session, write_table

# The API is re-queried every day for the current month, so bronze holds many
# revisions of the same quarter-hour. Spark keeps only the latest before the
# (single-node) panel builder sees the data.
LATEST_REVISION_SQL = """
SELECT batch_id, source_dataset, retrieved_at_utc, raw_path, record_index, record_json
FROM (
  SELECT *,
    row_number() OVER (
      PARTITION BY
        source_dataset,
        get_json_object(record_json, '$.PriceArea'),
        coalesce(get_json_object(record_json, '$.TimeUTC'), get_json_object(record_json, '$.HourUTC'))
      ORDER BY retrieved_at_utc DESC, batch_id DESC
    ) AS revision_rank
  FROM {table}
)
WHERE revision_rank = 1
"""


def main() -> None:
    args = parser(__doc__).parse_args()
    lake = lakehouse(args)
    spark = spark_session()

    bronze = spark.sql(LATEST_REVISION_SQL.format(table=lake.table(BRONZE_EDS_RECORDS))).toPandas()
    panel, qa = build_silver_price_panel(bronze)
    write_table(spark, panel_to_delta_frame(panel), lake.table(SILVER_PRICE_PANEL), mode="overwrite")
    write_table(spark, qa_record(qa), lake.table(SILVER_PRICE_PANEL_QA), mode="append")
    print(
        f"Wrote {len(panel)} panel rows ({qa['min_ds_utc']} -> {qa['max_ds_utc']}), "
        f"status {qa.get('artifact_status')}"
    )


if __name__ == "__main__":
    main()
