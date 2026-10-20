"""Task 2: append raw batches that are new since the last run to the bronze table."""

from __future__ import annotations

from dkenergy_databricks.cli import lakehouse, parser

from dkenergy_databricks.ingest import bronze_records_frame
from dkenergy_databricks.lakehouse import BRONZE_EDS_RECORDS
from dkenergy_databricks.spark_io import spark_session, table_exists, write_table


def main() -> None:
    args = parser(__doc__).parse_args()
    lake = lakehouse(args)
    spark = spark_session()
    table = lake.table(BRONZE_EDS_RECORDS)

    known = []
    if table_exists(spark, table):
        known = [row.batch_id for row in spark.table(table).select("batch_id").distinct().collect()]

    new_records = bronze_records_frame(lake.eds_raw_root, known_batch_ids=known)
    write_table(spark, new_records, table, mode="append")
    print(f"Appended {len(new_records)} records from {new_records['batch_id'].nunique()} new batch(es) to {table}")


if __name__ == "__main__":
    main()
