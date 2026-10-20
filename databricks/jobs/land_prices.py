"""Task 1: land raw Energi Data Service responses in the Unity Catalog volume."""

from __future__ import annotations

from dkenergy_databricks.cli import lakehouse, parser, run_date

from dkenergy_databricks.ingest import default_fetch_range, land_eds_prices


def main() -> None:
    cli = parser(__doc__)
    cli.add_argument("--history-days", type=int, default=450)
    args = cli.parse_args()
    lake = lakehouse(args)

    start, end = default_fetch_range(run_date(args), history_days=args.history_days)
    result = land_eds_prices(lake.eds_raw_root, start=start, end=end)
    print(f"Landing {start} -> {end} into {lake.eds_raw_root}: {result}")

    if result.status == "api_unreachable":
        if not any(lake.eds_raw_root.glob("*/fetched_at=*/*.json")):
            raise SystemExit(
                "Energi Data Service is unreachable from this workspace and no raw files "
                "have been landed. Run the 'Databricks' GitHub workflow to land them."
            )
        print("WARNING: API unreachable; continuing with raw files already in the volume.")


if __name__ == "__main__":
    main()
