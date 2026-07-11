# Operations runbook

## Independent jobs

The critical live path is intentionally short:

```bash
python scripts/run_daily_pipeline.py --with-weather --skip-backtest
```

It refreshes inputs, generates one live forecast, validates it, writes one
immutable run transactionally, and promotes latest atomically. It does not run
recent model diagnostics or published-history scoring.

Schedule the non-critical jobs independently:

```bash
python scripts/run_recent_diagnostics.py --allow-incomplete-panel
python scripts/score_published_forecasts.py --allow-incomplete-panel
```

Run the evaluation arena only for reviewed model-selection rounds, not as a
daily task.

## Schedule and deadline

The AWS default is 10:00 `Europe/Copenhagen`; the decision cutoff is local noon.
The two-hour margin covers ingestion, inference, validation, and upload. A late
live process fails instead of relabeling itself as a noon forecast.

## Publication sequence

1. Hydrate required state and the model artifact.
2. Refresh price and, normally, weather sources.
3. Require a complete current Danish price-delivery day.
4. Validate the model artifact schema, file hashes, and weather policy.
5. Generate and validate the requested horizon.
6. Build the run in a hidden sibling directory.
7. Hash its artifacts and atomically rename it into `forecast_runs/`.
8. Atomically replace consumer files under a promotion lock; write the latest
   manifest last as the local commit marker.
9. In cloud mode, upload `latest/forecast_dashboard.json` last.

The stable live run ID makes concurrent attempts for one origin collide. An
identical retry is idempotent; changed core artifacts under the same key fail.

## Before enabling a model

1. Export a schema-v2 artifact with an immutable base revision if possible.
2. Confirm its manifest has training-data and model-file hashes.
3. Run the real-artifact adapter smoke test.
4. Build the exact pipeline image using `constraints-production.txt`.
5. Run the frozen evaluation arena and review its JSON/Markdown report.
6. Upload the artifact, verify it, and only then enable the schedule.

## Monitoring signals

Alert on:

- No eligible live run before noon.
- Stale or incomplete current-day prices.
- Stale weather features or insufficient future weather coverage.
- Artifact checksum/schema/policy mismatch.
- Prediction-key or quantile validation failure.
- Latest-promotion lock contention or attempted origin regression.
- Missing published-history scoring runs.
- MAE/WIS/calibration or subgroup drift in independent diagnostics.

The Streamlit dashboard warns about stale artifacts, non-live runs, and
score-ineligible runs, but dashboard warnings are not a substitute for job and
log alerts.

## Recovery and rollback

- Never edit an immutable forecast run.
- If a live run fails before its atomic rename, rerun the same request.
- If latest is corrupt but the immutable run is valid, regenerate latest from
  that run; do not alter the run.
- To roll back a model, change the source-controlled registry/artifact reference
  and deploy. Preserve both model artifacts and all forecasts.
- Use `shadow` for a candidate under live information constraints.
- Use `replay` for historical investigation; it cannot update latest or enter
  published scoring.

## Verification

```bash
python -m pytest
python -m ruff check .
python -m compileall -q src scripts app
terraform -chdir=infra/aws fmt -check -recursive
terraform -chdir=infra/aws init -backend=false
terraform -chdir=infra/aws validate
docker compose build
```

See [codebase-tour.md](codebase-tour.md) for the implementation paths behind
each step.
