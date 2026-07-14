# Production operations

This is the runbook for the intentionally small AWS deployment.

## Normal daily behavior

At 10:00 Europe/Copenhagen, EventBridge Scheduler starts one Fargate task. The
task refreshes prices and point-in-time weather, publishes tomorrow's forecast,
updates registered-history diagnostics, replaces the static page, and exits.

Expected durable outputs in the private artifact store are:

```text
state/...
latest/price_panel_hourly_v1.parquet
forecast_runs/<run-id>/...
latest.json
dashboard/forecast_history.parquet
```

The public site bucket should contain a new version of `index.html`.

## Quick health check

Use the production profile and region:

```bash
export AWS_PROFILE=dkenergy-production
export AWS_REGION=eu-central-1
```

Check the newest task:

```bash
aws ecs list-tasks \
  --cluster dk-energy-forecasts-production \
  --desired-status STOPPED \
  --max-items 5 \
  --region "$AWS_REGION"
```

Describe its stop reason and container exit code, then inspect the corresponding
log stream in `/ecs/dk-energy-forecasts-production/pipeline`.

Check publication:

```bash
aws s3 cp \
  s3://dk-energy-forecasts-production-653044339519/dk-energy-forecasts/latest.json \
  - --region "$AWS_REGION"

aws s3api head-object \
  --bucket dk-energy-forecasts-site-653044339519 \
  --key index.html \
  --region "$AWS_REGION"
```

Confirm that the delivery date is tomorrow in Copenhagen, the run status is
completed, and `ContentType` is `text/html; charset=utf-8`.

## Failure interpretation

### Task never started

Inspect the ECS stopped reason. Typical causes are an invalid task revision,
missing ECR image permission, unavailable public IP/networking, or insufficient
Fargate capacity. Disable the schedule if repeated starts could cross the
publication deadline.

### Data fetch failed

The immutable latest pointer should remain unchanged. Inspect the API response
and retry manually only if the information cutoff/deadline contract is still
valid.

### Chronos failed but the task succeeded

This is a designed degraded run. `forecast_status` should be `degraded`, the
published model should be `weighted_median_v1`, and the public page should show
a fallback notice. Investigate Chronos before the next day, but do not rewrite
the completed run.

### Forecast published, dashboard failed

The task may exit non-zero after writing `latest.json`. The previous static page
remains public because `index.html` is uploaded only after a complete render.
Treat the forecast as valid; repair and rebuild the page separately.

### Publication missed the deadline

The cloud pipeline refuses to advance `latest.json` after the configured
decision deadline. Do not bypass this guard. Leave the previous pointer in
place, document the missed run, and fix the underlying latency before the next
day.

## Manual page rebuild

Build and inspect locally first:

```bash
make static-dashboard
python -m http.server 8000 --directory build/static-dashboard
```

Then replace only `index.html`:

```bash
aws s3 cp build/static-dashboard/index.html \
  s3://dk-energy-forecasts-site-653044339519/index.html \
  --content-type 'text/html; charset=utf-8' \
  --cache-control 'public,max-age=300' \
  --region "$AWS_REGION"
```

S3 versioning preserves the previous page.

## Disable and re-enable automation

Apply Terraform with `enable_pipeline_schedule=false` to stop future daily
starts. This does not delete artifacts, the task definition, or the public
site. Re-enable only after a manual task using the intended image revision exits
successfully.

## Deploying a code revision

1. Run the full quality gate.
2. Commit the exact source revision.
3. Build/push an image tagged by the full Git SHA.
4. Apply Terraform with the schedule disabled.
5. Run and verify one manual task.
6. Apply Terraform with the schedule enabled.

Never deploy uncommitted source under an unrelated image tag.

## Model changes

Training is not part of daily production. A new LoRA adapter is validated and
uploaded under a new content-addressed prefix. Update `config/production.json`,
create a new code/image revision, and perform a manual task. Existing model
artifact prefixes are immutable.

## Retention and cleanup

- ECR keeps five images.
- CloudWatch pipeline logs are retained for 14 days.
- S3 artifacts and page versions are not automatically deleted yet; review
  lifecycle needs after real usage is known.
- Local `data/`, `results/`, `artifacts/`, `runtime/`, and `build/` directories
  are generated and may be recreated.
