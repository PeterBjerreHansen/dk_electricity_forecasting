# AWS MVP Infrastructure

This Terraform stack can deploy the production system in stages. Start with the
beginner-oriented [deployment plan](../../docs/aws-deployment-plan.md).

- private S3 artifact bucket with versioning and SSE
- a pipeline ECR image repository, plus an optional web image repository
- optional ECS/Fargate Streamlit web service behind an ALB and CloudFront
- ECS/Fargate scheduled pipeline task
- EventBridge Scheduler daily trigger
- IAM roles and CloudWatch log groups

## Terraform State

Create a Terraform state bucket once before enabling the GitHub workflow. The
workflow expects the bucket name in the `TF_STATE_BUCKET` repository secret and
stores state at `dk-energy-forecasts/production.tfstate`.

For local validation without a backend, use:

```bash
terraform -chdir=infra/aws init -backend=false
terraform -chdir=infra/aws validate
```

## Bootstrap

The GitHub workflow expects repository secrets named `AWS_DEPLOY_ROLE_ARN` and
`TF_STATE_BUCKET`. Create the deploy role once with permissions to manage this
stack, or run Terraform locally with admin credentials for the first apply.

```bash
terraform -chdir=infra/aws init \
  -backend-config="bucket=$TF_STATE_BUCKET" \
  -backend-config="key=dk-energy-forecasts/production.tfstate" \
  -backend-config="region=eu-central-1" \
  -backend-config="use_lockfile=true"
terraform -chdir=infra/aws apply \
  -target=aws_ecr_repository.pipeline \
  -target=aws_ecr_lifecycle_policy.pipeline
```

Build and push the pipeline image, then apply with schedules and the web tier
disabled. These are the safe defaults:

```bash
export TF_STATE_BUCKET=<terraform-state-bucket>
export AWS_PROFILE=dkenergy-terraform
make aws-deploy
```

`dkenergy-terraform` is a process-credential profile that delegates to the
temporary `aws login` session in `dkenergy-production`; it does not store
long-lived access keys. See the deployment plan for its one-time setup.

Upload the trained Chronos LoRA artifact:

```bash
export AWS_MODEL_ARTIFACT_URI="$(terraform -chdir=infra/aws output -raw model_artifact_uri)"
make aws-bootstrap-model
```

Then enable the daily schedule:

```bash
AWS_ENABLE_PIPELINE_SCHEDULE=true make aws-deploy
```

In GitHub Actions, the same toggle is controlled by the repository variable
`ENABLE_PIPELINE_SCHEDULE`. Leave it unset or `false` for the first deploy, then
set it to `true` after the model artifact has been uploaded.

## Runtime Notes

The scheduled task runs `scripts/run_cloud_pipeline.py` with `WITH_WEATHER=1`.
The default schedule is 10:00 `Europe/Copenhagen`, leaving two hours before the
repository's noon decision cutoff. It downloads the production inference state
from S3, refreshes data, runs the live path with `--skip-backtest`, publishes a
transactional immutable forecast run, and updates
the root `latest.json` pointer last. Historical raw data remains in S3; the
container runtime only hydrates the slice it needs for inference.

Recent diagnostics and published-history scoring are separate jobs. Schedule
`scripts/run_recent_diagnostics.py` and `scripts/score_published_forecasts.py`
independently when those operational views should refresh; their failure must
not delay or replace a valid live publication.

The dashboard reads latest S3 artifacts plus published forecast performance
history. Notebook/backtest artifact folders are disabled in ECS via
`DKENERGY_ENABLE_LEGACY_BACKTESTS=0`.

The dashboard ECR repository, ECS service, ALB, and CloudFront distribution are
not created unless `enable_web=true`. A historical smoke run must use
`--run-kind replay`, an explicit `--information-cutoff-utc`, and a separate
artifact-store prefix so it cannot update the production `latest.json`.
