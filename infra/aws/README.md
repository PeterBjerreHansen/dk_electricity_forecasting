# AWS infrastructure

This directory contains one static-first production stack:

- private S3 storage for data state, immutable forecast runs, dashboard history,
  and the Chronos LoRA artifact;
- a private S3 origin containing one generated `index.html`;
- CloudFront for HTTPS, compression, and five-minute edge caching;
- ECR for one pipeline image;
- ECS/Fargate for a short-lived daily task;
- CloudWatch logs;
- an optional EventBridge schedule at 10:00 Europe/Copenhagen.

There is no always-on dashboard service, load balancer, database, model training
job, or automatic model selection.

## Terraform state

Terraform's own state lives in the separately created bucket:

```text
dk-energy-forecasts-tfstate-653044339519-eu-central-1
```

Initialize from the repository root:

```bash
export AWS_PROFILE=dkenergy-production
export AWS_REGION=eu-central-1
export TF_STATE_BUCKET=dk-energy-forecasts-tfstate-653044339519-eu-central-1

terraform -chdir=infra/aws init \
  -backend-config="bucket=$TF_STATE_BUCKET" \
  -backend-config="key=dk-energy-forecasts/production.tfstate" \
  -backend-config="region=$AWS_REGION" \
  -backend-config="use_lockfile=true"
```

Never delete or publish that bucket. It describes which real resources
Terraform owns.

## Safe review loop

```bash
terraform -chdir=infra/aws fmt -check -recursive
terraform -chdir=infra/aws validate

terraform -chdir=infra/aws plan \
  -var "aws_region=$AWS_REGION" \
  -var "pipeline_image_uri=<account>.dkr.ecr.eu-central-1.amazonaws.com/dk-energy-forecasts-pipeline:<git-sha>" \
  -var "build_git_sha=<full-git-sha>" \
  -var "enable_pipeline_schedule=false" \
  -out=deploy.tfplan

terraform -chdir=infra/aws apply deploy.tfplan
```

Keep the schedule disabled for an infrastructure/image change until one manual
task succeeds. Then plan and apply only `enable_pipeline_schedule=true`.

## Build and push the image

Use an immutable Git SHA tag:

```bash
export AWS_ACCOUNT_ID=653044339519
export GIT_SHA=$(git rev-parse HEAD)
export ECR="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
export IMAGE="$ECR/dk-energy-forecasts-pipeline:$GIT_SHA"

aws ecr get-login-password --region "$AWS_REGION" \
  | docker login --username AWS --password-stdin "$ECR"
docker build --platform linux/amd64 --build-arg "GIT_SHA=$GIT_SHA" \
  -f Dockerfile.pipeline -t "$IMAGE" .
docker push "$IMAGE"
```

Terraform never uses a mutable `latest` image tag.

## Manual task

After applying the task definition, use the current subnet and security-group
ids from Terraform state or the AWS CLI and start one task:

```bash
aws ecs run-task \
  --cluster dk-energy-forecasts-production \
  --task-definition dk-energy-forecasts-production-pipeline \
  --launch-type FARGATE \
  --network-configuration \
  'awsvpcConfiguration={subnets=[subnet-a,subnet-b],securityGroups=[sg-id],assignPublicIp=ENABLED}' \
  --region "$AWS_REGION"
```

Watch `/ecs/dk-energy-forecasts-production/pipeline` in CloudWatch. A successful
task exits with code zero after uploading the immutable run, `latest.json`, the
private dashboard history, and the site `index.html`.

## Static site

The site bucket is private and readable only by the CloudFront distribution.
The public HTTPS URL is:

```bash
terraform -chdir=infra/aws output -raw static_site_url
```

The bucket is versioned for simple rollback. CloudFront and the pipeline task
can access only `index.html`. CloudFront redirects HTTP to HTTPS, compresses the
page with Gzip or Brotli, and caches it for at most five minutes.

For a manual local page replacement:

```bash
aws s3 cp build/static-dashboard/index.html \
  "$(terraform -chdir=infra/aws output -raw static_site_s3_uri)/index.html" \
  --content-type 'text/html; charset=utf-8' \
  --cache-control 'public,max-age=300' \
  --region "$AWS_REGION"
```

After a manual replacement, wait at most five minutes or invalidate `/` and
`/index.html` using the `static_site_cloudfront_distribution_id` output.

## GitHub deployment identity

Terraform creates a GitHub Actions OIDC provider and deploy role. Its trust
policy accepts tokens only from this repository's `production` environment, so
GitHub does not store a long-lived AWS access key. The role has PowerUser access
for project resources plus narrowly scoped IAM permissions for the runtime
roles; it cannot modify its own trust or permissions.

Store `github_deploy_role_arn` as the production environment secret
`AWS_DEPLOY_ROLE_ARN`. Store the Terraform-state bucket as `TF_STATE_BUCKET` and
set `ENABLE_PIPELINE_SCHEDULE=true` before running the manual Production Deploy
workflow.

## Permissions

The task may read the private project prefix. It may write only runtime state,
forecast runs, published-history outputs, the dashboard history, and pointers.
The model prefix is read-only. In the site bucket it may write only
`index.html`.

## Schedule

The default expression is `cron(0 10 * * ? *)` with timezone
`Europe/Copenhagen`. EventBridge Scheduler handles the timezone, including DST.
Set `enable_pipeline_schedule=true` only after a manual task with the exact
deployed revision succeeds.
