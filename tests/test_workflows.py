from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_production_deploy_workflow_serializes_runs() -> None:
    workflow = (ROOT / ".github" / "workflows" / "production.yml").read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "push:" not in workflow
    assert "environment: production" in workflow
    assert "concurrency:" in workflow
    assert "group: production-deploy" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "id-token: write" in workflow
    assert "secrets.AWS_DEPLOY_ROLE_ARN" in workflow
    assert "vars.ENABLE_PIPELINE_SCHEDULE || 'true'" in workflow


def test_static_site_is_private_behind_compressed_https_cloudfront() -> None:
    terraform = (ROOT / "infra" / "aws" / "static_site.tf").read_text(
        encoding="utf-8"
    )

    assert 'resource "aws_cloudfront_distribution" "static_site"' in terraform
    assert 'resource "aws_cloudfront_origin_access_control" "static_site"' in terraform
    assert 'resource "aws_s3_bucket_website_configuration"' not in terraform
    assert "block_public_policy     = true" in terraform
    assert "restrict_public_buckets = true" in terraform
    assert 'viewer_protocol_policy = "redirect-to-https"' in terraform
    assert "compress               = true" in terraform
    assert "enable_accept_encoding_brotli = true" in terraform
    assert 'identifiers = ["cloudfront.amazonaws.com"]' in terraform


def test_github_deploy_identity_is_repository_and_environment_bound() -> None:
    terraform = (ROOT / "infra" / "aws" / "github_deploy.tf").read_text(
        encoding="utf-8"
    )

    assert 'url            = "https://token.actions.githubusercontent.com"' in terraform
    assert 'values   = ["sts.amazonaws.com"]' in terraform
    assert (
        'values   = ["repo:${var.github_repository}:environment:${var.github_environment}"]'
        in terraform
    )
    assert 'policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"' in terraform
