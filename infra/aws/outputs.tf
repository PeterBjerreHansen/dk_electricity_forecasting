output "artifact_bucket_name" {
  value = aws_s3_bucket.artifacts.bucket
}

output "artifact_store_uri" {
  value = local.artifact_store_uri
}

output "model_artifact_uri" {
  value = local.model_artifact_uri
}

output "pipeline_ecr_repository_url" {
  value = aws_ecr_repository.pipeline.repository_url
}

output "static_site_url" {
  value = "https://${aws_cloudfront_distribution.static_site.domain_name}"
}

output "static_site_cloudfront_distribution_id" {
  value = aws_cloudfront_distribution.static_site.id
}

output "static_site_s3_uri" {
  value = "s3://${aws_s3_bucket.static_site.bucket}"
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "github_deploy_role_arn" {
  value = aws_iam_role.github_deploy.arn
}

output "publication_marker_s3_uri" {
  value = "s3://${aws_s3_bucket.artifacts.bucket}/${local.publication_marker_key}"
}
