output "artifact_bucket_name" {
  value = aws_s3_bucket.artifacts.bucket
}

output "artifact_store_uri" {
  value = local.artifact_store_uri
}

output "model_artifact_uri" {
  value = local.model_artifact_uri
}

output "web_ecr_repository_url" {
  value = aws_ecr_repository.web.repository_url
}

output "pipeline_ecr_repository_url" {
  value = aws_ecr_repository.pipeline.repository_url
}

output "cloudfront_domain_name" {
  value = aws_cloudfront_distribution.web.domain_name
}

output "alb_dns_name" {
  value = aws_lb.web.dns_name
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "production_alert_topic_arn" {
  value = aws_sns_topic.production_alerts.arn
}

output "publication_marker_s3_uri" {
  value = "s3://${aws_s3_bucket.artifacts.bucket}/${local.publication_marker_key}"
}
