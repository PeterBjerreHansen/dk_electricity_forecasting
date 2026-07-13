variable "aws_region" {
  type        = string
  description = "AWS region for the MVP stack."
  default     = "eu-central-1"
}

variable "environment" {
  type        = string
  description = "Environment name used in resource names."
  default     = "production"
}

variable "project_name" {
  type        = string
  description = "Project name used in resource names."
  default     = "dk-energy-forecasts"
}

variable "artifact_prefix" {
  type        = string
  description = "Prefix inside the artifact S3 bucket."
  default     = "dk-energy-forecasts"
}

variable "artifact_bucket_name" {
  type        = string
  description = "Optional globally unique artifact bucket name. Defaults to a deterministic account/environment name."
  default     = ""
}

variable "web_image_uri" {
  type        = string
  description = "Container image URI deployed to the Streamlit ECS service."
  default     = "bootstrap"
}

variable "enable_web" {
  type        = bool
  description = "Whether to provision the always-on Streamlit service, ALB, and CloudFront distribution. Keep false for the low-cost pipeline-only stages."
  default     = false
}

variable "pipeline_image_uri" {
  type        = string
  description = "Container image URI deployed to the scheduled pipeline ECS task."
  default     = "bootstrap"
}

variable "build_git_sha" {
  type        = string
  description = "Git commit embedded in the deployed container images and task environment."
  default     = "unknown"

  validation {
    condition     = var.build_git_sha == "unknown" || can(regex("^[0-9a-f]{7,40}$", var.build_git_sha))
    error_message = "build_git_sha must be 'unknown' or a 7-40 character lowercase hexadecimal Git SHA."
  }
}

variable "forecast_schedule_expression" {
  type        = string
  description = "EventBridge Scheduler expression for the live pipeline; keep enough headroom before the market-noon decision cutoff."
  default     = "cron(0 10 * * ? *)"
}

variable "forecast_schedule_timezone" {
  type        = string
  description = "Timezone for the EventBridge Scheduler expression."
  default     = "Europe/Copenhagen"
}

variable "enable_pipeline_schedule" {
  type        = bool
  description = "Whether EventBridge Scheduler should invoke the daily pipeline task."
  default     = false
}

variable "published_scoring_schedule_expression" {
  type        = string
  description = "EventBridge Scheduler expression for independent published-forecast scoring."
  default     = "cron(0 14 * * ? *)"
}

variable "enable_published_scoring_schedule" {
  type        = bool
  description = "Whether EventBridge Scheduler should run background published-forecast scoring."
  default     = false
}

variable "forecast_deadline_check_schedule_expression" {
  type        = string
  description = "EventBridge Scheduler expression for checking that the day-ahead forecast was committed."
  default     = "cron(15 12 * * ? *)"
}

variable "enable_forecast_deadline_check" {
  type        = bool
  description = "Whether to check latest.json shortly after the publication deadline when the live schedule is enabled."
  default     = true
}

variable "publication_marker_relative_key" {
  type        = string
  description = "Artifact-store-relative S3 key for the atomically committed latest forecast pointer."
  default     = "latest.json"

  validation {
    condition     = trim(var.publication_marker_relative_key, "/") != ""
    error_message = "publication_marker_relative_key must contain a non-empty key."
  }
}

variable "forecast_delivery_date_offset_days" {
  type        = number
  description = "Expected local delivery-date offset from the deadline-check date; day-ahead forecasts use one day."
  default     = 1
}

variable "forecast_marker_max_age_minutes" {
  type        = number
  description = "Maximum age accepted for latest.json during the deadline check."
  default     = 360

  validation {
    condition     = var.forecast_marker_max_age_minutes > 0
    error_message = "forecast_marker_max_age_minutes must be positive."
  }
}

variable "alert_email_endpoint" {
  type        = string
  description = "Optional email address subscribed to the production SNS alert topic. Confirmation is required."
  default     = ""
}

variable "web_cpu" {
  type        = number
  description = "Fargate CPU units for the Streamlit service."
  default     = 1024
}

variable "web_memory" {
  type        = number
  description = "Fargate memory MiB for the Streamlit service."
  default     = 2048
}

variable "pipeline_cpu" {
  type        = number
  description = "Fargate CPU units for the scheduled pipeline task."
  default     = 2048
}

variable "pipeline_memory" {
  type        = number
  description = "Fargate memory MiB for the scheduled pipeline task."
  default     = 8192
}

variable "pipeline_ephemeral_storage_gib" {
  type        = number
  description = "Ephemeral disk GiB for the scheduled pipeline task workdir and model cache."
  default     = 50

  validation {
    condition     = var.pipeline_ephemeral_storage_gib >= 20 && var.pipeline_ephemeral_storage_gib <= 200
    error_message = "pipeline_ephemeral_storage_gib must be between 20 and 200 GiB for Fargate."
  }
}

variable "scoring_cpu" {
  type        = number
  description = "Fargate CPU units for background published-forecast scoring."
  default     = 1024
}

variable "scoring_memory" {
  type        = number
  description = "Fargate memory MiB for background published-forecast scoring."
  default     = 4096
}
