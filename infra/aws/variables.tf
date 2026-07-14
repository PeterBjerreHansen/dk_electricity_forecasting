variable "aws_region" {
  type        = string
  description = "AWS region for the production stack."
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
  description = "Prefix inside the private artifact bucket."
  default     = "dk-energy-forecasts"
}

variable "artifact_bucket_name" {
  type        = string
  description = "Optional artifact bucket name; the default includes account and environment."
  default     = ""
}

variable "static_site_bucket_name" {
  type        = string
  description = "Optional static-site bucket name; the default includes the account id."
  default     = ""
}

variable "github_repository" {
  type        = string
  description = "GitHub repository allowed to assume the production deployment role."
  default     = "PeterBjerreHansen/dk_electricity_forecasting"
}

variable "github_environment" {
  type        = string
  description = "GitHub environment allowed to assume the production deployment role."
  default     = "production"
}

variable "pipeline_image_uri" {
  type        = string
  description = "Immutable container image URI for the forecast task."
  default     = "bootstrap"
}

variable "build_git_sha" {
  type        = string
  description = "Git commit embedded in the task environment."
  default     = "unknown"

  validation {
    condition     = var.build_git_sha == "unknown" || can(regex("^[0-9a-f]{7,40}$", var.build_git_sha))
    error_message = "build_git_sha must be 'unknown' or a 7-40 character lowercase hexadecimal Git SHA."
  }
}

variable "forecast_schedule_expression" {
  type        = string
  description = "Daily EventBridge Scheduler expression."
  default     = "cron(0 10 * * ? *)"
}

variable "forecast_schedule_timezone" {
  type        = string
  description = "Timezone for the daily schedule."
  default     = "Europe/Copenhagen"
}

variable "enable_pipeline_schedule" {
  type        = bool
  description = "Whether EventBridge Scheduler invokes the daily forecast task."
  default     = false
}

variable "pipeline_cpu" {
  type        = number
  description = "Fargate CPU units for the forecast task."
  default     = 2048
}

variable "pipeline_memory" {
  type        = number
  description = "Fargate memory in MiB for the forecast task."
  default     = 8192
}

variable "pipeline_ephemeral_storage_gib" {
  type        = number
  description = "Ephemeral disk for data, model cache, and generated artifacts."
  default     = 50

  validation {
    condition     = var.pipeline_ephemeral_storage_gib >= 20 && var.pipeline_ephemeral_storage_gib <= 200
    error_message = "pipeline_ephemeral_storage_gib must be between 20 and 200 GiB."
  }
}
