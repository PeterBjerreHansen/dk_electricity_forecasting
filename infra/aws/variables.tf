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

variable "pipeline_image_uri" {
  type        = string
  description = "Container image URI deployed to the scheduled pipeline ECS task."
  default     = "bootstrap"
}

variable "forecast_schedule_expression" {
  type        = string
  description = "EventBridge Scheduler expression for the pipeline."
  default     = "cron(15 13 * * ? *)"
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

variable "score_max_origins" {
  type        = number
  description = "Recent scoring origin count for production pipeline runs."
  default     = 7
}
