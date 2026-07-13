locals {
  publication_marker_key = (
    local.artifact_prefix == ""
    ? trim(var.publication_marker_relative_key, "/")
    : "${local.artifact_prefix}/${trim(var.publication_marker_relative_key, "/")}"
  )
  enable_scheduled_operations = var.enable_pipeline_schedule || var.enable_published_scoring_schedule
  enable_deadline_check       = var.enable_pipeline_schedule && var.enable_forecast_deadline_check
  scheduled_task_definition_arns = concat(
    var.enable_pipeline_schedule ? [aws_ecs_task_definition.pipeline.arn] : [],
    var.enable_published_scoring_schedule ? [aws_ecs_task_definition.published_scoring[0].arn] : []
  )
}

resource "aws_sns_topic" "production_alerts" {
  count = local.enable_scheduled_operations ? 1 : 0
  name  = "${local.name}-production-alerts"
}

resource "aws_sns_topic_subscription" "production_alert_email" {
  count = local.enable_scheduled_operations && trimspace(var.alert_email_endpoint) != "" ? 1 : 0

  topic_arn = aws_sns_topic.production_alerts[0].arn
  protocol  = "email"
  endpoint  = trimspace(var.alert_email_endpoint)
}

resource "aws_cloudwatch_event_rule" "scheduled_task_nonzero_exit" {
  count       = local.enable_scheduled_operations ? 1 : 0
  name        = "${local.name}-scheduled-task-nonzero-exit"
  description = "Notify when a scheduled live or scoring container exits unsuccessfully."

  event_pattern = jsonencode({
    source        = ["aws.ecs"]
    "detail-type" = ["ECS Task State Change"]
    detail = {
      clusterArn        = [aws_ecs_cluster.main.arn]
      taskDefinitionArn = local.scheduled_task_definition_arns
      lastStatus        = ["STOPPED"]
      containers = {
        exitCode = [{ "anything-but" = 0 }]
      }
    }
  })
}

resource "aws_cloudwatch_event_rule" "scheduled_task_failed_to_start" {
  count       = local.enable_scheduled_operations ? 1 : 0
  name        = "${local.name}-scheduled-task-failed-to-start"
  description = "Notify when a scheduled live or scoring task cannot start."

  event_pattern = jsonencode({
    source        = ["aws.ecs"]
    "detail-type" = ["ECS Task State Change"]
    detail = {
      clusterArn        = [aws_ecs_cluster.main.arn]
      taskDefinitionArn = local.scheduled_task_definition_arns
      lastStatus        = ["STOPPED"]
      stopCode          = ["TaskFailedToStart"]
    }
  })
}

resource "aws_cloudwatch_event_target" "scheduled_task_nonzero_exit" {
  count = local.enable_scheduled_operations ? 1 : 0
  rule  = aws_cloudwatch_event_rule.scheduled_task_nonzero_exit[0].name
  arn   = aws_sns_topic.production_alerts[0].arn
}

resource "aws_cloudwatch_event_target" "scheduled_task_failed_to_start" {
  count = local.enable_scheduled_operations ? 1 : 0
  rule  = aws_cloudwatch_event_rule.scheduled_task_failed_to_start[0].name
  arn   = aws_sns_topic.production_alerts[0].arn
}

data "aws_iam_policy_document" "production_alerts" {
  count = local.enable_scheduled_operations ? 1 : 0

  statement {
    sid       = "AccountOwner"
    effect    = "Allow"
    actions   = ["SNS:*"]
    resources = [aws_sns_topic.production_alerts[0].arn]

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"]
    }
  }

  statement {
    sid       = "EventBridgeTaskFailureAlerts"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.production_alerts[0].arn]

    principals {
      type        = "Service"
      identifiers = ["events.amazonaws.com"]
    }

    condition {
      test     = "ArnEquals"
      variable = "aws:SourceArn"
      values = [
        aws_cloudwatch_event_rule.scheduled_task_nonzero_exit[0].arn,
        aws_cloudwatch_event_rule.scheduled_task_failed_to_start[0].arn
      ]
    }
  }

  statement {
    sid       = "CloudWatchAlarmAlerts"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [aws_sns_topic.production_alerts[0].arn]

    principals {
      type        = "Service"
      identifiers = ["cloudwatch.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [data.aws_caller_identity.current.account_id]
    }
  }
}

resource "aws_sns_topic_policy" "production_alerts" {
  count  = local.enable_scheduled_operations ? 1 : 0
  arn    = aws_sns_topic.production_alerts[0].arn
  policy = data.aws_iam_policy_document.production_alerts[0].json
}

data "archive_file" "forecast_deadline_check" {
  count       = local.enable_deadline_check ? 1 : 0
  type        = "zip"
  source_file = "${path.module}/functions/check_forecast_deadline.py"
  output_path = "${path.module}/.terraform/check_forecast_deadline.zip"
}

resource "aws_cloudwatch_log_group" "forecast_deadline_check" {
  count             = local.enable_deadline_check ? 1 : 0
  name              = "/aws/lambda/${local.name}-forecast-deadline-check"
  retention_in_days = 14
}

resource "aws_iam_role" "forecast_deadline_check" {
  count = local.enable_deadline_check ? 1 : 0
  name  = "${local.name}-forecast-deadline-check"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "forecast_deadline_check_logs" {
  count      = local.enable_deadline_check ? 1 : 0
  role       = aws_iam_role.forecast_deadline_check[0].name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

resource "aws_iam_role_policy" "forecast_deadline_check" {
  count = local.enable_deadline_check ? 1 : 0
  name  = "${local.name}-forecast-deadline-check"
  role  = aws_iam_role.forecast_deadline_check[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = local.artifact_object_arn
      },
      {
        Effect   = "Allow"
        Action   = ["sns:Publish"]
        Resource = aws_sns_topic.production_alerts[0].arn
      }
    ]
  })
}

resource "aws_lambda_function" "forecast_deadline_check" {
  count         = local.enable_deadline_check ? 1 : 0
  function_name = "${local.name}-forecast-deadline-check"
  role          = aws_iam_role.forecast_deadline_check[0].arn
  runtime       = "python3.12"
  handler       = "check_forecast_deadline.handler"
  filename      = data.archive_file.forecast_deadline_check[0].output_path
  architectures = ["arm64"]
  timeout       = 15

  source_code_hash = data.archive_file.forecast_deadline_check[0].output_base64sha256

  environment {
    variables = {
      ALERT_TOPIC_ARN           = aws_sns_topic.production_alerts[0].arn
      ARTIFACT_BUCKET           = aws_s3_bucket.artifacts.bucket
      ARTIFACT_PREFIX           = local.artifact_prefix
      DELIVERY_DATE_OFFSET_DAYS = tostring(var.forecast_delivery_date_offset_days)
      MARKER_MAX_AGE_MINUTES    = tostring(var.forecast_marker_max_age_minutes)
      PUBLICATION_MARKER_KEY    = local.publication_marker_key
      SCHEDULE_TIMEZONE         = var.forecast_schedule_timezone
    }
  }

  depends_on = [
    aws_cloudwatch_log_group.forecast_deadline_check[0],
    aws_iam_role_policy_attachment.forecast_deadline_check_logs[0]
  ]
}

resource "aws_iam_role" "deadline_scheduler" {
  count = local.enable_deadline_check ? 1 : 0
  name  = "${local.name}-deadline-scheduler"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "scheduler.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "deadline_scheduler" {
  count = local.enable_deadline_check ? 1 : 0
  name  = "${local.name}-invoke-deadline-check"
  role  = aws_iam_role.deadline_scheduler[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["lambda:InvokeFunction"]
        Resource = aws_lambda_function.forecast_deadline_check[0].arn
      }
    ]
  })
}

resource "aws_scheduler_schedule" "forecast_deadline_check" {
  count = local.enable_deadline_check ? 1 : 0

  name                         = "${local.name}-forecast-deadline-check"
  schedule_expression          = var.forecast_deadline_check_schedule_expression
  schedule_expression_timezone = var.forecast_schedule_timezone

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_lambda_function.forecast_deadline_check[0].arn
    role_arn = aws_iam_role.deadline_scheduler[0].arn

    retry_policy {
      maximum_event_age_in_seconds = 300
      maximum_retry_attempts       = 0
    }
  }
}

resource "aws_cloudwatch_metric_alarm" "forecast_deadline_check_errors" {
  count               = local.enable_deadline_check ? 1 : 0
  alarm_name          = "${local.name}-forecast-deadline-check-errors"
  alarm_description   = "The forecast deadline checker itself failed to execute."
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.production_alerts[0].arn]

  dimensions = {
    FunctionName = aws_lambda_function.forecast_deadline_check[0].function_name
  }
}
