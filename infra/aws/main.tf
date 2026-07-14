data "aws_caller_identity" "current" {}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  name            = "${var.project_name}-${var.environment}"
  artifact_prefix = trim(var.artifact_prefix, "/")
  production_config = jsondecode(
    file("${path.module}/../../config/production.json")
  )
  bucket_name = (
    var.artifact_bucket_name != ""
    ? var.artifact_bucket_name
    : "${var.project_name}-${var.environment}-${data.aws_caller_identity.current.account_id}"
  )
  artifact_store_uri = (
    local.artifact_prefix == ""
    ? "s3://${aws_s3_bucket.artifacts.bucket}"
    : "s3://${aws_s3_bucket.artifacts.bucket}/${local.artifact_prefix}"
  )
  artifact_object_arn = (
    local.artifact_prefix == ""
    ? "${aws_s3_bucket.artifacts.arn}/*"
    : "${aws_s3_bucket.artifacts.arn}/${local.artifact_prefix}/*"
  )
  artifact_write_arns = [
    for relative in [
      "dashboard/*",
      "forecast_runs/*",
      "latest/*",
      "latest.json",
      "published_forecast_history/*",
      "state/*",
    ] : "${aws_s3_bucket.artifacts.arn}/${local.artifact_prefix == "" ? relative : "${local.artifact_prefix}/${relative}"}"
  ]
  model_artifact_key = trimprefix(
    local.production_config.primary.artifact_path,
    "artifacts/"
  )
  model_artifact_uri = "${local.artifact_store_uri}/${local.model_artifact_key}"
  publication_marker_key = (
    local.artifact_prefix == "" ? "latest.json" : "${local.artifact_prefix}/latest.json"
  )
}

resource "aws_ecr_repository" "pipeline" {
  name                 = "${var.project_name}-pipeline"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "pipeline" {
  repository = aws_ecr_repository.pipeline.name
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the five most recent images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 5
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

resource "aws_s3_bucket" "artifacts" {
  bucket = local.bucket_name
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket                  = aws_s3_bucket.artifacts.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_vpc" "main" {
  cidr_block           = "10.42.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true

  tags = { Name = local.name }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = { Name = local.name }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = { Name = "${local.name}-public-${count.index + 1}" }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = { Name = "${local.name}-public" }
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "pipeline_tasks" {
  name        = "${local.name}-pipeline-tasks"
  description = "Scheduled pipeline ECS task security group"
  vpc_id      = aws_vpc.main.id

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_cloudwatch_log_group" "pipeline" {
  name              = "/ecs/${local.name}/pipeline"
  retention_in_days = 14
}

resource "aws_ecs_cluster" "main" {
  name = local.name
}

resource "aws_iam_role" "ecs_execution" {
  name = "${local.name}-ecs-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "pipeline_task" {
  name = "${local.name}-pipeline-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "ecs-tasks.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "pipeline_task_s3" {
  name = "${local.name}-s3-read-write"
  role = aws_iam_role.pipeline_task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["s3:ListBucket"]
        Resource = aws_s3_bucket.artifacts.arn
        Condition = {
          StringLike = {
            "s3:prefix" = local.artifact_prefix == "" ? ["*"] : ["${local.artifact_prefix}/*"]
          }
        }
      },
      {
        Effect   = "Allow"
        Action   = ["s3:GetObject"]
        Resource = local.artifact_object_arn
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = local.artifact_write_arns
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.static_site.arn}/index.html"
      }
    ]
  })
}

resource "aws_ecs_task_definition" "pipeline" {
  family                   = "${local.name}-pipeline"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.pipeline_cpu
  memory                   = var.pipeline_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.pipeline_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  ephemeral_storage {
    size_in_gib = var.pipeline_ephemeral_storage_gib
  }

  container_definitions = jsonencode([
    {
      name      = "pipeline"
      image     = var.pipeline_image_uri
      essential = true
      command = [
        "pipeline",
        "--artifact-store-uri",
        local.artifact_store_uri,
        "--model-artifact-uri",
        local.model_artifact_uri,
        "--static-site-uri",
        "s3://${aws_s3_bucket.static_site.bucket}",
        "--workdir",
        "/var/lib/dkenergy"
      ]
      environment = [
        { name = "DKENERGY_BUILD_GIT_SHA", value = var.build_git_sha },
        { name = "WITH_WEATHER", value = "1" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.pipeline.name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "pipeline"
        }
      }
    }
  ])
}

resource "aws_iam_role" "scheduler" {
  count = var.enable_pipeline_schedule ? 1 : 0
  name  = "${local.name}-scheduler"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Action = "sts:AssumeRole"
      Effect = "Allow"
      Principal = {
        Service = "scheduler.amazonaws.com"
      }
    }]
  })
}

resource "aws_iam_role_policy" "scheduler" {
  count = var.enable_pipeline_schedule ? 1 : 0
  name  = "${local.name}-run-task"
  role  = aws_iam_role.scheduler[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecs:RunTask"]
        Resource = aws_ecs_task_definition.pipeline.arn
      },
      {
        Effect = "Allow"
        Action = ["iam:PassRole"]
        Resource = [
          aws_iam_role.ecs_execution.arn,
          aws_iam_role.pipeline_task.arn
        ]
      }
    ]
  })
}

resource "aws_scheduler_schedule" "pipeline" {
  count                        = var.enable_pipeline_schedule ? 1 : 0
  name                         = "${local.name}-pipeline"
  schedule_expression          = var.forecast_schedule_expression
  schedule_expression_timezone = var.forecast_schedule_timezone

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_ecs_cluster.main.arn
    role_arn = aws_iam_role.scheduler[0].arn

    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.pipeline.arn
      launch_type         = "FARGATE"

      network_configuration {
        subnets          = aws_subnet.public[*].id
        security_groups  = [aws_security_group.pipeline_tasks.id]
        assign_public_ip = true
      }
    }

    retry_policy {
      maximum_event_age_in_seconds = 300
      maximum_retry_attempts       = 0
    }
  }
}
