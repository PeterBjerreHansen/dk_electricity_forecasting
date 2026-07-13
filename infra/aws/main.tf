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
  model_artifact_key = trimprefix(
    local.production_config.primary.artifact_path,
    "artifacts/"
  )
  model_artifact_uri = "${local.artifact_store_uri}/${local.model_artifact_key}"
}

resource "aws_ecr_repository" "web" {
  count                = var.enable_web ? 1 : 0
  name                 = "${var.project_name}-web"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "pipeline" {
  name                 = "${var.project_name}-pipeline"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "web" {
  count      = var.enable_web ? 1 : 0
  repository = aws_ecr_repository.web[0].name
  policy     = local.ecr_lifecycle_policy
}

resource "aws_ecr_lifecycle_policy" "pipeline" {
  repository = aws_ecr_repository.pipeline.name
  policy     = local.ecr_lifecycle_policy
}

locals {
  ecr_lifecycle_policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep the last 20 images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 20
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

  tags = {
    Name = local.name
  }
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id

  tags = {
    Name = local.name
  }
}

resource "aws_subnet" "public" {
  count                   = 2
  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(aws_vpc.main.cidr_block, 8, count.index)
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = {
    Name = "${local.name}-public-${count.index + 1}"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = {
    Name = "${local.name}-public"
  }
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

data "aws_ec2_managed_prefix_list" "cloudfront_origin_facing" {
  count = var.enable_web ? 1 : 0
  name  = "com.amazonaws.global.cloudfront.origin-facing"
}

resource "aws_security_group" "alb" {
  count       = var.enable_web ? 1 : 0
  name        = "${local.name}-alb"
  description = "CloudFront-only HTTP ingress for dashboard ALB"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 80
    to_port         = 80
    protocol        = "tcp"
    prefix_list_ids = [data.aws_ec2_managed_prefix_list.cloudfront_origin_facing[0].id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "web_tasks" {
  count       = var.enable_web ? 1 : 0
  name        = "${local.name}-web-tasks"
  description = "Streamlit ECS task security group"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 8501
    to_port         = 8501
    protocol        = "tcp"
    security_groups = [aws_security_group.alb[0].id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
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

resource "aws_lb" "web" {
  count              = var.enable_web ? 1 : 0
  name               = replace(substr(local.name, 0, 32), "_", "-")
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb[0].id]
  subnets            = aws_subnet.public[*].id
}

resource "aws_lb_target_group" "web" {
  count       = var.enable_web ? 1 : 0
  name        = replace(substr("${local.name}-web", 0, 32), "_", "-")
  port        = 8501
  protocol    = "HTTP"
  target_type = "ip"
  vpc_id      = aws_vpc.main.id

  health_check {
    enabled             = true
    path                = "/_stcore/health"
    matcher             = "200-399"
    interval            = 30
    timeout             = 5
    healthy_threshold   = 2
    unhealthy_threshold = 3
  }
}

resource "aws_lb_listener" "web" {
  count             = var.enable_web ? 1 : 0
  load_balancer_arn = aws_lb.web[0].arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.web[0].arn
  }
}

resource "aws_cloudfront_distribution" "web" {
  count           = var.enable_web ? 1 : 0
  enabled         = true
  is_ipv6_enabled = true
  comment         = "${local.name} Streamlit dashboard"

  origin {
    domain_name = aws_lb.web[0].dns_name
    origin_id   = "alb"

    custom_origin_config {
      http_port              = 80
      https_port             = 443
      origin_protocol_policy = "http-only"
      origin_ssl_protocols   = ["TLSv1.2"]
    }
  }

  default_cache_behavior {
    target_origin_id       = "alb"
    viewer_protocol_policy = "redirect-to-https"
    allowed_methods        = ["GET", "HEAD", "OPTIONS", "PUT", "POST", "PATCH", "DELETE"]
    cached_methods         = ["GET", "HEAD"]
    min_ttl                = 0
    default_ttl            = 0
    max_ttl                = 0

    forwarded_values {
      query_string = true
      headers      = ["*"]
      cookies {
        forward = "all"
      }
    }
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }
}

resource "aws_cloudwatch_log_group" "web" {
  count             = var.enable_web ? 1 : 0
  name              = "/ecs/${local.name}/web"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "pipeline" {
  name              = "/ecs/${local.name}/pipeline"
  retention_in_days = 14
}

resource "aws_cloudwatch_log_group" "scoring" {
  count             = var.enable_published_scoring_schedule ? 1 : 0
  name              = "/ecs/${local.name}/scoring"
  retention_in_days = 14
}

resource "aws_ecs_cluster" "main" {
  name = local.name
}

resource "aws_iam_role" "ecs_execution" {
  name = "${local.name}-ecs-execution"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "ecs_execution" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role" "web_task" {
  count = var.enable_web ? 1 : 0
  name  = "${local.name}-web-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "web_task_s3" {
  count = var.enable_web ? 1 : 0
  name  = "${local.name}-s3-read"
  role  = aws_iam_role.web_task[0].id
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
      }
    ]
  })
}


resource "aws_iam_role" "pipeline_task" {
  name = "${local.name}-pipeline-task"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ecs-tasks.amazonaws.com"
        }
      }
    ]
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
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject"
        ]
        Resource = local.artifact_object_arn
      }
    ]
  })
}

resource "aws_ecs_task_definition" "web" {
  count                    = var.enable_web ? 1 : 0
  family                   = "${local.name}-web"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.web_cpu
  memory                   = var.web_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.web_task[0].arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([
    {
      name      = "web"
      image     = var.web_image_uri
      essential = true
      command   = ["web"]
      portMappings = [
        {
          containerPort = 8501
          hostPort      = 8501
          protocol      = "tcp"
        }
      ]
      environment = [
        { name = "DKENERGY_BUILD_GIT_SHA", value = var.build_git_sha },
        { name = "DKENERGY_ARTIFACT_STORE_URI", value = local.artifact_store_uri },
        { name = "DKENERGY_LATEST_POINTER", value = "${local.artifact_store_uri}/latest.json" },
        { name = "DKENERGY_PANEL_PATH", value = "${local.artifact_store_uri}/latest/price_panel_hourly_v1.parquet" },
        { name = "DKENERGY_PUBLISHED_HISTORY_PREDICTIONS_PATH", value = "${local.artifact_store_uri}/published_forecast_history/predictions.parquet" },
        { name = "DKENERGY_PUBLISHED_HISTORY_SCORES_PATH", value = "${local.artifact_store_uri}/published_forecast_history/model_scores.parquet" },
        { name = "DKENERGY_ENABLE_LEGACY_BACKTESTS", value = "0" },
        { name = "DKENERGY_CACHE_DIR", value = "/tmp/dkenergy-dashboard-cache" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.web[0].name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "web"
        }
      }
    }
  ])
}

resource "aws_ecs_service" "web" {
  count           = var.enable_web ? 1 : 0
  name            = "${local.name}-web"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.web[0].arn
  desired_count   = 1
  launch_type     = "FARGATE"

  network_configuration {
    subnets          = aws_subnet.public[*].id
    security_groups  = [aws_security_group.web_tasks[0].id]
    assign_public_ip = true
  }

  deployment_circuit_breaker {
    enable   = true
    rollback = true
  }

  health_check_grace_period_seconds = 60

  load_balancer {
    target_group_arn = aws_lb_target_group.web[0].arn
    container_name   = "web"
    container_port   = 8501
  }

  depends_on = [aws_lb_listener.web]
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
        "--workdir",
        "/var/lib/dkenergy"
      ]
      environment = [
        { name = "DKENERGY_BUILD_GIT_SHA", value = var.build_git_sha },
        { name = "DKENERGY_ARTIFACT_STORE_URI", value = local.artifact_store_uri },
        { name = "DKENERGY_MODEL_ARTIFACT_URI", value = local.model_artifact_uri },
        { name = "DKENERGY_WORKDIR", value = "/var/lib/dkenergy" },
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

resource "aws_ecs_task_definition" "published_scoring" {
  count                    = var.enable_published_scoring_schedule ? 1 : 0
  family                   = "${local.name}-published-scoring"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.scoring_cpu
  memory                   = var.scoring_memory
  execution_role_arn       = aws_iam_role.ecs_execution.arn
  task_role_arn            = aws_iam_role.pipeline_task.arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([
    {
      name      = "published-scoring"
      image     = var.pipeline_image_uri
      essential = true
      command = [
        "score-published-cloud",
        "--artifact-store-uri",
        local.artifact_store_uri,
        "--workdir",
        "/var/lib/dkenergy-scoring"
      ]
      environment = [
        { name = "DKENERGY_BUILD_GIT_SHA", value = var.build_git_sha },
        { name = "DKENERGY_ARTIFACT_STORE_URI", value = local.artifact_store_uri },
        { name = "DKENERGY_WORKDIR", value = "/var/lib/dkenergy-scoring" }
      ]
      logConfiguration = {
        logDriver = "awslogs"
        options = {
          awslogs-group         = aws_cloudwatch_log_group.scoring[0].name
          awslogs-region        = var.aws_region
          awslogs-stream-prefix = "published-scoring"
        }
      }
    }
  ])
}

resource "aws_iam_role" "scheduler" {
  count = local.enable_scheduled_operations ? 1 : 0
  name  = "${local.name}-scheduler"
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

resource "aws_iam_role_policy" "scheduler" {
  count = local.enable_scheduled_operations ? 1 : 0

  name = "${local.name}-run-task"
  role = aws_iam_role.scheduler[0].id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["ecs:RunTask"]
        Resource = local.scheduled_task_definition_arns
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
  count = var.enable_pipeline_schedule ? 1 : 0

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

resource "aws_scheduler_schedule" "published_scoring" {
  count = var.enable_published_scoring_schedule ? 1 : 0

  name                         = "${local.name}-published-scoring"
  schedule_expression          = var.published_scoring_schedule_expression
  schedule_expression_timezone = var.forecast_schedule_timezone

  flexible_time_window {
    mode = "OFF"
  }

  target {
    arn      = aws_ecs_cluster.main.arn
    role_arn = aws_iam_role.scheduler[0].arn

    ecs_parameters {
      task_definition_arn = aws_ecs_task_definition.published_scoring[0].arn
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
