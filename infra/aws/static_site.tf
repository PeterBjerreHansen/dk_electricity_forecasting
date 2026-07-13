locals {
  static_site_bucket_name = (
    var.static_site_bucket_name != ""
    ? var.static_site_bucket_name
    : "${var.project_name}-site-${data.aws_caller_identity.current.account_id}"
  )
}

resource "aws_s3_bucket" "static_site" {
  count  = var.enable_static_site ? 1 : 0
  bucket = local.static_site_bucket_name

  tags = {
    Environment = var.environment
    Project     = var.project_name
    Purpose     = "public-static-dashboard"
  }
}

resource "aws_s3_bucket_ownership_controls" "static_site" {
  count  = var.enable_static_site ? 1 : 0
  bucket = aws_s3_bucket.static_site[0].id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "static_site" {
  count  = var.enable_static_site ? 1 : 0
  bucket = aws_s3_bucket.static_site[0].id

  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_server_side_encryption_configuration" "static_site" {
  count  = var.enable_static_site ? 1 : 0
  bucket = aws_s3_bucket.static_site[0].id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_website_configuration" "static_site" {
  count  = var.enable_static_site ? 1 : 0
  bucket = aws_s3_bucket.static_site[0].id

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "index.html"
  }
}

data "aws_iam_policy_document" "static_site" {
  count = var.enable_static_site ? 1 : 0

  statement {
    sid       = "PublicReadStaticDashboard"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.static_site[0].arn}/*"]

    principals {
      type        = "*"
      identifiers = ["*"]
    }
  }
}

resource "aws_s3_bucket_policy" "static_site" {
  count = var.enable_static_site ? 1 : 0

  bucket = aws_s3_bucket.static_site[0].id
  policy = data.aws_iam_policy_document.static_site[0].json

  depends_on = [aws_s3_bucket_public_access_block.static_site]
}
