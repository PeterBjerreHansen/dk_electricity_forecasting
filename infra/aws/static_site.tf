locals {
  static_site_bucket_name = (
    var.static_site_bucket_name != ""
    ? var.static_site_bucket_name
    : "${var.project_name}-site-${data.aws_caller_identity.current.account_id}"
  )
}

moved {
  from = aws_s3_bucket.static_site[0]
  to   = aws_s3_bucket.static_site
}

moved {
  from = aws_s3_bucket_ownership_controls.static_site[0]
  to   = aws_s3_bucket_ownership_controls.static_site
}

moved {
  from = aws_s3_bucket_public_access_block.static_site[0]
  to   = aws_s3_bucket_public_access_block.static_site
}

moved {
  from = aws_s3_bucket_server_side_encryption_configuration.static_site[0]
  to   = aws_s3_bucket_server_side_encryption_configuration.static_site
}

moved {
  from = aws_s3_bucket_website_configuration.static_site[0]
  to   = aws_s3_bucket_website_configuration.static_site
}

moved {
  from = aws_s3_bucket_policy.static_site[0]
  to   = aws_s3_bucket_policy.static_site
}

resource "aws_s3_bucket" "static_site" {
  bucket = local.static_site_bucket_name

  tags = {
    Environment = var.environment
    Project     = var.project_name
    Purpose     = "public-static-dashboard"
  }
}

resource "aws_s3_bucket_ownership_controls" "static_site" {
  bucket = aws_s3_bucket.static_site.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

resource "aws_s3_bucket_public_access_block" "static_site" {
  bucket = aws_s3_bucket.static_site.id

  block_public_acls       = true
  ignore_public_acls      = true
  block_public_policy     = false
  restrict_public_buckets = false
}

resource "aws_s3_bucket_versioning" "static_site" {
  bucket = aws_s3_bucket.static_site.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "static_site" {
  bucket = aws_s3_bucket.static_site.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_website_configuration" "static_site" {
  bucket = aws_s3_bucket.static_site.id

  index_document {
    suffix = "index.html"
  }

  error_document {
    key = "index.html"
  }
}

data "aws_iam_policy_document" "static_site" {
  statement {
    sid       = "PublicReadStaticDashboard"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.static_site.arn}/index.html"]

    principals {
      type        = "*"
      identifiers = ["*"]
    }
  }
}

resource "aws_s3_bucket_policy" "static_site" {
  bucket = aws_s3_bucket.static_site.id
  policy = data.aws_iam_policy_document.static_site.json

  depends_on = [aws_s3_bucket_public_access_block.static_site]
}
