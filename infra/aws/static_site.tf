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
  from = aws_s3_bucket_policy.static_site[0]
  to   = aws_s3_bucket_policy.static_site
}

resource "aws_s3_bucket" "static_site" {
  bucket = local.static_site_bucket_name

  tags = {
    Environment = var.environment
    Project     = var.project_name
    Purpose     = "static-dashboard-origin"
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
  block_public_policy     = true
  restrict_public_buckets = true
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

resource "aws_cloudfront_origin_access_control" "static_site" {
  name                              = "${local.name}-static-site"
  description                       = "Private S3 access for the forecast dashboard"
  origin_access_control_origin_type = "s3"
  signing_behavior                  = "always"
  signing_protocol                  = "sigv4"
}

resource "aws_cloudfront_cache_policy" "static_site" {
  name        = "${local.name}-static-site"
  comment     = "Five-minute compressed cache for the daily forecast dashboard"
  default_ttl = 300
  max_ttl     = 300
  min_ttl     = 0

  parameters_in_cache_key_and_forwarded_to_origin {
    enable_accept_encoding_brotli = true
    enable_accept_encoding_gzip   = true

    cookies_config {
      cookie_behavior = "none"
    }

    headers_config {
      header_behavior = "none"
    }

    query_strings_config {
      query_string_behavior = "none"
    }
  }
}

resource "aws_cloudfront_distribution" "static_site" {
  enabled             = true
  is_ipv6_enabled     = true
  comment             = "Danish electricity price forecast dashboard"
  default_root_object = "index.html"
  price_class         = "PriceClass_100"

  origin {
    domain_name              = aws_s3_bucket.static_site.bucket_regional_domain_name
    origin_id                = "static-site-s3"
    origin_access_control_id = aws_cloudfront_origin_access_control.static_site.id
  }

  default_cache_behavior {
    allowed_methods        = ["GET", "HEAD"]
    cached_methods         = ["GET", "HEAD"]
    target_origin_id       = "static-site-s3"
    viewer_protocol_policy = "redirect-to-https"
    compress               = true
    cache_policy_id        = aws_cloudfront_cache_policy.static_site.id
  }

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  viewer_certificate {
    cloudfront_default_certificate = true
  }

  tags = {
    Environment = var.environment
    Project     = var.project_name
    Purpose     = "static-dashboard-delivery"
  }
}

data "aws_iam_policy_document" "static_site" {
  statement {
    sid       = "CloudFrontReadStaticDashboard"
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.static_site.arn}/index.html"]

    principals {
      type        = "Service"
      identifiers = ["cloudfront.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "AWS:SourceArn"
      values   = [aws_cloudfront_distribution.static_site.arn]
    }
  }
}

resource "aws_s3_bucket_policy" "static_site" {
  bucket = aws_s3_bucket.static_site.id
  policy = data.aws_iam_policy_document.static_site.json

  depends_on = [aws_s3_bucket_public_access_block.static_site]
}
