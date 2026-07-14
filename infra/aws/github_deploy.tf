resource "aws_iam_openid_connect_provider" "github" {
  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]

  tags = {
    Environment = var.environment
    Project     = var.project_name
    Purpose     = "github-actions-oidc"
  }
}

data "aws_iam_policy_document" "github_deploy_trust" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:environment:${var.github_environment}"]
    }
  }
}

resource "aws_iam_role" "github_deploy" {
  name               = "${local.name}-github-deploy"
  assume_role_policy = data.aws_iam_policy_document.github_deploy_trust.json

  tags = {
    Environment = var.environment
    Project     = var.project_name
    Purpose     = "github-actions-deployment"
  }
}

resource "aws_iam_role_policy_attachment" "github_deploy_power_user" {
  role       = aws_iam_role.github_deploy.name
  policy_arn = "arn:aws:iam::aws:policy/PowerUserAccess"
}

locals {
  runtime_role_arns = concat(
    [
      aws_iam_role.ecs_execution.arn,
      aws_iam_role.pipeline_task.arn,
    ],
    aws_iam_role.scheduler[*].arn,
  )
}

data "aws_iam_policy_document" "github_deploy_iam" {
  statement {
    sid = "ReadDeploymentIdentity"
    actions = [
      "iam:GetOpenIDConnectProvider",
      "iam:GetRole",
      "iam:GetRolePolicy",
      "iam:ListAttachedRolePolicies",
      "iam:ListRolePolicies",
    ]
    resources = concat(
      local.runtime_role_arns,
      [
        aws_iam_openid_connect_provider.github.arn,
        aws_iam_role.github_deploy.arn,
      ],
    )
  }

  statement {
    sid = "ManageRuntimeRoles"
    actions = [
      "iam:AttachRolePolicy",
      "iam:CreateRole",
      "iam:DeleteRole",
      "iam:DeleteRolePolicy",
      "iam:DetachRolePolicy",
      "iam:PassRole",
      "iam:PutRolePolicy",
      "iam:TagRole",
      "iam:UntagRole",
      "iam:UpdateAssumeRolePolicy",
    ]
    resources = local.runtime_role_arns
  }

  statement {
    sid = "ReadManagedPolicies"
    actions = [
      "iam:GetPolicy",
      "iam:GetPolicyVersion",
      "iam:ListPolicyVersions",
    ]
    resources = ["arn:aws:iam::aws:policy/*"]
  }
}

resource "aws_iam_role_policy" "github_deploy_iam" {
  name   = "${local.name}-runtime-iam"
  role   = aws_iam_role.github_deploy.id
  policy = data.aws_iam_policy_document.github_deploy_iam.json
}
