terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "5.35.0"
    }
  }
  backend "s3" {
    bucket = "a-gn-cicd-data"
    key    = "terraform/state/reimpl-docker-image/terraform.tfstate"
    region = "us-east-1"
  }
}

# ECR repository for final images
resource "aws_ecr_repository" "reimpl-dev-containers" {
  name                 = "reimpl-dev-containers"
  image_tag_mutability = "MUTABLE"
}

# resource "aws_s3_bucket" "reimpl_dev_containers_cache" {
#   bucket = "reimpl_dev_containers_cache"
# }

# resource "aws_s3_bucket_acl" "reimpl_dev_containers_cache_acl" {
#   bucket = aws_s3_bucket.reimpl_dev_containers_cache.id
#   acl    = "private"
# }

# # CodeBuild project to build images
# resource "aws_codebuild_project" "reimpl_dev_docker_image" {
#   name          = "reimpl-dev-docker-image"
#   description   = "Build Docker image for the reimpl project."
#   build_timeout = 15
#   service_role  = aws_iam_role.reimpl_dev_docker_image_builder.arn

#   artifacts {
#     type = "NO_ARTIFACTS"
#   }

#   cache {
#     type     = "S3"
#     location = aws_s3_bucket.reimpl_dev_containers_cache.bucket
#   }

#   environment {
#     compute_type                = "BUILD_GENERAL1_SMALL"
#     image                       = "aws/codebuild/amazonlinux2-aarch64-standard:3.0"
#     type                        = "LINUX_CONTAINER"
#     image_pull_credentials_type = "CODEBUILD"

#     environment_variable {
#       name  = "SOME_KEY1"
#       value = "SOME_VALUE1"
#     }

#     environment_variable {
#       name  = "SOME_KEY2"
#       value = "SOME_VALUE2"
#       type  = "PARAMETER_STORE"
#     }
#   }

#   logs_config {
#     cloudwatch_logs {
#       group_name  = "log-group"
#       stream_name = "log-stream"
#     }

#     s3_logs {
#       status   = "ENABLED"
#       location = "${aws_s3_bucket.reimpl_dev_containers_cache.id}/build-log"
#     }
#   }

#   source {
#     type            = "GITHUB"
#     location        = "https://github.com/a-gn/reimpl.git"
#     git_clone_depth = 1

#     git_submodules_config {
#       fetch_submodules = false
#     }
#   }

#   source_version = "main"

#   vpc_config {
#     vpc_id = aws_vpc.example.id

#     subnets = [
#       aws_subnet.example1.id,
#       aws_subnet.example2.id,
#     ]

#     security_group_ids = [
#       aws_security_group.example1.id,
#       aws_security_group.example2.id,
#     ]
#   }

#   tags = {
#     Environment = "Test"
#   }
# }

# data "aws_iam_policy_document" "assume_role" {
#   statement {
#     effect = "Allow"

#     principals {
#       type        = "Service"
#       identifiers = ["codebuild.amazonaws.com"]
#     }

#     actions = ["sts:AssumeRole"]
#   }
# }

# resource "aws_iam_role" "reimpl_dev_docker_image_builder" {
#   name               = "reimpl-dev-docker-image-builder"
#   assume_role_policy = data.aws_iam_policy_document.assume_role.json
# }

# data "aws_iam_policy_document" "example" {
#   statement {
#     effect = "Allow"

#     actions = [
#       "logs:CreateLogGroup",
#       "logs:CreateLogStream",
#       "logs:PutLogEvents",
#     ]

#     resources = ["*"]
#   }

#   statement {
#     effect = "Allow"

#     actions = [
#       "ec2:CreateNetworkInterface",
#       "ec2:DescribeDhcpOptions",
#       "ec2:DescribeNetworkInterfaces",
#       "ec2:DeleteNetworkInterface",
#       "ec2:DescribeSubnets",
#       "ec2:DescribeSecurityGroups",
#       "ec2:DescribeVpcs",
#     ]

#     resources = ["*"]
#   }

#   statement {
#     effect    = "Allow"
#     actions   = ["ec2:CreateNetworkInterfacePermission"]
#     resources = ["arn:aws:ec2:us-east-1:123456789012:network-interface/*"]

#     condition {
#       test     = "StringEquals"
#       variable = "ec2:Subnet"

#       values = [
#         aws_subnet.example1.arn,
#         aws_subnet.example2.arn,
#       ]
#     }

#     condition {
#       test     = "StringEquals"
#       variable = "ec2:AuthorizedService"
#       values   = ["codebuild.amazonaws.com"]
#     }
#   }
# }

# resource "aws_iam_role_policy" "reimpl_dev_docker_image_builder_policy" {
#   role   = aws_iam_role.reimpl_dev_docker_image_builder.name
#   policy = data.aws_iam_policy_document.example.json
# }
