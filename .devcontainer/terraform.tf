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

resource "aws_ecr_repository" "reimpl-dev-containers" {
  name                 = "reimpl-dev-containers"
  image_tag_mutability = "MUTABLE"
}
