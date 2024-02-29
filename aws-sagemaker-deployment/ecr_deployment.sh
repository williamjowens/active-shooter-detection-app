#!/bin/bash

# Configuration variables
REGION="us-west-1"
ACCOUNT_ID="339712991889"
REPO_NAME="active-shooter-detector"
IMAGE_NAME="active-shooter-detector"
IMAGE_TAG="latest"

# Create ECR repository
aws ecr create-repository --repository-name ${REPO_NAME} || true

# Login to ECR
aws ecr get-login-password --region ${REGION} | docker login --username AWS --password-stdin ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com

# Build Docker image
docker build -t ${IMAGE_NAME}:${IMAGE_TAG} -f Dockerfile_sm .

# Tag Docker image
docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}

# Push Docker image to ECR
docker push ${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/${REPO_NAME}:${IMAGE_TAG}