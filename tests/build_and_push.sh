#!/bin/bash

# GCP Project ID
PROJECT_ID="solarflux-harmony"
# Artifact Registry region
REGION="us-central1"
# Repository name
REPOSITORY_NAME="sunlinkai-artifacts"

# Build and push the Worker image
docker buildx build \
  --platform=linux/amd64,linux/arm64 \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/sunlink-worker:latest \
  -f Dockerfile.worker \
  --push \
  .

# Build and push the API image
docker buildx build \
  --platform=linux/amd64,linux/arm64 \
  -t ${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPOSITORY_NAME}/sunlink-api:latest \
  -f Dockerfile.api \
  --push \
  .

echo "Docker images pushed to Artifact Registry successfully!"