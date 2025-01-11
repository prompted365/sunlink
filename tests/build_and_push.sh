#!/bin/bash

# GCP Project ID
PROJECT_ID="solarflux-harmony"

# Build and push the Worker image
docker buildx build \
  --platform=linux/amd64,linux/arm64 \
  -t sunlinkai-artifacts/${PROJECT_ID}/sunlink-worker:latest \
  -f Dockerfile.worker \
  --push

# Build and push the API image
docker buildx build \
  --platform=linux/amd64,linux/arm64 \
  -t sunlinkai-artifacts/${PROJECT_ID}/sunlink-api:latest \
  -f Dockerfile.api \
  --push

echo "Docker images pushed to GCR successfully!"