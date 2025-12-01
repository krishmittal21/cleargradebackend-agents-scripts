#!/bin/bash

set -e

PROJECT_ID=${1:-$(gcloud config get-value project)}
SERVICE_NAME="users-service"
REGION=${2:-"asia-south2"}
MEMORY="512Mi"
CPU="1"

if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: Project ID required"
    echo "Usage: ./deploy.sh <project-id> [region]"
    exit 1
fi

echo "🚀 Deploying to Cloud Run..."
echo "  Project: $PROJECT_ID"
echo "  Service: $SERVICE_NAME"
echo "  Region: $REGION"
echo ""

# Enable APIs
echo "📡 Enabling APIs..."
gcloud services enable \
    run.googleapis.com \
    firestore.googleapis.com \
    cloudbuild.googleapis.com \
    --project="$PROJECT_ID" \
    --quiet

# Deploy
echo "📦 Deploying..."
gcloud run deploy "$SERVICE_NAME" \
    --source=. \
    --platform=managed \
    --region="$REGION" \
    --memory="$MEMORY" \
    --cpu="$CPU" \
    --timeout="300" \
    --max-instances="3" \
    --set-env-vars="FIRESTORE_PROJECT_ID=${PROJECT_ID},LOG_LEVEL=INFO" \
    --allow-unauthenticated \
    --quiet \
    --project="$PROJECT_ID"

echo "✅ Deployment complete!"
echo ""

SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --platform=managed \
    --region="$REGION" \
    --format='value(status.url)' \
    --project="$PROJECT_ID")

echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "Test:"
echo "  curl $SERVICE_URL/health"
echo ""
echo "API Docs:"
echo "  $SERVICE_URL/docs"