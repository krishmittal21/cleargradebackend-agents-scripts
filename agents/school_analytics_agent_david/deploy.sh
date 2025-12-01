#!/bin/bash

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="${GCP_PROJECT_ID:-clearmarks}"
SERVICE_NAME="david-business-agent"
REGION="asia-south2"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Check if running on M1/ARM
if [[ $(uname -m) == "arm64" ]]; then
    echo -e "${YELLOW}M1/ARM64 detected. Building for linux/amd64...${NC}"
    PLATFORM="--platform linux/amd64"
else
    PLATFORM=""
fi

# Validate inputs
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: GCP_PROJECT_ID not set${NC}"
    echo "Usage: GCP_PROJECT_ID=your-project ./deploy-cloudrun.sh"
    exit 1
fi


# Step 3: Build Docker image
echo -e "${YELLOW}Step 3: Building Docker image for linux/amd64...${NC}"
docker build $PLATFORM \
    -t ${IMAGE_NAME}:latest \
    -t ${IMAGE_NAME}:$(date +%Y%m%d_%H%M%S) \
    .

# Step 4: Push to Google Container Registry
echo -e "${YELLOW}Step 4: Pushing image to GCR...${NC}"
docker push ${IMAGE_NAME}:latest

# Step 5: Deploy to Cloud Run
echo -e "${YELLOW}Step 5: Deploying to Cloud Run...${NC}"
gcloud run deploy $SERVICE_NAME \
    --image ${IMAGE_NAME}:latest \
    --platform managed \
    --region $REGION \
    --memory 1Gi \
    --cpu 2 \
    --timeout 3600 \
    --max-instances 1 \
    --min-instances 0 \
    --allow-unauthenticated \
    --set-env-vars \
        "OPENROUTER_API_KEY=sk-9McOzvw3H9Bm0O2jAYW4JadNHsvjhAzTXedpB8zUKONDvmoW,\
OPENROUTER_BASE_URL=https://api.moonshot.ai/v1,\
OPENROUTER_MODEL=kimi-k2-turbo-preview,\
OPENROUTER_SITE_URL=https://analyse.com,\
OPENROUTER_SITE_NAME=Analyse,\
GCP_PROJECT_ID=clearmarks,\
FIRESTORE_PROJECT_ID=clearmarks" \

# Step 6: Display service info
echo -e "${YELLOW}Step 6: Retrieving service information...${NC}"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --platform managed \
    --region $REGION \
    --format 'value(status.url)')

echo -e "${GREEN}✓ Deployment successful!${NC}"
echo -e "${GREEN}Service URL: ${SERVICE_URL}${NC}"
echo -e "${GREEN}Health check: ${SERVICE_URL}/health${NC}"


echo -e "${GREEN}✓ Setup complete!${NC}"
