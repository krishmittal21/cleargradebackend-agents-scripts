# LangSmith to GCS Exporter

A Cloud Run service that exports LangSmith traces to Google Cloud Storage.

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
cp .env.example .env
# Edit .env with your values

# Run locally
uvicorn main:app --reload --port 8080

# Test the export endpoint
curl -X POST http://localhost:8080/export
```

### Deploy to Cloud Run

```bash
# Set variables
export PROJECT_ID=your-gcp-project
export REGION=us-central1
export SERVICE_NAME=langsmith-exporter

# Build and deploy
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${SERVICE_NAME}

gcloud run deploy ${SERVICE_NAME} \
    --image gcr.io/${PROJECT_ID}/${SERVICE_NAME} \
    --platform managed \
    --region ${REGION} \
    --allow-unauthenticated \
    --memory 512Mi \
    --set-env-vars "LANGCHAIN_API_KEY=your-key" \
    --set-env-vars "LANGCHAIN_PROJECT=default" \
    --set-env-vars "GCS_BUCKET_NAME=your-bucket"
```

### Set Up Cloud Scheduler (Every 14 Days)

```bash
gcloud scheduler jobs create http ${SERVICE_NAME}-job \
    --schedule="0 0 */14 * *" \
    --uri="https://YOUR_SERVICE_URL/export" \
    --http-method=POST \
    --location ${REGION}
```

## API Endpoints

- `GET /health` - Health check
- `POST /export` - Export traces to GCS
  - Optional body: `{"days_back": 14, "project_name": "optional-override"}`
