"""
LangSmith to GCS Export Service

A Cloud Run service that fetches traces from LangSmith and exports them to GCS.
Triggered by Cloud Scheduler every 14 days.
"""

import os
import json
import logging
from datetime import datetime, timedelta, timezone
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

from exporter import LangSmithExporter

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LangSmith to GCS Exporter",
    description="Exports LangSmith traces to Google Cloud Storage",
    version="1.0.0"
)

# Initialize exporter
exporter = LangSmithExporter(
    api_key=os.getenv("LANGCHAIN_API_KEY"),
    project_name=os.getenv("LANGCHAIN_PROJECT", "default"),
    bucket_name=os.getenv("GCS_BUCKET_NAME")
)


class ExportRequest(BaseModel):
    """Request model for export endpoint."""
    days_back: int = 14
    project_name: Optional[str] = None


class ExportResponse(BaseModel):
    """Response model for export endpoint."""
    status: str
    message: str
    file_path: Optional[str] = None
    traces_count: Optional[int] = None


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.post("/export", response_model=ExportResponse)
async def export_traces(request: ExportRequest = ExportRequest()):
    """
    Export traces from LangSmith to GCS.
    
    This endpoint fetches all traces from the last N days (default 14)
    and uploads them as JSON to Google Cloud Storage.
    """
    try:
        logger.info(f"Starting export for last {request.days_back} days")
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=request.days_back)
        
        # Use custom project name if provided
        project = request.project_name or os.getenv("LANGCHAIN_PROJECT", "default")
        
        # Fetch and export traces
        result = exporter.export_traces(
            start_date=start_date,
            end_date=end_date,
            project_name=project
        )
        
        logger.info(f"Export completed: {result}")
        
        return ExportResponse(
            status="success",
            message=f"Successfully exported {result['traces_count']} traces",
            file_path=result['gcs_path'],
            traces_count=result['traces_count']
        )
        
    except Exception as e:
        logger.error(f"Export failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with service info."""
    return {
        "service": "LangSmith to GCS Exporter",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/export": "POST - Export traces to GCS"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
