"""
LangSmith Exporter Module

Handles fetching traces from LangSmith and uploading to GCS.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from langsmith import Client
from google.cloud import storage

logger = logging.getLogger(__name__)


class LangSmithExporter:
    """
    Exports traces from LangSmith to Google Cloud Storage.
    """
    
    def __init__(
        self,
        api_key: str,
        project_name: str = "default",
        bucket_name: Optional[str] = None
    ):
        """
        Initialize the exporter.
        
        Args:
            api_key: LangSmith API key
            project_name: LangSmith project name
            bucket_name: GCS bucket name for storing exports
        """
        self.api_key = api_key
        self.project_name = project_name
        self.bucket_name = bucket_name
        
        # Initialize LangSmith client
        self.langsmith_client = Client(api_key=api_key)
        
        # Initialize GCS client
        self.storage_client = storage.Client()
        
    def fetch_traces(
        self,
        start_date: datetime,
        end_date: datetime,
        project_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Fetch all traces from LangSmith within the date range.
        
        Args:
            start_date: Start of the date range
            end_date: End of the date range
            project_name: Optional project name override
            
        Returns:
            List of trace dictionaries
        """
        project = project_name or self.project_name
        logger.info(f"Fetching traces from {start_date} to {end_date} for project: {project}")
        
        traces = []
        
        # Fetch all runs (traces) from the project
        # The SDK handles pagination automatically when no limit is set
        runs = self.langsmith_client.list_runs(
            project_name=project,
            start_time=start_date,
            end_time=end_date,
            is_root=True  # Only get root runs (top-level traces)
        )
        
        for run in runs:
            # Calculate latency from start/end times if available
            latency_ms = None
            if run.start_time and run.end_time:
                latency_ms = (run.end_time - run.start_time).total_seconds() * 1000
            
            # Convert run to dictionary for JSON serialization
            # Use getattr with defaults for optional attributes
            trace_data = {
                "id": str(run.id),
                "name": run.name,
                "run_type": run.run_type,
                "status": run.status,
                "start_time": run.start_time.isoformat() if run.start_time else None,
                "end_time": run.end_time.isoformat() if run.end_time else None,
                "inputs": run.inputs,
                "outputs": run.outputs,
                "error": run.error,
                "latency_ms": latency_ms,
                "total_tokens": getattr(run, 'total_tokens', None),
                "prompt_tokens": getattr(run, 'prompt_tokens', None),
                "completion_tokens": getattr(run, 'completion_tokens', None),
                "total_cost": getattr(run, 'total_cost', None),
                "feedback_stats": getattr(run, 'feedback_stats', None),
                "tags": getattr(run, 'tags', None),
                "metadata": run.extra.get("metadata") if run.extra else None,
            }
            traces.append(trace_data)
            
        logger.info(f"Fetched {len(traces)} traces")
        return traces
    
    def upload_to_gcs(
        self,
        data: List[Dict[str, Any]],
        filename: str,
        bucket_name: Optional[str] = None
    ) -> str:
        """
        Upload data as JSON to GCS.
        
        Args:
            data: List of trace dictionaries
            filename: Name for the file in GCS
            bucket_name: Optional bucket name override
            
        Returns:
            GCS path of the uploaded file
        """
        bucket = bucket_name or self.bucket_name
        if not bucket:
            raise ValueError("No GCS bucket name provided")
            
        logger.info(f"Uploading to gs://{bucket}/{filename}")
        
        # Get bucket and create blob
        gcs_bucket = self.storage_client.bucket(bucket)
        blob = gcs_bucket.blob(filename)
        
        # Upload JSON data
        json_data = json.dumps(data, indent=2, default=str)
        blob.upload_from_string(json_data, content_type="application/json")
        
        gcs_path = f"gs://{bucket}/{filename}"
        logger.info(f"Upload complete: {gcs_path}")
        
        return gcs_path
    
    def export_traces(
        self,
        start_date: datetime,
        end_date: datetime,
        project_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Fetch traces from LangSmith and upload to GCS.
        
        Args:
            start_date: Start of the date range
            end_date: End of the date range
            project_name: Optional project name override
            
        Returns:
            Dictionary with export results
        """
        # Fetch traces
        traces = self.fetch_traces(start_date, end_date, project_name)
        
        if not traces:
            logger.warning("No traces found for the specified date range")
            return {
                "traces_count": 0,
                "gcs_path": None,
                "message": "No traces found"
            }
        
        # Generate filename with timestamp
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        project = project_name or self.project_name
        filename = f"langsmith-traces/{project}/{timestamp}_traces.json"
        
        # Upload to GCS
        gcs_path = self.upload_to_gcs(traces, filename)
        
        return {
            "traces_count": len(traces),
            "gcs_path": gcs_path,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat()
        }
