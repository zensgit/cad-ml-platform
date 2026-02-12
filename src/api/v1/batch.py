"""Batch Processing API endpoints.

Provides endpoints for:
- Job submission
- Job status tracking
- Job results retrieval
- Job cancellation
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.batch.processor import BatchJobStatus, get_batch_processor

logger = logging.getLogger(__name__)
router = APIRouter(tags=["batch"])


class BatchItemRequest(BaseModel):
    """Single item in a batch request."""

    data: Dict[str, Any] = Field(..., description="Item data for processing")


class BatchJobRequest(BaseModel):
    """Request to create a batch job."""

    operation: str = Field(..., description="Operation type (e.g., 'ocr', 'analyze')")
    items: List[Dict[str, Any]] = Field(..., description="Items to process")
    priority: int = Field(5, ge=1, le=10, description="Job priority (1=lowest, 10=highest)")
    callback_url: Optional[str] = Field(None, description="URL for completion callback")
    max_retries: int = Field(3, ge=0, le=10, description="Max retries per item")
    concurrency: Optional[int] = Field(None, ge=1, le=20, description="Concurrent items")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Custom metadata")


class BatchJobResponse(BaseModel):
    """Response for batch job creation."""

    job_id: str
    operation: str
    status: str
    total_items: int
    created_at: float


class BatchJobStatusResponse(BaseModel):
    """Response for batch job status."""

    job_id: str
    operation: str
    status: str
    priority: int
    progress: float
    total_items: int
    completed_items: int
    successful_items: int
    failed_items: int
    created_at: float
    started_at: Optional[float]
    completed_at: Optional[float]
    user_id: Optional[str]
    tenant_id: Optional[str]


class BatchItemResult(BaseModel):
    """Result for a single batch item."""

    id: str
    status: str
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    retries: int


class BatchJobResultsResponse(BaseModel):
    """Response with full batch job results."""

    job_id: str
    operation: str
    status: str
    progress: float
    total_items: int
    successful_items: int
    failed_items: int
    items: List[BatchItemResult]


class BatchMetricsResponse(BaseModel):
    """Response for batch processor metrics."""

    jobs_submitted: int
    jobs_completed: int
    jobs_failed: int
    items_processed: int
    active_jobs: int
    queued_jobs: int
    total_jobs: int


@router.post("/jobs", response_model=BatchJobResponse)
async def create_batch_job(
    request: BatchJobRequest,
    api_key: str = Depends(get_api_key),
    x_user_id: Optional[str] = None,
    x_tenant_id: Optional[str] = None,
) -> BatchJobResponse:
    """Create a new batch processing job.

    Submits a batch of items for processing. The job is queued and
    processed asynchronously. Use the returned job_id to check status.

    Available operations depend on registered handlers.
    """
    processor = get_batch_processor()

    try:
        job = await processor.submit_job(
            operation=request.operation,
            items=request.items,
            priority=request.priority,
            user_id=x_user_id,
            tenant_id=x_tenant_id,
            callback_url=request.callback_url,
            max_retries=request.max_retries,
            concurrency=request.concurrency,
            metadata=request.metadata,
        )

        return BatchJobResponse(
            job_id=job.id,
            operation=job.operation,
            status=job.status.value,
            total_items=job.total_items,
            created_at=job.created_at,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/jobs/{job_id}", response_model=BatchJobStatusResponse)
async def get_job_status(
    job_id: str,
    api_key: str = Depends(get_api_key),
) -> BatchJobStatusResponse:
    """Get the status of a batch job."""
    processor = get_batch_processor()
    status = await processor.get_job_status(job_id)

    if not status:
        raise HTTPException(status_code=404, detail="Job not found")

    return BatchJobStatusResponse(**status)


@router.get("/jobs/{job_id}/results", response_model=BatchJobResultsResponse)
async def get_job_results(
    job_id: str,
    api_key: str = Depends(get_api_key),
) -> BatchJobResultsResponse:
    """Get detailed results of a batch job including all item results."""
    processor = get_batch_processor()
    results = await processor.get_job_results(job_id)

    if not results:
        raise HTTPException(status_code=404, detail="Job not found")

    return BatchJobResultsResponse(
        job_id=results["job_id"],
        operation=results["operation"],
        status=results["status"],
        progress=results["progress"],
        total_items=results["total_items"],
        successful_items=results["successful_items"],
        failed_items=results["failed_items"],
        items=[BatchItemResult(**item) for item in results["items"]],
    )


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(
    job_id: str,
    api_key: str = Depends(get_api_key),
) -> Dict[str, Any]:
    """Cancel a pending or running batch job."""
    processor = get_batch_processor()
    success = await processor.cancel_job(job_id)

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Job not found or cannot be cancelled",
        )

    return {"status": "cancelled", "job_id": job_id}


@router.get("/jobs", response_model=List[BatchJobStatusResponse])
async def list_jobs(
    api_key: str = Depends(get_api_key),
    user_id: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> List[BatchJobStatusResponse]:
    """List batch jobs with optional filtering."""
    processor = get_batch_processor()

    # Convert status string to enum
    status_enum = None
    if status:
        try:
            status_enum = BatchJobStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    jobs = await processor.list_jobs(
        user_id=user_id,
        tenant_id=tenant_id,
        status=status_enum,
        limit=limit,
    )

    return [BatchJobStatusResponse(**job) for job in jobs]


@router.get("/metrics", response_model=BatchMetricsResponse)
async def get_batch_metrics(
    api_key: str = Depends(get_api_key),
) -> BatchMetricsResponse:
    """Get batch processor metrics."""
    processor = get_batch_processor()
    metrics = processor.get_metrics()

    return BatchMetricsResponse(**metrics)


# Register default handlers for common operations
def register_default_handlers() -> None:
    """Register default batch operation handlers."""
    processor = get_batch_processor()

    async def ocr_handler(data: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for OCR batch operations."""
        # Placeholder - integrate with actual OCR service
        return {
            "status": "processed",
            "text": data.get("text", ""),
            "confidence": 0.95,
        }

    async def analyze_handler(data: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for analysis batch operations."""
        # Placeholder - integrate with actual analysis service
        return {
            "status": "analyzed",
            "result": data,
        }

    async def validate_handler(data: Dict[str, Any]) -> Dict[str, Any]:
        """Handler for validation batch operations."""
        # Placeholder - integrate with actual validation service
        return {
            "status": "valid",
            "errors": [],
        }

    processor.register_handler("ocr", ocr_handler)
    processor.register_handler("analyze", analyze_handler)
    processor.register_handler("validate", validate_handler)


# Auto-register handlers on module import
register_default_handlers()
