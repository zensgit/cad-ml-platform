"""API v2 - Enhanced endpoints with improved response formats.

Changes from v1:
- Standardized response envelope
- Enhanced error responses
- Pagination support
- Field selection
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field
from pydantic.generics import GenericModel

router = APIRouter()

T = TypeVar("T")


class PaginationParams(BaseModel):
    """Pagination parameters."""

    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(20, ge=1, le=100, description="Items per page")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


class PaginationMeta(BaseModel):
    """Pagination metadata."""

    page: int
    page_size: int
    total_items: int
    total_pages: int
    has_next: bool
    has_prev: bool


class APIResponseEnvelope(GenericModel, Generic[T]):
    """Standard API response envelope."""

    success: bool
    data: Optional[T] = None
    error: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Error response format."""

    code: str
    message: str
    details: Optional[Dict[str, Any]] = None
    trace_id: Optional[str] = None


class HealthResponseV2(BaseModel):
    """Health check response v2."""

    status: str
    version: str = "2.0"
    uptime_seconds: float
    components: Dict[str, Dict[str, Any]]


def paginate_dependency(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
) -> PaginationParams:
    """Pagination dependency."""
    return PaginationParams(page=page, page_size=page_size)


@router.get("/health", response_model=APIResponseEnvelope[HealthResponseV2])
async def health_check_v2() -> APIResponseEnvelope[HealthResponseV2]:
    """Enhanced health check endpoint.

    Returns detailed health status with component breakdown.
    """
    import time

    # Mock uptime - in production, track actual start time
    uptime = time.time() % 86400  # Mock: reset daily

    health_data = HealthResponseV2(
        status="healthy",
        uptime_seconds=uptime,
        components={
            "api": {"status": "healthy", "latency_ms": 1.2},
            "database": {"status": "healthy", "connections": 10},
            "cache": {"status": "healthy", "hit_rate": 0.95},
            "model": {"status": "healthy", "loaded": True},
        },
    )

    return APIResponseEnvelope(
        success=True,
        data=health_data,
        meta={"version": "v2"},
    )


@router.get("/versions")
async def list_api_versions() -> APIResponseEnvelope[List[Dict[str, Any]]]:
    """List available API versions."""
    from src.api.versioning import get_version_manager

    manager = get_version_manager()
    versions = manager.list_versions()

    return APIResponseEnvelope(
        success=True,
        data=versions,
        meta={"current_version": manager.get_current_version()},
    )


class PredictRequestV2(BaseModel):
    """Prediction request v2."""

    features: Dict[str, Any] = Field(..., description="Input features")
    model_id: Optional[str] = Field(None, description="Specific model to use")
    include_explanation: bool = Field(False, description="Include prediction explanation")
    timeout_ms: Optional[int] = Field(None, ge=100, le=30000, description="Request timeout")


class PredictionResultV2(BaseModel):
    """Prediction result v2."""

    prediction: Any
    confidence: Optional[float] = None
    model_id: str
    model_version: str
    explanation: Optional[Dict[str, Any]] = None
    latency_ms: float


@router.post("/predict", response_model=APIResponseEnvelope[PredictionResultV2])
async def predict_v2(
    request: PredictRequestV2,
) -> APIResponseEnvelope[PredictionResultV2]:
    """Enhanced prediction endpoint.

    Features:
    - Model selection
    - Optional explanations
    - Configurable timeout
    """
    import time

    start = time.time()

    # Mock prediction - integrate with actual model service
    result = PredictionResultV2(
        prediction={"class": "A", "probabilities": [0.85, 0.15]},
        confidence=0.85,
        model_id=request.model_id or "default",
        model_version="1.0.0",
        explanation={"top_features": ["feature_1", "feature_2"]} if request.include_explanation else None,
        latency_ms=(time.time() - start) * 1000,
    )

    return APIResponseEnvelope(
        success=True,
        data=result,
        meta={"model_id": result.model_id},
    )


class BatchPredictRequestV2(BaseModel):
    """Batch prediction request v2."""

    items: List[Dict[str, Any]] = Field(..., description="Items to predict")
    model_id: Optional[str] = Field(None)
    async_mode: bool = Field(False, description="Process asynchronously")


class BatchPredictResponseV2(BaseModel):
    """Batch prediction response v2."""

    job_id: Optional[str] = None
    results: Optional[List[PredictionResultV2]] = None
    status: str


@router.post("/predict/batch", response_model=APIResponseEnvelope[BatchPredictResponseV2])
async def batch_predict_v2(
    request: BatchPredictRequestV2,
) -> APIResponseEnvelope[BatchPredictResponseV2]:
    """Batch prediction endpoint.

    Supports both synchronous and asynchronous processing.
    """
    import uuid

    if request.async_mode:
        # Create async job
        job_id = str(uuid.uuid4())
        result = BatchPredictResponseV2(
            job_id=job_id,
            status="processing",
        )
    else:
        # Synchronous processing
        results = []
        for item in request.items[:10]:  # Limit for sync
            results.append(
                PredictionResultV2(
                    prediction={"class": "A"},
                    confidence=0.9,
                    model_id=request.model_id or "default",
                    model_version="1.0.0",
                    latency_ms=1.0,
                )
            )
        result = BatchPredictResponseV2(
            results=results,
            status="completed",
        )

    return APIResponseEnvelope(
        success=True,
        data=result,
    )


# Feature flags for v2
V2_FEATURES = {
    "response_envelope": True,
    "pagination": True,
    "field_selection": True,
    "batch_operations": True,
    "async_jobs": True,
    "explanations": True,
}
