"""
Feedback API.

Collects user corrections to improve L3/L4 models (Data Flywheel).
"""

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.dedupcad_tenant_config import TenantDedup2DConfigStore
from src.core.feedback_log import append_feedback_entry, load_feedback_entries

router = APIRouter()


class FeedbackRequest(BaseModel):
    analysis_id: str = Field(..., description="The ID of the analysis being corrected")
    corrected_part_type: Optional[str] = Field(
        None, description="Correct classification"
    )
    corrected_process: Optional[str] = Field(
        None, description="Correct manufacturing process"
    )
    dfm_feedback: Optional[str] = Field(None, description="Comments on DFM accuracy")
    rating: int = Field(..., ge=1, le=5, description="1-5 star rating of the AI result")
    job_id: Optional[str] = Field(None, description="Optional dedup job identifier")
    candidate_id: Optional[str] = Field(
        None, description="Optional dedup candidate identifier"
    )
    correlation_id: Optional[str] = Field(
        None, description="Optional dedup workflow correlation id"
    )
    family_key: Optional[str] = Field(None, description="Optional dedup family key")
    reviewer: Optional[str] = Field(None, description="Reviewer/operator identifier")
    decision: Optional[str] = Field(None, description="Optional dedup review decision")
    reason_codes: list[str] = Field(
        default_factory=list,
        description="Optional structured reason codes for grounded feedback",
    )
    reuse_recommended: Optional[bool] = Field(
        None, description="Whether the candidate should be reused"
    )
    version_variant: Optional[bool] = Field(
        None, description="Whether the candidate is a version variant"
    )
    source: Optional[str] = Field(
        None,
        description="Optional source area, e.g. dedup_2d or analyze",
    )


class FeedbackResponse(BaseModel):
    status: str
    feedback_id: str
    message: str
    tenant_id: Optional[str] = None
    correlation_id: Optional[str] = None


class FeedbackEntryResponse(BaseModel):
    id: str
    event: Optional[str] = None
    timestamp: str
    logged_at: Optional[float] = None
    tenant_id: Optional[str] = None
    source: Optional[str] = None
    analysis_id: str
    rating: int
    corrected_part_type: Optional[str] = None
    corrected_process: Optional[str] = None
    dfm_feedback: Optional[str] = None
    job_id: Optional[str] = None
    candidate_id: Optional[str] = None
    correlation_id: Optional[str] = None
    family_key: Optional[str] = None
    reviewer: Optional[str] = None
    decision: Optional[str] = None
    reason_codes: List[str] = Field(default_factory=list)
    reuse_recommended: Optional[bool] = None
    version_variant: Optional[bool] = None


class FeedbackListResponse(BaseModel):
    items: List[FeedbackEntryResponse] = Field(default_factory=list)
    total: int = 0
    summary: Dict[str, Any] = Field(default_factory=dict)


def get_feedback_tenant_store() -> TenantDedup2DConfigStore:
    return TenantDedup2DConfigStore()


def _feedback_source(payload: FeedbackRequest) -> str:
    explicit = str(payload.source or "").strip()
    if explicit:
        return explicit
    if (
        payload.job_id
        or payload.candidate_id
        or payload.correlation_id
        or payload.family_key
    ):
        return "dedup_2d"
    return "analyze"


def _summarize_feedback(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_decision: Dict[str, int] = {}
    by_reason_code: Dict[str, int] = {}
    by_family_key: Dict[str, int] = {}
    by_source: Dict[str, int] = {}
    ratings: List[int] = []
    for item in items:
        ratings.append(int(item.get("rating") or 0))
        decision = str(item.get("decision") or "").strip()
        if decision:
            by_decision[decision] = by_decision.get(decision, 0) + 1
        source = str(item.get("source") or "").strip()
        if source:
            by_source[source] = by_source.get(source, 0) + 1
        family_key = str(item.get("family_key") or "").strip()
        if family_key:
            by_family_key[family_key] = by_family_key.get(family_key, 0) + 1
        for reason_code in list(item.get("reason_codes") or []):
            code = str(reason_code or "").strip()
            if code:
                by_reason_code[code] = by_reason_code.get(code, 0) + 1
    average_rating = round(sum(ratings) / len(ratings), 4) if ratings else None
    return {
        "total": len(items),
        "average_rating": average_rating,
        "by_decision": by_decision,
        "by_reason_code": by_reason_code,
        "by_family_key": by_family_key,
        "by_source": by_source,
    }


@router.post("/", response_model=FeedbackResponse)
async def submit_feedback(
    payload: FeedbackRequest,
    api_key: str = Depends(get_api_key),
    tenant_store: TenantDedup2DConfigStore = Depends(get_feedback_tenant_store),
):
    """
    Submit feedback for an analysis result.
    This data is logged and used to fine-tune future models.
    """
    feedback_id = str(uuid.uuid4())
    tenant_id = tenant_store.tenant_id(api_key)
    now = datetime.now()
    entry = {
        "id": feedback_id,
        "event": "submitted",
        "timestamp": now.isoformat(),
        "logged_at": now.timestamp(),
        "tenant_id": tenant_id,
        **payload.model_dump(),
        "source": _feedback_source(payload),
    }

    try:
        append_feedback_entry(entry)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save feedback: {str(e)}"
        )

    # Feed correction into the learning pipeline (non-blocking)
    learning_status = None
    if payload.corrected_part_type:
        try:
            from src.ml.learning import FeedbackLearningPipeline

            pipeline = FeedbackLearningPipeline()
            import asyncio

            learning_result = await pipeline.ingest_correction(
                file_id=payload.analysis_id,
                predicted_label="unknown",
                corrected_label=payload.corrected_part_type,
                confidence=0.0,
                source=_feedback_source(payload),
            )
            learning_status = learning_result.get("status", "ingested")
        except Exception as learn_err:
            import logging

            logging.getLogger(__name__).warning(
                "Learning pipeline ingestion failed (non-fatal): %s", learn_err
            )

    return FeedbackResponse(
        status="success",
        feedback_id=feedback_id,
        tenant_id=tenant_id,
        correlation_id=payload.correlation_id,
        message="Feedback received. Thank you for helping improve the AI."
        + (f" Learning status: {learning_status}" if learning_status else ""),
    )


@router.get("/dedup", response_model=FeedbackListResponse)
async def list_dedup_feedback(
    correlation_id: Optional[str] = Query(default=None),
    job_id: Optional[str] = Query(default=None),
    candidate_id: Optional[str] = Query(default=None),
    family_key: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    api_key: str = Depends(get_api_key),
    tenant_store: TenantDedup2DConfigStore = Depends(get_feedback_tenant_store),
):
    tenant_id = tenant_store.tenant_id(api_key)
    items = load_feedback_entries(
        tenant_id=tenant_id,
        source="dedup_2d",
        correlation_id=correlation_id,
        job_id=job_id,
        candidate_id=candidate_id,
        family_key=family_key,
        limit=limit,
    )
    return FeedbackListResponse(
        items=[FeedbackEntryResponse(**item) for item in items],
        total=len(items),
        summary=_summarize_feedback(items),
    )
