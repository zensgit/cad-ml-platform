"""
Active Learning API endpoints.
主动学习闭环接口：待审核样本、反馈、统计与导出
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.active_learning import ActiveLearningSample, get_active_learner
from src.core.errors_extended import ErrorCode, build_error

logger = logging.getLogger(__name__)
router = APIRouter()


class FeedbackRequest(BaseModel):
    sample_id: str = Field(..., description="样本ID")
    true_type: str = Field(..., description="人工标注类型")
    true_fine_type: Optional[str] = Field(default=None, description="人工标注细粒度类型")
    true_coarse_type: Optional[str] = Field(default=None, description="人工标注粗粒度类型")
    reviewer_id: Optional[str] = Field(default=None, description="审核人ID")


class FeedbackResponse(BaseModel):
    status: str
    is_correction: bool
    sample_id: str


class StatsResponse(BaseModel):
    stats: Dict[str, int]
    sample_type_stats: Dict[str, int] = Field(default_factory=dict)
    feedback_priority_stats: Dict[str, int] = Field(default_factory=dict)
    predicted_coarse_stats: Dict[str, int] = Field(default_factory=dict)
    predicted_fine_stats: Dict[str, int] = Field(default_factory=dict)
    labeled_true_coarse_stats: Dict[str, int] = Field(default_factory=dict)
    labeled_true_fine_stats: Dict[str, int] = Field(default_factory=dict)
    correction_count: int = 0
    retrain_ready: bool
    labeled_samples: int
    threshold: int
    remaining_samples: int = 0
    retrain_recommendation: str = ""


class ExportRequest(BaseModel):
    format: str = Field(default="jsonl", description="导出格式")
    only_labeled: bool = Field(default=True, description="仅导出已标注样本")


class ExportResponse(BaseModel):
    status: str
    count: Optional[int] = None
    file: Optional[str] = None
    message: Optional[str] = None


class ReviewQueueSummaryResponse(BaseModel):
    status: str
    total: int
    by_sample_type: Dict[str, int] = Field(default_factory=dict)
    by_feedback_priority: Dict[str, int] = Field(default_factory=dict)
    by_decision_source: Dict[str, int] = Field(default_factory=dict)
    by_uncertainty_reason: Dict[str, int] = Field(default_factory=dict)
    by_review_reason: Dict[str, int] = Field(default_factory=dict)
    critical_count: int = 0
    high_count: int = 0
    automation_ready_count: int = 0
    critical_ratio: float = 0.0
    high_ratio: float = 0.0
    automation_ready_ratio: float = 0.0
    operational_status: str = "under_control"


class ReviewQueueStatsResponse(ReviewQueueSummaryResponse):
    pass


class ReviewQueueResponse(BaseModel):
    total: int
    returned: int
    limit: int
    offset: int
    has_more: bool
    sort_by: str
    summary: ReviewQueueSummaryResponse
    items: List[ActiveLearningSample] = Field(default_factory=list)


class ReviewQueueExportResponse(BaseModel):
    status: str
    count: Optional[int] = None
    file: Optional[str] = None
    format: Optional[str] = None
    sort_by: Optional[str] = None
    summary: Optional[ReviewQueueSummaryResponse] = None
    message: Optional[str] = None


@router.get("/pending", response_model=List[ActiveLearningSample])
async def get_pending_samples(
    limit: int = 10,
    api_key: str = Depends(get_api_key),
):
    """获取待审核样本"""
    learner = get_active_learner()
    return learner.get_pending_samples(limit=limit)


@router.get("/review-queue", response_model=ReviewQueueResponse)
async def get_review_queue(
    limit: int = 20,
    offset: int = 0,
    status: str = "pending",
    sample_type: Optional[str] = None,
    feedback_priority: Optional[str] = None,
    sort_by: str = "priority",
    api_key: str = Depends(get_api_key),
):
    """获取带排序和筛选能力的审核队列。"""
    learner = get_active_learner()
    payload = learner.get_review_queue(
        limit=limit,
        offset=offset,
        status=status,
        sample_type=sample_type,
        feedback_priority=feedback_priority,
        sort_by=sort_by,
    )
    return ReviewQueueResponse(
        total=int(payload.get("total", 0)),
        returned=int(payload.get("returned", 0)),
        limit=int(payload.get("limit", limit)),
        offset=int(payload.get("offset", offset)),
        has_more=bool(payload.get("has_more", False)),
        sort_by=str(payload.get("sort_by", sort_by)),
        summary=ReviewQueueSummaryResponse(**payload.get("summary", {})),
        items=payload.get("items", []),
    )


@router.get("/review-queue/stats", response_model=ReviewQueueStatsResponse)
async def get_review_queue_stats(
    status: str = "pending",
    sample_type: Optional[str] = None,
    feedback_priority: Optional[str] = None,
    api_key: str = Depends(get_api_key),
):
    """获取审核队列聚合统计。"""
    learner = get_active_learner()
    payload = learner.get_review_queue_stats(
        status=status,
        sample_type=sample_type,
        feedback_priority=feedback_priority,
    )
    return ReviewQueueStatsResponse(**payload)


@router.get("/review-queue/export", response_model=ReviewQueueExportResponse)
async def export_review_queue(
    status: str = "pending",
    sample_type: Optional[str] = None,
    feedback_priority: Optional[str] = None,
    sort_by: str = "priority",
    format: str = "csv",  # noqa: A002
    api_key: str = Depends(get_api_key),
):
    """导出审核队列，便于人工复核与 benchmark 样本整理。"""
    learner = get_active_learner()
    payload = learner.export_review_queue(
        status=status,
        sample_type=sample_type,
        feedback_priority=feedback_priority,
        sort_by=sort_by,
        format=format,
    )
    if payload.get("status") != "ok":
        return ReviewQueueExportResponse(
            status="error",
            message=str(payload.get("message", "Export failed")),
        )
    return ReviewQueueExportResponse(
        status="ok",
        count=payload.get("count"),
        file=payload.get("file"),
        format=payload.get("format"),
        sort_by=payload.get("sort_by"),
        summary=ReviewQueueSummaryResponse(**payload.get("summary", {})),
    )


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    payload: FeedbackRequest,
    api_key: str = Depends(get_api_key),
):
    """提交人工反馈"""
    learner = get_active_learner()
    result = learner.submit_feedback(
        sample_id=payload.sample_id,
        true_type=payload.true_type,
        true_fine_type=payload.true_fine_type,
        true_coarse_type=payload.true_coarse_type,
        reviewer_id=payload.reviewer_id,
    )
    if result.get("status") != "ok":
        err = build_error(
            ErrorCode.DATA_NOT_FOUND,
            stage="active_learning_feedback",
            message=result.get("message", "Sample not found"),
            sample_id=payload.sample_id,
        )
        raise HTTPException(status_code=404, detail=err)
    return FeedbackResponse(
        status="ok",
        is_correction=bool(result.get("is_correction")),
        sample_id=payload.sample_id,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_active_learning_stats(api_key: str = Depends(get_api_key)):
    """获取主动学习统计"""
    learner = get_active_learner()
    summary = learner.get_stats_summary()
    retrain = learner.check_retrain_threshold()
    return StatsResponse(
        stats=summary.get("status", {}),
        sample_type_stats=summary.get("sample_type", {}),
        feedback_priority_stats=summary.get("feedback_priority", {}),
        predicted_coarse_stats=summary.get("predicted_coarse_type", {}),
        predicted_fine_stats=summary.get("predicted_fine_type", {}),
        labeled_true_coarse_stats=summary.get("labeled_true_coarse_type", {}),
        labeled_true_fine_stats=summary.get("labeled_true_fine_type", {}),
        correction_count=int(summary.get("correction_count", 0)),
        retrain_ready=bool(retrain.get("ready")),
        labeled_samples=int(retrain.get("labeled_samples", 0)),
        threshold=int(retrain.get("threshold", 0)),
        remaining_samples=int(retrain.get("remaining_samples", 0)),
        retrain_recommendation=str(retrain.get("recommendation", "")),
    )


@router.post("/export", response_model=ExportResponse)
async def export_training_data(
    payload: ExportRequest,
    api_key: str = Depends(get_api_key),
):
    """导出训练数据"""
    learner = get_active_learner()
    result = learner.export_training_data(
        format=payload.format,
        only_labeled=payload.only_labeled,
    )
    if result.get("status") != "ok":
        return ExportResponse(
            status="error",
            message=result.get("message", "Export failed"),
        )
    return ExportResponse(
        status="ok",
        count=result.get("count"),
        file=result.get("file"),
    )
