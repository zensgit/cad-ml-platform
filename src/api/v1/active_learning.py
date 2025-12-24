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
    reviewer_id: Optional[str] = Field(default=None, description="审核人ID")


class FeedbackResponse(BaseModel):
    status: str
    is_correction: bool
    sample_id: str


class StatsResponse(BaseModel):
    stats: Dict[str, int]
    retrain_ready: bool
    labeled_samples: int
    threshold: int


class ExportRequest(BaseModel):
    format: str = Field(default="jsonl", description="导出格式")
    only_labeled: bool = Field(default=True, description="仅导出已标注样本")


class ExportResponse(BaseModel):
    status: str
    count: Optional[int] = None
    file: Optional[str] = None
    message: Optional[str] = None


@router.get("/pending", response_model=List[ActiveLearningSample])
async def get_pending_samples(
    limit: int = 10,
    api_key: str = Depends(get_api_key),
):
    """获取待审核样本"""
    learner = get_active_learner()
    return learner.get_pending_samples(limit=limit)


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
        reviewer_id=payload.reviewer_id,
    )
    if result.get("status") != "ok":
        err = build_error(
            ErrorCode.NOT_FOUND,
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
    stats = learner.get_stats()
    retrain = learner.check_retrain_threshold()
    return StatsResponse(
        stats=stats,
        retrain_ready=bool(retrain.get("ready")),
        labeled_samples=int(retrain.get("labeled_samples", 0)),
        threshold=int(retrain.get("threshold", 0)),
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
