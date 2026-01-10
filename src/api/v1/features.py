"""
Features API endpoints
特征相关的API端点 - 包含特征差异比较等功能
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.api.dependencies import get_api_key
from src.core.errors_extended import ErrorCode, build_error, create_extended_error
from src.utils.analysis_metrics import (
    feature_cache_tuning_recommended_capacity,
    feature_cache_tuning_recommended_ttl_seconds,
    feature_cache_tuning_requests_total,
    features_diff_requests_total,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class FeatureSlotDiff(BaseModel):
    """特征槽位差异"""

    index: int = Field(..., description="槽位索引")
    name: str = Field(..., description="槽位名称")
    value_a: float = Field(..., description="向量A的值")
    value_b: float = Field(..., description="向量B的值")
    abs_diff: float = Field(..., description="绝对差值")
    rel_diff: Optional[float] = Field(None, description="相对差值百分比")


class FeaturesDiffResponse(BaseModel):
    """特征差异响应"""

    id_a: str = Field(..., description="向量A的ID")
    id_b: str = Field(..., description="向量B的ID")
    dimension: Optional[int] = Field(None, description="向量维度")
    diffs: List[FeatureSlotDiff] = Field(default_factory=list, description="差异列表")
    status: str = Field(..., description="状态: ok/not_found/dimension_mismatch")
    error: Optional[Dict[str, Any]] = Field(None, description="错误信息")


@router.get("/diff", response_model=FeaturesDiffResponse)
async def features_diff(id_a: str, id_b: str, api_key: str = Depends(get_api_key)):
    """
    比较两个向量的特征差异

    Args:
        id_a: 第一个向量ID
        id_b: 第二个向量ID
        api_key: API密钥

    Returns:
        特征差异详情
    """
    # Import locally to avoid circular imports
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

    # Check if both vectors exist
    if id_a not in _VECTOR_STORE or id_b not in _VECTOR_STORE:
        ext = create_extended_error(
            ErrorCode.DATA_NOT_FOUND,
            "Vector not found",
            stage="features_diff",
            context={"id": f"{id_a},{id_b}"},
        )
        features_diff_requests_total.labels(status="not_found").inc()
        return FeaturesDiffResponse(
            id_a=id_a, id_b=id_b, dimension=None, diffs=[], status="not_found", error=ext.to_dict()
        )

    # Get vectors
    va = _VECTOR_STORE[id_a]
    vb = _VECTOR_STORE[id_b]

    # Check dimension match
    if len(va) != len(vb):
        ext = create_extended_error(
            ErrorCode.DIMENSION_MISMATCH, "Dimension mismatch", stage="features_diff"
        )
        features_diff_requests_total.labels(status="dimension_mismatch").inc()
        return FeaturesDiffResponse(
            id_a=id_a,
            id_b=id_b,
            dimension=None,
            diffs=[],
            status="dimension_mismatch",
            error=ext.to_dict(),
        )

    # Build per-slot diff using feature_slots if available
    slots_a = _VECTOR_META.get(id_a, {}).get("feature_version")
    feature_version = (
        slots_a
        or _VECTOR_META.get(id_b, {}).get("feature_version")
        or os.getenv("FEATURE_VERSION", "v1")
    )

    from src.core.feature_extractor import FeatureExtractor

    extractor = FeatureExtractor()
    slot_defs = extractor.slots(feature_version)

    diffs = []
    for i in range(len(va)):
        abs_diff = abs(va[i] - vb[i])
        rel_diff = None
        if va[i] != 0:
            rel_diff = (abs_diff / abs(va[i])) * 100

        slot_name = f"slot_{i}"
        if i < len(slot_defs):
            slot_name = slot_defs[i].get("name", slot_name)

        diffs.append(
            FeatureSlotDiff(
                index=i,
                name=slot_name,
                value_a=va[i],
                value_b=vb[i],
                abs_diff=abs_diff,
                rel_diff=rel_diff,
            )
        )

    # Sort by absolute difference (descending)
    diffs.sort(key=lambda x: x.abs_diff, reverse=True)

    features_diff_requests_total.labels(status="ok").inc()

    return FeaturesDiffResponse(id_a=id_a, id_b=id_b, dimension=len(va), diffs=diffs, status="ok")


class FeatureSlotsResponse(BaseModel):
    version: str
    slots: List[Dict[str, str]]
    status: str
    error: Optional[Dict[str, Any]] = None


@router.get("/slots", response_model=FeatureSlotsResponse)
async def feature_slots(version: str = "v1", api_key: str = Depends(get_api_key)):
    allowed = {"v1", "v2", "v3", "v4"}
    if version not in allowed:
        err = build_error(
            ErrorCode.INPUT_VALIDATION_FAILED,
            stage="feature_slots",
            message="Unsupported version",
            version=version,
            allowed=list(sorted(allowed)),
        )
        raise HTTPException(status_code=422, detail=err)
    from src.core.feature_extractor import FeatureExtractor

    fx = FeatureExtractor(feature_version=version)
    return FeatureSlotsResponse(version=version, slots=fx.slots(version), status="ok")


class FeatureVersionsResponse(BaseModel):
    versions: List[Dict[str, Any]]
    status: str


class CacheTuningRequest(BaseModel):
    """Cache tuning input payload."""

    hit_rate: float = Field(..., ge=0.0, le=1.0, description="Cache hit ratio (0-1)")
    capacity: int = Field(..., gt=0, description="Current cache capacity")
    ttl_seconds: int = Field(..., gt=0, alias="ttl", description="Current cache TTL in seconds")
    window_hours: float = Field(24.0, gt=0.0, description="Observation window in hours")

    model_config = ConfigDict(validate_by_name=True)


class CacheTuningRecommendation(BaseModel):
    """Cache tuning recommendation response."""

    recommended_capacity: int = Field(..., description="Recommended cache capacity")
    recommended_ttl: int = Field(..., description="Recommended cache TTL in seconds")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Recommendation confidence")
    reasoning: List[str] = Field(default_factory=list, description="Recommendation reasoning")
    experimental: bool = Field(True, description="Recommendation is experimental")


def _adjust_ttl(ttl_seconds: int, window_hours: float) -> int:
    """Adjust TTL based on access window heuristics."""
    if window_hours <= 1:
        return max(int(ttl_seconds * 0.5), 60)
    if window_hours <= 6:
        return max(int(ttl_seconds * 0.8), 60)
    if window_hours >= 24:
        return int(ttl_seconds * 1.2)
    return ttl_seconds


@router.post("/cache/tuning", response_model=CacheTuningRecommendation)
async def cache_tuning(
    payload: CacheTuningRequest, api_key: str = Depends(get_api_key)
):
    """Generate cache tuning recommendations from supplied metrics."""
    try:
        if payload.capacity <= 0 or payload.ttl_seconds <= 0 or payload.window_hours <= 0:
            feature_cache_tuning_requests_total.labels(status="invalid").inc()
            err = build_error(
                ErrorCode.INPUT_VALIDATION_FAILED,
                stage="cache_tuning",
                message="Capacity, ttl, and window_hours must be positive",
                capacity=payload.capacity,
                ttl=payload.ttl_seconds,
                window_hours=payload.window_hours,
            )
            raise HTTPException(status_code=422, detail=err)

        recommended_capacity = payload.capacity
        recommended_ttl = payload.ttl_seconds
        reasoning: List[str] = []
        confidence = 0.5

        if payload.hit_rate < 0.4:
            recommended_capacity = max(int(round(payload.capacity * 1.5)), payload.capacity + 1)
            reasoning.append("Low hit rate suggests insufficient capacity")
            confidence = 0.75
        elif payload.hit_rate < 0.7:
            recommended_ttl = _adjust_ttl(payload.ttl_seconds, payload.window_hours)
            reasoning.append("Moderate hit rate, optimize TTL")
            confidence = 0.6
        elif payload.hit_rate > 0.85:
            recommended_capacity = max(int(round(payload.capacity * 0.8)), 1)
            reasoning.append("High hit rate, capacity can be reduced")
            confidence = 0.7
        else:
            reasoning.append("Hit rate within target band; keep current settings")
            confidence = 0.55

        try:
            feature_cache_tuning_recommended_capacity.set(recommended_capacity)
            feature_cache_tuning_recommended_ttl_seconds.set(recommended_ttl)
        except Exception:
            pass

        feature_cache_tuning_requests_total.labels(status="ok").inc()
        return CacheTuningRecommendation(
            recommended_capacity=recommended_capacity,
            recommended_ttl=recommended_ttl,
            confidence=round(confidence, 2),
            reasoning=reasoning,
            experimental=True,
        )
    except HTTPException:
        raise
    except Exception as exc:
        feature_cache_tuning_requests_total.labels(status="error").inc()
        logger.exception("Cache tuning recommendation failed")
        err = build_error(
            ErrorCode.INTERNAL_ERROR,
            stage="cache_tuning",
            message="Cache tuning recommendation failed",
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=err)


@router.get("/versions", response_model=FeatureVersionsResponse)
async def feature_versions(api_key: str = Depends(get_api_key)):
    from src.core.feature_extractor import SLOTS_V1, SLOTS_V2, SLOTS_V3, SLOTS_V4

    data = []
    data.append(
        {"version": "v1", "dimension": len(SLOTS_V1), "stable": True, "experimental": False}
    )
    data.append(
        {
            "version": "v2",
            "dimension": len(SLOTS_V1) + len(SLOTS_V2),
            "stable": True,
            "experimental": False,
        }
    )
    data.append(
        {
            "version": "v3",
            "dimension": len(SLOTS_V1) + len(SLOTS_V2) + len(SLOTS_V3),
            "stable": True,
            "experimental": False,
        }
    )
    data.append(
        {
            "version": "v4",
            "dimension": len(SLOTS_V1) + len(SLOTS_V2) + len(SLOTS_V3) + len(SLOTS_V4),
            "stable": False,
            "experimental": True,
        }
    )
    return FeatureVersionsResponse(versions=data, status="ok")
