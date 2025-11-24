"""
Features API endpoints
特征相关的API端点 - 包含特征差异比较等功能
"""

import os
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.dependencies import get_api_key
from src.core.errors_extended import ErrorCode, create_extended_error, build_error
from src.utils.analysis_metrics import features_diff_requests_total

logger = logging.getLogger(__name__)
router = APIRouter()


class FeatureSlotDiff(BaseModel):
    """特征槽位差异"""
    index: int = Field(..., description="槽位索引")
    name: str = Field(..., description="槽位名称")
    value_a: float = Field(..., description="向量A的值")
    value_b: float = Field(..., description="向量B的值")
    abs_diff: float = Field(..., description="绝对差值")
    rel_diff: float | None = Field(None, description="相对差值百分比")


class FeaturesDiffResponse(BaseModel):
    """特征差异响应"""
    id_a: str = Field(..., description="向量A的ID")
    id_b: str = Field(..., description="向量B的ID")
    dimension: int | None = Field(None, description="向量维度")
    diffs: List[FeatureSlotDiff] = Field(default_factory=list, description="差异列表")
    status: str = Field(..., description="状态: ok/not_found/dimension_mismatch")
    error: Dict[str, Any] | None = Field(None, description="错误信息")


@router.get("/diff", response_model=FeaturesDiffResponse)
async def features_diff(
    id_a: str,
    id_b: str,
    api_key: str = Depends(get_api_key)
):
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
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META  # type: ignore

    # Check if both vectors exist
    if id_a not in _VECTOR_STORE or id_b not in _VECTOR_STORE:
        ext = create_extended_error(
            ErrorCode.DATA_NOT_FOUND,
            "Vector not found",
            stage="features_diff",
            context={"id": f"{id_a},{id_b}"}
        )
        features_diff_requests_total.labels(status="not_found").inc()
        return FeaturesDiffResponse(
            id_a=id_a,
            id_b=id_b,
            dimension=None,
            diffs=[],
            status="not_found",
            error=ext.to_dict()
        )

    # Get vectors
    va = _VECTOR_STORE[id_a]
    vb = _VECTOR_STORE[id_b]

    # Check dimension match
    if len(va) != len(vb):
        ext = create_extended_error(
            ErrorCode.DIMENSION_MISMATCH,
            "Dimension mismatch",
            stage="features_diff"
        )
        features_diff_requests_total.labels(status="dimension_mismatch").inc()
        return FeaturesDiffResponse(
            id_a=id_a,
            id_b=id_b,
            dimension=None,
            diffs=[],
            status="dimension_mismatch",
            error=ext.to_dict()
        )

    # Build per-slot diff using feature_slots if available
    slots_a = _VECTOR_META.get(id_a, {}).get("feature_version")
    feature_version = (
        slots_a or
        _VECTOR_META.get(id_b, {}).get("feature_version") or
        os.getenv("FEATURE_VERSION", "v1")
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
                rel_diff=rel_diff
            )
        )

    # Sort by absolute difference (descending)
    diffs.sort(key=lambda x: x.abs_diff, reverse=True)

    features_diff_requests_total.labels(status="ok").inc()

    return FeaturesDiffResponse(
        id_a=id_a,
        id_b=id_b,
        dimension=len(va),
        diffs=diffs,
        status="ok"
    )


class FeatureSlotsResponse(BaseModel):
    version: str
    slots: list[dict[str, str]]
    status: str
    error: Dict[str, Any] | None = None


@router.get("/slots", response_model=FeatureSlotsResponse)
async def feature_slots(version: str = "v1", api_key: str = Depends(get_api_key)):
    allowed = {"v1", "v2", "v3", "v4"}
    if version not in allowed:
        err = build_error(ErrorCode.INPUT_VALIDATION_FAILED, stage="feature_slots", message="Unsupported version", version=version, allowed=list(sorted(allowed)))
        raise HTTPException(status_code=422, detail=err)
    from src.core.feature_extractor import FeatureExtractor
    fx = FeatureExtractor(feature_version=version)
    return FeatureSlotsResponse(version=version, slots=fx.slots(version), status="ok")


class FeatureVersionsResponse(BaseModel):
    versions: List[Dict[str, Any]]
    status: str


@router.get("/versions", response_model=FeatureVersionsResponse)
async def feature_versions(api_key: str = Depends(get_api_key)):
    from src.core.feature_extractor import (
        SLOTS_V1,
        SLOTS_V2,
        SLOTS_V3,
        SLOTS_V4,
    )
    data = []
    data.append({"version": "v1", "dimension": len(SLOTS_V1), "stable": True, "experimental": False})
    data.append({"version": "v2", "dimension": len(SLOTS_V1) + len(SLOTS_V2), "stable": True, "experimental": False})
    data.append({"version": "v3", "dimension": len(SLOTS_V1) + len(SLOTS_V2) + len(SLOTS_V3), "stable": True, "experimental": False})
    data.append({"version": "v4", "dimension": len(SLOTS_V1) + len(SLOTS_V2) + len(SLOTS_V3) + len(SLOTS_V4), "stable": False, "experimental": True})
    return FeatureVersionsResponse(versions=data, status="ok")
