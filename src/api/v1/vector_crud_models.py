from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class VectorDeleteRequest(BaseModel):
    id: str = Field(description="要删除的向量分析ID")


class VectorDeleteResponse(BaseModel):
    id: str
    status: str
    error: Optional[Dict[str, Any]] = None


class VectorRegisterRequest(BaseModel):
    id: str = Field(description="向量 ID")
    vector: list[float] = Field(description="向量数据")
    meta: Optional[Dict[str, str]] = Field(default=None, description="向量元数据")


class VectorRegisterResponse(BaseModel):
    id: str
    status: str
    dimension: Optional[int] = None
    error: Optional[Dict[str, Any]] = None


class VectorSearchRequest(BaseModel):
    vector: list[float] = Field(description="查询向量")
    k: int = Field(default=10, ge=1, le=100, description="返回数量")
    material_filter: Optional[str] = Field(default=None, description="材料过滤")
    complexity_filter: Optional[str] = Field(default=None, description="复杂度过滤")
    fine_part_type_filter: Optional[str] = Field(default=None, description="细分类过滤")
    coarse_part_type_filter: Optional[str] = Field(default=None, description="粗分类过滤")
    decision_source_filter: Optional[str] = Field(default=None, description="决策来源过滤")
    is_coarse_label_filter: Optional[bool] = Field(
        default=None,
        description="是否仅返回 coarse label 样本",
    )


class VectorSearchResponse(BaseModel):
    results: list[Dict[str, Any]]
    total: int


__all__ = [
    "VectorDeleteRequest",
    "VectorDeleteResponse",
    "VectorRegisterRequest",
    "VectorRegisterResponse",
    "VectorSearchRequest",
    "VectorSearchResponse",
]
