from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field
from src.api.v1.vector_migration_models import (
    VectorMigrateItem,
    VectorMigrateRequest,
    VectorMigrateResponse,
    VectorMigrationStatusResponse,
)


class VectorDeleteRequest(BaseModel):  # deprecated moved to vectors.py
    id: str = Field(description="要删除的向量分析ID")


class VectorDeleteResponse(BaseModel):  # deprecated moved to vectors.py
    id: str
    status: str


class VectorListItem(BaseModel):  # deprecated moved to vectors.py
    id: str
    dimension: int
    material: Optional[str] = None
    complexity: Optional[str] = None
    format: Optional[str] = None


class VectorListResponse(BaseModel):  # deprecated moved to vectors.py
    total: int
    vectors: list[VectorListItem]


class VectorUpdateRequest(BaseModel):
    id: str = Field(description="要更新的向量分析ID")
    replace: Optional[list[float]] = Field(
        default=None, description="新的向量 (维度需与原向量一致)"
    )
    append: Optional[list[float]] = Field(
        default=None, description="追加的向量片段 (若提供 replace 则忽略)"
    )
    material: Optional[str] = Field(default=None, description="更新材料元数据")
    complexity: Optional[str] = Field(default=None, description="更新复杂度元数据")
    format: Optional[str] = Field(default=None, description="更新格式元数据")


class VectorUpdateResponse(BaseModel):
    id: str
    status: str
    dimension: Optional[int] = None
    error: Optional[Dict[str, Any]] = None
    feature_version: Optional[str] = None


class VectorStatsResponse(BaseModel):  # deprecated moved to vectors_stats.py
    backend: str
    total: int
    by_material: Dict[str, int]
    by_complexity: Dict[str, int]
    by_format: Dict[str, int]
    versions: Optional[Dict[str, int]] = None


class VectorDistributionResponse(BaseModel):  # deprecated moved to vectors_stats.py
    total: int
    by_material: Dict[str, int]
    by_complexity: Dict[str, int]
    by_format: Dict[str, int]
    dominant_ratio: float
    feature_version: str
    average_dimension: Optional[float] = None
    versions: Optional[Dict[str, int]] = None


class ProcessRulesAuditResponse(BaseModel):
    version: str
    source: str
    hash: Optional[str] = None
    materials: list[str]
    complexities: Dict[str, list[str]]
    raw: Dict[str, Any]


class FeaturesDiffResponse(BaseModel):
    id_a: str
    id_b: str
    dimension: Optional[int] = None
    diffs: list[Dict[str, Any]]
    status: str
    error: Optional[Dict[str, Any]] = None


class ModelReloadRequest(BaseModel):
    path: str = Field(description="模型文件路径")
    expected_version: Optional[str] = Field(default=None, description="期望模型版本")
    force: bool = Field(default=False, description="强制重载忽略版本校验")


class ModelReloadResponse(BaseModel):
    status: str
    model_version: Optional[str] = None
    hash: Optional[str] = None
    error: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(protected_namespaces=())


class OrphanCleanupResponse(BaseModel):
    status: str
    cleaned: int
    total_orphans_detected: Optional[int] = None
    error: Optional[Dict[str, Any]] = None


class FeatureCacheStatsResponse(BaseModel):
    status: str
    size: int
    capacity: int
    ttl_seconds: int
    hit_ratio: Optional[float] = None
    hits: Optional[int] = None
    misses: Optional[int] = None
    evictions: Optional[int] = None


class FaissHealthResponse(BaseModel):
    available: bool
    index_size: Optional[int]
    dim: Optional[int]
    age_seconds: Optional[int]
    pending_delete: Optional[int]
    max_pending_delete: Optional[int]
    normalize: Optional[bool]
    status: str


__all__ = [
    "FaissHealthResponse",
    "FeatureCacheStatsResponse",
    "FeaturesDiffResponse",
    "ModelReloadRequest",
    "ModelReloadResponse",
    "OrphanCleanupResponse",
    "ProcessRulesAuditResponse",
    "VectorDeleteRequest",
    "VectorDeleteResponse",
    "VectorDistributionResponse",
    "VectorListItem",
    "VectorListResponse",
    "VectorMigrateItem",
    "VectorMigrateRequest",
    "VectorMigrateResponse",
    "VectorMigrationStatusResponse",
    "VectorStatsResponse",
    "VectorUpdateRequest",
    "VectorUpdateResponse",
]
