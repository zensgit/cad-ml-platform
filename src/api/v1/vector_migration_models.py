from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class VectorMigrateItem(BaseModel):
    id: str
    status: str
    from_version: Optional[str] = None
    to_version: Optional[str] = None
    dimension_before: Optional[int] = None
    dimension_after: Optional[int] = None
    error: Optional[str] = None


class VectorMigrateRequest(BaseModel):
    ids: list[str] = Field(description="需要迁移的向量ID列表")
    to_version: str = Field(description="目标特征版本")
    dry_run: bool = Field(default=False, description="是否为试运行 (不真正写入)")


class VectorMigrateResponse(BaseModel):
    total: int
    migrated: int
    skipped: int
    items: list[VectorMigrateItem]
    migration_id: Optional[str] = Field(default=None, description="迁移批次ID")
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    dry_run_total: Optional[int] = None


class VectorMigrationStatusResponse(BaseModel):
    last_migration_id: Optional[str] = None
    last_started_at: Optional[datetime] = None
    last_finished_at: Optional[datetime] = None
    last_total: Optional[int] = None
    last_migrated: Optional[int] = None
    last_skipped: Optional[int] = None
    pending_vectors: Optional[int] = None
    feature_versions: Optional[Dict[str, int]] = None
    history: Optional[list[Dict[str, Any]]] = None
    backend: str = "memory"
    current_total_vectors: Optional[int] = None
    scanned_vectors: Optional[int] = None
    scan_limit: Optional[int] = None
    distribution_complete: bool = True
    target_version: str = "v4"
    target_version_vectors: Optional[int] = None
    target_version_ratio: Optional[float] = None
    migration_ready: bool = False


__all__ = [
    "VectorMigrateItem",
    "VectorMigrateRequest",
    "VectorMigrateResponse",
    "VectorMigrationStatusResponse",
]
