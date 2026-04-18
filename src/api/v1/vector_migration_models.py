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


class VectorMigrationSummaryResponse(BaseModel):
    counts: Dict[str, int]
    total_migrations: int
    history_size: int
    statuses: list[str]
    backend: str = "memory"
    current_version_distribution: Optional[Dict[str, int]] = None
    current_total_vectors: Optional[int] = None
    scanned_vectors: Optional[int] = None
    scan_limit: Optional[int] = None
    distribution_complete: bool = True
    target_version: str = "v4"
    target_version_vectors: Optional[int] = None
    target_version_ratio: Optional[float] = None
    pending_vectors: Optional[int] = None
    migration_ready: bool = False


class VectorMigrationPendingItem(BaseModel):
    id: str
    from_version: str
    to_version: str


class VectorMigrationPendingResponse(BaseModel):
    target_version: str
    from_version_filter: Optional[str] = None
    items: list[VectorMigrationPendingItem]
    listed_count: int
    total_pending: Optional[int] = None
    backend: str = "memory"
    scanned_vectors: Optional[int] = None
    scan_limit: Optional[int] = None
    distribution_complete: bool = True


class VectorMigrationPendingSummaryResponse(BaseModel):
    target_version: str
    from_version_filter: Optional[str] = None
    observed_by_from_version: Dict[str, int]
    recommended_from_versions: list[str] = Field(default_factory=list)
    largest_pending_from_version: Optional[str] = None
    largest_pending_count: Optional[int] = None
    total_pending: Optional[int] = None
    pending_ratio: Optional[float] = None
    backend: str = "memory"
    scanned_vectors: Optional[int] = None
    scan_limit: Optional[int] = None
    distribution_complete: bool = True


class VectorMigrationPlanBatch(BaseModel):
    priority: int
    from_version: str
    pending_count: int
    suggested_run_limit: int
    allow_partial_scan_required: bool = False
    request_payload: Dict[str, Any]
    notes: list[str] = Field(default_factory=list)


class VectorMigrationPlanResponse(BaseModel):
    target_version: str
    from_version_filter: Optional[str] = None
    observed_by_from_version: Dict[str, int]
    recommended_from_versions: list[str] = Field(default_factory=list)
    largest_pending_from_version: Optional[str] = None
    largest_pending_count: Optional[int] = None
    total_pending: Optional[int] = None
    pending_ratio: Optional[float] = None
    backend: str = "memory"
    scanned_vectors: Optional[int] = None
    scan_limit: Optional[int] = None
    distribution_complete: bool = True
    max_batches: int
    default_run_limit: int
    estimated_total_runs: int = 0
    estimated_runs_by_version: Dict[str, int] = Field(default_factory=dict)
    plan_ready: bool = False
    blocking_reasons: list[str] = Field(default_factory=list)
    recommended_first_batch: Optional[VectorMigrationPlanBatch] = None
    recommended_first_request_payload: Optional[Dict[str, Any]] = None
    planned_pending_count: int = 0
    remaining_pending_count: Optional[int] = None
    planned_pending_ratio: Optional[float] = None
    coverage_complete: bool = False
    truncated_by_max_batches: bool = False
    unplanned_from_versions: list[str] = Field(default_factory=list)
    suggested_next_max_batches: Optional[int] = None
    batches: list[VectorMigrationPlanBatch] = Field(default_factory=list)


class VectorMigrationPendingRunRequest(BaseModel):
    limit: int = Field(default=50, ge=1, le=200, description="最多处理多少个待迁移向量")
    dry_run: bool = Field(default=False, description="是否只做试运行")
    from_version_filter: Optional[str] = Field(
        default=None,
        description="仅处理指定来源版本的待迁移向量",
    )
    allow_partial_scan: bool = Field(
        default=False,
        description="Qdrant 扫描不完整时是否仍允许按已扫描结果执行",
    )


class VectorMigrationPreviewResponse(BaseModel):
    total_vectors: int = Field(description="总向量数量")
    by_version: Dict[str, int] = Field(description="各版本向量数量统计")
    preview_items: list[VectorMigrateItem] = Field(description="预览前N个向量的迁移结果")
    estimated_dimension_changes: Dict[str, int] = Field(
        description="预计维度变化统计 (positive/negative/zero)"
    )
    migration_feasible: bool = Field(description="迁移是否可行")
    warnings: list[str] = Field(default_factory=list, description="潜在问题警告")
    avg_delta: Optional[float] = Field(default=None, description="采样维度变化平均值")
    median_delta: Optional[float] = Field(default=None, description="采样维度变化中位数")


class VectorMigrationTrendsResponse(BaseModel):
    total_migrations: int = Field(description="窗口内总迁移数量")
    success_rate: float = Field(description="迁移成功率 (0.0-1.0)")
    v4_adoption_rate: float = Field(description="v4版本采用率")
    avg_dimension_delta: float = Field(description="平均维度变化")
    window_hours: int = Field(description="统计窗口小时数")
    version_distribution: Dict[str, int] = Field(description="当前版本分布")
    migration_velocity: float = Field(description="每小时迁移数量")
    downgrade_rate: float = Field(description="降级比例")
    error_rate: float = Field(description="错误比例")
    time_range: Dict[str, Optional[str]] = Field(description="统计时间范围")
    current_total_vectors: Optional[int] = Field(default=None, description="当前向量总数")
    scanned_vectors: Optional[int] = Field(default=None, description="版本分布扫描数量")
    scan_limit: Optional[int] = Field(default=None, description="版本分布扫描上限")
    distribution_complete: bool = Field(default=True, description="版本分布是否为全量结果")
    target_version: str = Field(default="v4", description="readiness 目标特征版本")
    target_version_vectors: Optional[int] = Field(
        default=None, description="目标版本向量数，仅在全量分布时返回"
    )
    target_version_ratio: Optional[float] = Field(
        default=None, description="目标版本占比，仅在全量分布时返回"
    )
    pending_vectors: Optional[int] = Field(
        default=None, description="待迁移向量数，仅在全量分布时返回"
    )
    migration_ready: bool = Field(
        default=False, description="是否已经全部迁移到目标版本"
    )


__all__ = [
    "VectorMigrateItem",
    "VectorMigrateRequest",
    "VectorMigrateResponse",
    "VectorMigrationStatusResponse",
    "VectorMigrationSummaryResponse",
    "VectorMigrationPendingItem",
    "VectorMigrationPendingResponse",
    "VectorMigrationPendingSummaryResponse",
    "VectorMigrationPlanBatch",
    "VectorMigrationPlanResponse",
    "VectorMigrationPendingRunRequest",
    "VectorMigrationPreviewResponse",
    "VectorMigrationTrendsResponse",
]
