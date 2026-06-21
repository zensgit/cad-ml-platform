"""Pydantic request/response models for the 2D dedup API.

Extracted from src/api/v1/dedup.py (behavior-preserving slimming); re-exported by
dedup.py so route decorators (response_model=...) and external imports keep working.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, model_validator

from src.core.dedupcad_2d_jobs import Dedup2DJobStatus
from src.core.dedup_2d_gating import _VERSION_GATE_MODES


class Dedup2DHealthResponse(BaseModel):
    status: str
    service: Optional[str] = None
    version: Optional[str] = None
    indexes: Optional[Dict[str, Any]] = None


class Dedup2DMatchItem(BaseModel):
    drawing_id: str
    file_hash: str
    file_name: str
    fine_part_type: Optional[str] = None
    coarse_part_type: Optional[str] = None
    decision_source: Optional[str] = None
    is_coarse_label: Optional[bool] = None
    similarity: float
    visual_similarity: Optional[float] = None
    precision_score: Optional[float] = None
    precision_breakdown: Optional[Dict[str, float]] = None
    precision_diff_similarity: Optional[float] = None
    precision_diff: Optional[Dict[str, Any]] = None
    confidence: float
    match_level: int
    verdict: str
    levels: Dict[str, Any]
    diff_image_base64: Optional[str] = None
    diff_regions: Optional[List[Dict[str, Any]]] = None


class Dedup2DSearchResponse(BaseModel):
    success: bool
    total_matches: int
    duplicates: List[Dedup2DMatchItem] = Field(default_factory=list)
    similar: List[Dedup2DMatchItem] = Field(default_factory=list)
    final_level: int
    timing: Dict[str, Any] = Field(default_factory=dict)
    level_stats: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None


class Dedup2DSearchAsyncResponse(BaseModel):
    job_id: str
    status: Dedup2DJobStatus
    poll_url: str
    forced_async_reason: Optional[str] = None  # Phase 1: explain why async was forced


class Dedup2DSearchJobResponse(BaseModel):
    job_id: str
    tenant_id: str  # Phase 1: tenant isolation
    status: Dedup2DJobStatus
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Optional[Dedup2DSearchResponse] = None
    error: Optional[str] = None
    # Phase 3 (optional): webhook callback status (redis backend only)
    callback_status: Optional[str] = None  # pending|success|failed|skipped
    callback_attempts: Optional[int] = None
    callback_http_status: Optional[int] = None
    callback_finished_at: Optional[float] = None
    callback_last_error: Optional[str] = None


class Dedup2DJobCancelResponse(BaseModel):
    job_id: str
    tenant_id: str  # Phase 1: tenant isolation
    canceled: bool


class Dedup2DIndexAddResponse(BaseModel):
    success: bool
    drawing_id: Optional[int] = None
    file_hash: str
    message: str
    processing_time_ms: float
    s3_key: Optional[str] = None


class Dedup2DIndexRebuildResponse(BaseModel):
    success: bool
    message: str


class Dedup2DPrecisionCompareResponse(BaseModel):
    score: float
    breakdown: Dict[str, float] = Field(default_factory=dict)
    geom_hash_left: str
    geom_hash_right: str


class Dedup2DGeomExistsResponse(BaseModel):
    file_hash: str
    exists: bool


class Dedup2DPresetsResponse(BaseModel):
    presets: Dict[str, Dict[str, Any]]


class Dedup2DTenantConfig(BaseModel):
    preset: Optional[str] = None
    mode: Optional[str] = None
    precision_profile: Optional[str] = None
    version_gate: Optional[str] = None
    precision_top_n: Optional[int] = Field(default=None, ge=1)
    precision_visual_weight: Optional[float] = Field(default=None, ge=0.0)
    precision_geom_weight: Optional[float] = Field(default=None, ge=0.0)
    duplicate_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    similar_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_thresholds(self) -> "Dedup2DTenantConfig":
        if self.duplicate_threshold is not None and self.similar_threshold is not None:
            if float(self.similar_threshold) > float(self.duplicate_threshold):
                raise ValueError("similar_threshold must be <= duplicate_threshold")
        if self.version_gate is not None:
            key = str(self.version_gate).strip().lower()
            if key not in _VERSION_GATE_MODES:
                raise ValueError(f"Invalid version_gate: {self.version_gate}")
        return self


class Dedup2DTenantConfigResponse(BaseModel):
    tenant_id: str
    config: Dedup2DTenantConfig


class Dedup2DJobListItem(BaseModel):
    """Job item for list endpoint (excludes result to reduce payload size)."""

    job_id: str
    tenant_id: str
    status: Dedup2DJobStatus
    created_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    error: Optional[str] = None


class Dedup2DJobListResponse(BaseModel):
    """Response for job list endpoint."""

    jobs: List[Dedup2DJobListItem] = Field(default_factory=list)
    total: int = 0
