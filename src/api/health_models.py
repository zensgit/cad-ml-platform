"""Health and readiness response models."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class HealthServices(BaseModel):
    api: str
    ml: str
    redis: str


class HealthErrorRateEma(BaseModel):
    ocr: float
    vision: float


class HealthRuntime(BaseModel):
    python_version: str
    metrics_enabled: bool
    vision_max_base64_bytes: int
    error_rate_ema: HealthErrorRateEma


class HealthConfigLimits(BaseModel):
    vision_max_base64_bytes: int
    vision_max_base64_mb: float
    ocr_timeout_ms: int
    ocr_timeout_seconds: float


class HealthConfigProviders(BaseModel):
    ocr_default: str
    confidence_fallback: float


class HealthConfigMonitoring(BaseModel):
    error_ema_alpha: float
    metrics_enabled: bool
    redis_enabled: bool
    classifier_rate_limit_per_min: Optional[int] = None
    classifier_rate_limit_burst: Optional[int] = None
    classifier_cache_max_size: Optional[int] = None


class HealthConfigNetwork(BaseModel):
    cors_origins: List[str]
    allowed_hosts: List[str]


class HealthConfigDebug(BaseModel):
    debug_mode: bool
    log_level: str


class HealthConfigMlClassification(BaseModel):
    hybrid_enabled: bool
    hybrid_version: str
    hybrid_config_path: str
    graph2d_model_path: str
    filename_enabled: bool
    graph2d_enabled: bool
    titleblock_enabled: bool
    process_enabled: bool


class HealthConfigMlSampling(BaseModel):
    max_nodes: int
    strategy: str
    seed: int
    text_priority_ratio: float


class HealthConfigMl(BaseModel):
    classification: HealthConfigMlClassification
    sampling: HealthConfigMlSampling


class HealthConfig(BaseModel):
    limits: HealthConfigLimits
    providers: HealthConfigProviders
    monitoring: HealthConfigMonitoring
    network: HealthConfigNetwork
    debug: HealthConfigDebug
    ml: Optional[HealthConfigMl] = None


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: str
    timestamp: str
    services: HealthServices
    runtime: HealthRuntime
    config: HealthConfig
    resilience: Optional[Any] = None


class VectorStoreSummary(BaseModel):
    total: int
    versions: Dict[str, int]


class FaissSummary(BaseModel):
    enabled: bool
    imported: bool
    last_export_size: int
    last_export_age_seconds: Optional[float]


class ExtendedHealthResponse(HealthResponse):
    vector_store: VectorStoreSummary
    faiss: FaissSummary
    feature_version_env: str


class ReadinessCheck(BaseModel):
    status: str
    detail: Optional[str] = None
    duration_ms: Optional[float] = None
    timed_out: bool = False


class ReadinessResponse(BaseModel):
    status: str
    ready: bool
    timestamp: str
    checks: Dict[str, ReadinessCheck]
