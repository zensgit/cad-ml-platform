"""2D dedup endpoints (proxy to dedupcad-vision).

This router intentionally delegates 2D duplicate detection to the dedicated
`dedupcad-vision` service for visual recall, while cad-ml-platform can optionally
run a local L4 precision verification using plugin-provided v2 geometry JSON.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import anyio
import httpx
from fastapi import APIRouter, Depends, File, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field, model_validator

from src.api.dependencies import get_admin_token, get_api_key
from src.core.dedupcad_2d_pipeline import run_dedup_2d_pipeline
from src.core.dedupcad_precision import GeomJsonStoreProtocol, PrecisionVerifier, create_geom_store
from src.core.dedupcad_precision.vendor.json_diff import compare_json
from src.core.dedupcad_tenant_config import TenantDedup2DConfigStore
from src.core.dedupcad_vision import DedupCadVisionClient
from src.core.dedupcad_2d_jobs import (
    Dedup2DJob,
    Dedup2DJobStatus,
    JobForbiddenError,
    JobNotFoundError,
    JobQueueFullError,
    get_dedup2d_job_store,
    set_dedup2d_job_metrics_callback,
)
from src.core.dedupcad_2d_jobs_redis import (
    Dedup2DRedisJobConfig,
    cancel_dedup2d_job_for_tenant as cancel_dedup2d_job_for_tenant_redis,
    get_dedup2d_job_for_tenant as get_dedup2d_job_for_tenant_redis,
    get_dedup2d_queue_depth as get_dedup2d_queue_depth_redis,
    get_dedup2d_redis_pool,
    list_dedup2d_jobs_for_tenant as list_dedup2d_jobs_for_tenant_redis,
    submit_dedup2d_job as submit_dedup2d_job_redis,
)
from src.utils.analysis_metrics import (
    dedup2d_cancel_total,
    dedup2d_job_duration_seconds,
    dedup2d_job_queue_depth,
    dedup2d_jobs_total,
    dedup2d_queue_full_total,
    dedup2d_search_mode_total,
    dedup2d_tenant_access_denied_total,
)

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# Phase 1: Metrics Callback for Job Completion/Failure
# =============================================================================


class _Dedup2DJobMetricsCallback:
    """Callback to update Prometheus metrics when jobs complete/fail."""

    def on_job_completed(self, job: Dedup2DJob, duration_seconds: float) -> None:
        dedup2d_jobs_total.labels(status="completed").inc()
        dedup2d_job_duration_seconds.labels(status="completed").observe(duration_seconds)

    def on_job_failed(self, job: Dedup2DJob, duration_seconds: float) -> None:
        dedup2d_jobs_total.labels(status="failed").inc()
        dedup2d_job_duration_seconds.labels(status="failed").observe(duration_seconds)

    def on_job_canceled(self, job: Dedup2DJob) -> None:
        dedup2d_jobs_total.labels(status="canceled").inc()
        try:
            finished_at = float(job.finished_at or time.time())
            started_at = float(job.started_at or job.created_at)
            duration = max(0.0, finished_at - started_at)
            dedup2d_job_duration_seconds.labels(status="canceled").observe(duration)
        except Exception:
            logger.debug("dedup2d_job_duration_observe_failed", exc_info=True)

    def on_queue_depth_changed(self, depth: int) -> None:
        dedup2d_job_queue_depth.set(depth)


def register_dedup2d_job_metrics() -> None:
    """Register metrics callback for job completion/failure events.

    Call this once during application startup to enable accurate metrics
    for job duration histograms and completion counters.
    """
    set_dedup2d_job_metrics_callback(_Dedup2DJobMetricsCallback())
    logger.info("dedup2d_job_metrics_registered")


def _get_dedup2d_async_backend() -> str:
    return os.getenv("DEDUP2D_ASYNC_BACKEND", "inprocess").strip().lower() or "inprocess"


# =============================================================================
# Phase 4 Day 6: Forced Async Configuration
# =============================================================================

def _get_int_env(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("invalid_int_env", extra={"name": name, "value": raw})
        return default


def _get_bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value == "":
        return default
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    logger.warning("invalid_bool_env", extra={"name": name, "value": raw})
    return default


def _check_forced_async(
    file_size: int,
    enable_precision: bool,
    mode: str,
    query_geom: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Check if the request should be forced to async mode.

    Returns:
        forced_async_reason if async should be forced, None otherwise.
    """
    forced_async_file_size_bytes = _get_int_env(
        "DEDUP2D_FORCED_ASYNC_FILE_SIZE_BYTES",
        default=5 * 1024 * 1024,  # 5MB
    )
    forced_async_enable_precision = _get_bool_env(
        "DEDUP2D_FORCED_ASYNC_ON_PRECISION",
        default=True,
    )
    forced_async_mode_precise = _get_bool_env(
        "DEDUP2D_FORCED_ASYNC_ON_MODE_PRECISE",
        default=True,
    )

    # Check file size threshold
    if forced_async_file_size_bytes > 0 and file_size > forced_async_file_size_bytes:
        mb = max(1, forced_async_file_size_bytes // (1024 * 1024))
        return f"file_size>{mb}MB"

    # Check precision verification (slow operation)
    if forced_async_enable_precision and enable_precision and query_geom is not None:
        return "enable_precision_with_geom_json"

    # Check precise mode
    if forced_async_mode_precise and mode == "precise":
        return "mode=precise"

    return None


_SEARCH_PRESETS: Dict[str, Dict[str, Any]] = {
    # "strict duplicate": bias towards low false positives.
    "strict": {
        "mode": "balanced",
        "precision_profile": "strict",
        "precision_top_n": 20,
        "duplicate_threshold": 0.95,
        "similar_threshold": 0.80,
        "precision_visual_weight": 0.3,
        "precision_geom_weight": 0.7,
        "version_gate": "off",
    },
    # "version dedup": bias towards grouping different versions of the same drawing.
    "version": {
        "mode": "balanced",
        "precision_profile": "version",
        "precision_top_n": 50,
        "duplicate_threshold": 0.95,
        "similar_threshold": 0.70,
        "precision_visual_weight": 0.5,
        "precision_geom_weight": 0.5,
        # Prefer same-drawing candidates (meta/file_name) when possible.
        "version_gate": "auto",
    },
    # "loose similar": bias towards recall.
    "loose": {
        "mode": "balanced",
        "precision_profile": "version",
        "precision_top_n": 50,
        "duplicate_threshold": 0.90,
        "similar_threshold": 0.50,
        "precision_visual_weight": 0.6,
        "precision_geom_weight": 0.4,
        "version_gate": "off",
    },
}


def _apply_preset_defaults(
    request: Request,
    preset: Optional[str],
    values: Dict[str, Any],
) -> Dict[str, Any]:
    if preset is None:
        return values
    key = str(preset).strip().lower()
    if not key:
        return values
    if key not in _SEARCH_PRESETS:
        raise HTTPException(status_code=400, detail=f"Unknown preset: {preset}") from None

    merged = dict(values)
    params = request.query_params
    for k, v in _SEARCH_PRESETS[key].items():
        # Only apply when the caller did not explicitly set the query param.
        if k not in params:
            merged[k] = v
    return merged


def get_dedupcad_vision_client() -> DedupCadVisionClient:
    return DedupCadVisionClient()


def get_geom_store() -> GeomJsonStoreProtocol:
    return create_geom_store()


def get_precision_verifier() -> PrecisionVerifier:
    return PrecisionVerifier()


def get_tenant_config_store() -> TenantDedup2DConfigStore:
    return TenantDedup2DConfigStore()


def _normalize_weights(visual_w: float, geom_w: float) -> tuple[float, float]:
    if visual_w < 0 or geom_w < 0:
        raise ValueError("weights must be >= 0")
    total = visual_w + geom_w
    if total <= 0:
        raise ValueError("weights sum must be > 0")
    return visual_w / total, geom_w / total


_TENANT_DEFAULT_KEYS = {
    "mode",
    "precision_profile",
    "precision_top_n",
    "precision_visual_weight",
    "precision_geom_weight",
    "duplicate_threshold",
    "similar_threshold",
    "version_gate",
}


_VERSION_GATE_MODES = {"off", "auto", "file_name", "meta"}
_VERSION_SUFFIX_RE = re.compile(r"(?:[_\-\s]?v\d+)$", re.IGNORECASE)


def _extract_meta_drawing_key(v2: Dict[str, Any]) -> Optional[str]:
    meta = v2.get("meta")
    if not isinstance(meta, dict):
        return None
    for k in (
        "drawing_number",
        "drawing_no",
        "drawingNo",
        "drawingNumber",
        "drawing_id",
        "drawingId",
        "number",
    ):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _extract_file_stem_key(file_name: Optional[str]) -> Optional[str]:
    if not file_name:
        return None
    stem = Path(str(file_name)).stem.strip()
    if not stem:
        return None
    if _VERSION_SUFFIX_RE.search(stem) is None:
        return None
    base = _VERSION_SUFFIX_RE.sub("", stem).rstrip(" _-").strip()
    return base if base else None


def _apply_tenant_defaults(
    request: Request,
    tenant_cfg: Dict[str, Any],
    values: Dict[str, Any],
) -> Dict[str, Any]:
    if not tenant_cfg:
        return values
    merged = dict(values)
    params = request.query_params
    for k in _TENANT_DEFAULT_KEYS:
        # Only apply when the caller did not explicitly set the query param.
        if k in params:
            continue
        v = tenant_cfg.get(k)
        if v is None:
            continue
        merged[k] = v
    return merged


def _apply_precision_l4(
    response: Dict[str, Any],
    query_geom: Dict[str, Any],
    query_file_name: Optional[str],
    geom_store: GeomJsonStoreProtocol,
    precision_verifier: PrecisionVerifier,
    max_results: int,
    precision_profile: Optional[str],
    version_gate: Optional[str],
    precision_top_n: int,
    precision_visual_weight: float,
    precision_geom_weight: float,
    duplicate_threshold: float,
    similar_threshold: float,
    precision_compute_diff: bool,
    precision_diff_top_n: int,
    precision_diff_max_paths: int,
) -> Dict[str, Any]:
    """Apply local v2 JSON precision scoring to a vision response (sync)."""
    precision_top_n = int(precision_top_n)
    if precision_top_n <= 0:
        precision_top_n = 1

    visual_w, geom_w = _normalize_weights(
        float(precision_visual_weight),
        float(precision_geom_weight),
    )

    matches: List[Dict[str, Any]] = list(response.get("duplicates") or []) + list(
        response.get("similar") or []
    )
    if not matches:
        return response

    raw_profile = str(precision_profile or "strict").strip().lower() or "strict"
    gate_mode = str(version_gate or "off").strip().lower() or "off"
    if gate_mode not in _VERSION_GATE_MODES:
        raise ValueError(f"Invalid version_gate: {version_gate}")
    query_key: Optional[str] = None
    gate_mode_effective = "off"
    if raw_profile == "version" and gate_mode != "off":
        query_key_meta = _extract_meta_drawing_key(query_geom)
        query_key_file = _extract_file_stem_key(query_file_name)
        if gate_mode == "meta":
            query_key = query_key_meta
            gate_mode_effective = "meta" if query_key else "off"
        elif gate_mode == "file_name":
            query_key = query_key_file
            gate_mode_effective = "file_name" if query_key else "off"
        else:  # auto
            if query_key_meta:
                query_key = query_key_meta
                gate_mode_effective = "meta"
            elif query_key_file:
                query_key = query_key_file
                gate_mode_effective = "file_name"

    missing_geom = 0
    verified = 0
    verified_hashes: set[str] = set()
    gated_out = 0

    duplicate_threshold = float(duplicate_threshold)
    similar_threshold = float(similar_threshold)
    if not (0.0 <= similar_threshold <= duplicate_threshold <= 1.0):
        raise ValueError(
            "invalid thresholds: require 0 <= similar_threshold <= duplicate_threshold <= 1"
        )

    # Select Top-N by current similarity (visual) for precision verification
    matches_sorted = sorted(
        matches,
        key=lambda m: float(m.get("similarity") or 0.0),
        reverse=True,
    )
    diff_budget = int(precision_diff_top_n)
    if diff_budget < 0:
        diff_budget = 0
    diff_max_paths = int(precision_diff_max_paths)
    if diff_max_paths <= 0:
        diff_max_paths = 200

    def _diff_view(v2: Dict[str, Any]) -> Dict[str, Any]:
        # Avoid exploding diff size: keep only sections that are helpful for human review.
        # Entities are often large and order-sensitive; rely on scoring breakdown instead.
        out: Dict[str, Any] = {}
        for k in ("layers", "blocks", "dimensions", "hatches", "text_content"):
            if k in v2:
                out[k] = v2.get(k)
        return out

    for match in matches_sorted[: max(1, precision_top_n)]:
        file_hash = str(match.get("file_hash") or "")
        if not file_hash:
            continue

        if gate_mode_effective == "file_name" and query_key:
            cand_key = _extract_file_stem_key(str(match.get("file_name") or ""))
            if cand_key and cand_key != query_key:
                match["similarity"] = 0.0
                match["verdict"] = "different"
                levels = match.setdefault("levels", {})
                l4 = levels.setdefault("l4", {})
                l4["version_gate"] = {
                    "mode": gate_mode_effective,
                    "query_key": query_key,
                    "candidate_key": cand_key,
                    "match": False,
                }
                gated_out += 1
                continue

        candidate_geom = geom_store.load(file_hash)
        if candidate_geom is None:
            missing_geom += 1
            continue

        if gate_mode_effective == "meta" and query_key:
            cand_key = _extract_meta_drawing_key(candidate_geom)
            if cand_key and cand_key != query_key:
                match["similarity"] = 0.0
                match["verdict"] = "different"
                levels = match.setdefault("levels", {})
                l4 = levels.setdefault("l4", {})
                l4["version_gate"] = {
                    "mode": gate_mode_effective,
                    "query_key": query_key,
                    "candidate_key": cand_key,
                    "match": False,
                }
                gated_out += 1
                continue

        precision = precision_verifier.score_pair(
            query_geom,
            candidate_geom,
            profile=precision_profile,
        )
        visual_sim = float(match.get("similarity") or 0.0)
        final_sim = (visual_w * visual_sim) + (geom_w * precision.score)

        match["visual_similarity"] = visual_sim
        match["precision_score"] = precision.score
        match["precision_breakdown"] = precision.breakdown
        match["similarity"] = float(final_sim)

        levels = match.setdefault("levels", {})
        l4 = levels.setdefault("l4", {})
        l4["precision_score"] = precision.score
        l4["precision_breakdown"] = precision.breakdown
        l4["precision_profile"] = precision_profile or "strict"
        l4["geom_hash_left"] = precision.geom_hash_left
        l4["geom_hash_right"] = precision.geom_hash_right
        if precision_compute_diff and diff_budget > 0:
            try:
                diff, diff_sim = compare_json(
                    _diff_view(query_geom),
                    _diff_view(candidate_geom),
                    case_insensitive=True,
                    list_path_modes={
                        # Order of dimensions/hatches is generally not meaningful for human review.
                        "dimensions": "unordered",
                        "hatches": "unordered",
                    },
                    max_diff_paths=diff_max_paths,
                )
                match["precision_diff"] = diff
                match["precision_diff_similarity"] = float(diff_sim)
                l4["precision_diff_similarity"] = float(diff_sim)
            except Exception as e:
                match["precision_diff"] = {"meta": {"error": str(e)}}
                match["precision_diff_similarity"] = None
                l4["precision_diff_similarity"] = None
            diff_budget -= 1

        try:
            match["match_level"] = max(int(match.get("match_level") or 0), 4)
        except Exception:
            match["match_level"] = 4

        verified += 1
        verified_hashes.add(file_hash)

    # Safety: if a query provides geom_json, do not let unverified (vision-only) matches
    # be classified as duplicate/similar at high similarity thresholds.
    #
    # We down-weight any match that did not receive a precision score, using only the
    # visual component. This keeps results stable even when vision recall returns many
    # high-similarity candidates but geometry verification is capped by precision_top_n.
    for match in matches:
        file_hash = str(match.get("file_hash") or "")
        if file_hash and file_hash in verified_hashes:
            continue
        if match.get("precision_score") is not None:
            continue
        visual_sim = float(match.get("similarity") or 0.0)
        match["visual_similarity"] = match.get("visual_similarity") or visual_sim
        match["similarity"] = float(visual_w * visual_sim)

    # Re-categorize results using fused similarity
    dup_th = duplicate_threshold
    sim_th = similar_threshold
    new_duplicates: List[Dict[str, Any]] = []
    new_similar: List[Dict[str, Any]] = []
    for match in matches:
        sim = float(match.get("similarity") or 0.0)
        if sim >= dup_th:
            match["verdict"] = "duplicate"
            new_duplicates.append(match)
        elif sim >= sim_th:
            match["verdict"] = "similar"
            new_similar.append(match)
        else:
            match["verdict"] = "different"

    new_duplicates.sort(key=lambda m: float(m.get("similarity") or 0.0), reverse=True)
    new_similar.sort(key=lambda m: float(m.get("similarity") or 0.0), reverse=True)

    response["duplicates"] = new_duplicates[:max_results]
    response["similar"] = new_similar[:max_results]
    response["total_matches"] = len(response["duplicates"]) + len(response["similar"])
    if verified:
        response["final_level"] = max(int(response.get("final_level") or 0), 4)
    else:
        response["final_level"] = response.get("final_level")

    warnings = response.setdefault("warnings", [])
    if missing_geom:
        warnings.append(f"precision_missing_geom_json:{missing_geom}")
    if gated_out:
        warnings.append(f"precision_version_gate_filtered:{gated_out}")
    if verified:
        warnings.append(f"precision_verified:{verified}")

    return response


async def _run_dedup_2d_pipeline(
    *,
    client: DedupCadVisionClient,
    geom_store: GeomJsonStoreProtocol,
    precision_verifier: PrecisionVerifier,
    file_name: str,
    file_bytes: bytes,
    content_type: str,
    query_geom: Optional[Dict[str, Any]],
    mode: str,
    max_results: int,
    compute_diff: bool,
    enable_ml: bool,
    enable_geometric: bool,
    enable_precision: bool,
    precision_profile: Optional[str],
    version_gate: Optional[str],
    precision_top_n: int,
    precision_visual_weight: float,
    precision_geom_weight: float,
    precision_compute_diff: bool,
    precision_diff_top_n: int,
    precision_diff_max_paths: int,
    duplicate_threshold: float,
    similar_threshold: float,
) -> Dict[str, Any]:
    response = await client.search_2d(
        file_name=file_name or "unknown",
        file_bytes=file_bytes,
        content_type=content_type or "application/octet-stream",
        mode=mode,
        max_results=max_results,
        compute_diff=compute_diff,
        enable_ml=enable_ml,
        enable_geometric=enable_geometric,
    )

    should_run_precision = query_geom is not None and (
        enable_precision or enable_geometric or mode == "precise"
    )
    if not should_run_precision:
        return response

    precision_start = time.perf_counter()
    response = await anyio.to_thread.run_sync(
        _apply_precision_l4,
        response,
        query_geom,
        file_name,
        geom_store,
        precision_verifier,
        max_results,
        precision_profile,
        version_gate,
        precision_top_n,
        precision_visual_weight,
        precision_geom_weight,
        duplicate_threshold,
        similar_threshold,
        precision_compute_diff,
        precision_diff_top_n,
        precision_diff_max_paths,
    )
    precision_ms = (time.perf_counter() - precision_start) * 1000
    timing = response.setdefault("timing", {})
    timing["precision_ms"] = precision_ms
    try:
        timing["l4_ms"] = float(timing.get("l4_ms") or 0.0) + precision_ms
    except Exception:
        timing["l4_ms"] = precision_ms
    try:
        timing["total_ms"] = float(timing.get("total_ms") or 0.0) + precision_ms
    except Exception:
        timing["total_ms"] = precision_ms
    return response


class Dedup2DHealthResponse(BaseModel):
    status: str
    service: Optional[str] = None
    version: Optional[str] = None
    indexes: Optional[Dict[str, Any]] = None


class Dedup2DMatchItem(BaseModel):
    drawing_id: str
    file_hash: str
    file_name: str
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


@router.get("/2d/health", response_model=Dedup2DHealthResponse)
async def dedup_2d_health(
    api_key: str = Depends(get_api_key),
    client: DedupCadVisionClient = Depends(get_dedupcad_vision_client),
):
    """Proxy dedupcad-vision health for operational visibility."""
    try:
        return await client.health()
    except httpx.RequestError as e:
        logger.warning("dedupcad_vision_unavailable", extra={"error": str(e)})
        raise HTTPException(status_code=503, detail="dedupcad-vision unavailable") from e
    except httpx.HTTPStatusError as e:
        detail: Any
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        raise HTTPException(status_code=e.response.status_code, detail=detail) from e


@router.post("/2d/index/rebuild", response_model=Dedup2DIndexRebuildResponse)
async def dedup_2d_index_rebuild(
    api_key: str = Depends(get_api_key),
    client: DedupCadVisionClient = Depends(get_dedupcad_vision_client),
):
    """Trigger a rebuild of dedupcad-vision L1/L2 indexes (batch-friendly)."""
    try:
        return await client.rebuild_indexes()
    except httpx.RequestError as e:
        logger.warning("dedupcad_vision_unavailable", extra={"error": str(e)})
        raise HTTPException(status_code=503, detail="dedupcad-vision unavailable") from e
    except httpx.HTTPStatusError as e:
        detail: Any
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        raise HTTPException(status_code=e.response.status_code, detail=detail) from e


@router.get("/2d/presets", response_model=Dedup2DPresetsResponse)
async def dedup_2d_presets(
    api_key: str = Depends(get_api_key),
):
    return Dedup2DPresetsResponse(presets=_SEARCH_PRESETS)


@router.get(
    "/2d/config",
    response_model=Dedup2DTenantConfigResponse,
    response_model_exclude_none=True,
)
async def dedup_2d_get_tenant_config(
    api_key: str = Depends(get_api_key),
    store: TenantDedup2DConfigStore = Depends(get_tenant_config_store),
):
    cfg = await store.get(api_key)
    model = Dedup2DTenantConfig(**(cfg or {}))
    return Dedup2DTenantConfigResponse(
        tenant_id=store.tenant_id(api_key),
        config=model,
    )


@router.put(
    "/2d/config",
    response_model=Dedup2DTenantConfigResponse,
    response_model_exclude_none=True,
)
async def dedup_2d_set_tenant_config(
    payload: Dedup2DTenantConfig,
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(get_admin_token),
    store: TenantDedup2DConfigStore = Depends(get_tenant_config_store),
):
    cfg = payload.model_dump(exclude_none=True)
    preset = cfg.get("preset")
    if preset is not None:
        key = str(preset).strip().lower()
        if key not in _SEARCH_PRESETS:
            raise HTTPException(status_code=400, detail=f"Unknown preset: {preset}") from None
        cfg["preset"] = key
    mode = cfg.get("mode")
    if mode is not None and str(mode) not in {"fast", "balanced", "precise"}:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {mode}") from None
    precision_profile = cfg.get("precision_profile")
    if precision_profile is not None and str(precision_profile) not in {"strict", "version"}:
        raise HTTPException(
            status_code=400, detail=f"Invalid precision_profile: {precision_profile}"
        ) from None

    await store.set(api_key, cfg)
    return Dedup2DTenantConfigResponse(
        tenant_id=store.tenant_id(api_key),
        config=Dedup2DTenantConfig(**cfg),
    )


@router.delete(
    "/2d/config",
    response_model=Dedup2DTenantConfigResponse,
    response_model_exclude_none=True,
)
async def dedup_2d_delete_tenant_config(
    api_key: str = Depends(get_api_key),
    admin_token: str = Depends(get_admin_token),
    store: TenantDedup2DConfigStore = Depends(get_tenant_config_store),
):
    await store.delete(api_key)
    return Dedup2DTenantConfigResponse(
        tenant_id=store.tenant_id(api_key),
        config=Dedup2DTenantConfig(),
    )


@router.post("/2d/precision/compare", response_model=Dedup2DPrecisionCompareResponse)
async def dedup_2d_precision_compare(
    left_geom_json: UploadFile = File(..., description="Left v2 geometry JSON"),
    right_geom_json: UploadFile = File(..., description="Right v2 geometry JSON"),
    api_key: str = Depends(get_api_key),
    precision_verifier: PrecisionVerifier = Depends(get_precision_verifier),
):
    """Compare two v2 JSON files using the DedupCAD precision scorer."""
    left_bytes = await left_geom_json.read()
    right_bytes = await right_geom_json.read()
    if not left_bytes or not right_bytes:
        raise HTTPException(status_code=400, detail="Empty geom_json") from None
    try:
        left_obj = precision_verifier.load_json_bytes(left_bytes)
        right_obj = precision_verifier.load_json_bytes(right_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"Invalid geom_json: {e}") from e

    precision = await anyio.to_thread.run_sync(precision_verifier.score_pair, left_obj, right_obj)
    return Dedup2DPrecisionCompareResponse(
        score=precision.score,
        breakdown=precision.breakdown,
        geom_hash_left=precision.geom_hash_left,
        geom_hash_right=precision.geom_hash_right,
    )


@router.get("/2d/geom/{file_hash}/exists", response_model=Dedup2DGeomExistsResponse)
async def dedup_2d_geom_exists(
    file_hash: str,
    api_key: str = Depends(get_api_key),
    geom_store: GeomJsonStoreProtocol = Depends(get_geom_store),
):
    """Check whether a candidate v2 JSON exists locally for a given `file_hash`."""
    try:
        exists = geom_store.exists(file_hash)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    return Dedup2DGeomExistsResponse(file_hash=file_hash, exists=exists)


@router.post("/2d/search", response_model=Union[Dedup2DSearchResponse, Dedup2DSearchAsyncResponse])
async def dedup_2d_search(
    request: Request,
    file: UploadFile = File(..., description="2D drawing image/PDF (PNG/JPG/PDF)"),
    geom_json: Optional[UploadFile] = File(
        default=None, description="v2 geometry JSON from CAD plugin (for precision verification)"
    ),
    async_request: bool = Query(
        default=False,
        alias="async",
        description="Return job_id immediately and poll /api/v1/dedup/2d/jobs/{job_id} for results",
    ),
    callback_url: Optional[str] = Query(
        default=None,
        description=(
            "Optional webhook URL to notify when an async job finishes "
            "(requires DEDUP2D_ASYNC_BACKEND=redis)"
        ),
    ),
    mode: str = "balanced",
    max_results: int = 50,
    compute_diff: bool = True,
    enable_ml: bool = False,
    enable_geometric: bool = False,
    enable_precision: bool = True,
    preset: Optional[str] = None,
    precision_profile: Optional[str] = None,
    version_gate: Optional[str] = None,
    precision_top_n: int = 20,
    precision_visual_weight: float = 0.3,
    precision_geom_weight: float = 0.7,
    precision_compute_diff: bool = False,
    precision_diff_top_n: int = 5,
    precision_diff_max_paths: int = 200,
    duplicate_threshold: float = 0.95,
    similar_threshold: float = 0.80,
    api_key: str = Depends(get_api_key),
    client: DedupCadVisionClient = Depends(get_dedupcad_vision_client),
    geom_store: GeomJsonStoreProtocol = Depends(get_geom_store),
    precision_verifier: PrecisionVerifier = Depends(get_precision_verifier),
    tenant_store: TenantDedup2DConfigStore = Depends(get_tenant_config_store),
):
    """Run progressive 2D duplicate search via dedupcad-vision."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    content_type = file.content_type or "application/octet-stream"

    query_geom: Optional[Dict[str, Any]] = None
    if geom_json is not None:
        geom_bytes = await geom_json.read()
        if not geom_bytes:
            raise HTTPException(status_code=400, detail="Empty geom_json")
        try:
            query_geom = precision_verifier.load_json_bytes(geom_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid geom_json: {e}") from e

    try:
        tenant_cfg = await tenant_store.get(api_key) or {}
        explicit_preset = "preset" in request.query_params
        preset_effective = preset if explicit_preset else (tenant_cfg.get("preset") or preset)

        values = {
            "mode": mode,
            "precision_profile": precision_profile,
            "version_gate": version_gate,
            "precision_top_n": precision_top_n,
            "precision_visual_weight": precision_visual_weight,
            "precision_geom_weight": precision_geom_weight,
            "duplicate_threshold": duplicate_threshold,
            "similar_threshold": similar_threshold,
        }

        if explicit_preset:
            # Explicit request preset should override tenant defaults.
            values = _apply_tenant_defaults(request, tenant_cfg, values)
            effective = _apply_preset_defaults(request, preset_effective, values)
        else:
            # Tenant preset/overrides apply when the caller did not specify a preset.
            values = _apply_preset_defaults(request, preset_effective, values)
            effective = _apply_tenant_defaults(request, tenant_cfg, values)

        mode = str(effective["mode"])
        precision_profile = effective.get("precision_profile")
        version_gate = effective.get("version_gate")
        precision_top_n = int(effective["precision_top_n"])
        precision_visual_weight = float(effective["precision_visual_weight"])
        precision_geom_weight = float(effective["precision_geom_weight"])
        duplicate_threshold = float(effective["duplicate_threshold"])
        similar_threshold = float(effective["similar_threshold"])

        # Phase 4 Day 6: Check if request should be forced to async mode
        forced_async_reason = _check_forced_async(
            file_size=len(content),
            enable_precision=enable_precision,
            mode=mode,
            query_geom=query_geom,
        )

        if async_request or forced_async_reason:
            tenant_id = tenant_store.tenant_id(api_key)
            backend = _get_dedup2d_async_backend()

            if callback_url and backend != "redis":
                raise HTTPException(
                    status_code=400,
                    detail={
                        "error": "CALLBACK_UNSUPPORTED",
                        "message": "callback_url requires DEDUP2D_ASYNC_BACKEND=redis",
                    },
                ) from None

            if backend == "redis":
                request_params = {
                    "mode": mode,
                    "max_results": max_results,
                    "compute_diff": compute_diff,
                    "enable_ml": enable_ml,
                    "enable_geometric": enable_geometric,
                    "enable_precision": enable_precision,
                    "precision_profile": precision_profile,
                    "version_gate": version_gate,
                    "precision_top_n": precision_top_n,
                    "precision_visual_weight": precision_visual_weight,
                    "precision_geom_weight": precision_geom_weight,
                    "precision_compute_diff": precision_compute_diff,
                    "precision_diff_top_n": precision_diff_top_n,
                    "precision_diff_max_paths": precision_diff_max_paths,
                    "duplicate_threshold": duplicate_threshold,
                    "similar_threshold": similar_threshold,
                }

                try:
                    job = await submit_dedup2d_job_redis(
                        tenant_id=tenant_id,
                        file_name=file.filename or "unknown",
                        file_bytes=content,
                        content_type=content_type,
                        query_geom=query_geom,
                        request_params=request_params,
                        callback_url=callback_url,
                    )
                    dedup2d_jobs_total.labels(status="pending").inc()
                    dedup2d_search_mode_total.labels(mode=mode).inc()
                    try:
                        dedup2d_job_queue_depth.set(await get_dedup2d_queue_depth_redis())
                    except Exception:
                        logger.debug("dedup2d_queue_depth_fetch_failed", exc_info=True)
                except ValueError as e:
                    raise HTTPException(
                        status_code=400,
                        detail={
                            "error": "CALLBACK_URL_INVALID",
                            "message": str(e),
                        },
                    ) from None
                except JobQueueFullError as e:
                    dedup2d_queue_full_total.inc()
                    raise HTTPException(
                        status_code=429,
                        detail={
                            "error": "JOB_QUEUE_FULL",
                            "message": str(e),
                            "max_jobs": e.max_jobs,
                            "current_jobs": e.current_jobs,
                        },
                        headers={"Retry-After": "5"},
                    ) from e

                return Dedup2DSearchAsyncResponse(
                    job_id=job.job_id,
                    status=job.status,
                    poll_url=f"/api/v1/dedup/2d/jobs/{job.job_id}",
                    forced_async_reason=forced_async_reason,
                )

            # Default backend: in-process
            store = get_dedup2d_job_store()

            async def _runner() -> Dict[str, Any]:
                return await run_dedup_2d_pipeline(
                    client=client,
                    geom_store=geom_store,
                    precision_verifier=precision_verifier,
                    file_name=file.filename or "unknown",
                    file_bytes=content,
                    content_type=content_type,
                    query_geom=query_geom,
                    mode=mode,
                    max_results=max_results,
                    compute_diff=compute_diff,
                    enable_ml=enable_ml,
                    enable_geometric=enable_geometric,
                    enable_precision=enable_precision,
                    precision_profile=precision_profile,
                    version_gate=version_gate,
                    precision_top_n=precision_top_n,
                    precision_visual_weight=precision_visual_weight,
                    precision_geom_weight=precision_geom_weight,
                    precision_compute_diff=precision_compute_diff,
                    precision_diff_top_n=precision_diff_top_n,
                    precision_diff_max_paths=precision_diff_max_paths,
                    duplicate_threshold=duplicate_threshold,
                    similar_threshold=similar_threshold,
                )

            try:
                job = await store.submit(
                    _runner,
                    tenant_id=tenant_id,
                    meta={
                        "mode": mode,
                        "max_results": max_results,
                        "enable_ml": enable_ml,
                        "enable_geometric": enable_geometric,
                        "enable_precision": enable_precision,
                    },
                )
                # Phase 1: Record metrics
                dedup2d_jobs_total.labels(status="pending").inc()
                dedup2d_search_mode_total.labels(mode=mode).inc()
                dedup2d_job_queue_depth.set(store.get_queue_depth())
            except JobQueueFullError as e:
                dedup2d_queue_full_total.inc()
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "JOB_QUEUE_FULL",
                        "message": str(e),
                        "max_jobs": e.max_jobs,
                        "current_jobs": e.current_jobs,
                    },
                    headers={"Retry-After": "5"},  # Suggest retry in 5 seconds
                ) from e

            return Dedup2DSearchAsyncResponse(
                job_id=job.job_id,
                status=job.status,
                poll_url=f"/api/v1/dedup/2d/jobs/{job.job_id}",
                forced_async_reason=forced_async_reason,
            )

        try:
            return await run_dedup_2d_pipeline(
                client=client,
                geom_store=geom_store,
                precision_verifier=precision_verifier,
                file_name=file.filename or "unknown",
                file_bytes=content,
                content_type=content_type,
                query_geom=query_geom,
                mode=mode,
                max_results=max_results,
                compute_diff=compute_diff,
                enable_ml=enable_ml,
                enable_geometric=enable_geometric,
                enable_precision=enable_precision,
                precision_profile=precision_profile,
                version_gate=version_gate,
                precision_top_n=precision_top_n,
                precision_visual_weight=precision_visual_weight,
                precision_geom_weight=precision_geom_weight,
                precision_compute_diff=precision_compute_diff,
                precision_diff_top_n=precision_diff_top_n,
                precision_diff_max_paths=precision_diff_max_paths,
                duplicate_threshold=duplicate_threshold,
                similar_threshold=similar_threshold,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
    except httpx.RequestError as e:
        logger.warning("dedupcad_vision_unavailable", extra={"error": str(e), "mode": mode})
        raise HTTPException(status_code=503, detail="dedupcad-vision unavailable") from e
    except httpx.HTTPStatusError as e:
        detail: Any
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        raise HTTPException(status_code=e.response.status_code, detail=detail) from e


@router.get("/2d/jobs/{job_id}", response_model=Dedup2DSearchJobResponse)
async def dedup_2d_job_status(
    job_id: str,
    api_key: str = Depends(get_api_key),
    tenant_store: TenantDedup2DConfigStore = Depends(get_tenant_config_store),
):
    """Get async 2D dedup job status and (when ready) the final result.

    Phase 1: Tenant isolation - only the tenant who created the job can view it.
    """
    tenant_id = tenant_store.tenant_id(api_key)
    backend = _get_dedup2d_async_backend()

    try:
        if backend == "redis":
            job = await get_dedup2d_job_for_tenant_redis(job_id, tenant_id)
        else:
            store = get_dedup2d_job_store()
            job = await store.get_for_tenant(job_id, tenant_id)
    except JobNotFoundError:
        raise HTTPException(
            status_code=404,
            detail={"error": "JOB_NOT_FOUND", "job_id": job_id},
        ) from None
    except JobForbiddenError:
        dedup2d_tenant_access_denied_total.labels(operation="get").inc()
        raise HTTPException(
            status_code=403,
            detail={"error": "JOB_FORBIDDEN", "job_id": job_id},
        ) from None

    if backend == "redis" and job.is_finished():
        cfg = Dedup2DRedisJobConfig.from_env()
        try:
            pool = await get_dedup2d_redis_pool(cfg)
            claimed = await pool.hsetnx(f"{cfg.key_prefix}:job:{job_id}", "metrics_recorded", "1")
        except Exception:
            claimed = False
        if claimed:
            duration = max(
                0.0,
                float((job.finished_at or time.time()) - (job.started_at or job.created_at)),
            )
            status_label = job.status.value
            dedup2d_jobs_total.labels(status=status_label).inc()
            if status_label in {"completed", "failed", "canceled"}:
                dedup2d_job_duration_seconds.labels(status=status_label).observe(duration)
            try:
                dedup2d_job_queue_depth.set(await get_dedup2d_queue_depth_redis())
            except Exception:
                logger.debug("dedup2d_queue_depth_fetch_failed", exc_info=True)

    result_model: Optional[Dedup2DSearchResponse] = None
    if job.result is not None:
        result_model = Dedup2DSearchResponse(**job.result)

    callback_status: Optional[str] = None
    callback_attempts: Optional[int] = None
    callback_http_status: Optional[int] = None
    callback_finished_at: Optional[float] = None
    callback_last_error: Optional[str] = None
    try:
        callback_status = str(job.meta.get("callback_status") or "").strip() or None
        callback_last_error = str(job.meta.get("callback_last_error") or "").strip() or None
        attempts_raw = str(job.meta.get("callback_attempts") or "").strip()
        callback_attempts = int(attempts_raw) if attempts_raw else None
        http_raw = str(job.meta.get("callback_http_status") or "").strip()
        callback_http_status = int(http_raw) if http_raw else None
        finished_raw = str(job.meta.get("callback_finished_at") or "").strip()
        callback_finished_at = float(finished_raw) if finished_raw else None
    except Exception:
        logger.debug("dedup2d_callback_meta_parse_failed", exc_info=True)

    return Dedup2DSearchJobResponse(
        job_id=job.job_id,
        tenant_id=job.tenant_id,
        status=job.status,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        result=result_model,
        error=job.error,
        callback_status=callback_status,
        callback_attempts=callback_attempts,
        callback_http_status=callback_http_status,
        callback_finished_at=callback_finished_at,
        callback_last_error=callback_last_error,
    )


@router.post("/2d/jobs/{job_id}/cancel", response_model=Dedup2DJobCancelResponse)
async def dedup_2d_job_cancel(
    job_id: str,
    api_key: str = Depends(get_api_key),
    tenant_store: TenantDedup2DConfigStore = Depends(get_tenant_config_store),
):
    """Cancel an async 2D dedup job (best-effort).

    Phase 1: Tenant isolation - only the tenant who created the job can cancel it.
    """
    tenant_id = tenant_store.tenant_id(api_key)
    backend = _get_dedup2d_async_backend()

    try:
        if backend == "redis":
            await cancel_dedup2d_job_for_tenant_redis(job_id, tenant_id)
            dedup2d_cancel_total.labels(result="success").inc()
            try:
                dedup2d_job_queue_depth.set(await get_dedup2d_queue_depth_redis())
            except Exception:
                logger.debug("dedup2d_queue_depth_fetch_failed", exc_info=True)
        else:
            store = get_dedup2d_job_store()
            await store.cancel_for_tenant(job_id, tenant_id)
            dedup2d_cancel_total.labels(result="success").inc()
            dedup2d_job_queue_depth.set(store.get_queue_depth())
    except JobNotFoundError:
        dedup2d_cancel_total.labels(result="not_found").inc()
        raise HTTPException(
            status_code=404,
            detail={"error": "JOB_NOT_FOUND", "job_id": job_id},
        ) from None
    except JobForbiddenError:
        dedup2d_cancel_total.labels(result="forbidden").inc()
        dedup2d_tenant_access_denied_total.labels(operation="cancel").inc()
        raise HTTPException(
            status_code=403,
            detail={"error": "JOB_FORBIDDEN", "job_id": job_id},
        ) from None

    return Dedup2DJobCancelResponse(job_id=job_id, tenant_id=tenant_id, canceled=True)


@router.get("/2d/jobs", response_model=Dedup2DJobListResponse)
async def dedup_2d_job_list(
    status: Optional[str] = Query(
        default=None,
        description="Filter by job status (pending, in_progress, completed, failed, canceled)",
    ),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of jobs to return"),
    api_key: str = Depends(get_api_key),
    tenant_store: TenantDedup2DConfigStore = Depends(get_tenant_config_store),
):
    """List async 2D dedup jobs for the current tenant.

    Phase 4 Day 6: API usability - allows tenants to view all their jobs.
    Returns jobs sorted by created_at descending (newest first).
    Only returns active jobs (pending/in_progress) and recently finished jobs within TTL.
    """
    tenant_id = tenant_store.tenant_id(api_key)
    backend = _get_dedup2d_async_backend()

    # Parse status filter
    status_filter: Optional[Dedup2DJobStatus] = None
    if status is not None:
        status_value = status.strip().lower()
        try:
            status_filter = Dedup2DJobStatus(status_value)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "INVALID_STATUS_FILTER",
                    "message": (
                        f"Invalid status: {status}. Valid values: pending, in_progress, "
                        "completed, failed, canceled"
                    ),
                },
            ) from None

    if backend == "redis":
        jobs = await list_dedup2d_jobs_for_tenant_redis(
            tenant_id, status=status_filter, limit=limit
        )
    else:
        store = get_dedup2d_job_store()
        jobs = await store.list_for_tenant(tenant_id, status=status_filter, limit=limit)

    items = [
        Dedup2DJobListItem(
            job_id=job.job_id,
            tenant_id=job.tenant_id,
            status=job.status,
            created_at=job.created_at,
            started_at=job.started_at,
            finished_at=job.finished_at,
            error=job.error,
        )
        for job in jobs
    ]

    return Dedup2DJobListResponse(jobs=items, total=len(items))


@router.post("/2d/index/add", response_model=Dedup2DIndexAddResponse)
async def dedup_2d_index_add(
    file: UploadFile = File(..., description="2D drawing image/PDF (PNG/JPG/PDF)"),
    geom_json: Optional[UploadFile] = File(
        default=None,
        description="v2 geometry JSON from CAD plugin (stored for later precision checks)",
    ),
    user_name: str = "cad-ml-platform",
    upload_to_s3: bool = True,
    api_key: str = Depends(get_api_key),
    client: DedupCadVisionClient = Depends(get_dedupcad_vision_client),
    geom_store: GeomJsonStoreProtocol = Depends(get_geom_store),
    precision_verifier: PrecisionVerifier = Depends(get_precision_verifier),
):
    """Index a 2D drawing into dedupcad-vision (for future searches)."""
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")
    content_type = file.content_type or "application/octet-stream"
    try:
        response = await client.index_add_2d(
            file_name=file.filename or "unknown",
            file_bytes=content,
            content_type=content_type,
            user_name=user_name,
            upload_to_s3=upload_to_s3,
        )
        if geom_json is None:
            return response

        geom_bytes = await geom_json.read()
        if not geom_bytes:
            raise HTTPException(status_code=400, detail="Empty geom_json")
        try:
            geom_obj = precision_verifier.load_json_bytes(geom_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid geom_json: {e}") from e

        file_hash = str(response.get("file_hash") or "")
        if not file_hash:
            raise HTTPException(
                status_code=502,
                detail="dedupcad-vision response missing file_hash",
            )
        await anyio.to_thread.run_sync(geom_store.save, file_hash, geom_obj)
        try:
            response["message"] = f"{response.get('message')}; geom_json stored"
        except Exception:
            pass
        return response
    except httpx.RequestError as e:
        logger.warning("dedupcad_vision_unavailable", extra={"error": str(e)})
        raise HTTPException(status_code=503, detail="dedupcad-vision unavailable") from e
    except httpx.HTTPStatusError as e:
        detail: Any
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text
        raise HTTPException(status_code=e.response.status_code, detail=detail) from e
