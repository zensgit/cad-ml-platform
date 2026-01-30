"""Redis/ARQ-backed async job store for 2D dedup (Phase 2).

This backend is designed to replace the in-process job store when:
- running with multiple API workers, or
- running a separate worker deployment.

It stores job metadata + payload + (optionally) the final result in Redis,
and uses ARQ to execute jobs in a dedicated worker process.

Phase 4 Rolling Upgrade:
  During grayscale deployment, the API can write BOTH file_ref and file_bytes_b64
  to support old workers that only understand file_bytes_b64.

  Environment variables:
    - DEDUP2D_JOB_PAYLOAD_INCLUDE_BYTES_B64: "1" to include base64 bytes in payload
    - DEDUP2D_JOB_PAYLOAD_BYTES_B64_MAX_BYTES: max file size to include as b64 (default 10MB)

  Upgrade order:
    1. Deploy new workers (support both file_ref and file_bytes_b64)
    2. Optionally enable DEDUP2D_JOB_PAYLOAD_INCLUDE_BYTES_B64=1 during grayscale
    3. Once all workers are new, disable the flag (file_ref-only mode)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from arq import create_pool  # type: ignore
    from arq.connections import ArqRedis, RedisSettings  # type: ignore
    from arq.jobs import Job  # type: ignore
except Exception:  # pragma: no cover - optional at runtime
    create_pool = None  # type: ignore
    ArqRedis = Any  # type: ignore
    RedisSettings = None  # type: ignore
    Job = None  # type: ignore

from src.core.dedup2d_file_storage import Dedup2DFileRef, create_dedup2d_file_storage
from src.core.dedup2d_metrics import dedup2d_payload_format
from src.core.dedup2d_webhook import validate_dedup2d_callback_url
from src.core.dedupcad_2d_jobs import (
    Dedup2DJob,
    Dedup2DJobStatus,
    JobForbiddenError,
    JobNotFoundError,
    JobQueueFullError,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dedup2DRedisJobConfig:
    redis_url: str
    key_prefix: str
    queue_name: str
    ttl_seconds: int
    max_jobs: int
    job_timeout_seconds: int
    render_queue_name: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Dedup2DRedisJobConfig":
        key_prefix = os.getenv("DEDUP2D_REDIS_KEY_PREFIX", "dedup2d").strip() or "dedup2d"
        queue_name = os.getenv("DEDUP2D_ARQ_QUEUE_NAME")
        if queue_name is None or not queue_name.strip():
            queue_name = f"{key_prefix}:queue"
        render_queue_raw = os.getenv("DEDUP2D_RENDER_QUEUE_NAME", "").strip()
        render_queue_name = render_queue_raw or None
        redis_url = (
            os.getenv("DEDUP2D_REDIS_URL") or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
        )
        ttl_seconds = int(os.getenv("DEDUP2D_ASYNC_TTL_SECONDS", "3600"))
        max_jobs = int(os.getenv("DEDUP2D_ASYNC_MAX_JOBS", "200"))
        job_timeout_seconds = int(os.getenv("DEDUP2D_ASYNC_JOB_TIMEOUT_SECONDS", "300"))
        return cls(
            redis_url=redis_url,
            key_prefix=key_prefix,
            queue_name=queue_name,
            render_queue_name=render_queue_name,
            ttl_seconds=max(60, ttl_seconds),
            max_jobs=max(1, max_jobs),
            job_timeout_seconds=max(1, job_timeout_seconds),
        )


@dataclass(frozen=True)
class Dedup2DPayloadConfig:
    """Configuration for job payload format (Phase 4 rolling upgrade)."""

    include_bytes_b64: bool
    bytes_b64_max_bytes: int

    @classmethod
    def from_env(cls) -> "Dedup2DPayloadConfig":
        include_bytes_b64_raw = (
            os.getenv("DEDUP2D_JOB_PAYLOAD_INCLUDE_BYTES_B64", "").strip().lower()
        )
        include_bytes_b64 = include_bytes_b64_raw in {"1", "true", "yes", "on"}

        # Default 10MB max for embedded base64 (avoid Redis memory pressure)
        bytes_b64_max_bytes = int(
            os.getenv("DEDUP2D_JOB_PAYLOAD_BYTES_B64_MAX_BYTES", str(10 * 1024 * 1024))
        )
        return cls(
            include_bytes_b64=include_bytes_b64,
            bytes_b64_max_bytes=max(0, bytes_b64_max_bytes),
        )


def _encode_bytes_b64(data: bytes) -> str:
    """Encode bytes to base64 string."""
    return base64.b64encode(data).decode("ascii")


def _payload_format_label(payload: Dict[str, Any]) -> str:
    file_ref = payload.get("file_ref")
    has_ref = isinstance(file_ref, dict) and bool(file_ref)
    b64 = payload.get("file_bytes_b64")
    has_b64 = isinstance(b64, str) and b64 != ""
    if has_ref and has_b64:
        return "file_ref_and_b64"
    if has_ref:
        return "file_ref_only"
    if has_b64:
        return "b64_only"
    return "unknown"


def _record_payload_format(payload: Dict[str, Any]) -> str:
    label = _payload_format_label(payload)
    try:
        dedup2d_payload_format.labels(format=label).inc()
    except Exception:
        pass
    return label


_CAD_EXTENSIONS = {".dxf", ".dwg"}
_CAD_MIME_HINTS = {"application/dxf", "application/x-dxf", "image/vnd.dwg", "application/acad"}


def is_cad_file(file_name: str, content_type: str) -> bool:
    ext = Path(str(file_name or "")).suffix.lower()
    if ext in _CAD_EXTENSIONS:
        return True
    content_type = str(content_type or "").lower()
    if content_type in _CAD_MIME_HINTS:
        return True
    if "dxf" in content_type or "dwg" in content_type:
        return True
    return False


def _job_key(cfg: Dedup2DRedisJobConfig, job_id: str) -> str:
    return f"{cfg.key_prefix}:job:{job_id}"


def _payload_key(cfg: Dedup2DRedisJobConfig, job_id: str) -> str:
    return f"{cfg.key_prefix}:payload:{job_id}"


def _result_key(cfg: Dedup2DRedisJobConfig, job_id: str) -> str:
    return f"{cfg.key_prefix}:result:{job_id}"


def _active_set_key(cfg: Dedup2DRedisJobConfig) -> str:
    return f"{cfg.key_prefix}:active"


def _tenant_jobs_key(cfg: Dedup2DRedisJobConfig, tenant_id: str) -> str:
    tenant_id = str(tenant_id or "").strip()
    if not tenant_id:
        raise ValueError("tenant_id is empty")
    return f"{cfg.key_prefix}:tenant:{tenant_id}:jobs"


_POOL: Optional[ArqRedis] = None
_POOL_LOCK = asyncio.Lock()


async def get_dedup2d_redis_pool(cfg: Optional[Dedup2DRedisJobConfig] = None) -> ArqRedis:
    if create_pool is None or RedisSettings is None:
        raise RuntimeError("arq is required for DEDUP2D_ASYNC_BACKEND=redis")
    global _POOL
    if _POOL is not None:
        return _POOL
    async with _POOL_LOCK:
        if _POOL is not None:
            return _POOL
        effective_cfg = cfg or Dedup2DRedisJobConfig.from_env()
        pool = await create_pool(RedisSettings.from_dsn(effective_cfg.redis_url))
        _POOL = pool
        return pool


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _hgetall_str(data: Dict[Any, Any]) -> Dict[str, str]:
    return {_to_str(k): _to_str(v) for k, v in data.items()}


def _status_is_finished(status: str) -> bool:
    value = (status or "").strip().lower()
    return value in {
        Dedup2DJobStatus.COMPLETED.value,
        Dedup2DJobStatus.FAILED.value,
        Dedup2DJobStatus.CANCELED.value,
    }


async def _prune_active_set(pool: ArqRedis, cfg: Dedup2DRedisJobConfig) -> int:
    """Best-effort cleanup for leaked active set entries.

    Active set members can leak if:
      - a worker crashes after submit but before writing final state, or
      - TTL expires on the job hash while the set member remains.
    """
    active_key = _active_set_key(cfg)
    members = await pool.smembers(active_key)
    if not members:
        return 0

    removed = 0
    for raw_job_id in members:
        job_id = _to_str(raw_job_id).strip()
        if not job_id:
            continue
        job_key = _job_key(cfg, job_id)
        try:
            raw_status = await pool.hget(job_key, "status")
        except Exception:
            raw_status = None
        if not raw_status:
            # Job hash is missing/expired.
            await pool.srem(active_key, job_id)
            removed += 1
            continue
        if _status_is_finished(_to_str(raw_status)):
            await pool.srem(active_key, job_id)
            removed += 1

    return removed


async def submit_dedup2d_job(
    *,
    tenant_id: str,
    file_name: str,
    file_bytes: bytes,
    content_type: str,
    query_geom: Optional[Dict[str, Any]],
    request_params: Dict[str, Any],
    callback_url: Optional[str] = None,
    cfg: Optional[Dedup2DRedisJobConfig] = None,
) -> Dedup2DJob:
    """Create a Redis-backed job and enqueue it to ARQ.

    Stores:
      - Job meta: `{prefix}:job:{job_id}`
      - Payload: `{prefix}:payload:{job_id}`
      - Active set: `{prefix}:active`
    """
    effective_cfg = cfg or Dedup2DRedisJobConfig.from_env()
    pool = await get_dedup2d_redis_pool(effective_cfg)

    tenant_id = str(tenant_id or "").strip()
    if not tenant_id:
        raise ValueError("tenant_id is empty")

    normalized_callback_url = None
    if callback_url is not None:
        normalized_callback_url = validate_dedup2d_callback_url(callback_url)

    active_key = _active_set_key(effective_cfg)
    try:
        await _prune_active_set(pool, effective_cfg)
    except Exception:
        logger.debug("dedup2d_prune_active_set_failed", exc_info=True)
    active = int(await pool.scard(active_key))
    if active >= effective_cfg.max_jobs:
        raise JobQueueFullError(effective_cfg.max_jobs, active)

    created_at = time.time()
    job_id = str(uuid.uuid4())
    job_key = _job_key(effective_cfg, job_id)
    payload_key = _payload_key(effective_cfg, job_id)
    tenant_jobs_key = _tenant_jobs_key(effective_cfg, tenant_id)

    storage = create_dedup2d_file_storage()
    file_ref = await storage.save_bytes(
        job_id=job_id,
        file_name=file_name or "unknown",
        content_type=content_type or "application/octet-stream",
        data=file_bytes,
    )

    render_required = is_cad_file(file_name, content_type)
    payload: Dict[str, Any] = {
        "tenant_id": tenant_id,
        "file_name": file_name or "unknown",
        "content_type": content_type or "application/octet-stream",
        "file_ref": file_ref.to_dict(),
        "query_geom": query_geom,
        "request_params": dict(request_params),
        "callback_url": normalized_callback_url,
        "render_required": render_required,
    }

    # Phase 4: Optionally include file_bytes_b64 for backward compatibility during rolling upgrade
    payload_cfg = Dedup2DPayloadConfig.from_env()
    if payload_cfg.include_bytes_b64 and len(file_bytes) <= payload_cfg.bytes_b64_max_bytes:
        payload["file_bytes_b64"] = _encode_bytes_b64(file_bytes)
        logger.debug(
            "dedup2d_payload_include_bytes_b64",
            extra={"job_id": job_id, "size": len(file_bytes)},
        )

    try:
        await pool.hset(
            job_key,
            mapping={
                "job_id": job_id,
                "tenant_id": tenant_id,
                "status": Dedup2DJobStatus.PENDING.value,
                "created_at": str(created_at),
                "started_at": "",
                "finished_at": "",
                "error": "",
                "cancel_requested": "0",
                "callback_url": normalized_callback_url or "",
                "callback_status": "pending" if normalized_callback_url else "",
                "callback_attempts": "0",
                "callback_http_status": "",
                "callback_finished_at": "",
                "callback_last_error": "",
            },
        )
        await pool.expire(job_key, effective_cfg.ttl_seconds)

        await pool.setex(payload_key, effective_cfg.ttl_seconds, json.dumps(payload))
        await pool.sadd(active_key, job_id)
        # Keep a per-tenant index so the API can list recently finished jobs within TTL.
        await pool.zadd(tenant_jobs_key, {job_id: float(created_at)})
        await pool.expire(tenant_jobs_key, effective_cfg.ttl_seconds)

        queue_name = effective_cfg.queue_name
        if render_required and effective_cfg.render_queue_name:
            queue_name = effective_cfg.render_queue_name

        job = await pool.enqueue_job(
            "dedup2d_run_job",
            job_id,
            _job_id=job_id,
            _queue_name=queue_name,
            _expires=effective_cfg.ttl_seconds,
        )
        if job is None:
            raise RuntimeError("ARQ returned None (job already exists)")
        _record_payload_format(payload)
        return Dedup2DJob(job_id=job_id, tenant_id=tenant_id, status=Dedup2DJobStatus.PENDING)
    except Exception:
        # Best-effort cleanup so a failed enqueue doesn't leak capacity.
        try:
            await pool.delete(job_key)
            await pool.delete(payload_key)
            await pool.srem(active_key, job_id)
            await pool.zrem(tenant_jobs_key, job_id)
        except Exception:
            logger.debug("dedup2d_redis_enqueue_cleanup_failed", exc_info=True)
        try:
            await storage.delete(file_ref)
        except Exception:
            logger.debug(
                "dedup2d_file_cleanup_failed",
                extra={"job_id": job_id, "file_ref": getattr(file_ref, "to_dict", lambda: {})()},
                exc_info=True,
            )
        raise


async def get_dedup2d_job_for_tenant(
    job_id: str,
    tenant_id: str,
    *,
    cfg: Optional[Dedup2DRedisJobConfig] = None,
) -> Dedup2DJob:
    effective_cfg = cfg or Dedup2DRedisJobConfig.from_env()
    pool = await get_dedup2d_redis_pool(effective_cfg)
    job_key = _job_key(effective_cfg, job_id)
    raw_data = await pool.hgetall(job_key)
    if not raw_data:
        raise JobNotFoundError(job_id)
    data = _hgetall_str(raw_data)

    stored_tenant_id = data.get("tenant_id") or ""
    if stored_tenant_id != tenant_id:
        raise JobForbiddenError(job_id, tenant_id)

    status_raw = data.get("status") or Dedup2DJobStatus.PENDING.value
    try:
        status = Dedup2DJobStatus(status_raw)
    except Exception:
        status = Dedup2DJobStatus.PENDING

    def _maybe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    job = Dedup2DJob(
        job_id=data.get("job_id") or job_id,
        tenant_id=stored_tenant_id,
        status=status,
        created_at=_maybe_float(data.get("created_at")) or time.time(),
        started_at=_maybe_float(data.get("started_at")),
        finished_at=_maybe_float(data.get("finished_at")),
        error=(data.get("error") or "") or None,
        meta={},
    )

    job.meta = {
        "cancel_requested": str(data.get("cancel_requested") or "0"),
        "metrics_recorded": str(data.get("metrics_recorded") or ""),
        "callback_status": str(data.get("callback_status") or ""),
        "callback_attempts": str(data.get("callback_attempts") or ""),
        "callback_http_status": str(data.get("callback_http_status") or ""),
        "callback_finished_at": str(data.get("callback_finished_at") or ""),
        "callback_last_error": str(data.get("callback_last_error") or ""),
    }

    if job.is_finished():
        result_raw = await pool.get(_result_key(effective_cfg, job_id))
        if result_raw:
            try:
                job.result = json.loads(result_raw)
            except Exception:
                job.result = None

    return job


async def get_dedup2d_queue_depth(*, cfg: Optional[Dedup2DRedisJobConfig] = None) -> int:
    effective_cfg = cfg or Dedup2DRedisJobConfig.from_env()
    pool = await get_dedup2d_redis_pool(effective_cfg)
    try:
        await _prune_active_set(pool, effective_cfg)
    except Exception:
        logger.debug("dedup2d_prune_active_set_failed", exc_info=True)
    return int(await pool.scard(_active_set_key(effective_cfg)))


async def cancel_dedup2d_job_for_tenant(
    job_id: str,
    tenant_id: str,
    *,
    cfg: Optional[Dedup2DRedisJobConfig] = None,
) -> bool:
    effective_cfg = cfg or Dedup2DRedisJobConfig.from_env()
    pool = await get_dedup2d_redis_pool(effective_cfg)
    job_key = _job_key(effective_cfg, job_id)
    raw_data = await pool.hgetall(job_key)
    if not raw_data:
        raise JobNotFoundError(job_id)
    data = _hgetall_str(raw_data)
    stored_tenant_id = data.get("tenant_id") or ""
    if stored_tenant_id != tenant_id:
        raise JobForbiddenError(job_id, tenant_id)

    status_raw = (data.get("status") or "").strip()
    if _status_is_finished(status_raw):
        return True

    started_at = (data.get("started_at") or "").strip()
    finished_at = str(time.time())
    mapping: Dict[str, str] = {"cancel_requested": "1"}
    # If the job hasn't started yet, we can mark it canceled immediately to
    # avoid "stuck pending" when ARQ abort prevents the worker from running.
    if status_raw == Dedup2DJobStatus.PENDING.value and not started_at:
        mapping.update(
            {
                "status": Dedup2DJobStatus.CANCELED.value,
                "finished_at": finished_at,
                "error": "",
            }
        )
        if str(data.get("callback_url") or "").strip():
            mapping.update(
                {
                    "callback_status": "skipped",
                    "callback_finished_at": finished_at,
                    "callback_last_error": "canceled_before_start",
                }
            )

    await pool.hset(job_key, mapping=mapping)
    try:
        await pool.expire(job_key, effective_cfg.ttl_seconds)
    except Exception:
        pass

    if Job is not None:
        try:
            job = Job(job_id, pool, _queue_name=effective_cfg.queue_name)
            await job.abort()
        except Exception:
            logger.debug("dedup2d_job_abort_failed", extra={"job_id": job_id}, exc_info=True)

    if mapping.get("status") == Dedup2DJobStatus.CANCELED.value:
        try:
            await pool.srem(_active_set_key(effective_cfg), job_id)
        except Exception:
            logger.debug("dedup2d_active_set_remove_failed", exc_info=True)
        try:
            payload_raw = await pool.get(_payload_key(effective_cfg, job_id))
            if payload_raw:
                payload = json.loads(payload_raw)
                file_ref_raw = payload.get("file_ref")
                if isinstance(file_ref_raw, dict):
                    file_ref = Dedup2DFileRef.from_dict(file_ref_raw)
                    storage = create_dedup2d_file_storage()
                    if storage.config.cleanup_on_finish:
                        await storage.delete(file_ref)
        except Exception:
            logger.debug(
                "dedup2d_cancel_file_cleanup_failed",
                extra={"job_id": job_id},
                exc_info=True,
            )

    return True


async def mark_dedup2d_job_result(
    job_id: str,
    *,
    status: Dedup2DJobStatus,
    started_at: Optional[float],
    finished_at: Optional[float],
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    cfg: Optional[Dedup2DRedisJobConfig] = None,
    redis: Optional[ArqRedis] = None,
) -> None:
    """Worker helper: persist final status and result (best-effort)."""
    effective_cfg = cfg or Dedup2DRedisJobConfig.from_env()
    pool = redis or await get_dedup2d_redis_pool(effective_cfg)

    job_key = _job_key(effective_cfg, job_id)
    mapping: Dict[str, str] = {
        "status": status.value,
        "started_at": str(started_at or ""),
        "finished_at": str(finished_at or ""),
        "error": str(error or ""),
    }
    await pool.hset(job_key, mapping=mapping)
    await pool.expire(job_key, effective_cfg.ttl_seconds)

    active_key = _active_set_key(effective_cfg)
    try:
        await pool.srem(active_key, job_id)
    except Exception:
        pass

    if result is not None:
        result_key = _result_key(effective_cfg, job_id)
        await pool.setex(result_key, effective_cfg.ttl_seconds, json.dumps(result))


async def list_dedup2d_jobs_for_tenant(
    tenant_id: str,
    *,
    status: Optional[Dedup2DJobStatus] = None,
    limit: int = 100,
    cfg: Optional[Dedup2DRedisJobConfig] = None,
) -> list[Dedup2DJob]:
    """List recent jobs for a tenant (including finished ones within TTL).

    Args:
        tenant_id: Tenant identifier
        status: Optional status filter
        limit: Maximum number of jobs to return

    Returns:
        List of jobs belonging to the tenant, sorted by created_at descending.

    Notes:
        - Uses a per-tenant Redis ZSET (created_at as score) to include recently finished jobs
          within the same TTL window as the job hash.
        - Missing/expired job hashes are pruned lazily.
    """
    effective_cfg = cfg or Dedup2DRedisJobConfig.from_env()
    pool = await get_dedup2d_redis_pool(effective_cfg)

    jobs: list[Dedup2DJob] = []
    tenant_jobs_key = _tenant_jobs_key(effective_cfg, tenant_id)

    def _maybe_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        s = str(value).strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None

    # We may need to scan more than `limit` when applying status filters.
    # Use a small batch loop to stay efficient while ensuring enough matches.
    batch_size = max(50, int(limit) * 3)
    start = 0
    while len(jobs) < limit:
        ids = await pool.zrevrange(tenant_jobs_key, start, start + batch_size - 1)
        if not ids:
            break
        start += batch_size

        for raw_job_id in ids:
            job_id = _to_str(raw_job_id).strip()
            if not job_id:
                continue

            job_key = _job_key(effective_cfg, job_id)
            raw_data = await pool.hgetall(job_key)
            if not raw_data:
                # Job metadata expired; prune index entry.
                try:
                    await pool.zrem(tenant_jobs_key, job_id)
                except Exception:
                    pass
                continue

            data = _hgetall_str(raw_data)
            stored_tenant_id = data.get("tenant_id") or ""
            if stored_tenant_id != tenant_id:
                # Should not happen, but keep index clean.
                try:
                    await pool.zrem(tenant_jobs_key, job_id)
                except Exception:
                    pass
                continue

            status_raw = data.get("status") or Dedup2DJobStatus.PENDING.value
            try:
                job_status = Dedup2DJobStatus(status_raw)
            except Exception:
                job_status = Dedup2DJobStatus.PENDING

            if status is not None and job_status != status:
                continue

            job = Dedup2DJob(
                job_id=data.get("job_id") or job_id,
                tenant_id=stored_tenant_id,
                status=job_status,
                created_at=_maybe_float(data.get("created_at")) or time.time(),
                started_at=_maybe_float(data.get("started_at")),
                finished_at=_maybe_float(data.get("finished_at")),
                error=(data.get("error") or "") or None,
                meta={
                    "cancel_requested": str(data.get("cancel_requested") or "0"),
                    "callback_status": str(data.get("callback_status") or ""),
                    "callback_attempts": str(data.get("callback_attempts") or ""),
                },
            )
            jobs.append(job)
            if len(jobs) >= limit:
                break

    return jobs
