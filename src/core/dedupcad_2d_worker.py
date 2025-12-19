"""ARQ worker for Redis-backed 2D dedup jobs (Phase 2).

Run:
  `arq src.core.dedupcad_2d_worker.WorkerSettings`

Rolling Upgrade Compatibility (Phase 4):
  This worker supports both old and new payload formats:
  - New format: file_ref (Dedup2DFileRef pointing to local/S3 storage)
  - Old format: file_bytes_b64 (base64-encoded file bytes in Redis payload)

  Upgrade order:
  1. Deploy new worker first (supports both formats)
  2. Once all workers are upgraded, switch API to file_ref-only mode
  3. Old jobs with file_bytes_b64 will continue to work until TTL expires
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import time
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

from arq.connections import RedisSettings

from src.core.dedupcad_2d_jobs import Dedup2DJobStatus
from src.core.dedupcad_2d_jobs_redis import (
    Dedup2DRedisJobConfig,
    _payload_key,
    get_dedup2d_redis_pool,
    mark_dedup2d_job_result,
)
from src.core.dedupcad_2d_pipeline import run_dedup_2d_pipeline
from src.core.dedup2d_file_storage import Dedup2DFileRef
from src.core.dedupcad_precision import PrecisionVerifier, create_geom_store
from src.core.dedupcad_vision import DedupCadVisionClient
from src.core.dedup2d_webhook import send_dedup2d_webhook

logger = logging.getLogger(__name__)


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decode_bytes_b64(data_b64: str) -> bytes:
    """Decode base64-encoded file bytes (legacy payload format)."""
    return base64.b64decode(data_b64.encode("ascii"))


async def _load_file_bytes_from_payload(
    payload: Dict[str, Any],
) -> Tuple[bytes, Optional[Dedup2DFileRef], Optional["Dedup2DFileStorageProtocol"]]:
    """Load file bytes from payload, supporting both new and legacy formats.

    Returns:
        (file_bytes, file_ref, storage)
        - file_ref and storage are None when using legacy file_bytes_b64 format

    Raises:
        ValueError: If neither file_ref nor file_bytes_b64 is present
    """
    from src.core.dedup2d_file_storage import (
        Dedup2DFileStorageProtocol,
        create_dedup2d_file_storage,
    )

    file_ref_raw = payload.get("file_ref")
    file_ref: Optional[Dedup2DFileRef] = None
    storage: Optional[Dedup2DFileStorageProtocol] = None

    if isinstance(file_ref_raw, dict):
        # New format: file_ref pointing to external storage
        file_ref = Dedup2DFileRef.from_dict(file_ref_raw)
        storage = create_dedup2d_file_storage()
        file_bytes = await storage.load_bytes(file_ref)
        logger.debug(
            "dedup2d_worker_loaded_file_ref",
            extra={"backend": file_ref.backend, "path": file_ref.path, "key": file_ref.key},
        )
        return file_bytes, file_ref, storage

    # Legacy format: file_bytes_b64 embedded in payload
    file_bytes_b64 = payload.get("file_bytes_b64")
    if file_bytes_b64 and isinstance(file_bytes_b64, str):
        file_bytes = _decode_bytes_b64(file_bytes_b64)
        logger.debug(
            "dedup2d_worker_loaded_legacy_b64",
            extra={"size": len(file_bytes)},
        )
        return file_bytes, None, None

    raise ValueError("Missing file_ref and file_bytes_b64 in job payload")


async def dedup2d_run_job(ctx: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Execute a 2D dedup job.

    The API server stores payload in Redis; the worker:
      1) marks job running,
      2) calls dedupcad-vision (+ optional local precision),
      3) writes result to Redis.
    """
    cfg = Dedup2DRedisJobConfig.from_env()
    redis = ctx.get("redis")
    if redis is None:
        redis = await get_dedup2d_redis_pool(cfg)

    started_at = time.time()
    tenant_id: Optional[str] = None
    callback_url: Optional[str] = None
    file_ref: Optional[Dedup2DFileRef] = None
    storage = None
    try:
        job_key = f"{cfg.key_prefix}:job:{job_id}"

        cancel_requested = _to_str(await redis.hget(job_key, "cancel_requested"))
        if (cancel_requested or "0") == "1":
            await mark_dedup2d_job_result(
                job_id,
                status=Dedup2DJobStatus.CANCELED,
                started_at=None,
                finished_at=time.time(),
                result=None,
                error=None,
                cfg=cfg,
                redis=redis,
            )
            return {"status": "canceled"}

        await redis.hset(
            job_key,
            mapping={
                "status": Dedup2DJobStatus.IN_PROGRESS.value,
                "started_at": str(started_at),
            },
        )
        await redis.expire(job_key, cfg.ttl_seconds)

        payload_raw = await redis.get(_payload_key(cfg, job_id))
        if not payload_raw:
            raise RuntimeError("Missing payload in Redis")

        payload = json.loads(payload_raw)
        tenant_id = str(payload.get("tenant_id") or "").strip() or None
        file_name = str(payload.get("file_name") or "unknown")
        content_type = str(payload.get("content_type") or "application/octet-stream")

        # Phase 4: Support both new (file_ref) and legacy (file_bytes_b64) formats
        file_bytes, file_ref, storage = await _load_file_bytes_from_payload(payload)

        query_geom = payload.get("query_geom")
        request_params = payload.get("request_params") or {}
        callback_url = str(payload.get("callback_url") or "").strip() or None
        if not isinstance(request_params, dict):
            raise ValueError("Invalid request_params")

        client = DedupCadVisionClient()
        geom_store = create_geom_store()
        precision_verifier = PrecisionVerifier()

        result = await run_dedup_2d_pipeline(
            client=client,
            geom_store=geom_store,
            precision_verifier=precision_verifier,
            file_name=file_name,
            file_bytes=file_bytes,
            content_type=content_type,
            query_geom=query_geom,
            mode=str(request_params.get("mode") or "balanced"),
            max_results=int(request_params.get("max_results") or 50),
            compute_diff=bool(request_params.get("compute_diff", True)),
            enable_ml=bool(request_params.get("enable_ml", False)),
            enable_geometric=bool(request_params.get("enable_geometric", False)),
            enable_precision=bool(request_params.get("enable_precision", True)),
            precision_profile=request_params.get("precision_profile"),
            version_gate=request_params.get("version_gate"),
            precision_top_n=int(request_params.get("precision_top_n") or 20),
            precision_visual_weight=float(request_params.get("precision_visual_weight") or 0.3),
            precision_geom_weight=float(request_params.get("precision_geom_weight") or 0.7),
            precision_compute_diff=bool(request_params.get("precision_compute_diff", False)),
            precision_diff_top_n=int(request_params.get("precision_diff_top_n") or 5),
            precision_diff_max_paths=int(request_params.get("precision_diff_max_paths") or 200),
            duplicate_threshold=float(request_params.get("duplicate_threshold") or 0.95),
            similar_threshold=float(request_params.get("similar_threshold") or 0.8),
        )

        cancel_requested = _to_str(await redis.hget(job_key, "cancel_requested"))
        if (cancel_requested or "0") == "1":
            await mark_dedup2d_job_result(
                job_id,
                status=Dedup2DJobStatus.CANCELED,
                started_at=started_at,
                finished_at=time.time(),
                result=None,
                error=None,
                cfg=cfg,
                redis=redis,
            )
            return {"status": "canceled"}

        finished_at = time.time()
        await mark_dedup2d_job_result(
            job_id,
            status=Dedup2DJobStatus.COMPLETED,
            started_at=started_at,
            finished_at=finished_at,
            result=result,
            error=None,
            cfg=cfg,
            redis=redis,
        )

        if callback_url:
            await _try_send_callback(
                redis=redis,
                job_key=job_key,
                cfg=cfg,
                callback_url=callback_url,
                job_id=job_id,
                tenant_id=tenant_id,
                status=Dedup2DJobStatus.COMPLETED,
                started_at=started_at,
                finished_at=finished_at,
                result=result,
                error=None,
            )
        if file_ref is not None and storage is not None and storage.config.cleanup_on_finish:
            try:
                await storage.delete(file_ref)
            except Exception:
                logger.debug("dedup2d_file_cleanup_failed", extra={"job_id": job_id}, exc_info=True)
        return {"status": "completed"}
    except asyncio.CancelledError:
        finished_at = time.time()
        await mark_dedup2d_job_result(
            job_id,
            status=Dedup2DJobStatus.CANCELED,
            started_at=started_at,
            finished_at=finished_at,
            result=None,
            error=None,
            cfg=cfg,
            redis=redis,
        )
        if callback_url:
            await _try_send_callback(
                redis=redis,
                job_key=f"{cfg.key_prefix}:job:{job_id}",
                cfg=cfg,
                callback_url=callback_url,
                job_id=job_id,
                tenant_id=tenant_id,
                status=Dedup2DJobStatus.CANCELED,
                started_at=started_at,
                finished_at=finished_at,
                result=None,
                error=None,
            )
        if file_ref is not None and storage is not None and storage.config.cleanup_on_finish:
            try:
                await storage.delete(file_ref)
            except Exception:
                logger.debug("dedup2d_file_cleanup_failed", extra={"job_id": job_id}, exc_info=True)
        raise
    except Exception as e:
        logger.exception("dedup2d_job_failed", extra={"job_id": job_id, "error": str(e)})
        finished_at = time.time()
        await mark_dedup2d_job_result(
            job_id,
            status=Dedup2DJobStatus.FAILED,
            started_at=started_at,
            finished_at=finished_at,
            result=None,
            error=str(e),
            cfg=cfg,
            redis=redis,
        )
        if callback_url:
            await _try_send_callback(
                redis=redis,
                job_key=f"{cfg.key_prefix}:job:{job_id}",
                cfg=cfg,
                callback_url=callback_url,
                job_id=job_id,
                tenant_id=tenant_id,
                status=Dedup2DJobStatus.FAILED,
                started_at=started_at,
                finished_at=finished_at,
                result=None,
                error=str(e),
            )
        if file_ref is not None and storage is not None and storage.config.cleanup_on_finish:
            try:
                await storage.delete(file_ref)
            except Exception:
                logger.debug("dedup2d_file_cleanup_failed", extra={"job_id": job_id}, exc_info=True)
        return {"status": "failed"}


async def _try_send_callback(
    *,
    redis: Any,
    job_key: str,
    cfg: Dedup2DRedisJobConfig,
    callback_url: str,
    job_id: str,
    tenant_id: Optional[str],
    status: Dedup2DJobStatus,
    started_at: Optional[float],
    finished_at: Optional[float],
    result: Optional[Dict[str, Any]],
    error: Optional[str],
) -> None:
    payload = {
        "job_id": job_id,
        "tenant_id": tenant_id,
        "status": status.value,
        "started_at": started_at,
        "finished_at": finished_at,
        "result": result,
        "error": error,
    }
    try:
        ok, attempts, http_status, err = await send_dedup2d_webhook(
            callback_url=callback_url,
            payload=payload,
            job_id=job_id,
            tenant_id=tenant_id,
        )
        await redis.hset(
            job_key,
            mapping={
                "callback_status": "success" if ok else "failed",
                "callback_attempts": str(attempts),
                "callback_http_status": str(http_status or ""),
                "callback_finished_at": str(time.time()),
                "callback_last_error": str(err or ""),
            },
        )
        await redis.expire(job_key, cfg.ttl_seconds)
    except Exception as e:
        logger.debug(
            "dedup2d_callback_failed",
            extra={"job_id": job_id, "callback_url": callback_url, "error": str(e)},
            exc_info=True,
        )


class WorkerSettings:
    functions = [dedup2d_run_job]
    redis_settings = RedisSettings.from_dsn(
        os.getenv("DEDUP2D_REDIS_URL")
        or os.getenv("REDIS_URL")
        or "redis://localhost:6379/0"
    )
    queue_name = os.getenv("DEDUP2D_ARQ_QUEUE_NAME") or "dedup2d:queue"
    max_jobs = int(os.getenv("DEDUP2D_WORKER_MAX_JOBS", "10"))
    job_timeout = timedelta(seconds=int(os.getenv("DEDUP2D_ASYNC_JOB_TIMEOUT_SECONDS", "300")))
