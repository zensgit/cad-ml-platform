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
import tempfile
import time
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

try:
    from arq.connections import RedisSettings  # type: ignore
except Exception:  # pragma: no cover - optional dependency

    class RedisSettings:  # type: ignore
        @classmethod
        def from_dsn(cls, *args: Any, **kwargs: Any) -> None:
            return None

import src.core.dedup2d_file_storage as dedup2d_file_storage
from src.core.dedup2d_file_storage import Dedup2DFileRef
from src.core.dedup2d_webhook import send_dedup2d_webhook
from src.core.dedupcad_2d_jobs import Dedup2DJobStatus
from src.core.dedupcad_2d_jobs_redis import (
    Dedup2DRedisJobConfig,
    _payload_key,
    get_dedup2d_redis_pool,
    is_cad_file,
    mark_dedup2d_job_result,
)
from src.core.dedupcad_2d_pipeline import run_dedup_2d_pipeline
from src.core.dedupcad_precision import PrecisionVerifier, create_geom_store
from src.core.dedupcad_precision.cad_pipeline import (
    DxfRenderConfig,
    convert_dwg_to_dxf,
    render_dxf_to_png,
)
from src.core.dedupcad_vision import DedupCadVisionClient
from src.core.dedup2d_metrics import dedup2d_legacy_b64_fallback_total

if TYPE_CHECKING:
    from src.core.dedup2d_file_storage import Dedup2DFileStorageProtocol

logger = logging.getLogger(__name__)


def _env_flag(name: str, *, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


@contextmanager
def _temporary_env(updates: Dict[str, str]) -> Any:
    original: Dict[str, Optional[str]] = {}
    try:
        for key, value in updates.items():
            original[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, previous in original.items():
            if previous is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous


def _to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _decode_bytes_b64(data_b64: str) -> bytes:
    """Decode base64-encoded file bytes (legacy payload format)."""
    return base64.b64decode(data_b64.encode("ascii"))


def _render_config_from_env() -> DxfRenderConfig:
    size_px = int(os.getenv("DEDUPCAD2_RENDER_SIZE_PX", "1024") or "1024")
    dpi = int(os.getenv("DEDUPCAD2_RENDER_DPI", "200") or "200")
    margin_ratio = float(os.getenv("DEDUPCAD2_RENDER_MARGIN_RATIO", "0.05") or "0.05")
    return DxfRenderConfig(size_px=size_px, dpi=dpi, margin_ratio=margin_ratio)


def _render_fallback_config_from_env() -> DxfRenderConfig:
    size_px = int(os.getenv("DEDUPCAD2_RENDER_FALLBACK_SIZE_PX", "512") or "512")
    dpi = int(os.getenv("DEDUPCAD2_RENDER_FALLBACK_DPI", "100") or "100")
    margin_ratio = float(os.getenv("DEDUPCAD2_RENDER_MARGIN_RATIO", "0.05") or "0.05")
    return DxfRenderConfig(size_px=size_px, dpi=dpi, margin_ratio=margin_ratio)


def _resolve_cad_suffix(file_name: str, content_type: str) -> str:
    suffix = Path(str(file_name or "")).suffix.lower()
    if suffix in {".dxf", ".dwg"}:
        return suffix
    content_type = str(content_type or "").lower()
    if "dwg" in content_type:
        return ".dwg"
    if "dxf" in content_type:
        return ".dxf"
    return suffix


def _convert_dwg_to_dxf(in_path: Path, out_path: Path) -> None:
    convert_dwg_to_dxf(in_path, out_path)


def _render_cad_to_png(
    *,
    file_name: str,
    file_bytes: bytes,
    content_type: str,
) -> tuple[str, bytes, str]:
    suffix = _resolve_cad_suffix(file_name, content_type)
    if suffix not in {".dxf", ".dwg"}:
        raise RuntimeError(f"Unsupported CAD format for rendering: {suffix or 'unknown'}")

    render_cfg = _render_config_from_env()
    with tempfile.TemporaryDirectory(prefix="dedup2d_render_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        input_path = tmp_root / f"input{suffix}"
        input_path.write_bytes(file_bytes)

        dxf_path = input_path
        if suffix == ".dwg":
            dxf_path = tmp_root / "input.dxf"
            _convert_dwg_to_dxf(input_path, dxf_path)

        output_path = tmp_root / "render.png"
        try:
            render_dxf_to_png(dxf_path, output_path, config=render_cfg)
        except Exception as exc:
            if not _env_flag("DEDUPCAD2_RENDER_FALLBACK", default=True):
                raise
            fallback_cfg = _render_fallback_config_from_env()
            with _temporary_env(
                {
                    "DEDUPCAD2_RENDER_TEXT": "0",
                    "DEDUPCAD2_RENDER_HATCH": "0",
                }
            ):
                render_dxf_to_png(dxf_path, output_path, config=fallback_cfg)
            logger.warning(
                "dedup2d_render_fallback_used",
                extra={
                    "error": str(exc),
                    "fallback_size_px": fallback_cfg.size_px,
                    "fallback_dpi": fallback_cfg.dpi,
                },
            )
        png_bytes = output_path.read_bytes()

    stem = Path(str(file_name or "drawing")).stem or "drawing"
    return f"{stem}.png", png_bytes, "image/png"


async def _load_file_bytes_from_payload(
    payload: Dict[str, Any],
) -> Tuple[bytes, Optional[Dedup2DFileRef], Optional[Dedup2DFileStorageProtocol]]:
    """Load file bytes from payload, supporting both new and legacy formats.

    Returns:
        (file_bytes, file_ref, storage)
        - file_ref and storage are None when using legacy file_bytes_b64 format

    Raises:
        ValueError: If neither file_ref nor file_bytes_b64 is present
    """
    file_ref_raw = payload.get("file_ref")
    file_ref: Optional[Dedup2DFileRef] = None
    storage: Optional[Dedup2DFileStorageProtocol] = None

    if isinstance(file_ref_raw, dict):
        # New format: file_ref pointing to external storage
        file_ref = Dedup2DFileRef.from_dict(file_ref_raw)
        storage = dedup2d_file_storage.create_dedup2d_file_storage()
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
        try:
            dedup2d_legacy_b64_fallback_total.inc()
        except Exception:
            pass
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

        render_required = bool(payload.get("render_required"))
        if render_required or is_cad_file(file_name, content_type):
            render_start = time.time()
            file_name, file_bytes, content_type = _render_cad_to_png(
                file_name=file_name,
                file_bytes=file_bytes,
                content_type=content_type,
            )
            logger.info(
                "dedup2d_render_completed",
                extra={
                    "job_id": job_id,
                    "duration_seconds": round(time.time() - render_start, 3),
                    "output_bytes": len(file_bytes),
                },
            )

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
        os.getenv("DEDUP2D_REDIS_URL") or os.getenv("REDIS_URL") or "redis://localhost:6379/0"
    )
    queue_name = os.getenv("DEDUP2D_ARQ_QUEUE_NAME") or "dedup2d:queue"
    max_jobs = int(os.getenv("DEDUP2D_WORKER_MAX_JOBS", "10"))
    job_timeout = timedelta(seconds=int(os.getenv("DEDUP2D_ASYNC_JOB_TIMEOUT_SECONDS", "300")))

    async def on_startup(ctx: dict[str, Any]) -> None:
        if not _env_flag("DEDUPCAD2_RENDER_PREWARM_FONTS", default=False):
            return
        start = time.time()
        try:
            from ezdxf.fonts import fonts

            fonts.build_system_font_cache()
            logger.info(
                "dedup2d_font_cache_warmed",
                extra={"duration_seconds": round(time.time() - start, 3)},
            )
        except Exception as e:
            logger.warning(
                "dedup2d_font_cache_warm_failed",
                extra={"error": str(e)},
            )
