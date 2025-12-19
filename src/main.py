"""
CAD ML Platform - 主服务入口
智能CAD分析微服务平台
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware

try:
    from prometheus_client import make_asgi_app  # type: ignore

    _metrics_enabled = True
except Exception:  # module missing
    _metrics_enabled = False
import uvicorn

from src.api import api_router
from src.core.config import get_settings
from src.models.loader import load_models
from src.utils.cache import init_redis
from src.utils.logging import setup_logging
from src.utils.metrics import get_ocr_error_rate_ema, get_vision_error_rate_ema
from src.api.health_resilience import get_resilience_health
from src.core.similarity import background_prune_task
from src.core.similarity import FaissVectorStore, load_recovery_state  # type: ignore
from src.tasks.orphan_scan import orphan_scan_loop

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

# 加载配置
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("Starting CAD ML Platform...")

    # Phase 1: Register dedup2d job metrics callback
    from src.api.v1.dedup import register_dedup2d_job_metrics
    register_dedup2d_job_metrics()

    # 初始化Redis连接
    if settings.REDIS_ENABLED:
        await init_redis()
        logger.info("Redis initialized")

    # 加载ML模型
    await load_models()
    logger.info("ML models loaded")

    # 启动向量 TTL 定期清理任务
    import asyncio
    prune_interval = float(__import__("os").getenv("VECTOR_PRUNE_INTERVAL_SECONDS", "30"))
    _prune_handle = asyncio.create_task(background_prune_task(prune_interval))

    # Faiss periodic export task
    faiss_path = __import__("os").getenv("FAISS_INDEX_PATH", "data/faiss_index.bin")
    faiss_interval = float(__import__("os").getenv("FAISS_EXPORT_INTERVAL_SECONDS", "300"))
    # Auto import existing Faiss index at startup (best effort)
    try:
        # Load persisted recovery state first (next attempt timestamps etc.)
        load_recovery_state()
        if __import__("os").getenv("VECTOR_STORE_BACKEND", "memory") == "faiss":
            store = FaissVectorStore()
            imported = store.import_index(faiss_path)
            if imported:
                from src.core.similarity import _FAISS_INDEX  # type: ignore
                dim = getattr(_FAISS_INDEX, 'd', None) if _FAISS_INDEX is not None else None  # type: ignore
                size = getattr(_FAISS_INDEX, 'ntotal', None) if _FAISS_INDEX is not None else None  # type: ignore
                env_dim = int(__import__('os').getenv('FEATURE_VECTOR_EXPECTED_DIM', '0') or 0)
                if env_dim and dim and dim != env_dim:
                    from src.utils.analysis_metrics import faiss_index_dim_mismatch_total
                    faiss_index_dim_mismatch_total.inc()
                    logger.warning("Faiss index dim mismatch (imported=%s, expected=%s) - falling back to memory store", dim, env_dim)
                else:
                    logger.info("Faiss index auto-imported from %s (dim=%s, size=%s)", faiss_path, dim, size)
    except Exception:
        logger.warning("Faiss index auto-import failed")
    async def _faiss_export_loop():
        while True:
            try:
                if __import__("os").getenv("VECTOR_STORE_BACKEND", "memory") == "faiss":
                    store = FaissVectorStore()
                    store.export(faiss_path)
                    # update index age gauge (reset to 0 after export)
                    try:
                        from src.utils.analysis_metrics import faiss_index_age_seconds
                        faiss_index_age_seconds.set(0)
                    except Exception:
                        pass
            except Exception:
                pass
            await asyncio.sleep(faiss_interval)
    _faiss_export_handle = asyncio.create_task(_faiss_export_loop())

    # Faiss age increment loop (ticks every 60s) to show staleness when exports disabled or failing
    async def _faiss_age_loop():
        import asyncio, os
        while True:
            try:
                if os.getenv("VECTOR_STORE_BACKEND", "memory") == "faiss":
                    from src.utils.analysis_metrics import faiss_index_age_seconds
                    # increment age by 60s up to a cap for visibility; if export loop resets to 0 this shows freshness
                    current = getattr(faiss_index_age_seconds, "_value", 0) or 0
                    faiss_index_age_seconds.set(current + 60)
            except Exception:
                pass
            await asyncio.sleep(60)
    _faiss_age_handle = __import__("asyncio").create_task(_faiss_age_loop())

    # Faiss auto recovery loop
    try:
        from src.core.similarity import faiss_recovery_loop
        _faiss_recovery_handle = asyncio.create_task(faiss_recovery_loop())
    except Exception:
        _faiss_recovery_handle = None

    # Load drift baselines from Redis if present
    try:
        from src.utils.cache import get_client
        client = get_client()
        if client is not None:
            import json
            from src.api.v1 import analyze as _an
            raw_m = await client.get("baseline:material")  # type: ignore[attr-defined]
            raw_c = await client.get("baseline:class")  # type: ignore[attr-defined]
            ts_m = await client.get("baseline:material:ts")  # type: ignore[attr-defined]
            ts_c = await client.get("baseline:class:ts")  # type: ignore[attr-defined]
            if raw_m:
                _an._DRIFT_STATE["baseline_materials"] = json.loads(raw_m)
            if ts_m:
                try:
                    _an._DRIFT_STATE["baseline_materials_ts"] = int(ts_m)
                except Exception:
                    pass
            if raw_c:
                _an._DRIFT_STATE["baseline_predictions"] = json.loads(raw_c)
            if ts_c:
                try:
                    _an._DRIFT_STATE["baseline_predictions_ts"] = int(ts_c)
                except Exception:
                    pass
            if raw_m or raw_c:
                logger.info("Drift baselines loaded from Redis")
    except Exception:
        pass

    # Orphan scan task
    orphan_handle = asyncio.create_task(orphan_scan_loop())

    yield

    # 关闭时
    logger.info("Shutting down CAD ML Platform...")
    # 清理资源
    try:
        _prune_handle.cancel()
    except Exception:
        pass
    try:
        _faiss_export_handle.cancel()
    except Exception:
        pass
    try:
        _faiss_age_handle.cancel()
    except Exception:
        pass
    try:
        if _faiss_recovery_handle:
            _faiss_recovery_handle.cancel()
    except Exception:
        pass
    try:
        orphan_handle.cancel()
    except Exception:
        pass


# 创建FastAPI应用
app = FastAPI(
    title="CAD ML Platform",
    description="智能CAD分析微服务平台",
    version="1.0.0",
    lifespan=lifespan,
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置信任主机
app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.ALLOWED_HOSTS)

# 注册路由
app.include_router(api_router, prefix="/api")

if _metrics_enabled:
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
else:
    @app.get("/metrics")
    async def metrics_fallback() -> PlainTextResponse:  # type: ignore
        """Fallback metrics endpoint when prometheus_client is unavailable.

        Provides a minimal textual exposition so monitoring contracts expecting
        a 200 response do not fail in reduced environments.
        """
        body = (
            "# HELP app_metrics_disabled Metrics client disabled\n"
            "# TYPE app_metrics_disabled gauge\n"
            "app_metrics_disabled 1\n"
        )
        return PlainTextResponse(body, media_type="text/plain; version=0.0.4")


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "CAD ML Platform",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """健康检查（附加运行时与指标状态）"""
    import sys
    from datetime import datetime, timezone

    current_settings = get_settings()

    base = {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "services": {
            "api": "up",
            "ml": "up",
            "redis": "up" if settings.REDIS_ENABLED else "disabled",
        },
        "runtime": {
            "python_version": sys.version.split(" ")[0],
            "metrics_enabled": _metrics_enabled,
            "vision_max_base64_bytes": current_settings.VISION_MAX_BASE64_BYTES,
            "error_rate_ema": {
                "ocr": get_ocr_error_rate_ema(),
                "vision": get_vision_error_rate_ema(),
            },
        },
        "config": {
            # 运维可见的关键配置
            "limits": {
                "vision_max_base64_bytes": current_settings.VISION_MAX_BASE64_BYTES,
                # precompute to keep line length under linter threshold
                "vision_max_base64_mb": round(
                    current_settings.VISION_MAX_BASE64_BYTES / 1024 / 1024, 2
                ),
                "ocr_timeout_ms": current_settings.OCR_TIMEOUT_MS,
                "ocr_timeout_seconds": current_settings.OCR_TIMEOUT_MS / 1000,
            },
            "providers": {
                "ocr_default": current_settings.OCR_PROVIDER_DEFAULT,
                "confidence_fallback": current_settings.CONFIDENCE_FALLBACK,
            },
            "monitoring": {
                "error_ema_alpha": current_settings.ERROR_EMA_ALPHA,
                "metrics_enabled": _metrics_enabled,
                "redis_enabled": current_settings.REDIS_ENABLED,
            },
            "network": {
                "cors_origins": current_settings.CORS_ORIGINS,
                "allowed_hosts": current_settings.ALLOWED_HOSTS,
            },
            "debug": {
                "debug_mode": current_settings.DEBUG,
                "log_level": current_settings.LOG_LEVEL,
            },
        },
    }

    # Merge resilience health block
    try:
        base.update(get_resilience_health())
    except Exception:
        # Keep /health resilient if resilience module is unavailable
        pass

    return base


@app.get("/health/extended")
async def extended_health():
    """扩展健康检查: 包含向量与特征版本分布、Faiss索引状态、缓存命中情况等。"""
    from src.core.similarity import _VECTOR_STORE, _VECTOR_META  # type: ignore
    from src.core.similarity import _FAISS_IMPORTED, _FAISS_LAST_EXPORT_SIZE, _FAISS_LAST_EXPORT_TS  # type: ignore
    import os, time
    vector_total = len(_VECTOR_STORE)
    versions: dict[str, int] = {}
    for meta in _VECTOR_META.values():
        ver = meta.get("feature_version", os.environ.get("FEATURE_VERSION", "v1"))
        versions[ver] = versions.get(ver, 0) + 1
    faiss_enabled = os.environ.get("VECTOR_STORE_BACKEND", "memory") == "faiss"
    last_export_age = None
    if _FAISS_LAST_EXPORT_TS is not None:
        last_export_age = round(time.time() - _FAISS_LAST_EXPORT_TS, 2)
    return {
        "status": "healthy",
        "vector_store": {
            "total": vector_total,
            "versions": versions,
        },
        "faiss": {
            "enabled": faiss_enabled,
            "imported": _FAISS_IMPORTED if faiss_enabled else False,
            "last_export_size": _FAISS_LAST_EXPORT_SIZE if faiss_enabled else 0,
            "last_export_age_seconds": last_export_age,
        },
        "feature_version_env": os.environ.get("FEATURE_VERSION", "v1"),
    }


@app.get("/ready")
async def readiness_check():
    """就绪检查"""
    # 检查所有依赖服务
    try:
        # 检查模型是否加载
        from src.models.loader import models_loaded

        if not models_loaded():
            raise HTTPException(status_code=503, detail="Models not loaded")

        # 检查Redis连接
        if settings.REDIS_ENABLED:
            from src.utils.cache import redis_healthy

            if not await redis_healthy():
                raise HTTPException(status_code=503, detail="Redis not ready")

        return {"status": "ready"}
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="Service not ready")


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
