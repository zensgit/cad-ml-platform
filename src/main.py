"""
CAD ML Platform - 主服务入口
智能CAD分析微服务平台
"""

import asyncio
import inspect
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Any, Callable

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import PlainTextResponse
import uvicorn

from src.api import api_router
from src.api.health_models import (
    ExtendedHealthResponse,
    HealthResponse,
    ReadinessCheck,
    ReadinessResponse,
)
from src.api.health_utils import build_health_payload, metrics_enabled, record_health_request
from src.api.middleware.integration_auth import IntegrationAuthMiddleware
from src.core.config import get_settings
from src.core.similarity import (  # type: ignore
    FaissVectorStore,
    background_prune_task,
    load_recovery_state,
)
from src.models.loader import load_models
from src.tasks.orphan_scan import orphan_scan_loop
from src.utils.cache import init_redis
from src.utils.logging import setup_logging

_metrics_enabled = metrics_enabled()
if _metrics_enabled:
    from prometheus_client import make_asgi_app  # type: ignore

# 设置日志
setup_logging()
logger = logging.getLogger(__name__)

# 加载配置
settings = get_settings()
READINESS_CHECK_TIMEOUT_SECONDS = float(os.getenv("READINESS_CHECK_TIMEOUT_SECONDS", "0.5"))


def _validate_optional_feature_flags() -> None:
    graph2d_enabled = os.getenv("GRAPH2D_ENABLED", "false").lower() == "true"
    graph2d_model = os.getenv("GRAPH2D_MODEL_PATH", "models/graph2d_parts_upsampled_20260122.pth")
    fusion_enabled = os.getenv("FUSION_ANALYZER_ENABLED", "false").lower() == "true"
    fusion_override = os.getenv("FUSION_ANALYZER_OVERRIDE", "false").lower() == "true"
    graph2d_fusion = os.getenv("GRAPH2D_FUSION_ENABLED", "false").lower() == "true"
    min_conf_raw = os.getenv("FUSION_ANALYZER_OVERRIDE_MIN_CONF", "0.5")
    graph2d_override_labels = os.getenv("FUSION_GRAPH2D_OVERRIDE_LABELS", "").strip()
    graph2d_override_min_conf_raw = os.getenv("FUSION_GRAPH2D_OVERRIDE_MIN_CONF", "0.6")
    graph2d_override_low_conf_labels = os.getenv(
        "FUSION_GRAPH2D_OVERRIDE_LOW_CONF_LABELS", ""
    ).strip()
    graph2d_override_low_conf_min_raw = os.getenv(
        "FUSION_GRAPH2D_OVERRIDE_LOW_CONF_MIN_CONF", "0.6"
    )
    graph2d_min_conf_raw = os.getenv("GRAPH2D_MIN_CONF", "0.0")
    graph2d_exclude_raw = os.getenv("GRAPH2D_EXCLUDE_LABELS", "other")

    if graph2d_enabled and not os.path.exists(graph2d_model):
        logger.warning("GRAPH2D_ENABLED=true but model missing: %s", graph2d_model)
    if graph2d_fusion and not graph2d_enabled:
        logger.warning("GRAPH2D_FUSION_ENABLED=true requires GRAPH2D_ENABLED=true")
    if fusion_override and not fusion_enabled:
        logger.warning("FUSION_ANALYZER_OVERRIDE=true requires FUSION_ANALYZER_ENABLED=true")
    if graph2d_override_labels and not graph2d_fusion:
        logger.warning("FUSION_GRAPH2D_OVERRIDE_LABELS set but GRAPH2D_FUSION_ENABLED=false")
    if graph2d_override_low_conf_labels and not graph2d_fusion:
        logger.warning(
            "FUSION_GRAPH2D_OVERRIDE_LOW_CONF_LABELS set but GRAPH2D_FUSION_ENABLED=false"
        )
    try:
        min_conf = float(min_conf_raw)
        if not 0.0 <= min_conf <= 1.0:
            logger.warning(
                "FUSION_ANALYZER_OVERRIDE_MIN_CONF out of range: %s", min_conf_raw
            )
    except (TypeError, ValueError):
        logger.warning(
            "FUSION_ANALYZER_OVERRIDE_MIN_CONF is not a float: %s", min_conf_raw
        )
    try:
        graph2d_override_min = float(graph2d_override_min_conf_raw)
        if not 0.0 <= graph2d_override_min <= 1.0:
            logger.warning(
                "FUSION_GRAPH2D_OVERRIDE_MIN_CONF out of range: %s",
                graph2d_override_min_conf_raw,
            )
    except (TypeError, ValueError):
        logger.warning(
            "FUSION_GRAPH2D_OVERRIDE_MIN_CONF is not a float: %s",
            graph2d_override_min_conf_raw,
        )
    try:
        graph2d_override_low_conf_min = float(graph2d_override_low_conf_min_raw)
        if not 0.0 <= graph2d_override_low_conf_min <= 1.0:
            logger.warning(
                "FUSION_GRAPH2D_OVERRIDE_LOW_CONF_MIN_CONF out of range: %s",
                graph2d_override_low_conf_min_raw,
            )
    except (TypeError, ValueError):
        logger.warning(
            "FUSION_GRAPH2D_OVERRIDE_LOW_CONF_MIN_CONF is not a float: %s",
            graph2d_override_low_conf_min_raw,
        )
    try:
        graph2d_min_conf = float(graph2d_min_conf_raw)
        if not 0.0 <= graph2d_min_conf <= 1.0:
            logger.warning("GRAPH2D_MIN_CONF out of range: %s", graph2d_min_conf_raw)
    except (TypeError, ValueError):
        logger.warning("GRAPH2D_MIN_CONF is not a float: %s", graph2d_min_conf_raw)
    if graph2d_exclude_raw.strip() and not graph2d_enabled:
        logger.warning("GRAPH2D_EXCLUDE_LABELS set but GRAPH2D_ENABLED=false")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    logger.info("Starting CAD ML Platform...")
    _validate_optional_feature_flags()

    # Optional dev seeding of knowledge rules
    try:
        from src.utils.knowledge_seed import seed_knowledge_if_empty

        seed_knowledge_if_empty()
    except Exception:
        logger.warning("Knowledge seed failed", exc_info=True)

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

                dim = getattr(_FAISS_INDEX, "d", None) if _FAISS_INDEX is not None else None  # type: ignore
                size = getattr(_FAISS_INDEX, "ntotal", None) if _FAISS_INDEX is not None else None  # type: ignore
                env_dim = int(__import__("os").getenv("FEATURE_VECTOR_EXPECTED_DIM", "0") or 0)
                if env_dim and dim and dim != env_dim:
                    from src.utils.analysis_metrics import faiss_index_dim_mismatch_total

                    faiss_index_dim_mismatch_total.inc()
                    logger.warning(
                        "Faiss index dim mismatch (imported=%s, expected=%s) - falling back to memory store",
                        dim,
                        env_dim,
                    )
                else:
                    logger.info(
                        "Faiss index auto-imported from %s (dim=%s, size=%s)", faiss_path, dim, size
                    )
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
        import asyncio
        import os

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

    # Faiss auto recovery loop (only when faiss backend requested)
    try:
        if __import__("os").getenv("VECTOR_STORE_BACKEND", "memory") == "faiss":
            from src.core.similarity import faiss_recovery_loop

            _faiss_recovery_handle = asyncio.create_task(faiss_recovery_loop())
        else:
            _faiss_recovery_handle = None
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

    # Analysis result cleanup task (optional)
    cleanup_interval = float(
        __import__("os").getenv("ANALYSIS_RESULT_CLEANUP_INTERVAL_SECONDS", "0")
    )
    if cleanup_interval > 0:
        from src.utils.analysis_result_store import cleanup_analysis_results

        async def _analysis_result_cleanup_loop() -> None:
            while True:
                try:
                    result = await cleanup_analysis_results(dry_run=False, sample_limit=0)
                    if result.get("status") == "ok" and result.get("deleted_count", 0) > 0:
                        logger.info(
                            "analysis_result_cleanup status=%s deleted=%s eligible=%s",
                            result.get("status"),
                            result.get("deleted_count"),
                            result.get("eligible_count"),
                        )
                except Exception:
                    logger.warning("analysis_result_cleanup_failed", exc_info=True)
                await asyncio.sleep(cleanup_interval)

        _analysis_result_cleanup_handle = asyncio.create_task(_analysis_result_cleanup_loop())
    else:
        _analysis_result_cleanup_handle = None

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
    try:
        if _analysis_result_cleanup_handle:
            _analysis_result_cleanup_handle.cancel()
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

# Optional integration auth (JWT) for upstream platforms
app.add_middleware(IntegrationAuthMiddleware, settings=settings)

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


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查（附加运行时与指标状态）"""
    start = time.perf_counter()
    payload = build_health_payload(metrics_enabled_override=_metrics_enabled)
    record_health_request(
        "health", payload.get("status", "unknown"), time.perf_counter() - start
    )
    return payload


@app.get("/health/extended", response_model=ExtendedHealthResponse)
async def extended_health():
    """扩展健康检查: 包含向量与特征版本分布、Faiss索引状态、缓存命中情况等。"""
    start = time.perf_counter()
    from src.core.similarity import (  # type: ignore
        _FAISS_IMPORTED,
        _FAISS_LAST_EXPORT_SIZE,
        _FAISS_LAST_EXPORT_TS,
        _VECTOR_META,
        _VECTOR_STORE,
    )

    base_payload = build_health_payload(metrics_enabled_override=_metrics_enabled)
    vector_total = len(_VECTOR_STORE)
    versions: dict[str, int] = {}
    for meta in _VECTOR_META.values():
        ver = meta.get("feature_version", os.environ.get("FEATURE_VERSION", "v1"))
        versions[ver] = versions.get(ver, 0) + 1
    faiss_enabled = os.environ.get("VECTOR_STORE_BACKEND", "memory") == "faiss"
    last_export_age = None
    if _FAISS_LAST_EXPORT_TS is not None:
        last_export_age = round(time.time() - _FAISS_LAST_EXPORT_TS, 2)
    payload = {
        **base_payload,
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
    record_health_request(
        "health_extended", payload.get("status", "unknown"), time.perf_counter() - start
    )
    return payload


async def _run_readiness_check(
    check: Callable[[], Any],
    enabled: bool,
    timeout_seconds: float,
) -> ReadinessCheck:
    if not enabled:
        return ReadinessCheck(status="disabled", detail="disabled by config")

    start = time.perf_counter()
    try:
        result = check()
        if inspect.isawaitable(result):
            result = await asyncio.wait_for(result, timeout=timeout_seconds)
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        if result:
            return ReadinessCheck(status="ready", duration_ms=duration_ms)
        return ReadinessCheck(
            status="not_ready",
            detail="check returned false",
            duration_ms=duration_ms,
        )
    except asyncio.TimeoutError:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return ReadinessCheck(
            status="not_ready",
            detail=f"timeout after {timeout_seconds}s",
            duration_ms=duration_ms,
            timed_out=True,
        )
    except Exception as exc:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        return ReadinessCheck(
            status="not_ready",
            detail=f"{type(exc).__name__}: {exc}",
            duration_ms=duration_ms,
        )


@app.get("/ready", response_model=ReadinessResponse)
async def readiness_check(response: Response):
    """就绪检查"""
    start = time.perf_counter()
    from src.models.loader import models_loaded
    from src.utils.cache import redis_healthy

    checks = {
        "models_loaded": await _run_readiness_check(
            models_loaded,
            True,
            READINESS_CHECK_TIMEOUT_SECONDS,
        ),
        "redis": await _run_readiness_check(
            redis_healthy,
            settings.REDIS_ENABLED,
            READINESS_CHECK_TIMEOUT_SECONDS,
        ),
    }

    ready = all(check.status in {"ready", "disabled"} for check in checks.values())
    status = "ready" if ready else "not_ready"
    if not ready:
        response.status_code = 503
        failures = {
            name: check.detail or check.status
            for name, check in checks.items()
            if check.status != "ready"
        }
        logger.error("Readiness check failed: %s", failures)

    payload = ReadinessResponse(
        status=status,
        ready=ready,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        checks=checks,
    ).model_dump()
    record_health_request("ready", status, time.perf_counter() - start)
    return payload


if __name__ == "__main__":
    uvicorn.run(
        "src.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
