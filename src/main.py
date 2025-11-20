"""
CAD ML Platform - 主服务入口
智能CAD分析微服务平台
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
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

    # 初始化Redis连接
    if settings.REDIS_ENABLED:
        await init_redis()
        logger.info("Redis initialized")

    # 加载ML模型
    await load_models()
    logger.info("ML models loaded")

    yield

    # 关闭时
    logger.info("Shutting down CAD ML Platform...")
    # 清理资源
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
    return {
        "status": "healthy",
        "services": {
            "api": "up",
            "ml": "up",
            "redis": "up" if settings.REDIS_ENABLED else "disabled",
        },
        "runtime": {
            "python_version": sys.version.split(" ")[0],
            "metrics_enabled": _metrics_enabled,
            "vision_max_base64_bytes": get_settings().VISION_MAX_BASE64_BYTES,
            "error_rate_ema": {
                "ocr": get_ocr_error_rate_ema(),
                "vision": get_vision_error_rate_ema(),
            },
            "config": {
                "error_ema_alpha": get_settings().ERROR_EMA_ALPHA,
            },
        },
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
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
    )
