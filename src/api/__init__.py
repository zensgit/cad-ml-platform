"""API路由聚合"""
from fastapi import APIRouter

from src.api.v1 import (
    analyze, ocr, vision, drift, health, vectors, process, vectors_stats,
    features, model, maintenance
)

api_router = APIRouter()

# 注册v1版本API
v1_router = APIRouter(prefix="/v1")

# 核心分析模块
v1_router.include_router(analyze.router, prefix="/analyze", tags=["分析"])
v1_router.include_router(drift.router, prefix="/analyze", tags=["漂移"])

# 向量相关模块
v1_router.include_router(vectors.router, prefix="/analyze", tags=["向量"])
v1_router.include_router(vectors_stats.router, prefix="/analyze", tags=["向量统计"])

# 工艺和特征模块
v1_router.include_router(process.router, prefix="/analyze", tags=["工艺规则"])
v1_router.include_router(features.router, prefix="/features", tags=["特征"])

# 模型和维护模块
v1_router.include_router(model.router, prefix="/model", tags=["模型"])
v1_router.include_router(maintenance.router, prefix="/maintenance", tags=["维护"])

# 健康检查
v1_router.include_router(health.router, tags=["健康"])

# 子路由内部已移除前缀，这里统一加资源前缀
v1_router.include_router(vision.router, prefix="/vision", tags=["视觉"])
v1_router.include_router(ocr.router, prefix="/ocr", tags=["OCR"])

api_router.include_router(v1_router)

__all__ = ["api_router"]
