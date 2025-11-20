"""API路由聚合"""
from fastapi import APIRouter

from src.api.v1 import analyze, ocr, vision

api_router = APIRouter()

# 注册v1版本API
v1_router = APIRouter(prefix="/v1")
v1_router.include_router(analyze.router, prefix="/analyze", tags=["分析"])
# 子路由内部已移除前缀，这里统一加资源前缀
v1_router.include_router(vision.router, prefix="/vision", tags=["视觉"])
v1_router.include_router(ocr.router, prefix="/ocr", tags=["OCR"])

api_router.include_router(v1_router)

__all__ = ["api_router"]
