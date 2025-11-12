"""API路由聚合"""
from fastapi import APIRouter

from src.api.v1 import analyze, similarity, classify

api_router = APIRouter()

# 注册v1版本API
v1_router = APIRouter(prefix="/v1")
v1_router.include_router(analyze.router, prefix="/analyze", tags=["分析"])
v1_router.include_router(similarity.router, prefix="/similarity", tags=["相似度"])
v1_router.include_router(classify.router, prefix="/classify", tags=["分类"])

api_router.include_router(v1_router)

__all__ = ["api_router"]