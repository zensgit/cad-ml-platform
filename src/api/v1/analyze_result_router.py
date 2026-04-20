from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException

from src.api.dependencies import get_api_key
from src.core.analysis_result_lookup import load_analysis_result_with_cache

AsyncLoadResult = Callable[[str], Awaitable[Optional[dict[str, Any]]]]
AsyncSetCache = Callable[[str, dict[str, Any], int], Awaitable[Any]]


def build_result_router(
    *,
    get_cached_result_fn: AsyncLoadResult,
    load_analysis_result_fn: AsyncLoadResult,
    set_cache_fn: AsyncSetCache,
) -> APIRouter:
    router = APIRouter()

    # IMPORTANT: This catch-all route MUST remain the last analyze child route.
    @router.get("/{analysis_id}")
    async def get_analysis_result(
        analysis_id: str,
        api_key: str = Depends(get_api_key),
    ):
        """获取分析结果"""
        result = await load_analysis_result_with_cache(
            analysis_id=analysis_id,
            get_cached_result_fn=get_cached_result_fn,
            load_analysis_result_fn=load_analysis_result_fn,
            set_cache_fn=set_cache_fn,
            ttl_seconds=3600,
        )

        if not result:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return result

    return router


__all__ = ["build_result_router"]
