from __future__ import annotations

from typing import Any, Awaitable, Callable, Optional


AsyncLoadResult = Callable[[str], Awaitable[Optional[dict[str, Any]]]]
AsyncSetCache = Callable[[str, dict[str, Any], int], Awaitable[Any]]


async def load_analysis_result_with_cache(
    *,
    analysis_id: str,
    get_cached_result_fn: AsyncLoadResult,
    load_analysis_result_fn: AsyncLoadResult,
    set_cache_fn: AsyncSetCache,
    ttl_seconds: int = 3600,
) -> Optional[dict[str, Any]]:
    cache_key = f"analysis_result:{analysis_id}"
    result = await get_cached_result_fn(cache_key)

    if not result:
        result = await load_analysis_result_fn(analysis_id)
        if result:
            await set_cache_fn(cache_key, result, ttl_seconds=ttl_seconds)

    return result
