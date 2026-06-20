from __future__ import annotations

from typing import Any, Awaitable, Callable


async def collect_pending_run_candidates(
    *,
    payload: Any,
    target_version: str,
    collect_pending_candidates_fn: Callable[..., Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    return await collect_pending_candidates_fn(
        limit=payload.limit,
        target_version=target_version,
        from_version_filter=payload.from_version_filter,
    )


__all__ = ["collect_pending_run_candidates"]
