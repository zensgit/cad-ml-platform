from __future__ import annotations

from typing import Any, Awaitable, Callable

from src.core.vector_migration_pending_run_candidates import (
    collect_pending_run_candidates,
)
from src.core.vector_migration_pending_run_guard import (
    ensure_pending_run_scan_is_allowed,
)
from src.core.vector_migration_pending_run_request import (
    build_pending_run_migrate_request,
)


async def run_vector_migration_pending_run_pipeline(
    *,
    payload: Any,
    api_key: str,
    resolve_target_version_fn: Callable[[], str],
    collect_pending_candidates_fn: Callable[..., Awaitable[dict[str, Any]]],
    migrate_vectors_fn: Callable[..., Awaitable[Any]],
    request_cls: type[Any],
    error_code_cls: Any,
    build_error_fn: Callable[..., dict[str, Any]],
) -> Any:
    target_version = resolve_target_version_fn()
    pending = await collect_pending_run_candidates(
        payload=payload,
        target_version=target_version,
        collect_pending_candidates_fn=collect_pending_candidates_fn,
    )
    ensure_pending_run_scan_is_allowed(
        pending=pending,
        allow_partial_scan=payload.allow_partial_scan,
        target_version=target_version,
        error_code_cls=error_code_cls,
        build_error_fn=build_error_fn,
    )
    migrate_request = build_pending_run_migrate_request(
        pending=pending,
        target_version=target_version,
        dry_run=payload.dry_run,
        request_cls=request_cls,
    )
    return await migrate_vectors_fn(migrate_request, api_key=api_key)


__all__ = ["run_vector_migration_pending_run_pipeline"]
