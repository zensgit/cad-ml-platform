from __future__ import annotations

from typing import Any, Awaitable, Callable

from fastapi import HTTPException


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
    pending = await collect_pending_candidates_fn(
        limit=payload.limit,
        target_version=target_version,
        from_version_filter=payload.from_version_filter,
    )
    if pending["backend"] == "qdrant" and not pending["distribution_complete"]:
        if not payload.allow_partial_scan:
            raise HTTPException(
                status_code=409,
                detail=build_error_fn(
                    error_code_cls.CONSTRAINT_VIOLATION,
                    stage="vector_migrate_pending_run",
                    message=(
                        "Qdrant distribution scan is partial; raise VECTOR_MIGRATION_SCAN_LIMIT "
                        "or set allow_partial_scan=true"
                    ),
                    target_version=target_version,
                    scanned_vectors=pending["scanned_vectors"],
                    scan_limit=pending["scan_limit"],
                ),
            )
    return await migrate_vectors_fn(
        request_cls(
            ids=pending["pending_ids"],
            to_version=target_version,
            dry_run=payload.dry_run,
        ),
        api_key=api_key,
    )


__all__ = ["run_vector_migration_pending_run_pipeline"]
