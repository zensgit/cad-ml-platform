from __future__ import annotations

from typing import Any, Callable, Mapping

from fastapi import HTTPException


def ensure_pending_run_scan_is_allowed(
    *,
    pending: Mapping[str, Any],
    allow_partial_scan: bool,
    target_version: str,
    error_code_cls: Any,
    build_error_fn: Callable[..., dict[str, Any]],
) -> None:
    if pending["backend"] == "qdrant" and not pending["distribution_complete"]:
        if not allow_partial_scan:
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


__all__ = ["ensure_pending_run_scan_is_allowed"]
