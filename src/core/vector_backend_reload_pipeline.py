from __future__ import annotations

import os
from typing import Any, Callable, Optional

from fastapi import HTTPException


async def run_vector_backend_reload_pipeline(
    *,
    backend: Optional[str],
    reload_backend_fn: Callable[[], bool],
    reload_metric: Any,
    error_code_cls: Any,
    build_error_fn: Callable[..., dict[str, Any]],
    response_cls: type[Any],
) -> Any:
    allowed = {"memory", "faiss", "redis"}
    normalized_backend = backend
    if normalized_backend is not None:
        normalized_backend = normalized_backend.strip().lower()
        if normalized_backend not in allowed:
            reload_metric.labels(status="error", reason="invalid_backend").inc()
            err = build_error_fn(
                error_code_cls.INPUT_VALIDATION_FAILED,
                stage="backend_reload",
                message="Unsupported vector backend",
                backend=normalized_backend,
                supported=sorted(allowed),
            )
            raise HTTPException(status_code=400, detail=err)
        os.environ["VECTOR_STORE_BACKEND"] = normalized_backend

    effective_backend = normalized_backend or os.getenv("VECTOR_STORE_BACKEND", "memory")
    try:
        ok = reload_backend_fn()
    except Exception as exc:
        reload_metric.labels(status="error", reason="init_error").inc()
        err = build_error_fn(
            error_code_cls.INTERNAL_ERROR,
            stage="backend_reload",
            message="Exception during backend reload",
            backend=effective_backend,
            detail=str(exc),
        )
        raise HTTPException(status_code=500, detail=err)

    reload_metric.labels(
        status="success" if ok else "error",
        reason="ok" if ok else "init_error",
    ).inc()
    if not ok:
        err = build_error_fn(
            error_code_cls.INTERNAL_ERROR,
            stage="backend_reload",
            message="Vector store backend reload failed",
            backend=effective_backend,
        )
        raise HTTPException(status_code=500, detail=err)

    return response_cls(status="ok", backend=effective_backend)


__all__ = ["run_vector_backend_reload_pipeline"]
