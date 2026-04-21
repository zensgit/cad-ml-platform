from __future__ import annotations

from typing import Any, Callable

from fastapi import HTTPException


async def run_vector_delete_pipeline(
    *,
    payload: Any,
    response_cls: type[Any],
    error_code_cls: Any,
    build_error_fn: Callable[..., dict[str, Any]],
    get_qdrant_store_fn: Callable[[], Any],
    get_client_fn: Callable[[], Any],
) -> Any:
    qdrant_store = get_qdrant_store_fn()
    if qdrant_store is not None:
        existing = await qdrant_store.get_vector(payload.id)
        if existing is None:
            err = build_error_fn(
                error_code_cls.DATA_NOT_FOUND,
                stage="vector_delete",
                message="Vector not found",
                id=payload.id,
            )
            raise HTTPException(status_code=404, detail=err)
        deleted = await qdrant_store.delete_vector(payload.id)
        if deleted:
            return response_cls(id=payload.id, status="deleted")
        err = build_error_fn(
            error_code_cls.INTERNAL_ERROR,
            stage="vector_delete",
            message="Delete failed",
            id=payload.id,
        )
        raise HTTPException(status_code=500, detail=err)

    from src.core.similarity import (  # type: ignore
        _BACKEND,
        _VECTOR_META,
        _VECTOR_STORE,
        FaissVectorStore,
    )

    if payload.id not in _VECTOR_STORE:
        err = build_error_fn(
            error_code_cls.DATA_NOT_FOUND,
            stage="vector_delete",
            message="Vector not found",
            id=payload.id,
        )
        raise HTTPException(status_code=404, detail=err)
    try:
        if __import__("os").getenv("VECTOR_STORE_BACKEND", "memory") == "faiss":
            try:
                fstore = FaissVectorStore()
                fstore.mark_delete(payload.id)  # type: ignore[attr-defined]
            except Exception:
                pass
        del _VECTOR_STORE[payload.id]
        _VECTOR_META.pop(payload.id, None)
        if _BACKEND == "redis":
            client = get_client_fn()
            if client is not None:
                try:
                    await client.delete(f"vector:{payload.id}")
                except Exception:
                    pass
        return response_cls(id=payload.id, status="deleted")
    except Exception as exc:
        err = build_error_fn(
            error_code_cls.INTERNAL_ERROR,
            stage="vector_delete",
            message="Delete failed",
            id=payload.id,
            detail=str(exc),
        )
        raise HTTPException(status_code=500, detail=err)


__all__ = ["run_vector_delete_pipeline"]
