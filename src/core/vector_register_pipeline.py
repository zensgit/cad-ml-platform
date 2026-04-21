from __future__ import annotations

import os
from typing import Any, Callable


async def run_vector_register_pipeline(
    *,
    payload: Any,
    response_cls: type[Any],
    error_code_cls: Any,
    build_error_fn: Callable[..., dict[str, Any]],
    get_qdrant_store_fn: Callable[[], Any],
) -> Any:
    qdrant_store = get_qdrant_store_fn()
    if qdrant_store is not None:
        meta = dict(payload.meta or {})
        meta.setdefault("total_dim", str(len(payload.vector)))
        await qdrant_store.register_vector(payload.id, payload.vector, metadata=meta)
        return response_cls(
            id=payload.id,
            status="accepted",
            dimension=len(payload.vector),
        )

    from src.core.similarity import FaissVectorStore, last_vector_error, register_vector

    meta = dict(payload.meta or {})
    meta.setdefault("total_dim", str(len(payload.vector)))
    accepted = register_vector(payload.id, payload.vector, meta=meta)
    if accepted:
        if os.getenv("VECTOR_STORE_BACKEND", "memory") == "faiss":
            try:
                fstore = FaissVectorStore()
                fstore.add(payload.id, payload.vector)
            except Exception:
                pass
        return response_cls(
            id=payload.id,
            status="accepted",
            dimension=len(payload.vector),
        )

    err = last_vector_error()
    if err is None:
        err = build_error_fn(
            error_code_cls.DIMENSION_MISMATCH,
            stage="vector_register",
            message="Vector rejected",
            id=payload.id,
        )
    return response_cls(
        id=payload.id,
        status="rejected",
        error=err,
    )


__all__ = ["run_vector_register_pipeline"]
