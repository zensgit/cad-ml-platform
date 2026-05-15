from __future__ import annotations

from typing import Any, Callable, Optional

from fastapi import HTTPException

from src.core.vector_list_limits import resolve_vector_list_limits
from src.core.vector_list_qdrant import list_vectors_qdrant


async def run_vector_list_pipeline(
    *,
    source: str,
    offset: int,
    limit: int,
    material_filter: Optional[str],
    complexity_filter: Optional[str],
    fine_part_type_filter: Optional[str],
    coarse_part_type_filter: Optional[str],
    decision_source_filter: Optional[str],
    is_coarse_label_filter: Optional[bool],
    response_cls: type[Any],
    item_cls: type[Any],
    error_code_cls: Any,
    build_error_fn: Callable[..., dict[str, Any]],
    get_qdrant_store_fn: Callable[[], Any],
    resolve_list_source_fn: Callable[[str, str], str],
    build_filter_conditions_fn: Callable[..., dict[str, Any]],
    list_vectors_redis_fn: Callable[..., Any],
    list_vectors_memory_fn: Callable[..., Any],
    get_client_fn: Callable[[], Any],
) -> Any:
    from src.core.similarity import _BACKEND, _VECTOR_META, _VECTOR_STORE  # type: ignore

    allowed_sources = {"auto", "memory", "redis", "qdrant"}
    if source not in allowed_sources:
        err = build_error_fn(
            error_code_cls.INPUT_VALIDATION_FAILED,
            stage="vector_list",
            message="Invalid source",
            source=source,
            allowed=list(sorted(allowed_sources)),
        )
        raise HTTPException(status_code=400, detail=err)

    limit, scan_limit = resolve_vector_list_limits(limit)
    resolved = resolve_list_source_fn(source, _BACKEND)
    if resolved == "qdrant":
        qdrant_store = get_qdrant_store_fn()
        if qdrant_store is not None:
            from src.core.similarity import extract_vector_label_contract

            return await list_vectors_qdrant(
                qdrant_store,
                offset,
                limit,
                material_filter,
                complexity_filter,
                fine_part_type_filter,
                coarse_part_type_filter,
                decision_source_filter,
                is_coarse_label_filter,
                item_cls=item_cls,
                response_cls=response_cls,
                build_filter_conditions_fn=build_filter_conditions_fn,
                extract_label_contract_fn=extract_vector_label_contract,
            )
    if resolved == "redis":
        client = get_client_fn()
        if client is not None:
            return await list_vectors_redis_fn(
                client,
                offset,
                limit,
                scan_limit,
                material_filter,
                complexity_filter,
                fine_part_type_filter,
                coarse_part_type_filter,
                decision_source_filter,
                is_coarse_label_filter,
            )
    return list_vectors_memory_fn(
        _VECTOR_STORE,
        _VECTOR_META,
        offset,
        limit,
        material_filter,
        complexity_filter,
        fine_part_type_filter,
        coarse_part_type_filter,
        decision_source_filter,
        is_coarse_label_filter,
    )


__all__ = ["run_vector_list_pipeline"]
