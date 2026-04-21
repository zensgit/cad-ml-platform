from __future__ import annotations

import os
from typing import Any, Callable, Optional

from fastapi import HTTPException


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

    max_limit = int(os.getenv("VECTOR_LIST_LIMIT", "200"))
    limit = min(limit, max_limit)
    scan_limit = int(os.getenv("VECTOR_LIST_SCAN_LIMIT", "5000"))
    resolved = resolve_list_source_fn(source, _BACKEND)
    if resolved == "qdrant":
        qdrant_store = get_qdrant_store_fn()
        if qdrant_store is not None:
            from src.core.similarity import extract_vector_label_contract

            results, total = await qdrant_store.list_vectors(
                offset=offset,
                limit=limit,
                filter_conditions=build_filter_conditions_fn(
                    material_filter=material_filter,
                    complexity_filter=complexity_filter,
                    fine_part_type_filter=fine_part_type_filter,
                    coarse_part_type_filter=coarse_part_type_filter,
                    decision_source_filter=decision_source_filter,
                    is_coarse_label_filter=is_coarse_label_filter,
                ),
                with_vectors=True,
            )
            items = []
            for result in results:
                meta = result.metadata or {}
                label_contract = extract_vector_label_contract(meta)
                items.append(
                    item_cls(
                        id=result.id,
                        dimension=len(result.vector or []),
                        material=meta.get("material"),
                        complexity=meta.get("complexity"),
                        format=meta.get("format"),
                        part_type=label_contract.get("part_type"),
                        fine_part_type=label_contract.get("fine_part_type"),
                        coarse_part_type=label_contract.get("coarse_part_type"),
                        decision_source=label_contract.get("decision_source"),
                        is_coarse_label=label_contract.get("is_coarse_label"),
                    )
                )
            return response_cls(total=total, vectors=items)
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
