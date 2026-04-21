from __future__ import annotations

from typing import Any, Callable


async def run_vector_search_pipeline(
    *,
    payload: Any,
    response_cls: type[Any],
    get_qdrant_store_fn: Callable[[], Any],
    build_filter_conditions_fn: Callable[[Any], dict[str, Any]],
    matches_filters_fn: Callable[[Any, dict[str, Any], dict[str, Any]], bool],
    vector_item_payload_fn: Callable[[str, int, dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> Any:
    from src.core.similarity import (
        _VECTOR_META,
        _VECTOR_STORE,
        extract_vector_label_contract,
        get_vector_store,
    )

    qdrant_store = get_qdrant_store_fn()
    if qdrant_store is not None:
        results = await qdrant_store.search_similar(
            payload.vector,
            top_k=payload.k,
            filter_conditions=build_filter_conditions_fn(payload),
            with_vectors=True,
        )
        items = []
        for result in results:
            meta = result.metadata or {}
            label_contract = extract_vector_label_contract(meta)
            items.append(
                {
                    "id": result.id,
                    "score": round(float(result.score), 4),
                    **vector_item_payload_fn(
                        result.id,
                        len(result.vector or []),
                        meta,
                        label_contract,
                    ),
                }
            )
        return response_cls(results=items, total=len(items))

    store = get_vector_store()
    query_k = payload.k
    if any(
        [
            payload.material_filter,
            payload.complexity_filter,
            payload.fine_part_type_filter,
            payload.coarse_part_type_filter,
            payload.decision_source_filter,
            payload.is_coarse_label_filter is not None,
        ]
    ):
        query_k = min(payload.k * 5, payload.k + 100)
    results = store.query(payload.vector, top_k=query_k)
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for vid, score in results:
        if vid in seen:
            continue
        seen.add(vid)
        meta = _VECTOR_META.get(vid) or store.meta(vid) or {}
        label_contract = extract_vector_label_contract(meta)
        if not matches_filters_fn(payload, meta, label_contract):
            continue
        items.append(
            {"id": vid, "score": round(float(score), 4)}
            | vector_item_payload_fn(
                vid,
                len(_VECTOR_STORE.get(vid, [])),
                meta,
                label_contract,
            )
        )
        if len(items) >= payload.k:
            break

    return response_cls(results=items, total=len(items))


__all__ = ["run_vector_search_pipeline"]
