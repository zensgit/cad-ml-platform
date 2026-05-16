from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional


async def list_vectors_qdrant(
    qdrant_store: Any,
    offset: int,
    limit: int,
    material_filter: Optional[str],
    complexity_filter: Optional[str],
    fine_part_type_filter: Optional[str],
    coarse_part_type_filter: Optional[str],
    decision_source_filter: Optional[str],
    is_coarse_label_filter: Optional[bool],
    *,
    item_cls: type[Any],
    response_cls: type[Any],
    build_filter_conditions_fn: Callable[..., dict[str, Any]],
    extract_label_contract_fn: Callable[[dict[str, Any]], dict[str, Any]],
) -> Any:
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
    items: list[Any] = []
    for result in results:
        meta = result.metadata or {}
        label_contract = extract_label_contract_fn(meta)
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


__all__ = ["list_vectors_qdrant"]
