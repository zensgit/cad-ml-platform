from __future__ import annotations

from collections.abc import Callable
from typing import Any, Optional

from src.core.similarity import extract_vector_label_contract


def list_vectors_memory(
    vector_store: dict[str, list[float]],
    vector_meta: dict[str, dict[str, str]],
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
    matches_label_filters_fn: Callable[..., bool],
    extract_label_contract_fn: Callable[[dict[str, Any]], dict[str, Any]] = (
        extract_vector_label_contract
    ),
) -> Any:
    items: list[Any] = []
    matched_total = 0
    entries = list(vector_store.items())
    for vid, vec in entries:
        meta = vector_meta.get(vid, {})
        label_contract = extract_label_contract_fn(meta)
        if not matches_label_filters_fn(
            material_filter=material_filter,
            complexity_filter=complexity_filter,
            fine_part_type_filter=fine_part_type_filter,
            coarse_part_type_filter=coarse_part_type_filter,
            decision_source_filter=decision_source_filter,
            is_coarse_label_filter=is_coarse_label_filter,
            meta=meta,
            label_contract=label_contract,
        ):
            continue
        matched_total += 1
        if matched_total <= offset:
            continue
        if len(items) < limit:
            items.append(
                item_cls(
                    id=vid,
                    dimension=len(vec),
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
    return response_cls(total=matched_total, vectors=items)


__all__ = ["list_vectors_memory"]
