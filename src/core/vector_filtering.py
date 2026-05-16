from __future__ import annotations

from typing import Any, Dict, Optional


def build_vector_filter_conditions(
    *,
    material_filter: Optional[str],
    complexity_filter: Optional[str],
    fine_part_type_filter: Optional[str],
    coarse_part_type_filter: Optional[str],
    decision_source_filter: Optional[str],
    is_coarse_label_filter: Optional[bool],
) -> Dict[str, Any]:
    conditions: Dict[str, Any] = {}
    if material_filter:
        conditions["material"] = material_filter
    if complexity_filter:
        conditions["complexity"] = complexity_filter
    if fine_part_type_filter:
        conditions["fine_part_type"] = fine_part_type_filter
    if coarse_part_type_filter:
        conditions["coarse_part_type"] = coarse_part_type_filter
    if decision_source_filter:
        conditions["decision_source"] = decision_source_filter
    if is_coarse_label_filter is not None:
        conditions["is_coarse_label"] = is_coarse_label_filter
    return conditions


def build_vector_search_filter_conditions(payload: Any) -> Dict[str, Any]:
    return build_vector_filter_conditions(
        material_filter=payload.material_filter,
        complexity_filter=payload.complexity_filter,
        fine_part_type_filter=payload.fine_part_type_filter,
        coarse_part_type_filter=payload.coarse_part_type_filter,
        decision_source_filter=payload.decision_source_filter,
        is_coarse_label_filter=payload.is_coarse_label_filter,
    )


def vector_item_payload(
    vector_id: str,
    dimension: int,
    meta: Dict[str, Any],
    label_contract: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": vector_id,
        "dimension": dimension,
        "material": meta.get("material"),
        "complexity": meta.get("complexity"),
        "format": meta.get("format"),
        "part_type": label_contract.get("part_type"),
        "fine_part_type": label_contract.get("fine_part_type"),
        "coarse_part_type": label_contract.get("coarse_part_type"),
        "decision_source": label_contract.get("decision_source"),
        "is_coarse_label": label_contract.get("is_coarse_label"),
    }


def matches_vector_label_filters(
    *,
    material_filter: Optional[str],
    complexity_filter: Optional[str],
    fine_part_type_filter: Optional[str],
    coarse_part_type_filter: Optional[str],
    decision_source_filter: Optional[str],
    is_coarse_label_filter: Optional[bool],
    meta: Dict[str, Any],
    label_contract: Dict[str, Any],
) -> bool:
    if material_filter and meta.get("material") != material_filter:
        return False
    if complexity_filter and meta.get("complexity") != complexity_filter:
        return False
    if (
        fine_part_type_filter
        and label_contract.get("fine_part_type") != fine_part_type_filter
    ):
        return False
    if (
        coarse_part_type_filter
        and label_contract.get("coarse_part_type") != coarse_part_type_filter
    ):
        return False
    if (
        decision_source_filter
        and label_contract.get("decision_source") != decision_source_filter
    ):
        return False
    if (
        is_coarse_label_filter is not None
        and label_contract.get("is_coarse_label") is not is_coarse_label_filter
    ):
        return False
    return True


def matches_vector_search_filters(
    payload: Any,
    meta: Dict[str, Any],
    label_contract: Dict[str, Any],
) -> bool:
    return matches_vector_label_filters(
        material_filter=payload.material_filter,
        complexity_filter=payload.complexity_filter,
        fine_part_type_filter=payload.fine_part_type_filter,
        coarse_part_type_filter=payload.coarse_part_type_filter,
        decision_source_filter=payload.decision_source_filter,
        is_coarse_label_filter=payload.is_coarse_label_filter,
        meta=meta,
        label_contract=label_contract,
    )


__all__ = [
    "build_vector_filter_conditions",
    "build_vector_search_filter_conditions",
    "matches_vector_label_filters",
    "matches_vector_search_filters",
    "vector_item_payload",
]
