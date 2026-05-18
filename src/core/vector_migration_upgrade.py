from __future__ import annotations

from typing import Any

from src.core.vector_layouts import (
    VECTOR_LAYOUT_BASE,
    VECTOR_LAYOUT_LEGACY,
    layout_has_l3,
)
from src.core.vector_migration_config import coerce_optional_int


def prepare_vector_for_upgrade(
    extractor: Any,
    vector: list[float],
    meta: dict[str, Any],
    from_version: str,
) -> tuple[list[float], list[float], str]:
    layout = meta.get("vector_layout") or VECTOR_LAYOUT_BASE
    expected_len = extractor.expected_dim(from_version)
    l3_tail: list[float] = []
    base_vector = vector

    if layout_has_l3(layout):
        l3_dim = coerce_optional_int(meta.get("l3_3d_dim"))
        if l3_dim is None and len(vector) > expected_len:
            l3_dim = len(vector) - expected_len
        if not l3_dim or len(vector) < expected_len + l3_dim:
            raise ValueError("L3 layout length mismatch")
        base_vector = vector[: len(vector) - l3_dim]
        l3_tail = vector[-l3_dim:]

    if layout == VECTOR_LAYOUT_LEGACY:
        base_vector = extractor.reorder_legacy_vector(base_vector, from_version)

    return base_vector, l3_tail, layout


__all__ = ["prepare_vector_for_upgrade"]
