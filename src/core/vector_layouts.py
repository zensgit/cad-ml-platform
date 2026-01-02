"""Vector layout identifiers and helpers."""

from __future__ import annotations

from typing import Optional

VECTOR_LAYOUT_BASE = "base_sem_ext_v1"
VECTOR_LAYOUT_L3 = "base_sem_ext_v1+l3"
VECTOR_LAYOUT_LEGACY = "geom_all_sem_v1"


def layout_has_l3(layout: Optional[str]) -> bool:
    if not layout:
        return False
    return layout.endswith("+l3")


__all__ = [
    "VECTOR_LAYOUT_BASE",
    "VECTOR_LAYOUT_L3",
    "VECTOR_LAYOUT_LEGACY",
    "layout_has_l3",
]
