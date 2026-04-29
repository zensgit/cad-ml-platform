from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class VectorListItem(BaseModel):
    id: str
    dimension: int
    material: Optional[str] = None
    complexity: Optional[str] = None
    format: Optional[str] = None
    part_type: Optional[str] = None
    fine_part_type: Optional[str] = None
    coarse_part_type: Optional[str] = None
    decision_source: Optional[str] = None
    is_coarse_label: Optional[bool] = None


class VectorListResponse(BaseModel):
    total: int
    vectors: list[VectorListItem]


__all__ = ["VectorListItem", "VectorListResponse"]

