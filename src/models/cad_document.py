"""Unified internal CAD document representation.

Provides a minimal typed structure produced by format adapters (DXF, STL, STEP …)
so downstream feature extraction / classification logic can operate consistently.

Phase 1 scope: lightweight metadata + entity statistics without heavy geometry libs.
Later phases can extend with precise topology / B-Rep / mesh data.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    min_x: float = 0.0
    min_y: float = 0.0
    min_z: float = 0.0
    max_x: float = 0.0
    max_y: float = 0.0
    max_z: float = 0.0

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def depth(self) -> float:
        return self.max_z - self.min_z

    @property
    def volume_estimate(self) -> float:
        return max(self.width, 0.0) * max(self.height, 0.0) * max(self.depth, 0.0)


class CadEntity(BaseModel):
    kind: str = Field(description="Entity type e.g. LINE, CIRCLE, FACET")
    layer: Optional[str] = Field(default=None)
    attributes: Dict[str, Any] = Field(default_factory=dict)


class CadDocument(BaseModel):
    file_name: str
    format: str = Field(description="Canonical lowercase format id (dxf, stl, step, iges)")
    entities: List[CadEntity] = Field(default_factory=list)
    layers: Dict[str, int] = Field(default_factory=dict, description="Layer name -> entity count")
    bounding_box: BoundingBox = Field(default_factory=BoundingBox)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    raw_stats: Dict[str, Any] = Field(default_factory=dict, description="Quick scalar stats for fast access")

    def entity_count(self) -> int:
        return len(self.entities)

    def complexity_bucket(self) -> str:
        ec = self.entity_count()
        if ec < 50:
            return "low"
        if ec < 500:
            return "medium"
        return "high"

    def to_unified_dict(self) -> Dict[str, Any]:
        """Return legacy dict expected by existing analysis endpoint until refactor completes."""
        return {
            "entity_count": self.entity_count(),
            "layer_count": len(self.layers),
            "bounding_box": self.bounding_box.model_dump(),
            "complexity": self.complexity_bucket(),
            "format": self.format,
            "file_name": self.file_name,
            "metadata": self.metadata,
        }


__all__ = ["CadDocument", "CadEntity", "BoundingBox"]

