"""Data models for drawing version diff results.

Provides typed structures for representing entity-level changes between two
revisions of a CAD drawing, including spatial change-region clustering.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


@dataclass
class EntityChange:
    """Single entity-level change between two drawing revisions."""

    entity_type: str
    change_type: str  # "added" | "removed" | "modified"
    location: Tuple[float, float]
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffResult:
    """Aggregate diff output produced by geometry or annotation comparisons."""

    added: List[EntityChange] = field(default_factory=list)
    removed: List[EntityChange] = field(default_factory=list)
    modified: List[EntityChange] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    change_regions: List[Dict[str, Any]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.added and not self.removed and not self.modified


@dataclass
class DiffReport:
    """Envelope that pairs a DiffResult with file provenance and versioning."""

    file_a: str
    file_b: str
    diff_result: DiffResult
    timestamp: float = field(default_factory=time.time)
    format_version: str = "1.0"


__all__ = ["EntityChange", "DiffResult", "DiffReport"]
