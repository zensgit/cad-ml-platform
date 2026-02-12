"""
Spatial indexing for efficient geometric queries.

Uses R-tree or grid-based indexing for fast spatial lookups.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


@dataclass
class SpatialBounds:
    """Bounding box for spatial queries."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float

    def contains_point(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def intersects(self, other: "SpatialBounds") -> bool:
        return not (
            self.max_x < other.min_x or
            self.min_x > other.max_x or
            self.max_y < other.min_y or
            self.min_y > other.max_y
        )

    def contains_bounds(self, other: "SpatialBounds") -> bool:
        return (
            self.min_x <= other.min_x and
            self.max_x >= other.max_x and
            self.min_y <= other.min_y and
            self.max_y >= other.max_y
        )

    @property
    def width(self) -> float:
        return self.max_x - self.min_x

    @property
    def height(self) -> float:
        return self.max_y - self.min_y

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    def expand(self, margin: float) -> "SpatialBounds":
        return SpatialBounds(
            self.min_x - margin,
            self.min_y - margin,
            self.max_x + margin,
            self.max_y + margin,
        )

    def union(self, other: "SpatialBounds") -> "SpatialBounds":
        return SpatialBounds(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
        )

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.min_x, self.min_y, self.max_x, self.max_y)

    @classmethod
    def from_center(cls, cx: float, cy: float, width: float, height: float) -> "SpatialBounds":
        hw, hh = width / 2, height / 2
        return cls(cx - hw, cy - hh, cx + hw, cy + hh)


@dataclass
class SpatialEntry:
    """An entry in the spatial index."""
    entity_id: str
    bounds: SpatialBounds
    data: Any = None


class SpatialIndex(ABC):
    """Abstract base class for spatial indices."""

    @abstractmethod
    def insert(self, entity_id: str, bounds: SpatialBounds, data: Any = None) -> None:
        """Insert an entity into the index."""
        pass

    @abstractmethod
    def delete(self, entity_id: str) -> bool:
        """Delete an entity from the index."""
        pass

    @abstractmethod
    def query_point(self, x: float, y: float) -> List[SpatialEntry]:
        """Find all entries containing a point."""
        pass

    @abstractmethod
    def query_bounds(self, bounds: SpatialBounds) -> List[SpatialEntry]:
        """Find all entries intersecting a bounding box."""
        pass

    @abstractmethod
    def query_radius(self, cx: float, cy: float, radius: float) -> List[SpatialEntry]:
        """Find all entries within a radius."""
        pass

    @abstractmethod
    def nearest(self, x: float, y: float, k: int = 1) -> List[Tuple[SpatialEntry, float]]:
        """Find k nearest entries to a point."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear the index."""
        pass

    @property
    @abstractmethod
    def count(self) -> int:
        """Number of entries in the index."""
        pass


class GridIndex(SpatialIndex):
    """
    Grid-based spatial index.

    Simple but effective for uniformly distributed entities.
    """

    def __init__(
        self,
        cell_size: float = 100.0,
        bounds: Optional[SpatialBounds] = None,
    ):
        self.cell_size = cell_size
        self.bounds = bounds
        self._grid: Dict[Tuple[int, int], List[SpatialEntry]] = {}
        self._entries: Dict[str, SpatialEntry] = {}
        self._entry_cells: Dict[str, Set[Tuple[int, int]]] = {}

    def _get_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Get cell coordinates for a point."""
        return (int(x // self.cell_size), int(y // self.cell_size))

    def _get_cells_for_bounds(self, bounds: SpatialBounds) -> Iterator[Tuple[int, int]]:
        """Get all cells that intersect with bounds."""
        min_cell = self._get_cell(bounds.min_x, bounds.min_y)
        max_cell = self._get_cell(bounds.max_x, bounds.max_y)

        for i in range(min_cell[0], max_cell[0] + 1):
            for j in range(min_cell[1], max_cell[1] + 1):
                yield (i, j)

    def insert(self, entity_id: str, bounds: SpatialBounds, data: Any = None) -> None:
        """Insert an entity."""
        entry = SpatialEntry(entity_id, bounds, data)
        self._entries[entity_id] = entry

        # Add to relevant cells
        cells = set(self._get_cells_for_bounds(bounds))
        self._entry_cells[entity_id] = cells

        for cell in cells:
            if cell not in self._grid:
                self._grid[cell] = []
            self._grid[cell].append(entry)

    def delete(self, entity_id: str) -> bool:
        """Delete an entity."""
        if entity_id not in self._entries:
            return False

        entry = self._entries.pop(entity_id)
        cells = self._entry_cells.pop(entity_id, set())

        for cell in cells:
            if cell in self._grid:
                self._grid[cell] = [e for e in self._grid[cell] if e.entity_id != entity_id]
                if not self._grid[cell]:
                    del self._grid[cell]

        return True

    def query_point(self, x: float, y: float) -> List[SpatialEntry]:
        """Find entries containing a point."""
        cell = self._get_cell(x, y)
        if cell not in self._grid:
            return []

        return [
            entry for entry in self._grid[cell]
            if entry.bounds.contains_point(x, y)
        ]

    def query_bounds(self, bounds: SpatialBounds) -> List[SpatialEntry]:
        """Find entries intersecting bounds."""
        result = []
        seen = set()

        for cell in self._get_cells_for_bounds(bounds):
            if cell in self._grid:
                for entry in self._grid[cell]:
                    if entry.entity_id not in seen and entry.bounds.intersects(bounds):
                        result.append(entry)
                        seen.add(entry.entity_id)

        return result

    def query_radius(self, cx: float, cy: float, radius: float) -> List[SpatialEntry]:
        """Find entries within radius."""
        # Query bounding box first
        bounds = SpatialBounds(cx - radius, cy - radius, cx + radius, cy + radius)
        candidates = self.query_bounds(bounds)

        # Filter by actual distance
        result = []
        for entry in candidates:
            # Check distance to center of entry bounds
            ecx, ecy = entry.bounds.center
            dist = math.sqrt((ecx - cx) ** 2 + (ecy - cy) ** 2)
            if dist <= radius:
                result.append(entry)

        return result

    def nearest(self, x: float, y: float, k: int = 1) -> List[Tuple[SpatialEntry, float]]:
        """Find k nearest entries."""
        if not self._entries:
            return []

        # Compute distances to all entries
        distances = []
        for entry in self._entries.values():
            cx, cy = entry.bounds.center
            dist = math.sqrt((cx - x) ** 2 + (cy - y) ** 2)
            distances.append((entry, dist))

        # Sort by distance
        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def clear(self) -> None:
        """Clear the index."""
        self._grid.clear()
        self._entries.clear()
        self._entry_cells.clear()

    @property
    def count(self) -> int:
        return len(self._entries)


class RTreeIndex(SpatialIndex):
    """
    R-tree spatial index using rtree library if available.

    Falls back to GridIndex if rtree is not installed.
    """

    def __init__(self):
        self._use_rtree = False
        self._index = None
        self._entries: Dict[str, SpatialEntry] = {}
        self._id_counter = 0
        self._id_to_entity: Dict[int, str] = {}
        self._entity_to_id: Dict[str, int] = {}

        try:
            from rtree import index
            self._rtree_index = index
            p = index.Property()
            p.dimension = 2
            self._index = index.Index(properties=p)
            self._use_rtree = True
            logger.debug("Using rtree for spatial indexing")
        except ImportError:
            logger.debug("rtree not available, using GridIndex fallback")
            self._grid_fallback = GridIndex()

    def insert(self, entity_id: str, bounds: SpatialBounds, data: Any = None) -> None:
        """Insert an entity."""
        entry = SpatialEntry(entity_id, bounds, data)
        self._entries[entity_id] = entry

        if self._use_rtree:
            # rtree needs numeric IDs
            numeric_id = self._id_counter
            self._id_counter += 1
            self._id_to_entity[numeric_id] = entity_id
            self._entity_to_id[entity_id] = numeric_id
            self._index.insert(numeric_id, bounds.to_tuple())
        else:
            self._grid_fallback.insert(entity_id, bounds, data)

    def delete(self, entity_id: str) -> bool:
        """Delete an entity."""
        if entity_id not in self._entries:
            return False

        entry = self._entries.pop(entity_id)

        if self._use_rtree:
            numeric_id = self._entity_to_id.pop(entity_id, None)
            if numeric_id is not None:
                del self._id_to_entity[numeric_id]
                self._index.delete(numeric_id, entry.bounds.to_tuple())
            return True
        else:
            return self._grid_fallback.delete(entity_id)

    def query_point(self, x: float, y: float) -> List[SpatialEntry]:
        """Find entries containing a point."""
        if self._use_rtree:
            hits = list(self._index.intersection((x, y, x, y)))
            return [
                self._entries[self._id_to_entity[hit]]
                for hit in hits
                if hit in self._id_to_entity
            ]
        else:
            return self._grid_fallback.query_point(x, y)

    def query_bounds(self, bounds: SpatialBounds) -> List[SpatialEntry]:
        """Find entries intersecting bounds."""
        if self._use_rtree:
            hits = list(self._index.intersection(bounds.to_tuple()))
            return [
                self._entries[self._id_to_entity[hit]]
                for hit in hits
                if hit in self._id_to_entity
            ]
        else:
            return self._grid_fallback.query_bounds(bounds)

    def query_radius(self, cx: float, cy: float, radius: float) -> List[SpatialEntry]:
        """Find entries within radius."""
        bounds = SpatialBounds(cx - radius, cy - radius, cx + radius, cy + radius)
        candidates = self.query_bounds(bounds)

        result = []
        for entry in candidates:
            ecx, ecy = entry.bounds.center
            dist = math.sqrt((ecx - cx) ** 2 + (ecy - cy) ** 2)
            if dist <= radius:
                result.append(entry)

        return result

    def nearest(self, x: float, y: float, k: int = 1) -> List[Tuple[SpatialEntry, float]]:
        """Find k nearest entries."""
        if self._use_rtree:
            hits = list(self._index.nearest((x, y, x, y), k))
            result = []
            for hit in hits:
                if hit in self._id_to_entity:
                    entry = self._entries[self._id_to_entity[hit]]
                    cx, cy = entry.bounds.center
                    dist = math.sqrt((cx - x) ** 2 + (cy - y) ** 2)
                    result.append((entry, dist))
            return sorted(result, key=lambda x: x[1])[:k]
        else:
            return self._grid_fallback.nearest(x, y, k)

    def clear(self) -> None:
        """Clear the index."""
        self._entries.clear()
        self._id_to_entity.clear()
        self._entity_to_id.clear()
        self._id_counter = 0

        if self._use_rtree:
            from rtree import index
            p = index.Property()
            p.dimension = 2
            self._index = index.Index(properties=p)
        else:
            self._grid_fallback.clear()

    @property
    def count(self) -> int:
        return len(self._entries)


class SpatialQuery:
    """
    High-level spatial query interface.

    Provides convenient methods for common spatial queries.
    """

    def __init__(self, index: Optional[SpatialIndex] = None):
        self._index = index or RTreeIndex()

    def index_entities(self, entities: Iterator[Any]) -> int:
        """Index all entities from a DXF modelspace."""
        count = 0
        for entity in entities:
            bounds = self._get_entity_bounds(entity)
            if bounds:
                entity_id = self._get_entity_id(entity)
                self._index.insert(entity_id, bounds, entity)
                count += 1
        return count

    def find_at_point(self, x: float, y: float) -> List[Any]:
        """Find entities at a point."""
        return [e.data for e in self._index.query_point(x, y) if e.data]

    def find_in_region(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float,
    ) -> List[Any]:
        """Find entities in a rectangular region."""
        bounds = SpatialBounds(min_x, min_y, max_x, max_y)
        return [e.data for e in self._index.query_bounds(bounds) if e.data]

    def find_near(self, x: float, y: float, radius: float) -> List[Any]:
        """Find entities within radius of a point."""
        return [e.data for e in self._index.query_radius(x, y, radius) if e.data]

    def find_nearest(self, x: float, y: float, k: int = 1) -> List[Tuple[Any, float]]:
        """Find k nearest entities to a point."""
        return [(e.data, dist) for e, dist in self._index.nearest(x, y, k) if e.data]

    def _get_entity_id(self, entity: Any) -> str:
        """Get unique ID for entity."""
        if hasattr(entity.dxf, "handle"):
            return str(entity.dxf.handle)
        return str(id(entity))

    def _get_entity_bounds(self, entity: Any) -> Optional[SpatialBounds]:
        """Get bounding box for entity."""
        dxf_type = entity.dxftype()

        try:
            if dxf_type == "LINE":
                start = entity.dxf.start
                end = entity.dxf.end
                return SpatialBounds(
                    min(start.x, end.x),
                    min(start.y, end.y),
                    max(start.x, end.x),
                    max(start.y, end.y),
                )

            elif dxf_type == "CIRCLE":
                center = entity.dxf.center
                radius = entity.dxf.radius
                return SpatialBounds(
                    center.x - radius,
                    center.y - radius,
                    center.x + radius,
                    center.y + radius,
                )

            elif dxf_type == "ARC":
                center = entity.dxf.center
                radius = entity.dxf.radius
                return SpatialBounds(
                    center.x - radius,
                    center.y - radius,
                    center.x + radius,
                    center.y + radius,
                )

            elif dxf_type in ("POLYLINE", "LWPOLYLINE"):
                if hasattr(entity, 'get_points'):
                    points = list(entity.get_points())
                    if points:
                        xs = [p[0] for p in points]
                        ys = [p[1] for p in points]
                        return SpatialBounds(min(xs), min(ys), max(xs), max(ys))

            elif dxf_type == "POINT":
                loc = entity.dxf.location
                return SpatialBounds(loc.x, loc.y, loc.x, loc.y)

            elif dxf_type in ("TEXT", "MTEXT"):
                if hasattr(entity.dxf, "insert"):
                    pt = entity.dxf.insert
                    return SpatialBounds(pt.x, pt.y, pt.x, pt.y)

            elif dxf_type == "INSERT":
                pt = entity.dxf.insert
                return SpatialBounds(pt.x, pt.y, pt.x, pt.y)

        except Exception as e:
            logger.debug(f"Could not get bounds for {dxf_type}: {e}")

        return None

    @property
    def count(self) -> int:
        return self._index.count
