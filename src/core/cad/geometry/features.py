"""
Geometric feature extraction for CAD files.

Extracts geometric properties from DXF entities for ML analysis.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class GeometryType(str, Enum):
    """Types of geometric entities."""
    POINT = "point"
    LINE = "line"
    CIRCLE = "circle"
    ARC = "arc"
    ELLIPSE = "ellipse"
    SPLINE = "spline"
    POLYLINE = "polyline"
    POLYGON = "polygon"
    TEXT = "text"
    DIMENSION = "dimension"
    HATCH = "hatch"
    BLOCK = "block"
    UNKNOWN = "unknown"


@dataclass
class BoundingBox:
    """2D bounding box."""
    min_x: float
    min_y: float
    max_x: float
    max_y: float

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
    def aspect_ratio(self) -> float:
        if self.height == 0:
            return float('inf')
        return self.width / self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    @property
    def diagonal(self) -> float:
        return math.sqrt(self.width ** 2 + self.height ** 2)

    def contains(self, x: float, y: float) -> bool:
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def intersects(self, other: "BoundingBox") -> bool:
        return not (
            self.max_x < other.min_x or
            self.min_x > other.max_x or
            self.max_y < other.min_y or
            self.min_y > other.max_y
        )

    def union(self, other: "BoundingBox") -> "BoundingBox":
        return BoundingBox(
            min(self.min_x, other.min_x),
            min(self.min_y, other.min_y),
            max(self.max_x, other.max_x),
            max(self.max_y, other.max_y),
        )

    @classmethod
    def from_points(cls, points: List[Tuple[float, float]]) -> "BoundingBox":
        if not points:
            return cls(0, 0, 0, 0)
        xs, ys = zip(*points)
        return cls(min(xs), min(ys), max(xs), max(ys))


@dataclass
class GeometricFeatures:
    """Extracted geometric features from an entity."""
    entity_type: GeometryType
    bbox: Optional[BoundingBox] = None

    # Basic metrics
    length: float = 0.0
    area: float = 0.0
    perimeter: float = 0.0

    # Shape properties
    centroid: Optional[Tuple[float, float]] = None
    orientation: float = 0.0  # Angle in degrees
    curvature: float = 0.0

    # For circles/arcs
    radius: float = 0.0
    start_angle: float = 0.0
    end_angle: float = 0.0

    # For polylines
    vertex_count: int = 0
    is_closed: bool = False
    is_convex: bool = False

    # Raw data
    layer: str = "0"
    color: int = 0
    handle: str = ""

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML."""
        return np.array([
            self.length,
            self.area,
            self.perimeter,
            self.radius,
            self.orientation,
            self.curvature,
            self.vertex_count,
            float(self.is_closed),
            float(self.is_convex),
            self.bbox.width if self.bbox else 0,
            self.bbox.height if self.bbox else 0,
            self.bbox.aspect_ratio if self.bbox else 0,
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_type": self.entity_type.value,
            "bbox": {
                "min_x": self.bbox.min_x,
                "min_y": self.bbox.min_y,
                "max_x": self.bbox.max_x,
                "max_y": self.bbox.max_y,
            } if self.bbox else None,
            "length": self.length,
            "area": self.area,
            "perimeter": self.perimeter,
            "centroid": self.centroid,
            "orientation": self.orientation,
            "curvature": self.curvature,
            "radius": self.radius,
            "vertex_count": self.vertex_count,
            "is_closed": self.is_closed,
            "is_convex": self.is_convex,
            "layer": self.layer,
        }


class GeometryExtractor:
    """
    Extracts geometric features from DXF entities.

    Supports various entity types and provides standardized feature extraction.
    """

    # DXF type to GeometryType mapping
    TYPE_MAP = {
        "POINT": GeometryType.POINT,
        "LINE": GeometryType.LINE,
        "CIRCLE": GeometryType.CIRCLE,
        "ARC": GeometryType.ARC,
        "ELLIPSE": GeometryType.ELLIPSE,
        "SPLINE": GeometryType.SPLINE,
        "POLYLINE": GeometryType.POLYLINE,
        "LWPOLYLINE": GeometryType.POLYLINE,
        "TEXT": GeometryType.TEXT,
        "MTEXT": GeometryType.TEXT,
        "DIMENSION": GeometryType.DIMENSION,
        "LEADER": GeometryType.DIMENSION,
        "HATCH": GeometryType.HATCH,
        "INSERT": GeometryType.BLOCK,
    }

    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance

    def extract(self, entity: Any) -> GeometricFeatures:
        """Extract features from a single entity."""
        dxf_type = entity.dxftype()
        geom_type = self.TYPE_MAP.get(dxf_type, GeometryType.UNKNOWN)

        features = GeometricFeatures(
            entity_type=geom_type,
            layer=getattr(entity.dxf, "layer", "0"),
            color=getattr(entity.dxf, "color", 0),
            handle=str(entity.dxf.handle) if hasattr(entity.dxf, "handle") else "",
        )

        # Type-specific extraction
        extractors = {
            GeometryType.LINE: self._extract_line,
            GeometryType.CIRCLE: self._extract_circle,
            GeometryType.ARC: self._extract_arc,
            GeometryType.ELLIPSE: self._extract_ellipse,
            GeometryType.POLYLINE: self._extract_polyline,
            GeometryType.SPLINE: self._extract_spline,
            GeometryType.POINT: self._extract_point,
        }

        extractor = extractors.get(geom_type)
        if extractor:
            extractor(entity, features)

        return features

    def extract_all(self, entities: Iterator[Any]) -> List[GeometricFeatures]:
        """Extract features from all entities."""
        return [self.extract(e) for e in entities]

    def _extract_line(self, entity: Any, features: GeometricFeatures) -> None:
        """Extract line features."""
        start = entity.dxf.start
        end = entity.dxf.end

        dx = end.x - start.x
        dy = end.y - start.y

        features.length = math.sqrt(dx ** 2 + dy ** 2)
        features.centroid = ((start.x + end.x) / 2, (start.y + end.y) / 2)
        features.orientation = math.degrees(math.atan2(dy, dx))
        features.bbox = BoundingBox(
            min(start.x, end.x),
            min(start.y, end.y),
            max(start.x, end.x),
            max(start.y, end.y),
        )
        features.vertex_count = 2

    def _extract_circle(self, entity: Any, features: GeometricFeatures) -> None:
        """Extract circle features."""
        center = entity.dxf.center
        radius = entity.dxf.radius

        features.radius = radius
        features.centroid = (center.x, center.y)
        features.area = math.pi * radius ** 2
        features.perimeter = 2 * math.pi * radius
        features.curvature = 1 / radius if radius > 0 else 0
        features.is_closed = True
        features.bbox = BoundingBox(
            center.x - radius,
            center.y - radius,
            center.x + radius,
            center.y + radius,
        )

    def _extract_arc(self, entity: Any, features: GeometricFeatures) -> None:
        """Extract arc features."""
        center = entity.dxf.center
        radius = entity.dxf.radius
        start_angle = math.radians(entity.dxf.start_angle)
        end_angle = math.radians(entity.dxf.end_angle)

        # Arc length
        angle_diff = end_angle - start_angle
        if angle_diff < 0:
            angle_diff += 2 * math.pi

        features.radius = radius
        features.start_angle = entity.dxf.start_angle
        features.end_angle = entity.dxf.end_angle
        features.length = radius * angle_diff
        features.curvature = 1 / radius if radius > 0 else 0
        features.centroid = (center.x, center.y)
        features.is_closed = False

        # Bounding box (simplified)
        features.bbox = BoundingBox(
            center.x - radius,
            center.y - radius,
            center.x + radius,
            center.y + radius,
        )

    def _extract_ellipse(self, entity: Any, features: GeometricFeatures) -> None:
        """Extract ellipse features."""
        center = entity.dxf.center
        major_axis = entity.dxf.major_axis
        ratio = entity.dxf.ratio

        # Semi-axes
        a = math.sqrt(major_axis.x ** 2 + major_axis.y ** 2)
        b = a * ratio

        features.centroid = (center.x, center.y)
        features.area = math.pi * a * b
        features.orientation = math.degrees(math.atan2(major_axis.y, major_axis.x))

        # Approximate perimeter (Ramanujan)
        h = ((a - b) ** 2) / ((a + b) ** 2)
        features.perimeter = math.pi * (a + b) * (1 + 3 * h / (10 + math.sqrt(4 - 3 * h)))

        features.is_closed = True
        features.bbox = BoundingBox(
            center.x - a,
            center.y - b,
            center.x + a,
            center.y + b,
        )

    def _extract_polyline(self, entity: Any, features: GeometricFeatures) -> None:
        """Extract polyline features."""
        try:
            if hasattr(entity, 'get_points'):
                points = list(entity.get_points())
            elif hasattr(entity, 'vertices'):
                points = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]
            else:
                points = []
        except Exception:
            points = []

        if not points:
            return

        features.vertex_count = len(points)

        # Length (sum of segment lengths)
        total_length = 0.0
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            total_length += math.sqrt(dx ** 2 + dy ** 2)
        features.length = total_length

        # Check if closed
        if len(points) > 2:
            first, last = points[0], points[-1]
            dist = math.sqrt((first[0] - last[0]) ** 2 + (first[1] - last[1]) ** 2)
            features.is_closed = dist < self.tolerance or getattr(entity.dxf, 'closed', False)

        if features.is_closed:
            features.perimeter = total_length

        # Centroid
        xs, ys = zip(*[(p[0], p[1]) for p in points])
        features.centroid = (sum(xs) / len(xs), sum(ys) / len(ys))

        # Bounding box
        features.bbox = BoundingBox.from_points([(p[0], p[1]) for p in points])

        # Area (for closed polylines using shoelace formula)
        if features.is_closed and len(points) >= 3:
            area = 0.0
            for i in range(len(points)):
                j = (i + 1) % len(points)
                area += points[i][0] * points[j][1]
                area -= points[j][0] * points[i][1]
            features.area = abs(area) / 2

        # Convexity check
        if features.is_closed and len(points) >= 3:
            features.is_convex = self._is_convex([(p[0], p[1]) for p in points])

    def _extract_spline(self, entity: Any, features: GeometricFeatures) -> None:
        """Extract spline features."""
        try:
            control_points = list(entity.control_points)
            features.vertex_count = len(control_points)

            if control_points:
                points = [(p.x, p.y) for p in control_points]
                features.bbox = BoundingBox.from_points(points)

                xs, ys = zip(*points)
                features.centroid = (sum(xs) / len(xs), sum(ys) / len(ys))
        except Exception:
            pass

        features.is_closed = getattr(entity.dxf, 'closed', False)

    def _extract_point(self, entity: Any, features: GeometricFeatures) -> None:
        """Extract point features."""
        loc = entity.dxf.location
        features.centroid = (loc.x, loc.y)
        features.bbox = BoundingBox(loc.x, loc.y, loc.x, loc.y)
        features.vertex_count = 1

    def _is_convex(self, points: List[Tuple[float, float]]) -> bool:
        """Check if polygon is convex."""
        n = len(points)
        if n < 3:
            return False

        sign = None
        for i in range(n):
            p1 = points[i]
            p2 = points[(i + 1) % n]
            p3 = points[(i + 2) % n]

            # Cross product
            cross = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (p3[0] - p2[0])

            if sign is None:
                sign = cross > 0
            elif (cross > 0) != sign:
                return False

        return True


@dataclass
class DrawingStatistics:
    """Statistical features of an entire drawing."""
    total_entities: int = 0
    entity_type_counts: Dict[str, int] = field(default_factory=dict)
    layer_counts: Dict[str, int] = field(default_factory=dict)

    # Geometric stats
    total_length: float = 0.0
    total_area: float = 0.0
    bbox: Optional[BoundingBox] = None

    # Distribution stats
    mean_entity_length: float = 0.0
    std_entity_length: float = 0.0
    mean_entity_area: float = 0.0
    std_entity_area: float = 0.0

    # Complexity indicators
    line_count: int = 0
    circle_count: int = 0
    arc_count: int = 0
    polyline_count: int = 0
    text_count: int = 0
    dimension_count: int = 0
    block_count: int = 0

    # Ratios
    geometry_ratio: float = 0.0  # Lines+arcs+circles / total
    text_ratio: float = 0.0  # Text entities / total
    annotation_ratio: float = 0.0  # Dims+text / total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_entities": self.total_entities,
            "entity_type_counts": self.entity_type_counts,
            "layer_counts": self.layer_counts,
            "total_length": self.total_length,
            "total_area": self.total_area,
            "bbox": self.bbox.to_dict() if self.bbox and hasattr(self.bbox, "to_dict") else None,
            "line_count": self.line_count,
            "circle_count": self.circle_count,
            "arc_count": self.arc_count,
            "polyline_count": self.polyline_count,
            "text_count": self.text_count,
            "dimension_count": self.dimension_count,
            "block_count": self.block_count,
            "geometry_ratio": self.geometry_ratio,
            "text_ratio": self.text_ratio,
            "annotation_ratio": self.annotation_ratio,
        }

    def to_vector(self) -> np.ndarray:
        """Convert to feature vector for ML."""
        return np.array([
            self.total_entities,
            self.total_length,
            self.total_area,
            self.line_count,
            self.circle_count,
            self.arc_count,
            self.polyline_count,
            self.text_count,
            self.dimension_count,
            self.block_count,
            self.geometry_ratio,
            self.text_ratio,
            self.annotation_ratio,
            self.mean_entity_length,
            self.std_entity_length,
            self.bbox.width if self.bbox else 0,
            self.bbox.height if self.bbox else 0,
            self.bbox.aspect_ratio if self.bbox else 0,
        ], dtype=np.float32)


class DrawingAnalyzer:
    """
    Analyzes entire drawings to extract statistical features.
    """

    def __init__(self, extractor: Optional[GeometryExtractor] = None):
        self.extractor = extractor or GeometryExtractor()

    def analyze(self, entities: Iterator[Any]) -> DrawingStatistics:
        """Analyze all entities and compute statistics."""
        stats = DrawingStatistics()

        # Extract features
        features_list = []
        lengths = []
        areas = []
        combined_bbox = None

        for entity in entities:
            features = self.extractor.extract(entity)
            features_list.append(features)
            stats.total_entities += 1

            # Type counts
            type_name = features.entity_type.value
            stats.entity_type_counts[type_name] = stats.entity_type_counts.get(type_name, 0) + 1

            # Layer counts
            stats.layer_counts[features.layer] = stats.layer_counts.get(features.layer, 0) + 1

            # Accumulate metrics
            stats.total_length += features.length
            stats.total_area += features.area
            lengths.append(features.length)
            areas.append(features.area)

            # Expand bounding box
            if features.bbox:
                if combined_bbox is None:
                    combined_bbox = features.bbox
                else:
                    combined_bbox = combined_bbox.union(features.bbox)

            # Type-specific counts
            if features.entity_type == GeometryType.LINE:
                stats.line_count += 1
            elif features.entity_type == GeometryType.CIRCLE:
                stats.circle_count += 1
            elif features.entity_type == GeometryType.ARC:
                stats.arc_count += 1
            elif features.entity_type == GeometryType.POLYLINE:
                stats.polyline_count += 1
            elif features.entity_type == GeometryType.TEXT:
                stats.text_count += 1
            elif features.entity_type == GeometryType.DIMENSION:
                stats.dimension_count += 1
            elif features.entity_type == GeometryType.BLOCK:
                stats.block_count += 1

        stats.bbox = combined_bbox

        # Distribution stats
        if lengths:
            lengths_arr = np.array(lengths)
            stats.mean_entity_length = float(np.mean(lengths_arr))
            stats.std_entity_length = float(np.std(lengths_arr))

        if areas:
            areas_arr = np.array([a for a in areas if a > 0])
            if len(areas_arr) > 0:
                stats.mean_entity_area = float(np.mean(areas_arr))
                stats.std_entity_area = float(np.std(areas_arr))

        # Ratios
        if stats.total_entities > 0:
            geometry_count = stats.line_count + stats.circle_count + stats.arc_count + stats.polyline_count
            stats.geometry_ratio = geometry_count / stats.total_entities
            stats.text_ratio = stats.text_count / stats.total_entities
            stats.annotation_ratio = (stats.text_count + stats.dimension_count) / stats.total_entities

        return stats

    def analyze_file(self, dxf_path: str) -> DrawingStatistics:
        """Analyze a DXF file."""
        try:
            import ezdxf
        except ImportError:
            raise ImportError("ezdxf required: pip install ezdxf")

        doc = ezdxf.readfile(dxf_path)
        return self.analyze(doc.modelspace())
