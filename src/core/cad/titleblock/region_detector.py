"""
Titleblock region detection.

Provides:
- Automatic titleblock region detection
- Multiple detection methods
- Boundary calculation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DetectionMethod(str, Enum):
    """Method used for region detection."""
    CORNER_BASED = "corner_based"  # Bottom-right corner detection
    FRAME_BASED = "frame_based"  # Frame/border detection
    TEXT_DENSITY = "text_density"  # Text density analysis
    TEMPLATE = "template"  # Template matching


@dataclass
class BoundingBox:
    """Bounding box for a region."""
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
    def center(self) -> Tuple[float, float]:
        return ((self.min_x + self.max_x) / 2, (self.min_y + self.max_y) / 2)

    def contains_point(self, x: float, y: float) -> bool:
        """Check if point is inside bounding box."""
        return self.min_x <= x <= self.max_x and self.min_y <= y <= self.max_y

    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if overlaps with another bounding box."""
        return not (
            self.max_x < other.min_x or
            self.min_x > other.max_x or
            self.max_y < other.min_y or
            self.min_y > other.max_y
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "min_x": self.min_x,
            "min_y": self.min_y,
            "max_x": self.max_x,
            "max_y": self.max_y,
            "width": self.width,
            "height": self.height,
        }


@dataclass
class TitleblockRegion:
    """Detected titleblock region."""
    bounds: BoundingBox
    method: DetectionMethod
    confidence: float
    entities_inside: int = 0
    text_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bounds": self.bounds.to_dict(),
            "method": self.method.value,
            "confidence": round(self.confidence, 4),
            "entities_inside": self.entities_inside,
            "text_count": self.text_count,
            "metadata": self.metadata,
        }


class RegionDetector:
    """
    Detector for titleblock regions in CAD drawings.

    Supports multiple detection methods:
    - Corner-based: Assumes titleblock in bottom-right
    - Frame-based: Detects rectangular frames
    - Text density: Finds high text density regions
    """

    def __init__(
        self,
        default_width_ratio: float = 0.4,
        default_height_ratio: float = 0.25,
        min_text_entities: int = 3,
    ):
        """
        Initialize region detector.

        Args:
            default_width_ratio: Default titleblock width as ratio of drawing
            default_height_ratio: Default titleblock height as ratio of drawing
            min_text_entities: Minimum text entities for validation
        """
        self._default_width_ratio = default_width_ratio
        self._default_height_ratio = default_height_ratio
        self._min_text_entities = min_text_entities

    def detect(
        self,
        entities: List[Any],
        drawing_bounds: Optional[BoundingBox] = None,
        method: DetectionMethod = DetectionMethod.CORNER_BASED,
    ) -> Optional[TitleblockRegion]:
        """
        Detect titleblock region.

        Args:
            entities: List of DXF entities
            drawing_bounds: Overall drawing bounds
            method: Detection method to use

        Returns:
            TitleblockRegion or None if not found
        """
        # Calculate drawing bounds if not provided
        if drawing_bounds is None:
            drawing_bounds = self._calculate_drawing_bounds(entities)
            if drawing_bounds is None:
                return None

        if method == DetectionMethod.CORNER_BASED:
            return self._detect_corner_based(entities, drawing_bounds)
        elif method == DetectionMethod.FRAME_BASED:
            return self._detect_frame_based(entities, drawing_bounds)
        elif method == DetectionMethod.TEXT_DENSITY:
            return self._detect_text_density(entities, drawing_bounds)
        else:
            # Try multiple methods and return best result
            return self._detect_auto(entities, drawing_bounds)

    def _calculate_drawing_bounds(self, entities: List[Any]) -> Optional[BoundingBox]:
        """Calculate overall drawing bounds from entities."""
        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        has_geometry = False

        for entity in entities:
            try:
                entity_type = entity.dxftype() if hasattr(entity, "dxftype") else ""

                if entity_type == "LINE":
                    start = entity.dxf.start
                    end = entity.dxf.end
                    min_x = min(min_x, start.x, end.x)
                    max_x = max(max_x, start.x, end.x)
                    min_y = min(min_y, start.y, end.y)
                    max_y = max(max_y, start.y, end.y)
                    has_geometry = True

                elif entity_type == "CIRCLE":
                    center = entity.dxf.center
                    radius = entity.dxf.radius
                    min_x = min(min_x, center.x - radius)
                    max_x = max(max_x, center.x + radius)
                    min_y = min(min_y, center.y - radius)
                    max_y = max(max_y, center.y + radius)
                    has_geometry = True

                elif entity_type in ("TEXT", "MTEXT"):
                    pos = entity.dxf.insert if hasattr(entity.dxf, "insert") else None
                    if pos:
                        min_x = min(min_x, pos.x)
                        max_x = max(max_x, pos.x)
                        min_y = min(min_y, pos.y)
                        max_y = max(max_y, pos.y)
                        has_geometry = True

                elif entity_type == "LWPOLYLINE":
                    for point in entity.get_points():
                        min_x = min(min_x, point[0])
                        max_x = max(max_x, point[0])
                        min_y = min(min_y, point[1])
                        max_y = max(max_y, point[1])
                        has_geometry = True

            except Exception as e:
                logger.debug(f"Error processing entity for bounds: {e}")
                continue

        if not has_geometry:
            return None

        return BoundingBox(min_x=min_x, min_y=min_y, max_x=max_x, max_y=max_y)

    def _detect_corner_based(
        self,
        entities: List[Any],
        drawing_bounds: BoundingBox,
    ) -> TitleblockRegion:
        """
        Detect titleblock using bottom-right corner assumption.

        Standard engineering drawings typically have titleblock in bottom-right.
        """
        # Calculate expected titleblock region
        tb_width = drawing_bounds.width * self._default_width_ratio
        tb_height = drawing_bounds.height * self._default_height_ratio

        region_bounds = BoundingBox(
            min_x=drawing_bounds.max_x - tb_width,
            min_y=drawing_bounds.min_y,
            max_x=drawing_bounds.max_x,
            max_y=drawing_bounds.min_y + tb_height,
        )

        # Count entities in region
        entities_inside, text_count = self._count_entities_in_region(entities, region_bounds)

        # Calculate confidence based on text density
        confidence = min(1.0, text_count / self._min_text_entities) * 0.7

        return TitleblockRegion(
            bounds=region_bounds,
            method=DetectionMethod.CORNER_BASED,
            confidence=confidence,
            entities_inside=entities_inside,
            text_count=text_count,
        )

    def _detect_frame_based(
        self,
        entities: List[Any],
        drawing_bounds: BoundingBox,
    ) -> Optional[TitleblockRegion]:
        """
        Detect titleblock by finding rectangular frames.

        Looks for closed rectangles that could be titleblock borders.
        """
        # Find all rectangles/closed polylines
        rectangles = []

        for entity in entities:
            try:
                entity_type = entity.dxftype() if hasattr(entity, "dxftype") else ""

                if entity_type == "LWPOLYLINE":
                    if entity.is_closed:
                        points = list(entity.get_points())
                        if len(points) >= 4:
                            # Calculate bounding box
                            xs = [p[0] for p in points]
                            ys = [p[1] for p in points]
                            rect = BoundingBox(
                                min_x=min(xs),
                                min_y=min(ys),
                                max_x=max(xs),
                                max_y=max(ys),
                            )

                            # Check if it's in the typical titleblock location
                            if self._is_titleblock_candidate(rect, drawing_bounds):
                                rectangles.append(rect)

            except Exception as e:
                logger.debug(f"Error processing entity for frame detection: {e}")
                continue

        if not rectangles:
            return None

        # Find the best candidate (largest in bottom-right area)
        best_rect = max(rectangles, key=lambda r: r.area)

        entities_inside, text_count = self._count_entities_in_region(entities, best_rect)

        confidence = 0.8 if text_count >= self._min_text_entities else 0.5

        return TitleblockRegion(
            bounds=best_rect,
            method=DetectionMethod.FRAME_BASED,
            confidence=confidence,
            entities_inside=entities_inside,
            text_count=text_count,
        )

    def _detect_text_density(
        self,
        entities: List[Any],
        drawing_bounds: BoundingBox,
    ) -> Optional[TitleblockRegion]:
        """
        Detect titleblock using text density analysis.

        Titleblocks typically have higher text density than drawing area.
        """
        # Collect all text positions
        text_positions = []

        for entity in entities:
            try:
                entity_type = entity.dxftype() if hasattr(entity, "dxftype") else ""

                if entity_type in ("TEXT", "MTEXT"):
                    pos = entity.dxf.insert if hasattr(entity.dxf, "insert") else None
                    if pos:
                        text_positions.append((pos.x, pos.y))

                elif entity_type == "ATTRIB":
                    pos = entity.dxf.insert if hasattr(entity.dxf, "insert") else None
                    if pos:
                        text_positions.append((pos.x, pos.y))

            except Exception:
                continue

        if len(text_positions) < self._min_text_entities:
            return None

        # Cluster text positions to find dense regions
        # Simple approach: find region with most text in bottom-right quadrant
        quadrant_bounds = BoundingBox(
            min_x=drawing_bounds.min_x + drawing_bounds.width * 0.5,
            min_y=drawing_bounds.min_y,
            max_x=drawing_bounds.max_x,
            max_y=drawing_bounds.min_y + drawing_bounds.height * 0.5,
        )

        texts_in_quadrant = [
            (x, y) for x, y in text_positions
            if quadrant_bounds.contains_point(x, y)
        ]

        if len(texts_in_quadrant) < self._min_text_entities:
            return None

        # Calculate tight bounds around text cluster
        xs = [p[0] for p in texts_in_quadrant]
        ys = [p[1] for p in texts_in_quadrant]

        # Add padding
        padding_x = (max(xs) - min(xs)) * 0.1
        padding_y = (max(ys) - min(ys)) * 0.1

        region_bounds = BoundingBox(
            min_x=min(xs) - padding_x,
            min_y=min(ys) - padding_y,
            max_x=max(xs) + padding_x,
            max_y=max(ys) + padding_y,
        )

        entities_inside, text_count = self._count_entities_in_region(entities, region_bounds)

        confidence = min(1.0, text_count / 10) * 0.75

        return TitleblockRegion(
            bounds=region_bounds,
            method=DetectionMethod.TEXT_DENSITY,
            confidence=confidence,
            entities_inside=entities_inside,
            text_count=text_count,
        )

    def _detect_auto(
        self,
        entities: List[Any],
        drawing_bounds: BoundingBox,
    ) -> Optional[TitleblockRegion]:
        """
        Try multiple detection methods and return best result.
        """
        candidates = []

        # Try frame-based first (most accurate if successful)
        frame_result = self._detect_frame_based(entities, drawing_bounds)
        if frame_result:
            candidates.append(frame_result)

        # Try text density
        text_result = self._detect_text_density(entities, drawing_bounds)
        if text_result:
            candidates.append(text_result)

        # Always include corner-based as fallback
        corner_result = self._detect_corner_based(entities, drawing_bounds)
        candidates.append(corner_result)

        if not candidates:
            return None

        # Return highest confidence result
        return max(candidates, key=lambda r: r.confidence)

    def _is_titleblock_candidate(
        self,
        rect: BoundingBox,
        drawing_bounds: BoundingBox,
    ) -> bool:
        """Check if rectangle is a potential titleblock."""
        # Should be in bottom-right area
        center_x, center_y = rect.center
        if center_x < drawing_bounds.min_x + drawing_bounds.width * 0.5:
            return False
        if center_y > drawing_bounds.min_y + drawing_bounds.height * 0.5:
            return False

        # Size constraints
        width_ratio = rect.width / drawing_bounds.width
        height_ratio = rect.height / drawing_bounds.height

        if width_ratio < 0.1 or width_ratio > 0.6:
            return False
        if height_ratio < 0.05 or height_ratio > 0.4:
            return False

        return True

    def _count_entities_in_region(
        self,
        entities: List[Any],
        region: BoundingBox,
    ) -> Tuple[int, int]:
        """Count entities and text entities in a region."""
        total_count = 0
        text_count = 0

        for entity in entities:
            try:
                entity_type = entity.dxftype() if hasattr(entity, "dxftype") else ""

                # Get entity position
                pos = None
                if entity_type in ("TEXT", "MTEXT", "ATTRIB"):
                    pos = entity.dxf.insert if hasattr(entity.dxf, "insert") else None
                elif entity_type == "LINE":
                    start = entity.dxf.start
                    end = entity.dxf.end
                    # Use midpoint
                    pos_x = (start.x + end.x) / 2
                    pos_y = (start.y + end.y) / 2
                    if region.contains_point(pos_x, pos_y):
                        total_count += 1
                    continue
                elif entity_type == "CIRCLE":
                    pos = entity.dxf.center

                if pos and region.contains_point(pos.x, pos.y):
                    total_count += 1
                    if entity_type in ("TEXT", "MTEXT", "ATTRIB"):
                        text_count += 1

            except Exception:
                continue

        return total_count, text_count
