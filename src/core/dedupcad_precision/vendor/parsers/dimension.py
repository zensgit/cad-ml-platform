"""
DIMENSION Entity Parser

Extracts structured dimension data from DXF files for enhanced comparison.
"""

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def parse_dimensions(dxf_entities: List[Any]) -> List[Dict[str, Any]]:
    """Parse DIMENSION entities from DXF file."""
    dimensions: List[Dict[str, Any]] = []

    for entity in dxf_entities:
        if entity.dxftype() != "DIMENSION":
            continue

        try:
            pts = extract_dimension_points(entity)
            xs = [p[0] for p in pts] if pts else []
            ys = [p[1] for p in pts] if pts else []
            if hasattr(getattr(entity.dxf, "text_midpoint", None), "x"):
                xs.append(getattr(entity.dxf, "text_midpoint").x)
                ys.append(getattr(entity.dxf, "text_midpoint").y)
            bbox = None
            if xs and ys:
                bbox = {
                    "min_x": float(min(xs)),
                    "min_y": float(min(ys)),
                    "max_x": float(max(xs)),
                    "max_y": float(max(ys)),
                }

            dim_data = {
                "dimension_type": classify_dimension_type(entity),
                "measurement_text": extract_dimension_text(entity),
                "actual_measurement": getattr(entity.dxf, "actual_measurement", 0.0),
                "override_text": getattr(entity.dxf, "text", None) or None,
                "points": extract_dimension_points(entity),
                "text_midpoint": (
                    (
                        getattr(entity.dxf, "text_midpoint", (0, 0)).x
                        if hasattr(getattr(entity.dxf, "text_midpoint", None), "x")
                        else 0
                    ),
                    (
                        getattr(entity.dxf, "text_midpoint", (0, 0)).y
                        if hasattr(getattr(entity.dxf, "text_midpoint", None), "y")
                        else 0
                    ),
                ),
                "text_rotation": getattr(entity.dxf, "text_rotation", 0.0),
                "layer": getattr(entity.dxf, "layer", "0"),
                "style": getattr(entity.dxf, "dimstyle", "STANDARD"),
                "bbox": bbox,
                "text_matches_value": _text_matches_value(entity),
            }
            dimensions.append(dim_data)
        except Exception as e:
            logger.warning(f"Failed to parse DIMENSION entity: {e}")
            continue

    return dimensions


def _text_matches_value(dim_entity: Any) -> bool:
    """Heuristic: does displayed text represent actual measurement (within rounding)?"""
    try:
        override = getattr(dim_entity.dxf, "text", None)
        actual = float(getattr(dim_entity.dxf, "actual_measurement", 0.0) or 0.0)
        if not override:
            return True
        import re

        m = re.search(r"([-+]?\d*\.?\d+)", str(override))
        if not m:
            return False
        shown = float(m.group(1))
        if actual <= 1e-9:
            return abs(shown - actual) <= 1e-3
        return abs(shown - actual) / actual <= 0.02  # 2% tolerance
    except Exception:
        return False


def extract_dimension_text(dim_entity: Any) -> str:
    """Extract dimension measurement text (handles overrides)."""
    override = getattr(dim_entity.dxf, "text", None)
    if override:
        return override
    measurement = getattr(dim_entity.dxf, "actual_measurement", 0.0)
    return f"{measurement:.2f}"


def classify_dimension_type(dim_entity: Any) -> str:
    """Classify DIMENSION entity type."""
    dimtype_raw = getattr(dim_entity.dxf, "dimtype", 0)
    dimtype = dimtype_raw & 0x07
    if dimtype in (0, 1):
        return "linear"
    if dimtype in (2, 5):
        return "angular"
    if dimtype in (3, 4):
        return "radial"
    if dimtype == 6:
        return "ordinate"
    return "linear"


def extract_dimension_points(dim_entity: Any) -> List[tuple]:
    """Extract critical points from DIMENSION entity."""
    points = []
    defpoint = getattr(dim_entity.dxf, "defpoint", None)
    if defpoint:
        points.append((defpoint.x, defpoint.y))
    defpoint2 = getattr(dim_entity.dxf, "defpoint2", None)
    if defpoint2:
        points.append((defpoint2.x, defpoint2.y))
    defpoint3 = getattr(dim_entity.dxf, "defpoint3", None)
    if defpoint3:
        points.append((defpoint3.x, defpoint3.y))
    return points

