"""
HATCH Entity Parser

Extracts structured hatch pattern data from DXF files for enhanced comparison.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def parse_hatches(dxf_entities: List[Any]) -> List[Dict[str, Any]]:
    """Parse HATCH entities from DXF file."""
    hatches: List[Dict[str, Any]] = []

    for entity in dxf_entities:
        if entity.dxftype() != "HATCH":
            continue

        try:
            hatch_type = classify_hatch_type(entity)
            pattern_info = extract_pattern_definition(entity)
            boundary_paths = extract_boundary_paths(entity)
            area = compute_hatch_area(boundary_paths)
            fill_color = extract_fill_color(entity)
            hatch_data = {
                "hatch_type": hatch_type,
                "pattern_name": pattern_info["pattern_name"],
                "pattern_scale": pattern_info["pattern_scale"],
                "pattern_angle": pattern_info["pattern_angle"],
                "boundary_paths": boundary_paths,
                "area": area,
                "islands_count": len(boundary_paths) - 1 if len(boundary_paths) > 0 else 0,
                "layer": getattr(entity.dxf, "layer", "0"),
                "color": getattr(entity.dxf, "color", 256),
                "fill_color": fill_color,
            }
            hatches.append(hatch_data)
        except Exception as e:
            logger.warning(f"Failed to parse HATCH entity: {e}")
            continue

    return hatches


def classify_hatch_type(hatch_entity: Any) -> str:
    if hasattr(hatch_entity, "gradient") and hatch_entity.gradient is not None:
        return "gradient"
    pattern_name = getattr(hatch_entity.dxf, "pattern_name", "SOLID")
    if pattern_name and str(pattern_name).upper() != "SOLID":
        return "pattern"
    return "solid"


def extract_pattern_definition(hatch_entity: Any) -> Dict[str, Any]:
    pattern_name = getattr(hatch_entity.dxf, "pattern_name", None)
    pattern_scale = getattr(hatch_entity.dxf, "pattern_scale", 1.0)
    pattern_angle = getattr(hatch_entity.dxf, "pattern_angle", 0.0)
    return {
        "pattern_name": pattern_name
        if pattern_name and str(pattern_name).upper() != "SOLID"
        else None,
        "pattern_scale": pattern_scale,
        "pattern_angle": pattern_angle,
    }


def extract_boundary_paths(hatch_entity: Any) -> List[Dict[str, Any]]:
    boundary_paths: List[Dict[str, Any]] = []
    try:
        paths = getattr(hatch_entity, "paths", [])
        for path in paths:
            path_type = getattr(path, "path_type_flags", 0)
            edges = getattr(path, "edges", [])
            vertices_count = len(edges)
            boundary_paths.append(
                {
                    "type": "external" if path_type & 1 else "internal",
                    "vertices_count": vertices_count,
                    "is_polyline": bool(path_type & 2),
                }
            )
    except Exception as e:
        logger.debug(f"Could not extract boundary paths: {e}")
    return boundary_paths


def compute_hatch_area(boundary_paths: List[Dict[str, Any]]) -> float:
    if not boundary_paths:
        return 0.0
    acc = 0.0
    for p in boundary_paths:
        vc = int(p.get("vertices_count") or 0)
        if vc <= 2:
            continue
        base = max(0.0, vc - 2)
        if p.get("is_polyline"):
            base *= 1.2
        if p.get("type") == "external":
            acc += base
        else:
            acc -= base * 0.8
    return max(0.0, acc)


def extract_fill_color(hatch_entity: Any) -> Optional[Tuple[int, int, int]]:
    try:
        true_color = getattr(hatch_entity.dxf, "true_color", None)
        if true_color:
            r = (true_color >> 16) & 0xFF
            g = (true_color >> 8) & 0xFF
            b = true_color & 0xFF
            return (r, g, b)
    except Exception:
        pass
    return None
