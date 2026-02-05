"""DXF feature extraction helpers shared by inference paths."""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def extract_features_v6(dxf_path: str, *, log: Optional[logging.Logger] = None) -> Optional[np.ndarray]:
    """Extract the 48-dim DXF feature vector used by V6/V14/V16 models."""
    active_logger = log or logger
    try:
        import ezdxf

        doc = ezdxf.readfile(dxf_path)
        msp = doc.modelspace()

        entity_types = []
        layer_names = []
        all_points = []
        circle_radii = []
        arc_radii = []
        arc_angles = []
        line_lengths = []
        polyline_vertex_counts = []
        dimension_count = 0
        hatch_count = 0
        block_names = []

        for entity in msp:
            etype = entity.dxftype()
            entity_types.append(etype)

            if hasattr(entity.dxf, "layer"):
                layer_names.append(entity.dxf.layer)

            try:
                if etype == "LINE":
                    start = (entity.dxf.start.x, entity.dxf.start.y)
                    end = (entity.dxf.end.x, entity.dxf.end.y)
                    all_points.extend([start, end])
                    length = np.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
                    line_lengths.append(length)
                elif etype == "CIRCLE":
                    center = (entity.dxf.center.x, entity.dxf.center.y)
                    all_points.append(center)
                    circle_radii.append(entity.dxf.radius)
                elif etype == "ARC":
                    center = (entity.dxf.center.x, entity.dxf.center.y)
                    all_points.append(center)
                    arc_radii.append(entity.dxf.radius)
                    angle = abs(entity.dxf.end_angle - entity.dxf.start_angle)
                    if angle > 180:
                        angle = 360 - angle
                    arc_angles.append(angle)
                elif etype in ["TEXT", "MTEXT"]:
                    if hasattr(entity.dxf, "insert"):
                        all_points.append((entity.dxf.insert.x, entity.dxf.insert.y))
                elif etype in ["LWPOLYLINE", "POLYLINE"]:
                    if hasattr(entity, "get_points"):
                        pts = list(entity.get_points())
                        polyline_vertex_counts.append(len(pts))
                        for pt in pts:
                            all_points.append((pt[0], pt[1]))
                elif etype == "INSERT":
                    if hasattr(entity.dxf, "insert"):
                        all_points.append((entity.dxf.insert.x, entity.dxf.insert.y))
                    if hasattr(entity.dxf, "name"):
                        block_names.append(entity.dxf.name)
                elif etype == "DIMENSION":
                    dimension_count += 1
                elif etype == "HATCH":
                    hatch_count += 1
            except Exception as exc:
                active_logger.debug("DXF entity parse skipped: %s", exc)

        type_counts = Counter(entity_types)
        total_entities = len(entity_types)

        features = []

        # 1-12: entity type ratios
        for etype in [
            "LINE",
            "CIRCLE",
            "ARC",
            "LWPOLYLINE",
            "POLYLINE",
            "SPLINE",
            "ELLIPSE",
            "TEXT",
            "MTEXT",
            "DIMENSION",
            "HATCH",
            "INSERT",
        ]:
            features.append(type_counts.get(etype, 0) / max(total_entities, 1))

        # 13-16: base geometry
        features.append(np.log1p(total_entities) / 10)
        if all_points:
            xs = [p[0] for p in all_points]
            ys = [p[1] for p in all_points]
            width = max(xs) - min(xs)
            height = max(ys) - min(ys)
            features.extend(
                [
                    np.log1p(width) / 10,
                    np.log1p(height) / 10,
                    np.clip(width / max(height, 0.001), 0, 10) / 10,
                ]
            )
        else:
            features.extend([0, 0, 0.5])

        # 17-22: circles/arcs
        if circle_radii:
            features.extend(
                [
                    np.log1p(np.mean(circle_radii)) / 5,
                    np.log1p(np.std(circle_radii)) / 5 if len(circle_radii) > 1 else 0,
                    len(circle_radii) / max(total_entities, 1),
                ]
            )
        else:
            features.extend([0, 0, 0])

        if arc_radii:
            features.extend(
                [
                    np.log1p(np.mean(arc_radii)) / 5,
                    np.mean(arc_angles) / 180 if arc_angles else 0,
                    len(arc_radii) / max(total_entities, 1),
                ]
            )
        else:
            features.extend([0, 0, 0])

        # 23-26: line lengths
        if line_lengths:
            features.extend(
                [
                    np.log1p(np.mean(line_lengths)) / 5,
                    np.log1p(np.std(line_lengths)) / 5 if len(line_lengths) > 1 else 0,
                    np.log1p(np.max(line_lengths)) / 5,
                    np.log1p(np.min(line_lengths)) / 5,
                ]
            )
        else:
            features.extend([0, 0, 0, 0])

        # 27-32: layers
        unique_layers = len(set(layer_names))
        features.append(np.log1p(unique_layers) / 3)
        layer_lower = [l.lower() for l in layer_names]
        features.append(1.0 if any("dim" in l for l in layer_lower) else 0.0)
        features.append(1.0 if any("text" in l for l in layer_lower) else 0.0)
        features.append(1.0 if any("center" in l for l in layer_lower) else 0.0)
        features.append(1.0 if any("hidden" in l or "hid" in l for l in layer_lower) else 0.0)
        features.append(1.0 if any("section" in l or "cut" in l for l in layer_lower) else 0.0)

        # 33-36: complexity
        features.append((type_counts.get("CIRCLE", 0) + type_counts.get("ARC", 0)) / max(total_entities, 1))
        features.append((type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0)) / max(total_entities, 1))
        features.append(type_counts.get("INSERT", 0) / max(total_entities, 1))
        features.append(dimension_count / max(total_entities, 1))

        # 37-40: spatial distribution
        if all_points and len(all_points) > 1:
            xs = np.array([p[0] for p in all_points])
            ys = np.array([p[1] for p in all_points])
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            features.append(np.log1p(len(all_points) / max(area, 0.001)) / 10)
            features.append(np.std(xs) / max(max(xs) - min(xs), 0.001))
            features.append(np.std(ys) / max(max(ys) - min(ys), 0.001))
            center_offset = np.sqrt(
                (np.mean(xs) - (max(xs) + min(xs)) / 2) ** 2
                + (np.mean(ys) - (max(ys) + min(ys)) / 2) ** 2
            )
            features.append(np.log1p(center_offset) / 5)
        else:
            features.extend([0, 0.5, 0.5, 0])

        # 41-44: shape complexity
        if polyline_vertex_counts:
            features.extend(
                [
                    np.log1p(np.mean(polyline_vertex_counts)) / 3,
                    np.log1p(np.max(polyline_vertex_counts)) / 4,
                ]
            )
        else:
            features.extend([0, 0])
        features.append(hatch_count / max(total_entities, 1))
        features.append(np.log1p(len(set(block_names))) / 3)

        # 45-48: type ratios
        curved = type_counts.get("CIRCLE", 0) + type_counts.get("ARC", 0) + type_counts.get("ELLIPSE", 0)
        straight = type_counts.get("LINE", 0)
        features.append(np.clip(curved / max(straight, 1), 0, 5) / 5)
        annotation = type_counts.get("TEXT", 0) + type_counts.get("MTEXT", 0) + dimension_count
        features.append(annotation / max(total_entities, 1))
        geometry = straight + curved + type_counts.get("LWPOLYLINE", 0) + type_counts.get("POLYLINE", 0)
        features.append(np.clip(geometry / max(annotation, 1), 0, 20) / 20)
        features.append(len(circle_radii) / max(total_entities, 1))

        return np.array(features, dtype=np.float32)
    except Exception as exc:
        active_logger.error("DXF feature extraction failed: %s", exc)
        return None
