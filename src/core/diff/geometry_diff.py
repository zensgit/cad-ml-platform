"""Geometry-level diff between two DXF drawing revisions.

Uses ezdxf for parsing, scipy KDTree for spatial entity matching, and numpy
for numeric operations.  Falls back to an empty DiffResult when ezdxf is not
installed so callers never see an import failure.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from src.core.diff.models import DiffResult, EntityChange

logger = logging.getLogger(__name__)

# Distance threshold (drawing units) below which two entity centroids are
# considered candidates for the same entity.
_DEFAULT_MATCH_RADIUS: float = 1.0

# Annotation entity types (used by annotation_diff but excluded here when
# the caller wants geometry-only comparison).
_ANNOTATION_TYPES = frozenset({"TEXT", "MTEXT", "DIMENSION"})

try:
    import ezdxf

    _HAS_EZDXF = True
except ImportError:  # pragma: no cover
    _HAS_EZDXF = False


class GeometryDiff:
    """Compare geometric content of two DXF files and produce a DiffResult."""

    def __init__(self, match_radius: float = _DEFAULT_MATCH_RADIUS) -> None:
        self.match_radius = match_radius

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(self, file_a: str, file_b: str) -> DiffResult:
        """Compare two DXF files and return a structured diff.

        Args:
            file_a: Path to the baseline (old) DXF file.
            file_b: Path to the revised (new) DXF file.

        Returns:
            DiffResult populated with added / removed / modified changes,
            summary counts, and spatial change regions.
        """
        if not _HAS_EZDXF:
            logger.warning("ezdxf not available; returning empty DiffResult")
            return DiffResult(summary={"added": 0, "removed": 0, "modified": 0})

        try:
            doc_a = ezdxf.readfile(file_a)
            doc_b = ezdxf.readfile(file_b)
        except Exception:
            logger.exception("Failed to read DXF files for diff")
            return DiffResult(summary={"added": 0, "removed": 0, "modified": 0})

        entities_a = self._extract_entities(doc_a)
        entities_b = self._extract_entities(doc_b)

        matched, added_indices, removed_indices = self._match_entities(
            entities_a, entities_b
        )

        added: List[EntityChange] = []
        removed: List[EntityChange] = []
        modified: List[EntityChange] = []

        for idx in added_indices:
            ent = entities_b[idx]
            added.append(
                EntityChange(
                    entity_type=ent["dxftype"],
                    change_type="added",
                    location=(ent["cx"], ent["cy"]),
                    details={"layer": ent.get("layer", "0")},
                )
            )

        for idx in removed_indices:
            ent = entities_a[idx]
            removed.append(
                EntityChange(
                    entity_type=ent["dxftype"],
                    change_type="removed",
                    location=(ent["cx"], ent["cy"]),
                    details={"layer": ent.get("layer", "0")},
                )
            )

        for idx_a, idx_b in matched:
            attr_diffs = self._compare_attributes(entities_a[idx_a], entities_b[idx_b])
            if attr_diffs:
                ent_b = entities_b[idx_b]
                modified.append(
                    EntityChange(
                        entity_type=ent_b["dxftype"],
                        change_type="modified",
                        location=(ent_b["cx"], ent_b["cy"]),
                        details={"attribute_changes": attr_diffs},
                    )
                )

        summary = {
            "added": len(added),
            "removed": len(removed),
            "modified": len(modified),
        }

        all_changes = added + removed + modified
        change_regions = self._compute_change_regions(all_changes)

        return DiffResult(
            added=added,
            removed=removed,
            modified=modified,
            summary=summary,
            change_regions=change_regions,
        )

    # ------------------------------------------------------------------
    # Entity extraction
    # ------------------------------------------------------------------

    def _extract_entities(self, doc: Any) -> List[Dict[str, Any]]:
        """Extract entity metadata from an ezdxf document.

        Returns a list of dicts with keys: dxftype, cx, cy, layer, color,
        and type-specific attributes.
        """
        entities: List[Dict[str, Any]] = []
        msp = doc.modelspace()

        for entity in msp:
            dxftype = entity.dxftype()
            cx, cy = self._compute_centroid(entity)

            layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"
            color = entity.dxf.color if hasattr(entity.dxf, "color") else 256

            info: Dict[str, Any] = {
                "dxftype": dxftype,
                "cx": cx,
                "cy": cy,
                "layer": layer,
                "color": color,
            }

            # Capture type-specific attributes for attribute comparison.
            try:
                if dxftype == "LINE":
                    info["start"] = (entity.dxf.start.x, entity.dxf.start.y)
                    info["end"] = (entity.dxf.end.x, entity.dxf.end.y)
                elif dxftype == "CIRCLE":
                    info["radius"] = entity.dxf.radius
                elif dxftype == "ARC":
                    info["radius"] = entity.dxf.radius
                    info["start_angle"] = entity.dxf.start_angle
                    info["end_angle"] = entity.dxf.end_angle
                elif dxftype in ("TEXT", "MTEXT"):
                    text = ""
                    if dxftype == "TEXT" and hasattr(entity.dxf, "text"):
                        text = entity.dxf.text
                    elif dxftype == "MTEXT" and hasattr(entity, "text"):
                        text = entity.text
                    info["text"] = text
                elif dxftype in ("LWPOLYLINE", "POLYLINE"):
                    if hasattr(entity, "get_points"):
                        info["vertex_count"] = len(list(entity.get_points()))
                elif dxftype == "DIMENSION":
                    if hasattr(entity.dxf, "text"):
                        info["text"] = entity.dxf.text
                elif dxftype == "INSERT":
                    if hasattr(entity.dxf, "name"):
                        info["block_name"] = entity.dxf.name
                elif dxftype == "ELLIPSE":
                    if hasattr(entity.dxf, "major_axis"):
                        ma = entity.dxf.major_axis
                        info["major_axis"] = (ma.x, ma.y)
                    if hasattr(entity.dxf, "ratio"):
                        info["ratio"] = entity.dxf.ratio
            except Exception:
                logger.debug("Skipping attribute extraction for %s entity", dxftype)

            entities.append(info)

        return entities

    # ------------------------------------------------------------------
    # Centroid computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_centroid(entity: Any) -> Tuple[float, float]:
        """Compute a representative (cx, cy) point for an entity.

        Different entity types use different strategies:
        - LINE: midpoint
        - CIRCLE / ARC: center
        - TEXT / MTEXT / INSERT: insertion point
        - LWPOLYLINE / POLYLINE: average of vertices
        - DIMENSION: definition point if available, else (0, 0)
        - ELLIPSE: center
        - Others: (0, 0) fallback
        """
        dxftype = entity.dxftype()

        try:
            if dxftype == "LINE":
                sx, sy = entity.dxf.start.x, entity.dxf.start.y
                ex, ey = entity.dxf.end.x, entity.dxf.end.y
                return ((sx + ex) / 2.0, (sy + ey) / 2.0)

            if dxftype in ("CIRCLE", "ARC", "ELLIPSE"):
                return (entity.dxf.center.x, entity.dxf.center.y)

            if dxftype in ("TEXT", "MTEXT", "INSERT"):
                if hasattr(entity.dxf, "insert"):
                    return (entity.dxf.insert.x, entity.dxf.insert.y)
                return (0.0, 0.0)

            if dxftype in ("LWPOLYLINE", "POLYLINE"):
                if hasattr(entity, "get_points"):
                    pts = list(entity.get_points())
                    if pts:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        return (sum(xs) / len(xs), sum(ys) / len(ys))
                return (0.0, 0.0)

            if dxftype == "DIMENSION":
                if hasattr(entity.dxf, "defpoint"):
                    dp = entity.dxf.defpoint
                    return (dp.x, dp.y)
                return (0.0, 0.0)

            if dxftype == "HATCH":
                # Hatches can be complex; use (0,0) as a safe fallback.
                return (0.0, 0.0)

        except Exception:
            logger.debug("Centroid fallback for %s", dxftype)

        return (0.0, 0.0)

    # ------------------------------------------------------------------
    # Spatial matching via KDTree
    # ------------------------------------------------------------------

    def _match_entities(
        self,
        entities_a: List[Dict[str, Any]],
        entities_b: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match entities between two revisions using spatial proximity + type.

        Returns:
            (matched, added_indices, removed_indices)
            - matched: list of (idx_a, idx_b) pairs for entities that exist in
              both revisions (may still have attribute modifications).
            - added_indices: indices into *entities_b* with no match in A.
            - removed_indices: indices into *entities_a* with no match in B.
        """
        if not entities_a and not entities_b:
            return ([], [], [])

        if not entities_a:
            return ([], list(range(len(entities_b))), [])

        if not entities_b:
            return ([], [], list(range(len(entities_a))))

        import numpy as np
        from scipy.spatial import cKDTree

        coords_a = np.array([[e["cx"], e["cy"]] for e in entities_a], dtype=np.float64)
        coords_b = np.array([[e["cx"], e["cy"]] for e in entities_b], dtype=np.float64)

        tree_a = cKDTree(coords_a)

        # For every entity in B, find nearest neighbor in A.
        distances, indices = tree_a.query(coords_b)

        matched: List[Tuple[int, int]] = []
        matched_a_set: set[int] = set()
        matched_b_set: set[int] = set()

        # First pass: greedily match B -> A where distance is within threshold
        # and types agree.
        for idx_b, (dist, idx_a) in enumerate(zip(distances, indices)):
            if dist > self.match_radius:
                continue
            if idx_a in matched_a_set:
                continue
            if entities_a[idx_a]["dxftype"] != entities_b[idx_b]["dxftype"]:
                continue
            matched.append((int(idx_a), idx_b))
            matched_a_set.add(int(idx_a))
            matched_b_set.add(idx_b)

        added_indices = [i for i in range(len(entities_b)) if i not in matched_b_set]
        removed_indices = [i for i in range(len(entities_a)) if i not in matched_a_set]

        return (matched, added_indices, removed_indices)

    # ------------------------------------------------------------------
    # Attribute comparison
    # ------------------------------------------------------------------

    @staticmethod
    def _compare_attributes(
        entity_a: Dict[str, Any], entity_b: Dict[str, Any]
    ) -> List[str]:
        """Return a list of human-readable attribute differences."""
        diffs: List[str] = []

        if entity_a.get("layer") != entity_b.get("layer"):
            diffs.append(
                f"layer: '{entity_a.get('layer')}' -> '{entity_b.get('layer')}'"
            )

        if entity_a.get("color") != entity_b.get("color"):
            diffs.append(
                f"color: {entity_a.get('color')} -> {entity_b.get('color')}"
            )

        # Type-specific attribute comparisons
        dxftype = entity_a.get("dxftype", "")

        if dxftype == "CIRCLE" or dxftype == "ARC":
            ra = entity_a.get("radius")
            rb = entity_b.get("radius")
            if ra is not None and rb is not None and not math.isclose(ra, rb, rel_tol=1e-6):
                diffs.append(f"radius: {ra} -> {rb}")

        if dxftype == "ARC":
            for attr in ("start_angle", "end_angle"):
                va = entity_a.get(attr)
                vb = entity_b.get(attr)
                if va is not None and vb is not None and not math.isclose(va, vb, rel_tol=1e-6):
                    diffs.append(f"{attr}: {va} -> {vb}")

        if dxftype in ("TEXT", "MTEXT", "DIMENSION"):
            ta = entity_a.get("text", "")
            tb = entity_b.get("text", "")
            if ta != tb:
                diffs.append(f"text: '{ta}' -> '{tb}'")

        if dxftype in ("LWPOLYLINE", "POLYLINE"):
            va = entity_a.get("vertex_count")
            vb = entity_b.get("vertex_count")
            if va is not None and vb is not None and va != vb:
                diffs.append(f"vertex_count: {va} -> {vb}")

        if dxftype == "INSERT":
            na = entity_a.get("block_name")
            nb = entity_b.get("block_name")
            if na != nb:
                diffs.append(f"block_name: '{na}' -> '{nb}'")

        return diffs

    # ------------------------------------------------------------------
    # Change-region clustering
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_change_regions(
        changes: List[EntityChange],
        cluster_radius: float = 50.0,
    ) -> List[Dict[str, Any]]:
        """Cluster nearby changes into bounding-box regions.

        Uses a simple single-linkage approach: iterate through changes sorted
        by x-coordinate and merge into the current cluster when within
        *cluster_radius*.  Each resulting cluster is returned as a bounding
        box dict with min_x, min_y, max_x, max_y, and a count of changes.
        """
        if not changes:
            return []

        points = [(c.location[0], c.location[1]) for c in changes]
        # Sort by x then y for deterministic clustering
        sorted_indices = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))

        clusters: List[List[int]] = []
        current_cluster: List[int] = [sorted_indices[0]]

        for i in range(1, len(sorted_indices)):
            idx = sorted_indices[i]
            prev_idx = current_cluster[-1]
            px, py = points[prev_idx]
            cx, cy = points[idx]
            dist = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
            if dist <= cluster_radius:
                current_cluster.append(idx)
            else:
                clusters.append(current_cluster)
                current_cluster = [idx]
        clusters.append(current_cluster)

        regions: List[Dict[str, Any]] = []
        for cluster in clusters:
            xs = [points[i][0] for i in cluster]
            ys = [points[i][1] for i in cluster]
            regions.append(
                {
                    "min_x": min(xs),
                    "min_y": min(ys),
                    "max_x": max(xs),
                    "max_y": max(ys),
                    "change_count": len(cluster),
                }
            )

        return regions


__all__ = ["GeometryDiff"]
