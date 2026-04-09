"""Annotation-focused diff between two DXF drawing revisions.

Compares TEXT, MTEXT, and DIMENSION entities between two files, detecting
text content changes, dimension value changes, and positional shifts.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

from src.core.diff.models import DiffResult, EntityChange

logger = logging.getLogger(__name__)

_ANNOTATION_TYPES = frozenset({"TEXT", "MTEXT", "DIMENSION"})

_DEFAULT_MATCH_RADIUS: float = 2.0

try:
    import ezdxf

    _HAS_EZDXF = True
except ImportError:  # pragma: no cover
    _HAS_EZDXF = False


class AnnotationDiff:
    """Compare annotation (text / dimension) content of two DXF files."""

    def __init__(self, match_radius: float = _DEFAULT_MATCH_RADIUS) -> None:
        self.match_radius = match_radius

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compare(self, file_a: str, file_b: str) -> DiffResult:
        """Compare annotations in two DXF files.

        Focuses exclusively on TEXT, MTEXT, and DIMENSION entities.

        Args:
            file_a: Path to baseline DXF file.
            file_b: Path to revised DXF file.

        Returns:
            DiffResult with annotation-specific changes.
        """
        if not _HAS_EZDXF:
            logger.warning("ezdxf not available; returning empty DiffResult")
            return DiffResult(summary={"added": 0, "removed": 0, "modified": 0})

        try:
            doc_a = ezdxf.readfile(file_a)
            doc_b = ezdxf.readfile(file_b)
        except Exception:
            logger.exception("Failed to read DXF files for annotation diff")
            return DiffResult(summary={"added": 0, "removed": 0, "modified": 0})

        annots_a = self._extract_annotations(doc_a)
        annots_b = self._extract_annotations(doc_b)

        matched, added_indices, removed_indices = self._match_annotations(
            annots_a, annots_b
        )

        added: List[EntityChange] = []
        removed: List[EntityChange] = []
        modified: List[EntityChange] = []

        for idx in added_indices:
            ann = annots_b[idx]
            added.append(
                EntityChange(
                    entity_type=ann["dxftype"],
                    change_type="added",
                    location=(ann["cx"], ann["cy"]),
                    details={
                        "text": ann.get("text", ""),
                        "style": ann.get("style", ""),
                    },
                )
            )

        for idx in removed_indices:
            ann = annots_a[idx]
            removed.append(
                EntityChange(
                    entity_type=ann["dxftype"],
                    change_type="removed",
                    location=(ann["cx"], ann["cy"]),
                    details={
                        "text": ann.get("text", ""),
                        "style": ann.get("style", ""),
                    },
                )
            )

        for idx_a, idx_b in matched:
            text_diff = self._compare_text_content(annots_a[idx_a], annots_b[idx_b])
            if text_diff:
                ann_b = annots_b[idx_b]
                modified.append(
                    EntityChange(
                        entity_type=ann_b["dxftype"],
                        change_type="modified",
                        location=(ann_b["cx"], ann_b["cy"]),
                        details=text_diff,
                    )
                )

        summary = {
            "added": len(added),
            "removed": len(removed),
            "modified": len(modified),
        }

        all_changes = added + removed + modified
        change_regions = self._compute_annotation_regions(all_changes)

        return DiffResult(
            added=added,
            removed=removed,
            modified=modified,
            summary=summary,
            change_regions=change_regions,
        )

    # ------------------------------------------------------------------
    # Annotation extraction
    # ------------------------------------------------------------------

    def _extract_annotations(self, doc: Any) -> List[Dict[str, Any]]:
        """Extract annotation entities from an ezdxf document.

        Returns a list of dicts with keys: dxftype, cx, cy, text, style, layer.
        """
        annotations: List[Dict[str, Any]] = []
        msp = doc.modelspace()

        for entity in msp:
            dxftype = entity.dxftype()
            if dxftype not in _ANNOTATION_TYPES:
                continue

            cx, cy = 0.0, 0.0
            text = ""
            style = ""

            try:
                if dxftype == "TEXT":
                    if hasattr(entity.dxf, "insert"):
                        cx, cy = entity.dxf.insert.x, entity.dxf.insert.y
                    if hasattr(entity.dxf, "text"):
                        text = entity.dxf.text
                    if hasattr(entity.dxf, "style"):
                        style = entity.dxf.style

                elif dxftype == "MTEXT":
                    if hasattr(entity.dxf, "insert"):
                        cx, cy = entity.dxf.insert.x, entity.dxf.insert.y
                    if hasattr(entity, "text"):
                        text = entity.text
                    if hasattr(entity.dxf, "style"):
                        style = entity.dxf.style

                elif dxftype == "DIMENSION":
                    if hasattr(entity.dxf, "defpoint"):
                        dp = entity.dxf.defpoint
                        cx, cy = dp.x, dp.y
                    if hasattr(entity.dxf, "text"):
                        text = entity.dxf.text or ""
                    if hasattr(entity.dxf, "dimstyle"):
                        style = entity.dxf.dimstyle

            except Exception:
                logger.debug("Skipping annotation extraction for %s", dxftype)
                continue

            layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"

            annotations.append(
                {
                    "dxftype": dxftype,
                    "cx": cx,
                    "cy": cy,
                    "text": text,
                    "style": style,
                    "layer": layer,
                }
            )

        return annotations

    # ------------------------------------------------------------------
    # Annotation matching
    # ------------------------------------------------------------------

    def _match_annotations(
        self,
        annots_a: List[Dict[str, Any]],
        annots_b: List[Dict[str, Any]],
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """Match annotation entities between revisions by proximity and type.

        Uses scipy KDTree, same approach as GeometryDiff._match_entities.
        """
        if not annots_a and not annots_b:
            return ([], [], [])
        if not annots_a:
            return ([], list(range(len(annots_b))), [])
        if not annots_b:
            return ([], [], list(range(len(annots_a))))

        import numpy as np
        from scipy.spatial import cKDTree

        coords_a = np.array([[a["cx"], a["cy"]] for a in annots_a], dtype=np.float64)
        coords_b = np.array([[a["cx"], a["cy"]] for a in annots_b], dtype=np.float64)

        tree_a = cKDTree(coords_a)
        distances, indices = tree_a.query(coords_b)

        matched: List[Tuple[int, int]] = []
        matched_a_set: set[int] = set()
        matched_b_set: set[int] = set()

        for idx_b, (dist, idx_a) in enumerate(zip(distances, indices)):
            if dist > self.match_radius:
                continue
            if idx_a in matched_a_set:
                continue
            if annots_a[idx_a]["dxftype"] != annots_b[idx_b]["dxftype"]:
                continue
            matched.append((int(idx_a), idx_b))
            matched_a_set.add(int(idx_a))
            matched_b_set.add(idx_b)

        added_indices = [i for i in range(len(annots_b)) if i not in matched_b_set]
        removed_indices = [i for i in range(len(annots_a)) if i not in matched_a_set]

        return (matched, added_indices, removed_indices)

    # ------------------------------------------------------------------
    # Text content comparison
    # ------------------------------------------------------------------

    @staticmethod
    def _compare_text_content(
        text_a: Dict[str, Any], text_b: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare text content and style between matched annotations.

        Returns a dict of differences, or empty dict if identical.
        """
        diffs: Dict[str, Any] = {}

        content_a = text_a.get("text", "")
        content_b = text_b.get("text", "")
        if content_a != content_b:
            diffs["text_old"] = content_a
            diffs["text_new"] = content_b

        style_a = text_a.get("style", "")
        style_b = text_b.get("style", "")
        if style_a != style_b:
            diffs["style_old"] = style_a
            diffs["style_new"] = style_b

        layer_a = text_a.get("layer", "0")
        layer_b = text_b.get("layer", "0")
        if layer_a != layer_b:
            diffs["layer_old"] = layer_a
            diffs["layer_new"] = layer_b

        return diffs

    # ------------------------------------------------------------------
    # Change-region clustering for annotations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_annotation_regions(
        changes: List[EntityChange],
        cluster_radius: float = 100.0,
    ) -> List[Dict[str, Any]]:
        """Cluster nearby annotation changes into bounding-box regions.

        Uses a larger default cluster radius than geometry diff because
        annotation labels tend to be spatially dispersed around the geometry
        they describe.
        """
        if not changes:
            return []

        points = [(c.location[0], c.location[1]) for c in changes]
        sorted_indices = sorted(
            range(len(points)), key=lambda i: (points[i][0], points[i][1])
        )

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


__all__ = ["AnnotationDiff"]
