"""Pure version-gating constants + key-extraction helpers for 2D dedup.

Extracted from src/api/v1/dedup.py (behavior-preserving slimming); re-exported by
dedup.py so the precision-L4 path and the tenant-config validator keep resolving
them unchanged.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Optional


_VERSION_GATE_MODES = {"off", "auto", "file_name", "meta"}


_VERSION_SUFFIX_RE = re.compile(r"(?:[_\-\s]?v\d+)$", re.IGNORECASE)


def _normalize_weights(visual_w: float, geom_w: float) -> tuple[float, float]:
    if visual_w < 0 or geom_w < 0:
        raise ValueError("weights must be >= 0")
    total = visual_w + geom_w
    if total <= 0:
        raise ValueError("weights sum must be > 0")
    return visual_w / total, geom_w / total


def _extract_meta_drawing_key(v2: Dict[str, Any]) -> Optional[str]:
    meta = v2.get("meta")
    if not isinstance(meta, dict):
        return None
    for k in (
        "drawing_number",
        "drawing_no",
        "drawingNo",
        "drawingNumber",
        "drawing_id",
        "drawingId",
        "number",
    ):
        v = meta.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _extract_file_stem_key(file_name: Optional[str]) -> Optional[str]:
    if not file_name:
        return None
    stem = Path(str(file_name)).stem.strip()
    if not stem:
        return None
    if _VERSION_SUFFIX_RE.search(stem) is None:
        return None
    base = _VERSION_SUFFIX_RE.sub("", stem).rstrip(" _-").strip()
    return base if base else None
