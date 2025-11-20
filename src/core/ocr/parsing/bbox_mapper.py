"""Assign bounding boxes to parsed dimensions/symbols from OCR line boxes.

Strategy (heuristic):
- Given OCR lines: list of dicts {"text": str, "bbox": [x,y,w,h]}
- For each parsed item with raw text, find the first line whose text contains
  the raw (case-insensitive, stripped). If none, try matching numeric value.
- Assign that line bbox to the item if not already set.
"""

from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import difflib
import re

from src.core.ocr.base import DimensionInfo, SymbolInfo


def _normalize_text(s: str) -> str:
    return " ".join(str(s).strip().lower().split())


def _similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def _best_line_for_value(line_text: str, value: float) -> float:
    """Score proximity of numeric tokens in line_text to value.

    Returns a score in [0,1], higher is better.
    """
    nums = re.findall(r"\d+(?:\.\d+)?", line_text)
    if not nums:
        return 0.0
    diffs = []
    for n in nums:
        try:
            diffs.append(abs(float(n) - float(value)))
        except Exception:
            continue
    if not diffs:
        return 0.0
    d = min(diffs)
    # Map small diffs to high score; allow values within 0.1 to score strongly
    return max(0.0, min(1.0, 1.0 / (1.0 + d * 10.0)))


def _type_hint_boost(text: str, dim_type: Optional[str]) -> float:
    if not dim_type:
        return 0.0
    t = text
    if dim_type == "diameter" and any(ch in t for ch in ["Φ", "⌀", "∅"]):
        return 0.1
    if dim_type == "radius" and "R" in t:
        return 0.1
    if dim_type == "thread" and "M" in t:
        return 0.1
    return 0.0


def assign_bboxes(
    dimensions: List[DimensionInfo], symbols: List[SymbolInfo], ocr_lines: List[Dict]
) -> None:
    # Extract (normalized_text, bbox, score) tuples; score is optional
    lines: List[Tuple[str, Optional[list[int]], Optional[float]]] = []
    for line in ocr_lines:
        norm = _normalize_text(str(line.get("text", "")))
        lines.append((norm, line.get("bbox"), line.get("score")))

    for d in dimensions:
        if d.bbox:
            continue
        raw = d.raw or ""
        needle = _normalize_text(raw)
        # 1) Exact/substring match on normalized raw
        for t, bbox, score in lines:
            if needle and needle in t and bbox:
                d.bbox = bbox
                if d.confidence is None and isinstance(score, (int, float)):
                    d.confidence = float(score)
                break
        if d.bbox:
            continue
        # 2) Value substring match (e.g., '20' in '20.00mm')
        val_token = _normalize_text(str(d.value))
        for t, bbox, score in lines:
            if val_token and val_token in t and bbox:
                d.bbox = bbox
                if d.confidence is None and isinstance(score, (int, float)):
                    d.confidence = float(score)
                break
        if d.bbox:
            continue
        # 3) Heuristic scoring: combine string similarity, numeric proximity, and type hint
        best_score = 0.0
        best_bbox = None
        # Build a compact needle: prefer raw, else type prefix + value
        compact = needle or (f"{d.type.value} {d.value}")
        for t, bbox, line_score in lines:
            if not bbox:
                continue
            sim = _similarity(compact, t)
            prox = _best_line_for_value(t, d.value)
            hint = _type_hint_boost(t, d.type.value if hasattr(d.type, "value") else None)
            ls = 0.0
            if isinstance(line_score, (int, float)):
                # normalize to [0,1] (scores usually 0-1 already; clamp for safety)
                ls = max(0.0, min(1.0, float(line_score)))
            score_combined = sim * 0.5 + prox * 0.25 + hint * 0.1 + ls * 0.15
            if score_combined > best_score:
                best_score = score_combined
                best_bbox = bbox
        if best_bbox and best_score >= 0.6:
            d.bbox = best_bbox

    for s in symbols:
        if s.bbox:
            continue
        raw = getattr(s, "raw", None)
        needle = _normalize_text(raw or s.value)
        for t, bbox, score in lines:
            if needle and needle in t and bbox:
                s.bbox = bbox
                if s.confidence is None and isinstance(score, (int, float)):
                    s.confidence = float(score)
                break
        if s.bbox:
            continue
        # Simple similarity-based fallback for symbols
        best_score = 0.0
        best_bbox = None
        for t, bbox, line_score in lines:
            if not bbox:
                continue
            sim = _similarity(needle, t)
            ls = 0.0
            if isinstance(line_score, (int, float)):
                ls = max(0.0, min(1.0, float(line_score)))
            score_combined = sim * 0.7 + ls * 0.3
            if score_combined > best_score:
                best_score = score_combined
                best_bbox = bbox
        if best_bbox and best_score >= 0.65:
            s.bbox = best_bbox


def polygon_to_bbox(poly: List[List[float]] | List[float]) -> List[int]:
    if not poly:
        return []
    if isinstance(poly[0], (int, float)):
        # already a bbox-like list
        return [int(poly[0]), int(poly[1]), int(poly[2]), int(poly[3])]  # type: ignore[index]
    xs = [p[0] for p in poly]  # type: ignore[index]
    ys = [p[1] for p in poly]  # type: ignore[index]
    x, y = int(min(xs)), int(min(ys))
    w, h = int(max(xs) - x), int(max(ys) - y)
    return [x, y, w, h]
