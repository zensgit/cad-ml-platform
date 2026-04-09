"""Helpers for stable coarse-label normalization across AI branches."""

from __future__ import annotations

from typing import Optional

from src.ml.label_normalization import normalize_dxf_label


def normalize_coarse_label(label: Optional[str]) -> Optional[str]:
    """Normalize a label into the coarse DXF taxonomy when possible."""
    cleaned = str(label or "").strip()
    if not cleaned:
        return None
    normalized = normalize_dxf_label(cleaned, default=cleaned)
    normalized_clean = str(normalized or "").strip()
    return normalized_clean or None


def labels_conflict(left: Optional[str], right: Optional[str]) -> bool:
    """Return True when two labels disagree after coarse normalization."""
    left_norm = normalize_coarse_label(left)
    right_norm = normalize_coarse_label(right)
    if not left_norm or not right_norm:
        return False
    return left_norm != right_norm
