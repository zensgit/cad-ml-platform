"""Upload filename gate for ReviewReuse tasks.

Rejects non-drawing / non-raster inputs as ``unsupported_file_type``.
Does not inspect magic bytes (operator still controls isolated-sample policy).
"""

from __future__ import annotations

from pathlib import Path

# Drawings + rasters commonly accepted by the private dedup2d / vision path.
ALLOWED_SUFFIXES = frozenset(
    {
        ".dxf",
        ".dwg",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".webp",
    }
)


def file_suffix(file_name: str) -> str:
    return Path(file_name or "").suffix.lower()


def is_allowed_review_reuse_filename(file_name: str) -> bool:
    suffix = file_suffix(file_name)
    return bool(suffix) and suffix in ALLOWED_SUFFIXES
