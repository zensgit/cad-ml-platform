"""Text sanitization helpers.

These helpers are used to bound and normalize text that may be surfaced to
clients via health/readiness endpoints.
"""

from __future__ import annotations

from typing import Any, Optional


def sanitize_single_line_text(value: Any, *, max_len: int = 300) -> Optional[str]:
    """Best-effort sanitization for public-facing text fields.

    Behavior:
    - Convert to string via ``str(value)``.
    - Collapse whitespace (including newlines) into a single line.
    - Truncate to ``max_len`` characters (ellipsis when room allows).
    """
    if value is None:
        return None

    try:
        text = " ".join(str(value).split())
    except Exception:
        return None

    if not text:
        return None

    if max_len <= 0:
        max_len = 1

    if len(text) > max_len:
        if max_len <= 3:
            return text[:max_len]
        return text[: max_len - 3] + "..."

    return text

