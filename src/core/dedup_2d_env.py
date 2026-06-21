"""Pure env-driven helpers for the 2D dedup router (no FastAPI).

Extracted from src/api/v1/dedup.py (behavior-preserving router slimming). The
router re-exports these so handlers and the existing
`from src.api.v1.dedup import _check_forced_async` test import keep working.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _get_dedup2d_async_backend() -> str:
    return os.getenv("DEDUP2D_ASYNC_BACKEND", "inprocess").strip().lower() or "inprocess"


def _get_int_env(name: str, *, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip()
    if value == "":
        return default
    try:
        return int(value)
    except ValueError:
        logger.warning("invalid_int_env", extra={"env_name": name, "value": raw})
        return default


def _get_bool_env(name: str, *, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value == "":
        return default
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    logger.warning("invalid_bool_env", extra={"env_name": name, "value": raw})
    return default


def _check_forced_async(
    file_size: int,
    enable_precision: bool,
    mode: str,
    query_geom: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Check if the request should be forced to async mode.

    Returns:
        forced_async_reason if async should be forced, None otherwise.
    """
    forced_async_file_size_bytes = _get_int_env(
        "DEDUP2D_FORCED_ASYNC_FILE_SIZE_BYTES",
        default=5 * 1024 * 1024,  # 5MB
    )
    forced_async_enable_precision = _get_bool_env(
        "DEDUP2D_FORCED_ASYNC_ON_PRECISION",
        default=True,
    )
    forced_async_mode_precise = _get_bool_env(
        "DEDUP2D_FORCED_ASYNC_ON_MODE_PRECISE",
        default=True,
    )

    # Check file size threshold
    if forced_async_file_size_bytes > 0 and file_size > forced_async_file_size_bytes:
        mb = max(1, forced_async_file_size_bytes // (1024 * 1024))
        return f"file_size>{mb}MB"

    # Check precision verification (slow operation)
    if forced_async_enable_precision and enable_precision and query_geom is not None:
        return "enable_precision_with_geom_json"

    # Check precise mode
    if forced_async_mode_precise and mode == "precise":
        return "mode=precise"

    return None
