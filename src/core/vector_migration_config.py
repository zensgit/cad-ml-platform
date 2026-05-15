from __future__ import annotations

import os
from typing import Optional


ALLOWED_VECTOR_MIGRATION_TARGET_VERSIONS = {"v1", "v2", "v3", "v4"}


def resolve_vector_migration_scan_limit(default: int = 5000) -> int:
    try:
        resolved = int(os.getenv("VECTOR_MIGRATION_SCAN_LIMIT", str(default)))
    except (TypeError, ValueError):
        resolved = default
    return max(int(resolved or 0), 1)


def resolve_vector_migration_target_version(default: str = "v4") -> str:
    raw = str(
        os.getenv("VECTOR_MIGRATION_TARGET_VERSION", default) or default
    ).strip().lower()
    if raw in ALLOWED_VECTOR_MIGRATION_TARGET_VERSIONS:
        return raw
    return default


def coerce_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "ALLOWED_VECTOR_MIGRATION_TARGET_VERSIONS",
    "coerce_optional_int",
    "resolve_vector_migration_scan_limit",
    "resolve_vector_migration_target_version",
]
