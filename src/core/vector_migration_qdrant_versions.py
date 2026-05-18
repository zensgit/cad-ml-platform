from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.core.vector_migration_config import resolve_vector_migration_scan_limit


async def collect_qdrant_feature_versions(
    qdrant_store: Any,
    *,
    scan_limit: int | None = None,
    resolve_scan_limit_fn: Callable[[], int] = resolve_vector_migration_scan_limit,
) -> tuple[dict[str, int], int, int]:
    resolved_scan_limit = scan_limit
    if resolved_scan_limit is None:
        resolved_scan_limit = resolve_scan_limit_fn()
    resolved_scan_limit = max(int(resolved_scan_limit or 0), 1)

    total_available = int(await qdrant_store.count())
    versions: dict[str, int] = {}
    scanned = 0
    offset = 0

    while scanned < min(total_available, resolved_scan_limit):
        batch_limit = min(200, resolved_scan_limit - scanned)
        items, _ = await qdrant_store.list_vectors(
            offset=offset,
            limit=batch_limit,
            with_vectors=False,
        )
        if not items:
            break
        for item in items:
            meta = item.metadata or {}
            version = str(meta.get("feature_version") or "unknown")
            versions[version] = versions.get(version, 0) + 1
        consumed = len(items)
        scanned += consumed
        offset += consumed

    return versions, total_available, scanned


__all__ = ["collect_qdrant_feature_versions"]
