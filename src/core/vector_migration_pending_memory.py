from __future__ import annotations

from typing import Any, Mapping, Optional


def collect_memory_migration_pending_candidates(
    *,
    limit: int,
    target_version: str,
    normalized_filter: Optional[str],
    scan_limit: int,
    item_cls: type[Any],
    vector_meta: Mapping[str, Any],
    vector_store: Mapping[str, Any],
) -> dict[str, Any]:
    """Collect exact in-memory pending migration candidates.

    ``normalized_filter`` is expected to be normalized by the shared facade.
    ``scan_limit`` is echoed for response compatibility; memory scans remain
    exact so totals and version distribution stay complete.
    """
    items: list[Any] = []
    pending_ids: list[str] = []
    scanned = 0
    total_pending = 0
    observed_by_from_version: dict[str, int] = {}
    for vid, meta in vector_meta.items():
        if vid not in vector_store:
            continue
        scanned += 1
        from_version = str(meta.get("feature_version") or "unknown")
        if from_version == target_version:
            continue
        if normalized_filter and from_version != normalized_filter:
            continue
        total_pending += 1
        observed_by_from_version[from_version] = (
            observed_by_from_version.get(from_version, 0) + 1
        )
        pending_ids.append(str(vid))
        if len(items) < limit:
            items.append(
                item_cls(
                    id=str(vid),
                    from_version=from_version,
                    to_version=target_version,
                )
            )

    return {
        "target_version": target_version,
        "from_version_filter": normalized_filter,
        "items": items,
        "pending_ids": pending_ids[:limit],
        "listed_count": len(items),
        "total_pending": total_pending,
        "observed_by_from_version": observed_by_from_version,
        "backend": "memory",
        "scanned_vectors": scanned,
        "scan_limit": scan_limit,
        "distribution_complete": True,
    }


__all__ = ["collect_memory_migration_pending_candidates"]
