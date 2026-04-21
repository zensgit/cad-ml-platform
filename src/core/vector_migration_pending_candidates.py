from __future__ import annotations

from typing import Any, Mapping, Optional


async def collect_vector_migration_pending_candidates(
    *,
    limit: int,
    target_version: str,
    from_version_filter: Optional[str],
    qdrant_store: Any,
    scan_limit: int,
    item_cls: type[Any],
    vector_meta: Mapping[str, Any],
    vector_store: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_filter = str(from_version_filter or "").strip() or None

    if qdrant_store is not None:
        total_available = int(await qdrant_store.count())
        max_scan = min(total_available, scan_limit)
        scanned = 0
        offset = 0
        items: list[Any] = []
        pending_ids: list[str] = []
        total_pending = 0
        observed_by_from_version: dict[str, int] = {}
        while scanned < max_scan:
            batch_limit = min(200, max_scan - scanned)
            points, _ = await qdrant_store.list_vectors(
                offset=offset,
                limit=batch_limit,
                with_vectors=False,
            )
            if not points:
                break
            for point in points:
                meta = point.metadata or {}
                from_version = str(meta.get("feature_version") or "unknown")
                if from_version == target_version:
                    continue
                if normalized_filter and from_version != normalized_filter:
                    continue
                total_pending += 1
                observed_by_from_version[from_version] = (
                    observed_by_from_version.get(from_version, 0) + 1
                )
                pending_ids.append(str(point.id))
                if len(items) < limit:
                    items.append(
                        item_cls(
                            id=str(point.id),
                            from_version=from_version,
                            to_version=target_version,
                        )
                    )
            consumed = len(points)
            scanned += consumed
            offset += consumed

        distribution_complete = scanned >= total_available
        return {
            "target_version": target_version,
            "from_version_filter": normalized_filter,
            "items": items,
            "pending_ids": pending_ids[:limit],
            "listed_count": len(items),
            "total_pending": total_pending if distribution_complete else None,
            "observed_by_from_version": observed_by_from_version,
            "backend": "qdrant",
            "scanned_vectors": scanned,
            "scan_limit": scan_limit,
            "distribution_complete": distribution_complete,
        }

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
        observed_by_from_version[from_version] = observed_by_from_version.get(from_version, 0) + 1
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


__all__ = ["collect_vector_migration_pending_candidates"]
