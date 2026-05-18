from __future__ import annotations

from typing import Any


async def collect_qdrant_preview_samples(
    qdrant_store: Any,
    *,
    limit: int,
) -> tuple[list[tuple[str, list[float], dict[str, Any]]], int, dict[str, int]]:
    total_available = int(await qdrant_store.count())
    items, _ = await qdrant_store.list_vectors(
        offset=0,
        limit=max(limit, 1),
        with_vectors=True,
    )
    samples: list[tuple[str, list[float], dict[str, Any]]] = []
    by_version: dict[str, int] = {}
    for item in items:
        meta = dict(item.metadata or {})
        version = str(meta.get("feature_version") or "v1")
        by_version[version] = by_version.get(version, 0) + 1
        samples.append((str(item.id), list(item.vector or []), meta))
    if total_available > len(items):
        offset = len(items)
        scan_limit = min(total_available, 5000)
        while offset < scan_limit:
            batch, _ = await qdrant_store.list_vectors(
                offset=offset,
                limit=min(200, scan_limit - offset),
                with_vectors=False,
            )
            if not batch:
                break
            for item in batch:
                meta = dict(item.metadata or {})
                version = str(meta.get("feature_version") or "v1")
                by_version[version] = by_version.get(version, 0) + 1
            offset += len(batch)
    return samples, total_available, by_version


__all__ = ["collect_qdrant_preview_samples"]
