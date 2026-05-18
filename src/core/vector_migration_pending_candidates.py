from __future__ import annotations

from typing import Any, Mapping, Optional

from src.core.vector_migration_pending_memory import (
    collect_memory_migration_pending_candidates,
)
from src.core.vector_migration_pending_qdrant import (
    collect_qdrant_migration_pending_candidates,
)


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
        return await collect_qdrant_migration_pending_candidates(
            qdrant_store=qdrant_store,
            limit=limit,
            target_version=target_version,
            normalized_filter=normalized_filter,
            scan_limit=scan_limit,
            item_cls=item_cls,
        )

    return collect_memory_migration_pending_candidates(
        limit=limit,
        target_version=target_version,
        normalized_filter=normalized_filter,
        scan_limit=scan_limit,
        item_cls=item_cls,
        vector_meta=vector_meta,
        vector_store=vector_store,
    )


__all__ = ["collect_vector_migration_pending_candidates"]
