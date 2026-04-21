from __future__ import annotations

from datetime import datetime
from typing import Any, Awaitable, Callable


def _collect_memory_feature_versions() -> tuple[dict[str, int], int, int]:
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE  # type: ignore

    versions: dict[str, int] = {}
    total_vectors = 0
    for vector_id, meta in _VECTOR_META.items():
        if vector_id not in _VECTOR_STORE:
            continue
        total_vectors += 1
        version = str(meta.get("feature_version") or "unknown")
        versions[version] = versions.get(version, 0) + 1
    return versions, total_vectors, total_vectors


async def collect_vector_migration_distribution_snapshot(
    *,
    qdrant_store: Any,
    scan_limit: int,
    collect_qdrant_feature_versions_fn: Callable[..., Awaitable[tuple[dict[str, int], int, int]]],
    build_readiness_fn: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    if qdrant_store is not None:
        versions, total_vectors, scanned_vectors = await collect_qdrant_feature_versions_fn(
            qdrant_store,
            scan_limit=scan_limit,
        )
        backend = "qdrant"
        distribution_complete = scanned_vectors >= total_vectors
    else:
        versions, total_vectors, scanned_vectors = _collect_memory_feature_versions()
        backend = "memory"
        distribution_complete = True

    readiness = build_readiness_fn(
        versions,
        total_vectors=total_vectors,
        distribution_complete=distribution_complete,
    )
    return {
        "backend": backend,
        "versions": versions,
        "total_vectors": total_vectors,
        "scanned_vectors": scanned_vectors,
        "scan_limit": scan_limit,
        "distribution_complete": distribution_complete,
        "readiness": readiness,
    }


def _parse_history_datetime(entry: dict[str, Any] | None, key: str) -> datetime | None:
    if not entry:
        return None
    raw = entry.get(key)
    if not raw:
        return None
    return datetime.fromisoformat(raw)


def build_vector_migration_status_payload(
    *,
    history: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    last = history[-1] if history else None
    readiness = snapshot["readiness"]
    return {
        "last_migration_id": last.get("migration_id") if last else None,
        "last_started_at": _parse_history_datetime(last, "started_at"),
        "last_finished_at": _parse_history_datetime(last, "finished_at"),
        "last_total": last.get("total") if last else None,
        "last_migrated": last.get("migrated") if last else None,
        "last_skipped": last.get("skipped") if last else None,
        "pending_vectors": readiness["pending_vectors"],
        "feature_versions": snapshot["versions"],
        "history": history,
        "backend": snapshot["backend"],
        "current_total_vectors": snapshot["total_vectors"],
        "scanned_vectors": snapshot["scanned_vectors"],
        "scan_limit": snapshot["scan_limit"],
        "distribution_complete": snapshot["distribution_complete"],
        "target_version": readiness["target_version"],
        "target_version_vectors": readiness["target_version_vectors"],
        "target_version_ratio": readiness["target_version_ratio"],
        "migration_ready": readiness["migration_ready"],
    }


def build_vector_migration_summary_payload(
    *,
    history: list[dict[str, Any]],
    snapshot: dict[str, Any],
) -> dict[str, Any]:
    aggregate: dict[str, int] = {}
    for entry in history:
        counts = entry.get("counts", {})
        for key, value in counts.items():
            aggregate[key] = aggregate.get(key, 0) + int(value)

    readiness = snapshot["readiness"]
    return {
        "counts": aggregate,
        "total_migrations": sum(aggregate.values()),
        "history_size": len(history),
        "statuses": sorted(aggregate.keys()),
        "backend": snapshot["backend"],
        "current_version_distribution": snapshot["versions"],
        "current_total_vectors": snapshot["total_vectors"],
        "scanned_vectors": snapshot["scanned_vectors"],
        "scan_limit": snapshot["scan_limit"],
        "distribution_complete": snapshot["distribution_complete"],
        "target_version": readiness["target_version"],
        "target_version_vectors": readiness["target_version_vectors"],
        "target_version_ratio": readiness["target_version_ratio"],
        "pending_vectors": readiness["pending_vectors"],
        "migration_ready": readiness["migration_ready"],
    }


__all__ = [
    "build_vector_migration_status_payload",
    "build_vector_migration_summary_payload",
    "collect_vector_migration_distribution_snapshot",
]
