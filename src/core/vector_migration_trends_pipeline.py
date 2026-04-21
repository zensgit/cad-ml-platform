from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Awaitable, Callable

from src.core.vector_migration_reporting_pipeline import (
    collect_vector_migration_distribution_snapshot,
)


def _filter_history_by_window(
    history: list[dict[str, Any]],
    *,
    window_hours: int,
) -> list[dict[str, Any]]:
    cutoff = datetime.utcnow() - timedelta(hours=window_hours)
    filtered_history: list[dict[str, Any]] = []
    for entry in history:
        try:
            entry_time = datetime.fromisoformat(entry.get("started_at", ""))
            if entry_time >= cutoff:
                filtered_history.append(entry)
        except (ValueError, TypeError):
            filtered_history.append(entry)
    return filtered_history


async def run_vector_migration_trends_pipeline(
    *,
    window_hours: int,
    history: list[dict[str, Any]],
    response_cls: type[Any],
    get_qdrant_store_fn: Callable[[], Any],
    resolve_scan_limit_fn: Callable[[], int],
    collect_qdrant_feature_versions_fn: Callable[..., Awaitable[tuple[dict[str, int], int, int]]],
    build_readiness_fn: Callable[..., dict[str, Any]],
) -> Any:
    filtered_history = _filter_history_by_window(history, window_hours=window_hours)

    total_migrations = 0
    total_migrated = 0
    total_downgraded = 0
    total_errors = 0

    for entry in filtered_history:
        counts = entry.get("counts", {})
        total_migrations += entry.get("total", 0)
        total_migrated += counts.get("migrated", 0)
        total_downgraded += counts.get("downgraded", 0)
        total_errors += counts.get("error", 0) + counts.get("not_found", 0)

    attempted = total_migrated + total_downgraded + total_errors
    success_rate = (total_migrated + total_downgraded) / max(attempted, 1)
    downgrade_rate = total_downgraded / max(attempted, 1)
    error_rate = total_errors / max(attempted, 1)

    snapshot = await collect_vector_migration_distribution_snapshot(
        qdrant_store=get_qdrant_store_fn(),
        scan_limit=resolve_scan_limit_fn(),
        collect_qdrant_feature_versions_fn=collect_qdrant_feature_versions_fn,
        build_readiness_fn=build_readiness_fn,
    )
    version_distribution = snapshot["versions"]
    total_vectors = snapshot["total_vectors"]
    v4_adoption_rate = version_distribution.get("v4", 0) / max(total_vectors, 1)

    avg_dimension_delta = 0.0
    if total_migrated > 0:
        avg_dimension_delta = (total_migrated * 2 - total_downgraded * 2) / max(
            total_migrated + total_downgraded, 1
        )

    migration_velocity = total_migrations / max(window_hours, 1)
    time_range = {
        "start": (datetime.utcnow() - timedelta(hours=window_hours)).isoformat()
        if window_hours > 0
        else None,
        "end": datetime.utcnow().isoformat(),
    }
    readiness = snapshot["readiness"]

    return response_cls(
        total_migrations=total_migrations,
        success_rate=round(success_rate, 4),
        v4_adoption_rate=round(v4_adoption_rate, 4),
        avg_dimension_delta=round(avg_dimension_delta, 2),
        window_hours=window_hours,
        version_distribution=version_distribution,
        migration_velocity=round(migration_velocity, 2),
        downgrade_rate=round(downgrade_rate, 4),
        error_rate=round(error_rate, 4),
        time_range=time_range,
        current_total_vectors=snapshot["total_vectors"],
        scanned_vectors=snapshot["scanned_vectors"],
        scan_limit=snapshot["scan_limit"],
        distribution_complete=snapshot["distribution_complete"],
        target_version=readiness["target_version"],
        target_version_vectors=readiness["target_version_vectors"],
        target_version_ratio=readiness["target_version_ratio"],
        pending_vectors=readiness["pending_vectors"],
        migration_ready=readiness["migration_ready"],
    )


__all__ = ["run_vector_migration_trends_pipeline"]
