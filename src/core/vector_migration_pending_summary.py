from __future__ import annotations

from typing import Any

from src.core.vector_migration_plan_batches import rank_observed_versions


def build_vector_migration_pending_summary_payload(
    *,
    pending: dict[str, Any],
) -> dict[str, Any]:
    recommended_from_versions = rank_observed_versions(
        pending["observed_by_from_version"]
    )
    largest_pending_from_version = (
        recommended_from_versions[0] if recommended_from_versions else None
    )
    largest_pending_count = None
    if largest_pending_from_version is not None:
        largest_pending_count = int(
            pending["observed_by_from_version"].get(largest_pending_from_version, 0)
        )

    pending_ratio = None
    if pending["distribution_complete"]:
        scanned_vectors = int(pending["scanned_vectors"] or 0)
        pending_ratio = round(
            int(pending["total_pending"] or 0) / max(scanned_vectors, 1),
            4,
        )

    return {
        "target_version": pending["target_version"],
        "from_version_filter": pending["from_version_filter"],
        "observed_by_from_version": pending["observed_by_from_version"],
        "recommended_from_versions": recommended_from_versions,
        "largest_pending_from_version": largest_pending_from_version,
        "largest_pending_count": largest_pending_count,
        "total_pending": pending["total_pending"],
        "pending_ratio": pending_ratio,
        "backend": pending["backend"],
        "scanned_vectors": pending["scanned_vectors"],
        "scan_limit": pending["scan_limit"],
        "distribution_complete": pending["distribution_complete"],
    }


__all__ = ["build_vector_migration_pending_summary_payload"]
