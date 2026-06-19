from __future__ import annotations

from typing import Any

from src.core.vector_migration_plan_batches import (
    build_vector_migration_plan_batches,
    estimate_migration_runs_by_version,
)
from src.core.vector_migration_pending_summary import (
    build_vector_migration_pending_summary_payload,
)
from src.core.vector_migration_plan_outcome import (
    build_vector_migration_plan_outcome,
)


def build_vector_migration_plan_payload(
    *,
    pending: dict[str, Any],
    max_batches: int,
    default_run_limit: int,
) -> dict[str, Any]:
    summary = build_vector_migration_pending_summary_payload(pending=pending)
    allow_partial_scan_required = pending["backend"] == "qdrant" and not pending["distribution_complete"]
    batches = build_vector_migration_plan_batches(
        observed_by_from_version=pending["observed_by_from_version"],
        max_batches=max_batches,
        default_run_limit=default_run_limit,
        allow_partial_scan_required=allow_partial_scan_required,
    )
    outcome = build_vector_migration_plan_outcome(
        summary=summary,
        pending=pending,
        batches=batches,
        allow_partial_scan_required=allow_partial_scan_required,
    )
    estimated_runs_by_version = estimate_migration_runs_by_version(
        observed_by_from_version=pending["observed_by_from_version"],
        default_run_limit=default_run_limit,
    )
    return {
        **summary,
        "max_batches": max_batches,
        "default_run_limit": default_run_limit,
        "estimated_total_runs": sum(estimated_runs_by_version.values()),
        "estimated_runs_by_version": estimated_runs_by_version,
        **outcome,
        "batches": batches,
    }


__all__ = [
    "build_vector_migration_pending_summary_payload",
    "build_vector_migration_plan_payload",
]
