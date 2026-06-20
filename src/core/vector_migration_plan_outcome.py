from __future__ import annotations

from typing import Any, Mapping, Sequence


def build_vector_migration_plan_outcome(
    *,
    summary: Mapping[str, Any],
    pending: Mapping[str, Any],
    batches: Sequence[Mapping[str, Any]],
    allow_partial_scan_required: bool,
) -> dict[str, Any]:
    blocking_reasons: list[str] = []
    if allow_partial_scan_required:
        blocking_reasons.append("partial_scan_override_required")
    if not batches:
        blocking_reasons.append("no_pending_vectors")

    recommended_first_batch = batches[0] if batches else None
    recommended_first_request_payload = (
        dict(recommended_first_batch["request_payload"])
        if recommended_first_batch is not None
        else None
    )
    planned_pending_count = sum(int(batch["pending_count"]) for batch in batches)
    planned_from_versions = {str(batch["from_version"]) for batch in batches}
    recommended_from_versions = list(summary["recommended_from_versions"])
    unplanned_from_versions = [
        from_version
        for from_version in recommended_from_versions
        if str(from_version) not in planned_from_versions
    ]

    remaining_pending_count = None
    planned_pending_ratio = None
    if pending["distribution_complete"] and pending["total_pending"] is not None:
        remaining_pending_count = max(
            int(pending["total_pending"]) - planned_pending_count,
            0,
        )
        planned_pending_ratio = round(
            planned_pending_count / max(int(pending["total_pending"]), 1),
            4,
        )

    coverage_complete = bool(batches) and not unplanned_from_versions
    suggested_next_max_batches = (
        len(recommended_from_versions) if unplanned_from_versions else None
    )

    return {
        "plan_ready": bool(batches) and not blocking_reasons,
        "blocking_reasons": blocking_reasons,
        "recommended_first_batch": recommended_first_batch,
        "recommended_first_request_payload": recommended_first_request_payload,
        "planned_pending_count": planned_pending_count,
        "remaining_pending_count": remaining_pending_count,
        "planned_pending_ratio": planned_pending_ratio,
        "coverage_complete": coverage_complete,
        "truncated_by_max_batches": bool(unplanned_from_versions),
        "unplanned_from_versions": unplanned_from_versions,
        "suggested_next_max_batches": suggested_next_max_batches,
    }


__all__ = ["build_vector_migration_plan_outcome"]
