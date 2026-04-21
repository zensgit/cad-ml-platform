from __future__ import annotations

from typing import Any


def _rank_observed_versions(observed_by_from_version: dict[str, int]) -> list[str]:
    return [
        key
        for key, _ in sorted(
            observed_by_from_version.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
    ]


def _build_plan_batches(
    *,
    observed_by_from_version: dict[str, int],
    max_batches: int,
    default_run_limit: int,
    allow_partial_scan_required: bool,
) -> list[dict[str, Any]]:
    ordered = _rank_observed_versions(observed_by_from_version)[:max_batches]
    batches: list[dict[str, Any]] = []
    for index, from_version in enumerate(ordered):
        pending_count = int(observed_by_from_version.get(from_version, 0))
        suggested_run_limit = min(pending_count, default_run_limit)
        notes = ["split_batch_required"] if pending_count > suggested_run_limit else ["single_batch_ready"]
        if allow_partial_scan_required:
            notes.append("partial_scan_override_required")
        batches.append(
            {
                "priority": index + 1,
                "from_version": str(from_version),
                "pending_count": pending_count,
                "suggested_run_limit": suggested_run_limit,
                "allow_partial_scan_required": allow_partial_scan_required,
                "request_payload": {
                    "limit": suggested_run_limit,
                    "dry_run": True,
                    "from_version_filter": str(from_version),
                    "allow_partial_scan": allow_partial_scan_required,
                },
                "notes": notes,
            }
        )
    return batches


def build_vector_migration_pending_summary_payload(
    *,
    pending: dict[str, Any],
) -> dict[str, Any]:
    recommended_from_versions = _rank_observed_versions(pending["observed_by_from_version"])
    largest_pending_from_version = recommended_from_versions[0] if recommended_from_versions else None
    largest_pending_count = None
    if largest_pending_from_version is not None:
        largest_pending_count = int(
            pending["observed_by_from_version"].get(largest_pending_from_version, 0)
        )

    pending_ratio = None
    if pending["distribution_complete"]:
        scanned_vectors = int(pending["scanned_vectors"] or 0)
        pending_ratio = round(int(pending["total_pending"] or 0) / max(scanned_vectors, 1), 4)

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


def build_vector_migration_plan_payload(
    *,
    pending: dict[str, Any],
    max_batches: int,
    default_run_limit: int,
) -> dict[str, Any]:
    summary = build_vector_migration_pending_summary_payload(pending=pending)
    allow_partial_scan_required = pending["backend"] == "qdrant" and not pending["distribution_complete"]
    batches = _build_plan_batches(
        observed_by_from_version=pending["observed_by_from_version"],
        max_batches=max_batches,
        default_run_limit=default_run_limit,
        allow_partial_scan_required=allow_partial_scan_required,
    )
    blocking_reasons: list[str] = []
    if allow_partial_scan_required:
        blocking_reasons.append("partial_scan_override_required")
    if not batches:
        blocking_reasons.append("no_pending_vectors")
    recommended_first_batch = batches[0] if batches else None
    recommended_first_request_payload = (
        dict(recommended_first_batch["request_payload"]) if recommended_first_batch is not None else None
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
        remaining_pending_count = max(int(pending["total_pending"]) - planned_pending_count, 0)
        planned_pending_ratio = round(
            planned_pending_count / max(int(pending["total_pending"]), 1),
            4,
        )
    coverage_complete = bool(batches) and not unplanned_from_versions
    suggested_next_max_batches = len(recommended_from_versions) if unplanned_from_versions else None
    estimated_runs_by_version = {
        str(from_version): max((int(count) + default_run_limit - 1) // default_run_limit, 1)
        for from_version, count in pending["observed_by_from_version"].items()
    }
    return {
        **summary,
        "max_batches": max_batches,
        "default_run_limit": default_run_limit,
        "estimated_total_runs": sum(estimated_runs_by_version.values()),
        "estimated_runs_by_version": estimated_runs_by_version,
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
        "batches": batches,
    }


__all__ = [
    "build_vector_migration_pending_summary_payload",
    "build_vector_migration_plan_payload",
]
