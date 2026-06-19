from __future__ import annotations

from typing import Any, Mapping


def rank_observed_versions(observed_by_from_version: Mapping[str, int]) -> list[str]:
    return [
        key
        for key, _ in sorted(
            observed_by_from_version.items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )
    ]


def build_vector_migration_plan_batches(
    *,
    observed_by_from_version: Mapping[str, int],
    max_batches: int,
    default_run_limit: int,
    allow_partial_scan_required: bool,
) -> list[dict[str, Any]]:
    ordered = rank_observed_versions(observed_by_from_version)[:max_batches]
    batches: list[dict[str, Any]] = []
    for index, from_version in enumerate(ordered):
        pending_count = int(observed_by_from_version.get(from_version, 0))
        suggested_run_limit = min(pending_count, default_run_limit)
        notes = (
            ["split_batch_required"]
            if pending_count > suggested_run_limit
            else ["single_batch_ready"]
        )
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


def estimate_migration_runs_by_version(
    *,
    observed_by_from_version: Mapping[str, int],
    default_run_limit: int,
) -> dict[str, int]:
    return {
        str(from_version): max(
            (int(count) + default_run_limit - 1) // default_run_limit,
            1,
        )
        for from_version, count in observed_by_from_version.items()
    }


__all__ = [
    "build_vector_migration_plan_batches",
    "estimate_migration_runs_by_version",
    "rank_observed_versions",
]
