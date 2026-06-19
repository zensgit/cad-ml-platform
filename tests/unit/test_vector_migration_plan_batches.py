from __future__ import annotations

from src.core.vector_migration_plan_batches import (
    build_vector_migration_plan_batches,
    estimate_migration_runs_by_version,
    rank_observed_versions,
)


def test_rank_observed_versions_orders_by_count_then_name() -> None:
    assert rank_observed_versions({"v3": 1, "v2": 2, "v1": 1}) == [
        "v2",
        "v1",
        "v3",
    ]


def test_build_vector_migration_plan_batches_caps_and_marks_split() -> None:
    batches = build_vector_migration_plan_batches(
        observed_by_from_version={"v3": 1, "v2": 3, "v1": 1},
        max_batches=2,
        default_run_limit=2,
        allow_partial_scan_required=False,
    )

    assert batches == [
        {
            "priority": 1,
            "from_version": "v2",
            "pending_count": 3,
            "suggested_run_limit": 2,
            "allow_partial_scan_required": False,
            "request_payload": {
                "limit": 2,
                "dry_run": True,
                "from_version_filter": "v2",
                "allow_partial_scan": False,
            },
            "notes": ["split_batch_required"],
        },
        {
            "priority": 2,
            "from_version": "v1",
            "pending_count": 1,
            "suggested_run_limit": 1,
            "allow_partial_scan_required": False,
            "request_payload": {
                "limit": 1,
                "dry_run": True,
                "from_version_filter": "v1",
                "allow_partial_scan": False,
            },
            "notes": ["single_batch_ready"],
        },
    ]


def test_build_vector_migration_plan_batches_marks_partial_override() -> None:
    batches = build_vector_migration_plan_batches(
        observed_by_from_version={"v3": 1},
        max_batches=3,
        default_run_limit=50,
        allow_partial_scan_required=True,
    )

    assert batches[0]["request_payload"] == {
        "limit": 1,
        "dry_run": True,
        "from_version_filter": "v3",
        "allow_partial_scan": True,
    }
    assert batches[0]["notes"] == [
        "single_batch_ready",
        "partial_scan_override_required",
    ]


def test_estimate_migration_runs_by_version_uses_ceiling_batches() -> None:
    assert estimate_migration_runs_by_version(
        observed_by_from_version={"v2": 5, "v3": 1},
        default_run_limit=2,
    ) == {"v2": 3, "v3": 1}
