from __future__ import annotations

from src.core.vector_migration_plan_pipeline import (
    build_vector_migration_pending_summary_payload,
    build_vector_migration_plan_payload,
)


def test_build_vector_migration_pending_summary_payload():
    payload = build_vector_migration_pending_summary_payload(
        pending={
            "target_version": "v4",
            "from_version_filter": None,
            "observed_by_from_version": {"v3": 1, "v2": 2, "v1": 1},
            "total_pending": 4,
            "backend": "memory",
            "scanned_vectors": 5,
            "scan_limit": 5000,
            "distribution_complete": True,
        }
    )

    assert payload["recommended_from_versions"] == ["v2", "v1", "v3"]
    assert payload["largest_pending_from_version"] == "v2"
    assert payload["largest_pending_count"] == 2
    assert payload["pending_ratio"] == 0.8


def test_build_vector_migration_plan_payload_partial_scan():
    payload = build_vector_migration_plan_payload(
        pending={
            "target_version": "v4",
            "from_version_filter": None,
            "observed_by_from_version": {"v3": 1},
            "total_pending": None,
            "backend": "qdrant",
            "scanned_vectors": 2,
            "scan_limit": 2,
            "distribution_complete": False,
        },
        max_batches=3,
        default_run_limit=50,
    )

    assert payload["plan_ready"] is False
    assert payload["blocking_reasons"] == ["partial_scan_override_required"]
    assert payload["recommended_first_request_payload"] == {
        "limit": 1,
        "dry_run": True,
        "from_version_filter": "v3",
        "allow_partial_scan": True,
    }
    assert payload["batches"] == [
        {
            "priority": 1,
            "from_version": "v3",
            "pending_count": 1,
            "suggested_run_limit": 1,
            "allow_partial_scan_required": True,
            "request_payload": {
                "limit": 1,
                "dry_run": True,
                "from_version_filter": "v3",
                "allow_partial_scan": True,
            },
            "notes": ["single_batch_ready", "partial_scan_override_required"],
        }
    ]
