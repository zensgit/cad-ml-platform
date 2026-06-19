from __future__ import annotations

from src.core.vector_migration_plan_outcome import (
    build_vector_migration_plan_outcome,
)


def test_build_plan_outcome_complete_truncated_distribution() -> None:
    outcome = build_vector_migration_plan_outcome(
        summary={"recommended_from_versions": ["v2", "v1", "v3"]},
        pending={"distribution_complete": True, "total_pending": 4},
        batches=[
            {
                "from_version": "v2",
                "pending_count": 2,
                "request_payload": {"from_version_filter": "v2"},
            },
            {
                "from_version": "v1",
                "pending_count": 1,
                "request_payload": {"from_version_filter": "v1"},
            },
        ],
        allow_partial_scan_required=False,
    )

    assert outcome["plan_ready"] is True
    assert outcome["blocking_reasons"] == []
    assert outcome["recommended_first_batch"]["from_version"] == "v2"
    assert outcome["recommended_first_request_payload"] == {
        "from_version_filter": "v2"
    }
    assert outcome["planned_pending_count"] == 3
    assert outcome["remaining_pending_count"] == 1
    assert outcome["planned_pending_ratio"] == 0.75
    assert outcome["coverage_complete"] is False
    assert outcome["truncated_by_max_batches"] is True
    assert outcome["unplanned_from_versions"] == ["v3"]
    assert outcome["suggested_next_max_batches"] == 3


def test_build_plan_outcome_partial_scan_blocks_ready() -> None:
    outcome = build_vector_migration_plan_outcome(
        summary={"recommended_from_versions": ["v3"]},
        pending={"distribution_complete": False, "total_pending": None},
        batches=[
            {
                "from_version": "v3",
                "pending_count": 1,
                "request_payload": {
                    "from_version_filter": "v3",
                    "allow_partial_scan": True,
                },
            }
        ],
        allow_partial_scan_required=True,
    )

    assert outcome["plan_ready"] is False
    assert outcome["blocking_reasons"] == ["partial_scan_override_required"]
    assert outcome["remaining_pending_count"] is None
    assert outcome["planned_pending_ratio"] is None
    assert outcome["coverage_complete"] is True
    assert outcome["recommended_first_request_payload"] == {
        "from_version_filter": "v3",
        "allow_partial_scan": True,
    }


def test_build_plan_outcome_no_batches_blocks_ready() -> None:
    outcome = build_vector_migration_plan_outcome(
        summary={"recommended_from_versions": []},
        pending={"distribution_complete": True, "total_pending": 0},
        batches=[],
        allow_partial_scan_required=False,
    )

    assert outcome["plan_ready"] is False
    assert outcome["blocking_reasons"] == ["no_pending_vectors"]
    assert outcome["recommended_first_batch"] is None
    assert outcome["recommended_first_request_payload"] is None
    assert outcome["planned_pending_count"] == 0
    assert outcome["remaining_pending_count"] == 0
    assert outcome["planned_pending_ratio"] == 0.0
    assert outcome["coverage_complete"] is False
    assert outcome["truncated_by_max_batches"] is False
