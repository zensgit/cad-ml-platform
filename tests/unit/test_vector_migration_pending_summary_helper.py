from __future__ import annotations

from src.core.vector_migration_pending_summary import (
    build_vector_migration_pending_summary_payload,
)


def test_build_pending_summary_payload_complete_distribution() -> None:
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

    assert payload == {
        "target_version": "v4",
        "from_version_filter": None,
        "observed_by_from_version": {"v3": 1, "v2": 2, "v1": 1},
        "recommended_from_versions": ["v2", "v1", "v3"],
        "largest_pending_from_version": "v2",
        "largest_pending_count": 2,
        "total_pending": 4,
        "pending_ratio": 0.8,
        "backend": "memory",
        "scanned_vectors": 5,
        "scan_limit": 5000,
        "distribution_complete": True,
    }


def test_build_pending_summary_payload_partial_distribution_hides_ratio() -> None:
    payload = build_vector_migration_pending_summary_payload(
        pending={
            "target_version": "v4",
            "from_version_filter": "v3",
            "observed_by_from_version": {"v3": 1},
            "total_pending": None,
            "backend": "qdrant",
            "scanned_vectors": 2,
            "scan_limit": 2,
            "distribution_complete": False,
        }
    )

    assert payload["from_version_filter"] == "v3"
    assert payload["recommended_from_versions"] == ["v3"]
    assert payload["largest_pending_from_version"] == "v3"
    assert payload["largest_pending_count"] == 1
    assert payload["total_pending"] is None
    assert payload["pending_ratio"] is None


def test_build_pending_summary_payload_empty_distribution() -> None:
    payload = build_vector_migration_pending_summary_payload(
        pending={
            "target_version": "v4",
            "from_version_filter": None,
            "observed_by_from_version": {},
            "total_pending": 0,
            "backend": "memory",
            "scanned_vectors": 0,
            "scan_limit": 5000,
            "distribution_complete": True,
        }
    )

    assert payload["recommended_from_versions"] == []
    assert payload["largest_pending_from_version"] is None
    assert payload["largest_pending_count"] is None
    assert payload["pending_ratio"] == 0.0
