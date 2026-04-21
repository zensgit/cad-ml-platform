from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _auth_headers() -> dict[str, str]:
    return {"X-API-Key": "test"}


def test_migrate_pending_summary_delegates_to_plan_helper():
    async def _pending(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["target_version"] == "v4"
        return {
            "target_version": "v4",
            "from_version_filter": None,
            "observed_by_from_version": {"v2": 2},
            "total_pending": 2,
            "backend": "memory",
            "scanned_vectors": 4,
            "scan_limit": 5000,
            "distribution_complete": True,
        }

    def _payload(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["pending"]["observed_by_from_version"] == {"v2": 2}
        return {
            "target_version": "v4",
            "from_version_filter": None,
            "observed_by_from_version": {"v2": 2},
            "recommended_from_versions": ["v2"],
            "largest_pending_from_version": "v2",
            "largest_pending_count": 2,
            "total_pending": 2,
            "pending_ratio": 0.5,
            "backend": "memory",
            "scanned_vectors": 4,
            "scan_limit": 5000,
            "distribution_complete": True,
        }

    with patch(
        "src.api.v1.vectors._collect_vector_migration_pending_candidates",
        _pending,
    ), patch(
        "src.api.v1.vectors.build_vector_migration_pending_summary_payload",
        _payload,
    ):
        response = client.get("/api/v1/vectors/migrate/pending/summary", headers=_auth_headers())

    assert response.status_code == 200
    assert response.json()["recommended_from_versions"] == ["v2"]


def test_migrate_plan_delegates_to_plan_helper():
    async def _pending(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["target_version"] == "v4"
        return {
            "target_version": "v4",
            "from_version_filter": None,
            "observed_by_from_version": {"v3": 1},
            "total_pending": 1,
            "backend": "memory",
            "scanned_vectors": 2,
            "scan_limit": 5000,
            "distribution_complete": True,
        }

    def _payload(**kwargs):  # noqa: ANN003, ANN202
        assert kwargs["max_batches"] == 3
        assert kwargs["default_run_limit"] == 50
        return {
            "target_version": "v4",
            "from_version_filter": None,
            "observed_by_from_version": {"v3": 1},
            "recommended_from_versions": ["v3"],
            "largest_pending_from_version": "v3",
            "largest_pending_count": 1,
            "total_pending": 1,
            "pending_ratio": 0.5,
            "backend": "memory",
            "scanned_vectors": 2,
            "scan_limit": 5000,
            "distribution_complete": True,
            "max_batches": 3,
            "default_run_limit": 50,
            "estimated_total_runs": 1,
            "estimated_runs_by_version": {"v3": 1},
            "plan_ready": True,
            "blocking_reasons": [],
            "recommended_first_batch": {
                "priority": 1,
                "from_version": "v3",
                "pending_count": 1,
                "suggested_run_limit": 1,
                "allow_partial_scan_required": False,
                "request_payload": {
                    "limit": 1,
                    "dry_run": True,
                    "from_version_filter": "v3",
                    "allow_partial_scan": False,
                },
                "notes": ["single_batch_ready"],
            },
            "recommended_first_request_payload": {
                "limit": 1,
                "dry_run": True,
                "from_version_filter": "v3",
                "allow_partial_scan": False,
            },
            "planned_pending_count": 1,
            "remaining_pending_count": 0,
            "planned_pending_ratio": 1.0,
            "coverage_complete": True,
            "truncated_by_max_batches": False,
            "unplanned_from_versions": [],
            "suggested_next_max_batches": None,
            "batches": [
                {
                    "priority": 1,
                    "from_version": "v3",
                    "pending_count": 1,
                    "suggested_run_limit": 1,
                    "allow_partial_scan_required": False,
                    "request_payload": {
                        "limit": 1,
                        "dry_run": True,
                        "from_version_filter": "v3",
                        "allow_partial_scan": False,
                    },
                    "notes": ["single_batch_ready"],
                }
            ],
        }

    with patch(
        "src.api.v1.vectors._collect_vector_migration_pending_candidates",
        _pending,
    ), patch(
        "src.api.v1.vectors.build_vector_migration_plan_payload",
        _payload,
    ):
        response = client.get("/api/v1/vectors/migrate/plan", headers=_auth_headers())

    assert response.status_code == 200
    assert response.json()["plan_ready"] is True
