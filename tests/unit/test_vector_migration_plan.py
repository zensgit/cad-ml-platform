from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_vector_migration_plan_memory_returns_ranked_batches():
    vectors = {
        "vec1": [1.0] * 24,
        "vec2": [2.0] * 22,
        "vec3": [3.0] * 12,
        "vec4": [4.0] * 12,
        "vec5": [5.0] * 7,
    }
    meta = {
        "vec1": {"feature_version": "v4"},
        "vec2": {"feature_version": "v3"},
        "vec3": {"feature_version": "v2"},
        "vec4": {"feature_version": "v2"},
        "vec5": {"feature_version": "v1"},
    }
    with patch("src.core.similarity._VECTOR_STORE", vectors), patch(
        "src.core.similarity._VECTOR_META", meta
    ):
        response = client.get(
            "/api/v1/vectors/migrate/plan?max_batches=2&default_run_limit=1",
            headers={"x-api-key": "test"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["recommended_from_versions"] == ["v2", "v1", "v3"]
    assert data["largest_pending_from_version"] == "v2"
    assert data["largest_pending_count"] == 2
    assert data["total_pending"] == 4
    assert data["pending_ratio"] == 0.8
    assert data["max_batches"] == 2
    assert data["default_run_limit"] == 1
    assert data["estimated_runs_by_version"] == {"v3": 1, "v2": 2, "v1": 1}
    assert data["estimated_total_runs"] == 4
    assert data["plan_ready"] is True
    assert data["blocking_reasons"] == []
    assert data["recommended_first_request_payload"] == {
        "limit": 1,
        "dry_run": True,
        "from_version_filter": "v2",
        "allow_partial_scan": False,
    }
    assert data["recommended_first_batch"] == {
        "priority": 1,
        "from_version": "v2",
        "pending_count": 2,
        "suggested_run_limit": 1,
        "allow_partial_scan_required": False,
        "request_payload": {
            "limit": 1,
            "dry_run": True,
            "from_version_filter": "v2",
            "allow_partial_scan": False,
        },
        "notes": ["split_batch_required"],
    }
    assert data["batches"] == [
        {
            "priority": 1,
            "from_version": "v2",
            "pending_count": 2,
            "suggested_run_limit": 1,
            "allow_partial_scan_required": False,
            "request_payload": {
                "limit": 1,
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


def test_vector_migration_plan_qdrant_partial_requires_override():
    class DummyPoint:
        def __init__(self, point_id, metadata):
            self.id = point_id
            self.metadata = metadata

    class DummyQdrantStore:
        async def count(self):
            return 5

        async def list_vectors(self, offset=0, limit=50, with_vectors=False):
            items = [
                DummyPoint("vec1", {"feature_version": "v4"}),
                DummyPoint("vec2", {"feature_version": "v3"}),
                DummyPoint("vec3", {"feature_version": "v2"}),
                DummyPoint("vec4", {"feature_version": "v2"}),
                DummyPoint("vec5", {"feature_version": "v1"}),
            ]
            return items[offset : offset + limit], 5

    with patch.dict(
        "os.environ",
        {"VECTOR_STORE_BACKEND": "qdrant", "VECTOR_MIGRATION_SCAN_LIMIT": "2"},
    ), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        response = client.get(
            "/api/v1/vectors/migrate/plan",
            headers={"x-api-key": "test"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "qdrant"
    assert data["distribution_complete"] is False
    assert data["total_pending"] is None
    assert data["pending_ratio"] is None
    assert data["estimated_runs_by_version"] == {"v3": 1}
    assert data["estimated_total_runs"] == 1
    assert data["plan_ready"] is False
    assert data["blocking_reasons"] == ["partial_scan_override_required"]
    assert data["recommended_first_request_payload"] == {
        "limit": 1,
        "dry_run": True,
        "from_version_filter": "v3",
        "allow_partial_scan": True,
    }
    assert data["recommended_first_batch"] == {
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
    assert data["batches"] == [
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


def test_vector_migration_plan_applies_from_version_filter():
    vectors = {
        "vec1": [1.0] * 24,
        "vec2": [2.0] * 22,
        "vec3": [3.0] * 12,
        "vec4": [4.0] * 12,
    }
    meta = {
        "vec1": {"feature_version": "v4"},
        "vec2": {"feature_version": "v3"},
        "vec3": {"feature_version": "v2"},
        "vec4": {"feature_version": "v2"},
    }
    with patch("src.core.similarity._VECTOR_STORE", vectors), patch(
        "src.core.similarity._VECTOR_META", meta
    ):
        response = client.get(
            "/api/v1/vectors/migrate/plan?from_version_filter=v2",
            headers={"x-api-key": "test"},
        )

    assert response.status_code == 200
    data = response.json()
    assert data["from_version_filter"] == "v2"
    assert data["recommended_from_versions"] == ["v2"]
    assert data["largest_pending_from_version"] == "v2"
    assert data["largest_pending_count"] == 2
    assert data["estimated_runs_by_version"] == {"v2": 1}
    assert data["estimated_total_runs"] == 1
    assert data["plan_ready"] is True
    assert data["blocking_reasons"] == []
    assert data["recommended_first_request_payload"] == {
        "limit": 2,
        "dry_run": True,
        "from_version_filter": "v2",
        "allow_partial_scan": False,
    }
    assert data["recommended_first_batch"] == {
        "priority": 1,
        "from_version": "v2",
        "pending_count": 2,
        "suggested_run_limit": 2,
        "allow_partial_scan_required": False,
        "request_payload": {
            "limit": 2,
            "dry_run": True,
            "from_version_filter": "v2",
            "allow_partial_scan": False,
        },
        "notes": ["single_batch_ready"],
    }
    assert data["batches"] == [
        {
            "priority": 1,
            "from_version": "v2",
            "pending_count": 2,
            "suggested_run_limit": 2,
            "allow_partial_scan_required": False,
            "request_payload": {
                "limit": 2,
                "dry_run": True,
                "from_version_filter": "v2",
                "allow_partial_scan": False,
            },
            "notes": ["single_batch_ready"],
        }
    ]
