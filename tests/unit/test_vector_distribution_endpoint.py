from fastapi.testclient import TestClient
from unittest.mock import patch

from src.core.similarity import register_vector
from src.main import app

client = TestClient(app)


def test_vector_distribution_endpoint():
    register_vector(
        "dist_a",
        [0.1] * 7,
        meta={
            "material": "steel",
            "complexity": "low",
            "format": "dxf",
            "coarse_part_type": "开孔件",
            "final_decision_source": "hybrid",
        },
    )
    register_vector(
        "dist_b",
        [0.2] * 7,
        meta={
            "material": "steel",
            "complexity": "medium",
            "format": "stl",
            "coarse_part_type": "开孔件",
            "final_decision_source": "hybrid",
        },
    )
    register_vector(
        "dist_c",
        [0.3] * 7,
        meta={
            "material": "aluminum",
            "complexity": "high",
            "format": "step",
            "coarse_part_type": "传动件",
            "final_decision_source": "graph2d",
        },
    )
    resp = client.get("/api/v1/vectors_stats/distribution", headers={"x-api-key": "test"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] >= 3
    assert "steel" in data["by_material"] and "aluminum" in data["by_material"]
    assert 0.0 <= data["dominant_ratio"] <= 1.0
    assert data["by_coarse_part_type"]["开孔件"] >= 2
    assert data["by_decision_source"]["hybrid"] >= 2
    assert 0.0 <= data["dominant_coarse_ratio"] <= 1.0


def test_vector_distribution_qdrant_backend():
    class DummyQdrantResult:
        def __init__(self, vector_id, metadata, vector):
            self.id = vector_id
            self.metadata = metadata
            self.vector = vector

    class DummyQdrantStore:
        async def list_vectors(self, offset=0, limit=5000, with_vectors=False, **kwargs):
            return (
                [
                    DummyQdrantResult(
                        "dist-q1",
                        {
                            "material": "steel",
                            "complexity": "low",
                            "format": "dxf",
                            "coarse_part_type": "开孔件",
                            "final_decision_source": "hybrid",
                        },
                        [0.1, 0.2, 0.3],
                    ),
                    DummyQdrantResult(
                        "dist-q2",
                        {
                            "material": "steel",
                            "complexity": "mid",
                            "format": "step",
                            "coarse_part_type": "开孔件",
                            "final_decision_source": "hybrid",
                        },
                        [0.4, 0.5, 0.6],
                    ),
                    DummyQdrantResult(
                        "dist-q3",
                        {
                            "material": "aluminum",
                            "complexity": "high",
                            "format": "step",
                            "coarse_part_type": "传动件",
                            "final_decision_source": "graph2d",
                        },
                        [0.7, 0.8],
                    ),
                ],
                3,
            )

        async def get_stats(self):
            return {
                "collection_name": "cad_vectors",
                "points_count": 5,
                "indexed_vectors_count": 4,
                "status": "GREEN",
                "config": {"vector_size": 7, "distance": "Cosine"},
            }

        async def inspect_collection(self):
            return {
                "reachable": True,
                "collection_name": "cad_vectors",
                "collection_exists": True,
                "collection_status": "green",
                "points_count": 5,
                "vectors_count": 5,
                "indexed_vectors_count": 4,
                "unindexed_vectors_count": 1,
                "indexing_progress": 0.8,
                "requested_config": {
                    "vector_size": 7,
                    "distance": "Cosine",
                    "on_disk": False,
                    "timeout_seconds": 30.0,
                },
                "error": None,
            }

    with patch.dict(
        "os.environ",
        {"VECTOR_STORE_BACKEND": "qdrant", "VECTOR_STATS_SCAN_LIMIT": "2"},
    ), patch(
        "src.api.v1.vectors_stats._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ), patch("src.core.similarity._BACKEND", "qdrant"):
        resp = client.get("/api/v1/vectors_stats/distribution", headers={"x-api-key": "test"})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["total"] == 3
    assert data["by_coarse_part_type"]["开孔件"] == 2
    assert data["by_decision_source"]["hybrid"] == 2
    assert data["dominant_coarse_ratio"] == 0.6667
    assert data["backend_health"]["scan_truncated"] is True
    assert data["backend_health"]["reachable"] is True
    assert data["backend_health"]["collection_exists"] is True
    assert data["backend_health"]["observed_vectors_count"] == 3
    assert data["backend_health"]["points_count"] == 5
    assert data["backend_health"]["indexed_ratio"] == 0.8
    assert data["backend_health"]["unindexed_vectors_count"] == 1
    assert data["backend_health"]["readiness"] == "partial_scan"
    assert "scan_truncated_use_list_or_migration_for_exact_coverage" in data["backend_health"][
        "readiness_hints"
    ]
    assert "vector_index_backfill_in_progress" in data["backend_health"]["readiness_hints"]
