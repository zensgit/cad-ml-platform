import json
import uuid
from typing import Dict, Optional
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.core.similarity import register_vector
from src.main import app

client = TestClient(app)


class DummyQdrantResult:
    def __init__(self, vector_id: str, metadata: Dict[str, str], vector: list[float]) -> None:
        self.id = vector_id
        self.metadata = metadata
        self.vector = vector


class DummyRedis:
    def __init__(self, data: Dict[object, Dict[object, str]]) -> None:
        self._data = data

    async def scan(self, cursor: int = 0, match: Optional[str] = None, count: int = 500):
        return 0, list(self._data.keys())

    async def hgetall(self, key):
        return self._data[key]


def test_vector_stats_counts():
    # register some vectors
    for m, coarse, source in [
        ("steel", "开孔件", "hybrid"),
        ("aluminum", "传动件", "graph2d"),
        ("steel", "开孔件", "hybrid"),
    ]:
        vid = str(uuid.uuid4())
        register_vector(
            vid,
            [0.1, 0.2, 0.3],
            meta={
                "material": m,
                "complexity": "low",
                "format": "dxf",
                "coarse_part_type": coarse,
                "final_decision_source": source,
            },
        )
    resp = client.get("/api/v1/vectors_stats/stats", headers={"X-API-Key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 3
    assert "steel" in data["by_material"]
    assert data["by_material"]["steel"] >= 2
    assert data["by_coarse_part_type"]["开孔件"] >= 2
    assert data["by_decision_source"]["hybrid"] >= 2


def test_vector_stats_redis_backend():
    redis_data = {
        b"vector:alpha": {
            b"v": "0.1,0.2,0.3",
            b"m": json.dumps(
                {"material": "steel", "complexity": "low", "format": "dxf", "feature_version": "v4"}
            ),
        },
        b"vector:beta": {
            b"v": "0.4,0.5",
            b"m": json.dumps(
                {
                    "material": "aluminum",
                    "complexity": "mid",
                    "format": "dwg",
                    "coarse_part_type": "传动件",
                    "final_decision_source": "graph2d",
                    "feature_version": "v3",
                }
            ),
        },
    }
    dummy = DummyRedis(redis_data)
    with patch("src.api.v1.vectors_stats.get_client", return_value=dummy), patch(
        "src.core.similarity._BACKEND", "redis"
    ):
        resp = client.get("/api/v1/vectors_stats/stats", headers={"X-API-Key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["backend"] == "redis"
    assert data["total"] == 2
    assert data["by_material"]["steel"] == 1
    assert data["by_material"]["aluminum"] == 1
    assert data["by_coarse_part_type"]["传动件"] == 1
    assert data["by_decision_source"]["graph2d"] == 1
    assert data["versions"]["v4"] == 1


def test_vector_stats_qdrant_backend():
    class DummyQdrantStore:
        async def list_vectors(self, offset=0, limit=5000, with_vectors=False, **kwargs):
            assert offset == 0
            assert limit == 5000
            assert with_vectors is True
            return (
                [
                    DummyQdrantResult(
                        "q1",
                        {
                            "material": "steel",
                            "complexity": "low",
                            "format": "dxf",
                            "coarse_part_type": "开孔件",
                            "final_decision_source": "hybrid",
                            "feature_version": "v4",
                        },
                        [0.1, 0.2, 0.3],
                    ),
                    DummyQdrantResult(
                        "q2",
                        {
                            "material": "aluminum",
                            "complexity": "mid",
                            "format": "step",
                            "coarse_part_type": "传动件",
                            "final_decision_source": "graph2d",
                            "feature_version": "v3",
                        },
                        [0.4, 0.5],
                    ),
                ],
                2,
            )

    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.vectors_stats._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ), patch("src.core.similarity._BACKEND", "qdrant"):
        resp = client.get("/api/v1/vectors_stats/stats", headers={"X-API-Key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["backend"] == "qdrant"
    assert data["total"] == 2
    assert data["by_coarse_part_type"]["开孔件"] == 1
    assert data["by_decision_source"]["graph2d"] == 1
