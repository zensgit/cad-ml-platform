import json
import uuid
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.core.similarity import register_vector
from src.main import app

client = TestClient(app)


class DummyRedis:
    def __init__(self, data: dict[object, dict[object, str]]) -> None:
        self._data = data

    async def scan(self, cursor: int = 0, match: str | None = None, count: int = 500):
        return 0, list(self._data.keys())

    async def hgetall(self, key):
        return self._data[key]


def test_vector_stats_counts():
    # register some vectors
    for m in ["steel", "aluminum", "steel"]:
        vid = str(uuid.uuid4())
        register_vector(
            vid, [0.1, 0.2, 0.3], meta={"material": m, "complexity": "low", "format": "dxf"}
        )
    resp = client.get("/api/v1/vectors_stats/stats", headers={"X-API-Key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 3
    assert "steel" in data["by_material"]
    assert data["by_material"]["steel"] >= 2


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
    assert data["versions"]["v4"] == 1
