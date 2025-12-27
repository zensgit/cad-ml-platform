import json
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.core.errors_extended import ErrorCode
from src.main import app


class DummyRedis:
    def __init__(self, data: dict[object, dict[object, str]]) -> None:
        self._data = data

    async def scan(self, cursor: int = 0, match: str | None = None, count: int = 500):
        return 0, list(self._data.keys())

    async def hgetall(self, key):
        return self._data[key]


def test_vectors_list_endpoint():
    client = TestClient(app)
    r = client.get("/api/v1/vectors", headers={"X-API-Key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert "total" in data and "vectors" in data


def test_vectors_update_not_found():
    client = TestClient(app)
    r = client.post(
        "/api/v1/vectors/update",
        json={"id": "nope", "replace": [1.0, 2.0]},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "not_found"


def test_vectors_migrate_dry_run():
    client = TestClient(app)
    r = client.post(
        "/api/v1/vectors/migrate",
        json={"ids": ["none"], "to_version": "v2", "dry_run": True},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert "items" in data


def test_vectors_list_invalid_source():
    client = TestClient(app)
    r = client.get("/api/v1/vectors?source=invalid", headers={"X-API-Key": "test"})
    assert r.status_code == 400
    data = r.json()
    assert data["detail"]["code"] == ErrorCode.INPUT_VALIDATION_FAILED.value


def test_vectors_list_pagination_memory():
    client = TestClient(app)
    vector_store = {"vec1": [1.0], "vec2": [2.0], "vec3": [3.0]}
    vector_meta = {
        "vec1": {"material": "steel", "complexity": "low", "format": "dxf"},
        "vec2": {"material": "aluminum", "complexity": "mid", "format": "dwg"},
        "vec3": {"material": "steel", "complexity": "high", "format": "step"},
    }
    with patch("src.core.similarity._VECTOR_STORE", vector_store), patch(
        "src.core.similarity._VECTOR_META", vector_meta
    ), patch("src.core.similarity._BACKEND", "memory"):
        r = client.get(
            "/api/v1/vectors?source=memory&offset=1&limit=1", headers={"X-API-Key": "test"}
        )
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 3
    assert len(data["vectors"]) == 1
    assert data["vectors"][0]["id"] == "vec2"


def test_vectors_list_redis_source():
    client = TestClient(app)
    redis_data = {
        b"vector:vec1": {
            b"v": "1,2,3",
            b"m": json.dumps({"material": "steel", "complexity": "low", "format": "dxf"}),
        },
        b"vector:vec2": {
            b"v": "4,5",
            b"m": json.dumps({"material": "aluminum", "complexity": "mid", "format": "dwg"}),
        },
    }
    dummy = DummyRedis(redis_data)
    with patch("src.api.v1.vectors.get_client", return_value=dummy), patch(
        "src.core.similarity._BACKEND", "redis"
    ):
        r = client.get(
            "/api/v1/vectors?source=redis&offset=1&limit=1", headers={"X-API-Key": "test"}
        )
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert len(data["vectors"]) == 1
    assert data["vectors"][0]["id"] == "vec2"
    assert data["vectors"][0]["dimension"] == 2


def test_vectors_register_and_search():
    client = TestClient(app)
    payload = {
        "id": "vec1",
        "vector": [0.1] * 7,
        "meta": {"material": "steel", "complexity": "low"},
    }
    resp = client.post(
        "/api/v1/vectors/register",
        json=payload,
        headers={"X-API-Key": "test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "accepted"

    search = client.post(
        "/api/v1/vectors/search",
        json={"vector": [0.1] * 7, "k": 5},
        headers={"X-API-Key": "test"},
    )
    assert search.status_code == 200
    results = search.json()["results"]
    assert any(item["id"] == "vec1" for item in results)


def test_vectors_search_with_filters():
    client = TestClient(app)
    client.post(
        "/api/v1/vectors/register",
        json={
            "id": "steel_vec",
            "vector": [0.2] * 7,
            "meta": {"material": "steel"},
        },
        headers={"X-API-Key": "test"},
    )
    client.post(
        "/api/v1/vectors/register",
        json={
            "id": "alu_vec",
            "vector": [0.2] * 7,
            "meta": {"material": "aluminum"},
        },
        headers={"X-API-Key": "test"},
    )

    search = client.post(
        "/api/v1/vectors/search",
        json={
            "vector": [0.2] * 7,
            "k": 5,
            "material_filter": "steel",
        },
        headers={"X-API-Key": "test"},
    )
    assert search.status_code == 200
    results = search.json()["results"]
    assert results
    assert all(item["material"] == "steel" for item in results)
