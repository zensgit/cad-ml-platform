import json
from typing import Dict, Optional
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.core.errors_extended import ErrorCode
from src.main import app


class DummyQdrantResult:
    def __init__(
        self,
        vector_id: str,
        score: float,
        metadata: Dict[str, object],
        vector: Optional[list[float]] = None,
    ) -> None:
        self.id = vector_id
        self.score = score
        self.metadata = metadata
        self.vector = vector


class DummyRedis:
    def __init__(self, data: Dict[object, Dict[object, str]]) -> None:
        self._data = data

    async def scan(self, cursor: int = 0, match: Optional[str] = None, count: int = 500):
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
        "vec2": {
            "material": "aluminum",
            "complexity": "mid",
            "format": "dwg",
            "part_type": "人孔",
            "fine_part_type": "人孔",
            "coarse_part_type": "开孔件",
            "final_decision_source": "hybrid",
            "is_coarse_label": "false",
        },
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
    assert data["vectors"][0]["part_type"] == "人孔"
    assert data["vectors"][0]["fine_part_type"] == "人孔"
    assert data["vectors"][0]["coarse_part_type"] == "开孔件"
    assert data["vectors"][0]["decision_source"] == "hybrid"
    assert data["vectors"][0]["is_coarse_label"] is False


def test_vectors_list_supports_coarse_contract_filters_memory():
    client = TestClient(app)
    vector_store = {"vec1": [1.0], "vec2": [2.0], "vec3": [3.0]}
    vector_meta = {
        "vec1": {
            "material": "steel",
            "complexity": "low",
            "format": "dxf",
            "part_type": "人孔",
            "fine_part_type": "人孔",
            "coarse_part_type": "开孔件",
            "final_decision_source": "hybrid",
            "is_coarse_label": "false",
        },
        "vec2": {
            "material": "aluminum",
            "complexity": "mid",
            "format": "dwg",
            "part_type": "法兰",
            "fine_part_type": "法兰",
            "coarse_part_type": "法兰",
            "final_decision_source": "filename",
            "is_coarse_label": "true",
        },
        "vec3": {"material": "steel", "complexity": "high", "format": "step"},
    }
    with patch("src.core.similarity._VECTOR_STORE", vector_store), patch(
        "src.core.similarity._VECTOR_META", vector_meta
    ), patch("src.core.similarity._BACKEND", "memory"):
        r = client.get(
            "/api/v1/vectors?source=memory&coarse_part_type_filter=开孔件"
            "&decision_source_filter=hybrid&is_coarse_label_filter=false",
            headers={"X-API-Key": "test"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1
    assert len(data["vectors"]) == 1
    assert data["vectors"][0]["id"] == "vec1"
    assert data["vectors"][0]["coarse_part_type"] == "开孔件"
    assert data["vectors"][0]["decision_source"] == "hybrid"
    assert data["vectors"][0]["is_coarse_label"] is False


def test_vectors_list_redis_source():
    client = TestClient(app)
    redis_data = {
        b"vector:vec1": {
            b"v": "1,2,3",
            b"m": json.dumps({"material": "steel", "complexity": "low", "format": "dxf"}),
        },
        b"vector:vec2": {
            b"v": "4,5",
            b"m": json.dumps(
                {
                    "material": "aluminum",
                    "complexity": "mid",
                    "format": "dwg",
                    "part_type": "人孔",
                    "fine_part_type": "人孔",
                    "coarse_part_type": "开孔件",
                    "final_decision_source": "hybrid",
                    "is_coarse_label": "false",
                }
            ),
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
    assert data["vectors"][0]["coarse_part_type"] == "开孔件"
    assert data["vectors"][0]["decision_source"] == "hybrid"
    assert data["vectors"][0]["is_coarse_label"] is False


def test_vectors_list_redis_supports_coarse_contract_filters():
    client = TestClient(app)
    redis_data = {
        b"vector:vec1": {
            b"v": "1,2,3",
            b"m": json.dumps(
                {
                    "material": "steel",
                    "complexity": "low",
                    "format": "dxf",
                    "part_type": "人孔",
                    "fine_part_type": "人孔",
                    "coarse_part_type": "开孔件",
                    "final_decision_source": "hybrid",
                    "is_coarse_label": "false",
                }
            ),
        },
        b"vector:vec2": {
            b"v": "4,5",
            b"m": json.dumps(
                {
                    "material": "steel",
                    "complexity": "mid",
                    "format": "dwg",
                    "part_type": "法兰",
                    "fine_part_type": "法兰",
                    "coarse_part_type": "法兰",
                    "final_decision_source": "filename",
                    "is_coarse_label": "true",
                }
            ),
        },
    }
    dummy = DummyRedis(redis_data)
    with patch("src.api.v1.vectors.get_client", return_value=dummy), patch(
        "src.core.similarity._BACKEND", "redis"
    ):
        r = client.get(
            "/api/v1/vectors?source=redis&coarse_part_type_filter=开孔件"
            "&decision_source_filter=hybrid&is_coarse_label_filter=false",
            headers={"X-API-Key": "test"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 1
    assert len(data["vectors"]) == 1
    assert data["vectors"][0]["id"] == "vec1"
    assert data["vectors"][0]["coarse_part_type"] == "开孔件"
    assert data["vectors"][0]["decision_source"] == "hybrid"
    assert data["vectors"][0]["is_coarse_label"] is False


def test_vectors_register_and_search():
    client = TestClient(app)
    payload = {
        "id": "vec1",
        "vector": [0.1] * 7,
        "meta": {
            "material": "steel",
            "complexity": "low",
            "part_type": "人孔",
            "fine_part_type": "人孔",
            "coarse_part_type": "开孔件",
            "final_decision_source": "hybrid",
            "is_coarse_label": "false",
        },
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
    vec1 = next(item for item in results if item["id"] == "vec1")
    assert vec1["part_type"] == "人孔"
    assert vec1["fine_part_type"] == "人孔"
    assert vec1["coarse_part_type"] == "开孔件"
    assert vec1["decision_source"] == "hybrid"
    assert vec1["is_coarse_label"] is False


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


def test_vectors_search_with_coarse_contract_filters():
    client = TestClient(app)
    client.post(
        "/api/v1/vectors/register",
        json={
            "id": "manhole_vec",
            "vector": [0.25] * 7,
            "meta": {
                "part_type": "人孔",
                "fine_part_type": "人孔",
                "coarse_part_type": "开孔件",
                "final_decision_source": "hybrid",
                "is_coarse_label": "false",
            },
        },
        headers={"X-API-Key": "test"},
    )
    client.post(
        "/api/v1/vectors/register",
        json={
            "id": "flange_vec",
            "vector": [0.25] * 7,
            "meta": {
                "part_type": "法兰",
                "fine_part_type": "法兰",
                "coarse_part_type": "法兰",
                "final_decision_source": "filename",
                "is_coarse_label": "true",
            },
        },
        headers={"X-API-Key": "test"},
    )

    search = client.post(
        "/api/v1/vectors/search",
        json={
            "vector": [0.25] * 7,
            "k": 5,
            "coarse_part_type_filter": "开孔件",
            "decision_source_filter": "hybrid",
            "is_coarse_label_filter": False,
        },
        headers={"X-API-Key": "test"},
    )
    assert search.status_code == 200
    results = search.json()["results"]
    assert results
    ids = {item["id"] for item in results}
    assert "manhole_vec" in ids
    assert "flange_vec" not in ids
    assert all(item["coarse_part_type"] == "开孔件" for item in results)
    assert all(item["decision_source"] == "hybrid" for item in results)
    assert all(item["is_coarse_label"] is False for item in results)


def test_vectors_list_qdrant_source_supports_coarse_contract_filters():
    client = TestClient(app)

    class DummyQdrantStore:
        async def list_vectors(self, offset, limit, filter_conditions=None, with_vectors=False):
            assert offset == 0
            assert limit == 10
            assert filter_conditions == {
                "coarse_part_type": "开孔件",
                "decision_source": "hybrid",
                "is_coarse_label": False,
            }
            return (
                [
                    DummyQdrantResult(
                        "qdrant-1",
                        1.0,
                        {
                            "part_type": "人孔",
                            "fine_part_type": "人孔",
                            "coarse_part_type": "开孔件",
                            "decision_source": "hybrid",
                            "is_coarse_label": False,
                            "material": "steel",
                        },
                        vector=[0.1] * 7,
                    )
                ],
                1,
            )

    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        resp = client.get(
            "/api/v1/vectors?source=qdrant&limit=10&coarse_part_type_filter=开孔件"
            "&decision_source_filter=hybrid&is_coarse_label_filter=false",
            headers={"X-API-Key": "test"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert len(data["vectors"]) == 1
    assert data["vectors"][0]["id"] == "qdrant-1"
    assert data["vectors"][0]["coarse_part_type"] == "开孔件"
    assert data["vectors"][0]["decision_source"] == "hybrid"
    assert data["vectors"][0]["is_coarse_label"] is False


def test_vectors_search_uses_qdrant_native_filters_when_enabled():
    client = TestClient(app)

    class DummyQdrantStore:
        async def search_similar(self, query_vector, top_k=10, filter_conditions=None, **kwargs):
            assert query_vector == [0.2] * 7
            assert top_k == 5
            assert filter_conditions == {
                "coarse_part_type": "开孔件",
                "decision_source": "hybrid",
                "is_coarse_label": False,
            }
            return [
                DummyQdrantResult(
                    "qdrant-search-1",
                    0.93,
                    {
                        "part_type": "人孔",
                        "fine_part_type": "人孔",
                        "coarse_part_type": "开孔件",
                        "decision_source": "hybrid",
                        "is_coarse_label": False,
                        "material": "steel",
                    },
                )
            ]

    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        resp = client.post(
            "/api/v1/vectors/search",
            json={
                "vector": [0.2] * 7,
                "k": 5,
                "coarse_part_type_filter": "开孔件",
                "decision_source_filter": "hybrid",
                "is_coarse_label_filter": False,
            },
            headers={"X-API-Key": "test"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 1
    assert data["results"][0]["id"] == "qdrant-search-1"
    assert data["results"][0]["coarse_part_type"] == "开孔件"
    assert data["results"][0]["decision_source"] == "hybrid"
    assert data["results"][0]["is_coarse_label"] is False


def test_vectors_register_uses_qdrant_when_enabled():
    client = TestClient(app)

    class DummyQdrantStore:
        def __init__(self) -> None:
            self.calls = []

        async def register_vector(self, vector_id, vector, metadata=None):
            self.calls.append((vector_id, vector, metadata))
            return True

    store = DummyQdrantStore()
    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value=store,
    ):
        resp = client.post(
            "/api/v1/vectors/register",
            json={"id": "qdrant-reg", "vector": [0.1] * 7, "meta": {"material": "steel"}},
            headers={"X-API-Key": "test"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "accepted"
    assert store.calls
    _, _, metadata = store.calls[0]
    assert metadata["material"] == "steel"
    assert metadata["total_dim"] == "7"


def test_vectors_delete_uses_qdrant_when_enabled():
    client = TestClient(app)

    class DummyQdrantStore:
        async def get_vector(self, vector_id):
            return DummyQdrantResult(vector_id, 1.0, {}, vector=[0.1] * 7)

        async def delete_vector(self, vector_id):
            return vector_id == "qdrant-del"

    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        resp = client.post(
            "/api/v1/vectors/delete",
            json={"id": "qdrant-del"},
            headers={"X-API-Key": "test"},
        )
    assert resp.status_code == 200
    assert resp.json()["status"] == "deleted"


def test_vectors_delete_qdrant_failure_returns_internal_error():
    client = TestClient(app)

    class DummyQdrantStore:
        async def get_vector(self, vector_id):
            return DummyQdrantResult(vector_id, 1.0, {}, vector=[0.1] * 7)

        async def delete_vector(self, vector_id):
            return False

    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value=DummyQdrantStore(),
    ):
        resp = client.post(
            "/api/v1/vectors/delete",
            json={"id": "qdrant-del-fail"},
            headers={"X-API-Key": "test"},
        )

    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert detail["code"] == ErrorCode.INTERNAL_ERROR.value
    assert detail["stage"] == "vector_delete"


def test_vectors_update_uses_qdrant_when_enabled():
    client = TestClient(app)

    class DummyQdrantStore:
        def __init__(self) -> None:
            self.calls = []

        async def get_vector(self, vector_id):
            return DummyQdrantResult(
                vector_id,
                1.0,
                {"material": "steel", "feature_version": "v2", "created_at": "x"},
                vector=[0.1] * 7,
            )

        async def register_vector(self, vector_id, vector, metadata=None):
            self.calls.append((vector_id, vector, metadata))
            return True

    store = DummyQdrantStore()
    with patch.dict("os.environ", {"VECTOR_STORE_BACKEND": "qdrant"}), patch(
        "src.api.v1.vectors._get_qdrant_store_or_none",
        return_value=store,
    ):
        resp = client.post(
            "/api/v1/vectors/update",
            json={"id": "qdrant-upd", "replace": [0.2] * 7, "complexity": "high"},
            headers={"X-API-Key": "test"},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "updated"
    assert data["dimension"] == 7
    assert data["feature_version"] == "v2"
    _, vector, metadata = store.calls[0]
    assert vector == [0.2] * 7
    assert metadata["complexity"] == "high"
    assert metadata["total_dim"] == "7"
