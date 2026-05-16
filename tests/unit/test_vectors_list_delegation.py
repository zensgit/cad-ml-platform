from __future__ import annotations

from fastapi.testclient import TestClient

from src.api.v1 import vectors as vectors_module
from src.api.v1.vector_list_models import VectorListItem, VectorListResponse
from src.main import app

client = TestClient(app)


def test_vectors_list_route_delegates_to_shared_helper(monkeypatch):
    captured = {}

    async def _stub_run_vector_list_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {"total": 1, "vectors": [{"id": "vec1", "dimension": 7}]}

    def _sentinel_resolve_list_source(source, backend):  # noqa: ANN001, ANN201
        return f"{source}:{backend}"

    monkeypatch.setattr(
        "src.api.v1.vectors.run_vector_list_pipeline",
        _stub_run_vector_list_pipeline,
    )
    monkeypatch.setattr(
        "src.api.v1.vectors._resolve_list_source",
        _sentinel_resolve_list_source,
    )

    response = client.get(
        (
            "/api/v1/vectors/?source=memory&offset=2&limit=5"
            "&material_filter=steel&complexity_filter=high"
            "&fine_part_type_filter=轴&coarse_part_type_filter=回转件"
            "&decision_source_filter=classifier&is_coarse_label_filter=true"
        ),
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["total"] == 1
    assert body["vectors"][0]["id"] == "vec1"
    assert captured["source"] == "memory"
    assert captured["offset"] == 2
    assert captured["limit"] == 5
    assert captured["material_filter"] == "steel"
    assert captured["complexity_filter"] == "high"
    assert captured["fine_part_type_filter"] == "轴"
    assert captured["coarse_part_type_filter"] == "回转件"
    assert captured["decision_source_filter"] == "classifier"
    assert captured["is_coarse_label_filter"] is True
    assert captured["response_cls"] is VectorListResponse
    assert captured["item_cls"] is VectorListItem
    assert captured["get_qdrant_store_fn"] is vectors_module._get_qdrant_store_or_none
    assert captured["resolve_list_source_fn"] is _sentinel_resolve_list_source
    assert captured["build_filter_conditions_fn"] is (
        vectors_module._build_vector_filter_conditions
    )
    assert captured["list_vectors_redis_fn"] is vectors_module._list_vectors_redis
    assert captured["list_vectors_memory_fn"] is vectors_module._list_vectors_memory
    assert captured["get_client_fn"] is vectors_module.get_client
