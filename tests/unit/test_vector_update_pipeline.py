from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.vector_update_pipeline import run_vector_update_pipeline


@pytest.mark.asyncio
async def test_run_vector_update_pipeline_memory_not_found():
    payload = SimpleNamespace(
        id="missing",
        replace=None,
        append=None,
        material=None,
        complexity=None,
        format=None,
    )

    result = await run_vector_update_pipeline(payload=payload)

    assert result["status"] == "not_found"
    assert result["error"]["code"] == "DATA_NOT_FOUND"


@pytest.mark.asyncio
async def test_run_vector_update_pipeline_qdrant_updates_metadata():
    captured = {}

    class DummyPoint:
        vector = [0.1, 0.2, 0.3]
        metadata = {"feature_version": "v2"}

    class DummyStore:
        async def get_vector(self, vector_id):
            assert vector_id == "vec-1"
            return DummyPoint()

        async def register_vector(self, vector_id, vector, metadata):
            captured["id"] = vector_id
            captured["vector"] = vector
            captured["metadata"] = metadata

    payload = SimpleNamespace(
        id="vec-1",
        replace=[0.9, 0.8, 0.7],
        append=None,
        material="steel",
        complexity="low",
        format="dxf",
    )

    result = await run_vector_update_pipeline(payload=payload, qdrant_store=DummyStore())

    assert result["status"] == "updated"
    assert result["dimension"] == 3
    assert captured["id"] == "vec-1"
    assert captured["vector"] == [0.9, 0.8, 0.7]
    assert captured["metadata"]["material"] == "steel"
    assert captured["metadata"]["complexity"] == "low"
    assert captured["metadata"]["format"] == "dxf"
    assert captured["metadata"]["total_dim"] == "3"
