from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.errors_extended import ErrorCode
from src.core.vector_batch_similarity import run_vector_batch_similarity


class _Item:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Response(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _QdrantVector:
    def __init__(self, vector):
        self.vector = vector


class _QdrantResult:
    def __init__(self, vector_id, score, metadata, vector):
        self.id = vector_id
        self.score = score
        self.metadata = metadata
        self.vector = vector


@pytest.mark.asyncio
async def test_run_vector_batch_similarity_uses_qdrant_branch():
    payload = SimpleNamespace(
        ids=["vec1"],
        top_k=2,
        material="steel",
        complexity="high",
        format=None,
        min_score=None,
    )

    class _Store:
        async def get_vector(self, vector_id):  # noqa: ANN001, ANN201
            assert vector_id == "vec1"
            return _QdrantVector([1.0, 0.0, 0.0])

        async def search_similar(self, *_args, **kwargs):  # noqa: ANN002, ANN003, ANN201
            assert kwargs["top_k"] == 3
            assert kwargs["filter_conditions"] == {
                "material_filter": "steel",
                "complexity_filter": "high",
            }
            return [
                _QdrantResult("vec1", 1.0, {"material": "steel"}, [1.0, 0.0, 0.0]),
                _QdrantResult(
                    "vec2",
                    0.92,
                    {
                        "material": "steel",
                        "complexity": "high",
                        "format": "step",
                        "part_type": "support",
                        "fine_part_type": "support",
                        "coarse_part_type": "support",
                        "final_decision_source": "hybrid",
                        "is_coarse_label": "false",
                    },
                    [0.9, 0.1, 0.0],
                ),
            ]

    def _build_error(code, **kwargs):  # noqa: ANN001, ANN202
        return {"code": code.value, **kwargs}

    result = await run_vector_batch_similarity(
        payload=payload,
        batch_item_cls=_Item,
        batch_response_cls=_Response,
        error_code_cls=ErrorCode,
        build_error_fn=_build_error,
        get_qdrant_store_fn=lambda: _Store(),
        build_filter_conditions_fn=lambda **kwargs: {
            key: value for key, value in kwargs.items() if value is not None
        },
    )

    assert result["total"] == 1
    assert result["successful"] == 1
    assert result["failed"] == 0
    item = result["items"][0]
    assert item.status == "success"
    assert item.similar[0]["id"] == "vec2"
    assert item.similar[0]["decision_source"] == "hybrid"
    assert item.similar[0]["is_coarse_label"] is False
