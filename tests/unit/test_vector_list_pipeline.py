from __future__ import annotations

import pytest

from src.core.errors_extended import ErrorCode
from src.core.vector_list_pipeline import run_vector_list_pipeline


class _Item:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Response(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _QdrantResult:
    def __init__(self, vector_id, metadata, vector):
        self.id = vector_id
        self.metadata = metadata
        self.vector = vector


@pytest.mark.asyncio
async def test_run_vector_list_pipeline_uses_qdrant_branch():
    class _Store:
        async def list_vectors(self, offset, limit, filter_conditions=None, with_vectors=False):  # noqa: ANN001, ANN201
            assert offset == 0
            assert limit == 10
            assert filter_conditions == {
                "coarse_part_type": "开孔件",
                "decision_source": "hybrid",
                "is_coarse_label": False,
            }
            assert with_vectors is True
            return (
                [
                    _QdrantResult(
                        "qdrant-1",
                        {
                            "material": "steel",
                            "coarse_part_type": "开孔件",
                            "decision_source": "hybrid",
                            "is_coarse_label": False,
                        },
                        [0.1] * 7,
                    )
                ],
                1,
            )

    result = await run_vector_list_pipeline(
        source="qdrant",
        offset=0,
        limit=10,
        material_filter=None,
        complexity_filter=None,
        fine_part_type_filter=None,
        coarse_part_type_filter="开孔件",
        decision_source_filter="hybrid",
        is_coarse_label_filter=False,
        response_cls=_Response,
        item_cls=_Item,
        error_code_cls=ErrorCode,
        build_error_fn=lambda code, **kwargs: {"code": code.value, **kwargs},
        get_qdrant_store_fn=lambda: _Store(),
        resolve_list_source_fn=lambda source, _backend: source,
        build_filter_conditions_fn=lambda **kwargs: {
            "coarse_part_type": kwargs["coarse_part_type_filter"],
            "decision_source": kwargs["decision_source_filter"],
            "is_coarse_label": kwargs["is_coarse_label_filter"],
        },
        list_vectors_redis_fn=lambda *_args, **_kwargs: None,
        list_vectors_memory_fn=lambda *_args, **_kwargs: None,
        get_client_fn=lambda: None,
    )

    assert result["total"] == 1
    assert result["vectors"][0].id == "qdrant-1"
    assert result["vectors"][0].coarse_part_type == "开孔件"


@pytest.mark.asyncio
async def test_run_vector_list_pipeline_applies_limits_to_redis_branch(monkeypatch):
    monkeypatch.setenv("VECTOR_LIST_LIMIT", "4")
    monkeypatch.setenv("VECTOR_LIST_SCAN_LIMIT", "9")
    captured = {}

    async def _list_redis(*args):  # noqa: ANN002, ANN202
        captured["args"] = args
        return _Response(total=0, vectors=[])

    result = await run_vector_list_pipeline(
        source="redis",
        offset=1,
        limit=10,
        material_filter=None,
        complexity_filter=None,
        fine_part_type_filter=None,
        coarse_part_type_filter=None,
        decision_source_filter=None,
        is_coarse_label_filter=None,
        response_cls=_Response,
        item_cls=_Item,
        error_code_cls=ErrorCode,
        build_error_fn=lambda code, **kwargs: {"code": code.value, **kwargs},
        get_qdrant_store_fn=lambda: None,
        resolve_list_source_fn=lambda source, _backend: source,
        build_filter_conditions_fn=lambda **_kwargs: {},
        list_vectors_redis_fn=_list_redis,
        list_vectors_memory_fn=lambda *_args, **_kwargs: None,
        get_client_fn=lambda: "redis-client",
    )

    assert result == {"total": 0, "vectors": []}
    assert captured["args"][:4] == ("redis-client", 1, 4, 9)
