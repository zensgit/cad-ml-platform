from __future__ import annotations

from typing import Any

import pytest

from src.core.vector_list_qdrant import list_vectors_qdrant
from src.core.vector_list_pipeline import run_vector_list_pipeline
from src.core.errors_extended import ErrorCode


class _Item:
    def __init__(self, **kwargs: Any) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _Response(dict):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)


class _QdrantResult:
    def __init__(
        self,
        vector_id: str,
        metadata: dict[str, Any] | None,
        vector: list[float] | None,
    ) -> None:
        self.id = vector_id
        self.metadata = metadata
        self.vector = vector


class _QdrantStore:
    def __init__(self, results: list[_QdrantResult], total: int) -> None:
        self.results = results
        self.total = total
        self.calls: list[dict[str, Any]] = []

    async def list_vectors(
        self,
        offset: int,
        limit: int,
        filter_conditions: dict[str, Any] | None = None,
        with_vectors: bool = False,
    ) -> tuple[list[_QdrantResult], int]:
        self.calls.append(
            {
                "offset": offset,
                "limit": limit,
                "filter_conditions": filter_conditions,
                "with_vectors": with_vectors,
            }
        )
        return self.results, self.total


def _label_contract(meta: dict[str, Any]) -> dict[str, Any]:
    return dict(meta.get("label_contract", {}))


@pytest.mark.asyncio
async def test_list_vectors_qdrant_builds_filter_and_items() -> None:
    store = _QdrantStore(
        [
            _QdrantResult(
                "qdrant-1",
                {
                    "material": "steel",
                    "complexity": "high",
                    "format": "step",
                    "label_contract": {
                        "part_type": "shaft",
                        "fine_part_type": "shaft",
                        "coarse_part_type": "rotary",
                        "decision_source": "classifier",
                        "is_coarse_label": False,
                    },
                },
                [0.1, 0.2, 0.3],
            )
        ],
        total=7,
    )
    captured_filters: list[dict[str, Any]] = []

    def _build_filters(**kwargs: Any) -> dict[str, Any]:
        captured_filters.append(kwargs)
        return {"material": kwargs["material_filter"]}

    response = await list_vectors_qdrant(
        store,
        offset=2,
        limit=5,
        material_filter="steel",
        complexity_filter="high",
        fine_part_type_filter="shaft",
        coarse_part_type_filter="rotary",
        decision_source_filter="classifier",
        is_coarse_label_filter=False,
        item_cls=_Item,
        response_cls=_Response,
        build_filter_conditions_fn=_build_filters,
        extract_label_contract_fn=_label_contract,
    )

    assert store.calls == [
        {
            "offset": 2,
            "limit": 5,
            "filter_conditions": {"material": "steel"},
            "with_vectors": True,
        }
    ]
    assert captured_filters[0]["coarse_part_type_filter"] == "rotary"
    assert captured_filters[0]["is_coarse_label_filter"] is False
    assert response["total"] == 7
    assert response["vectors"][0].id == "qdrant-1"
    assert response["vectors"][0].dimension == 3
    assert response["vectors"][0].part_type == "shaft"
    assert response["vectors"][0].decision_source == "classifier"


@pytest.mark.asyncio
async def test_list_vectors_qdrant_handles_missing_metadata_and_vector() -> None:
    store = _QdrantStore([_QdrantResult("qdrant-empty", None, None)], total=1)

    response = await list_vectors_qdrant(
        store,
        offset=0,
        limit=10,
        material_filter=None,
        complexity_filter=None,
        fine_part_type_filter=None,
        coarse_part_type_filter=None,
        decision_source_filter=None,
        is_coarse_label_filter=None,
        item_cls=_Item,
        response_cls=_Response,
        build_filter_conditions_fn=lambda **_kwargs: {},
        extract_label_contract_fn=_label_contract,
    )

    assert response["total"] == 1
    assert response["vectors"][0].id == "qdrant-empty"
    assert response["vectors"][0].dimension == 0
    assert response["vectors"][0].material is None
    assert response["vectors"][0].part_type is None


@pytest.mark.asyncio
async def test_list_vectors_qdrant_passes_raw_meta_to_extractor() -> None:
    meta = {"material": "steel"}
    captured: list[dict[str, Any]] = []
    store = _QdrantStore([_QdrantResult("qdrant-1", meta, [0.1])], total=1)

    def _extract(raw_meta: dict[str, Any]) -> dict[str, Any]:
        captured.append(raw_meta)
        return {"part_type": "shaft", "is_coarse_label": True}

    response = await list_vectors_qdrant(
        store,
        offset=0,
        limit=10,
        material_filter=None,
        complexity_filter=None,
        fine_part_type_filter=None,
        coarse_part_type_filter=None,
        decision_source_filter=None,
        is_coarse_label_filter=None,
        item_cls=_Item,
        response_cls=_Response,
        build_filter_conditions_fn=lambda **_kwargs: {},
        extract_label_contract_fn=_extract,
    )

    assert captured == [meta]
    assert response["vectors"][0].part_type == "shaft"
    assert response["vectors"][0].is_coarse_label is True


@pytest.mark.asyncio
async def test_run_vector_list_pipeline_qdrant_preserves_extractor_patch(
    monkeypatch,
) -> None:
    from src.core import similarity as similarity_module

    monkeypatch.setattr(
        similarity_module,
        "extract_vector_label_contract",
        lambda _meta: {"part_type": "patched"},
    )

    result = await run_vector_list_pipeline(
        source="qdrant",
        offset=0,
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
        get_qdrant_store_fn=lambda: _QdrantStore(
            [_QdrantResult("qdrant-1", {}, [0.1])],
            total=1,
        ),
        resolve_list_source_fn=lambda source, _backend: source,
        build_filter_conditions_fn=lambda **_kwargs: {},
        list_vectors_redis_fn=lambda *_args, **_kwargs: None,
        list_vectors_memory_fn=lambda *_args, **_kwargs: None,
        get_client_fn=lambda: None,
    )

    assert result["total"] == 1
    assert result["vectors"][0].part_type == "patched"


@pytest.mark.asyncio
async def test_run_vector_list_pipeline_qdrant_without_store_falls_back_to_memory() -> None:
    captured: dict[str, Any] = {}

    def _list_memory(*args: Any) -> _Response:
        captured["args"] = args
        return _Response(total=0, vectors=[])

    result = await run_vector_list_pipeline(
        source="qdrant",
        offset=3,
        limit=7,
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
        list_vectors_redis_fn=lambda *_args, **_kwargs: None,
        list_vectors_memory_fn=_list_memory,
        get_client_fn=lambda: None,
    )

    assert result == {"total": 0, "vectors": []}
    assert captured["args"][2:4] == (3, 7)
