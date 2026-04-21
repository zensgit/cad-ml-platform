from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.vector_search_pipeline import run_vector_search_pipeline


class _Response(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _QdrantResult:
    def __init__(self, vector_id, score, metadata, vector):
        self.id = vector_id
        self.score = score
        self.metadata = metadata
        self.vector = vector


@pytest.mark.asyncio
async def test_run_vector_search_pipeline_uses_qdrant_branch():
    payload = SimpleNamespace(
        vector=[0.2] * 7,
        k=5,
        material_filter=None,
        complexity_filter=None,
        fine_part_type_filter=None,
        coarse_part_type_filter="开孔件",
        decision_source_filter="hybrid",
        is_coarse_label_filter=False,
    )

    class _Store:
        async def search_similar(self, query_vector, top_k=10, filter_conditions=None, **kwargs):  # noqa: ANN001, ANN201
            assert query_vector == [0.2] * 7
            assert top_k == 5
            assert filter_conditions == {
                "coarse_part_type_filter": "开孔件",
                "decision_source_filter": "hybrid",
                "is_coarse_label_filter": False,
            }
            assert kwargs["with_vectors"] is True
            return [
                _QdrantResult(
                    "qdrant-1",
                    0.93,
                    {
                        "material": "steel",
                        "coarse_part_type": "开孔件",
                        "decision_source": "hybrid",
                        "is_coarse_label": False,
                    },
                    [0.1] * 7,
                )
            ]

    result = await run_vector_search_pipeline(
        payload=payload,
        response_cls=_Response,
        get_qdrant_store_fn=lambda: _Store(),
        build_filter_conditions_fn=lambda p: {
            "coarse_part_type_filter": p.coarse_part_type_filter,
            "decision_source_filter": p.decision_source_filter,
            "is_coarse_label_filter": p.is_coarse_label_filter,
        },
        matches_filters_fn=lambda *_args, **_kwargs: True,
        vector_item_payload_fn=lambda vector_id, dimension, meta, _label_contract: {
            "id": vector_id,
            "dimension": dimension,
            "material": meta.get("material"),
            "coarse_part_type": meta.get("coarse_part_type"),
            "decision_source": meta.get("decision_source"),
            "is_coarse_label": meta.get("is_coarse_label"),
        },
    )

    assert result["total"] == 1
    assert result["results"][0]["id"] == "qdrant-1"
    assert result["results"][0]["coarse_part_type"] == "开孔件"

