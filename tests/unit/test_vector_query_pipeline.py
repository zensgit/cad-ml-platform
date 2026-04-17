import asyncio
from types import SimpleNamespace

from src.core import similarity as sim_module
from src.core.vector_query_pipeline import (
    run_similarity_query_pipeline,
    run_similarity_topk_pipeline,
)
def test_run_similarity_query_pipeline_memory_contract_payload():
    sim_module._VECTOR_STORE["vector-query-ref"] = [0.1, 0.2, 0.3]
    sim_module._VECTOR_STORE["vector-query-tgt"] = [0.1, 0.2, 0.31]
    sim_module._VECTOR_META["vector-query-ref"] = {
        "part_type": "人孔",
        "fine_part_type": "人孔",
        "coarse_part_type": "开孔件",
        "final_decision_source": "hybrid",
        "is_coarse_label": "false",
    }
    sim_module._VECTOR_META["vector-query-tgt"] = {
        "part_type": "传动件",
        "fine_part_type": "搅拌轴组件",
        "coarse_part_type": "传动件",
        "final_decision_source": "graph2d",
        "is_coarse_label": "false",
    }
    payload = SimpleNamespace(reference_id="vector-query-ref", target_id="vector-query-tgt")

    try:
        result = asyncio.run(
            run_similarity_query_pipeline(payload, get_qdrant_store=lambda: None)
        )
    finally:
        sim_module._VECTOR_STORE.pop("vector-query-ref", None)
        sim_module._VECTOR_STORE.pop("vector-query-tgt", None)
        sim_module._VECTOR_META.pop("vector-query-ref", None)
        sim_module._VECTOR_META.pop("vector-query-tgt", None)

    assert result["reference_id"] == "vector-query-ref"
    assert result["target_id"] == "vector-query-tgt"
    assert result["reference_part_type"] == "人孔"
    assert result["reference_decision_source"] == "hybrid"
    assert result["target_part_type"] == "传动件"
    assert result["target_decision_source"] == "graph2d"
    assert result["status"] is None


def test_run_similarity_query_pipeline_qdrant_missing_reference_records_error():
    recorded: list[str] = []

    class DummyQdrantStore:
        async def get_vector(self, vector_id):
            return None

    payload = SimpleNamespace(reference_id="missing-ref", target_id="target")
    result = asyncio.run(
        run_similarity_query_pipeline(
            payload,
            get_qdrant_store=lambda: DummyQdrantStore(),
            error_recorder=recorded.append,
        )
    )

    assert result["status"] == "reference_not_found"
    assert result["error"]["code"] == "DATA_NOT_FOUND"
    assert recorded == ["DATA_NOT_FOUND"]


def test_run_similarity_topk_pipeline_qdrant_filter_and_contract_payload():
    captured: dict[str, object] = {}

    class DummyQdrantResult:
        def __init__(self, vector_id, score, metadata=None, vector=None):
            self.id = vector_id
            self.score = score
            self.metadata = metadata or {}
            self.vector = vector

    class DummyQdrantStore:
        async def get_vector(self, vector_id):
            assert vector_id == "target-qdrant"
            return DummyQdrantResult("target-qdrant", 1.0, vector=[0.3] * 7)

        async def search_similar(self, query_vector, top_k=10, filter_conditions=None):
            captured["query_vector"] = query_vector
            captured["top_k"] = top_k
            captured["filter_conditions"] = filter_conditions
            return [
                DummyQdrantResult(
                    "target-qdrant",
                    1.0,
                    {
                        "part_type": "人孔",
                        "fine_part_type": "人孔",
                        "coarse_part_type": "开孔件",
                        "decision_source": "hybrid",
                        "is_coarse_label": False,
                    },
                ),
                DummyQdrantResult(
                    "neighbor-qdrant",
                    0.88,
                    {
                        "part_type": "人孔",
                        "fine_part_type": "人孔",
                        "coarse_part_type": "开孔件",
                        "decision_source": "hybrid",
                        "is_coarse_label": False,
                    },
                ),
            ]

    payload = SimpleNamespace(
        target_id="target-qdrant",
        k=2,
        exclude_self=True,
        offset=0,
        material_filter=None,
        complexity_filter=None,
        fine_part_type_filter=None,
        coarse_part_type_filter="开孔件",
        decision_source_filter="hybrid",
        is_coarse_label_filter=False,
    )
    result = asyncio.run(
        run_similarity_topk_pipeline(payload, get_qdrant_store=lambda: DummyQdrantStore())
    )

    assert captured["query_vector"] == [0.3] * 7
    assert captured["filter_conditions"] == {
        "coarse_part_type": "开孔件",
        "decision_source": "hybrid",
        "is_coarse_label": False,
    }
    assert len(result["results"]) == 1
    assert result["results"][0]["id"] == "neighbor-qdrant"
    assert result["results"][0]["coarse_part_type"] == "开孔件"
    assert result["results"][0]["decision_source"] == "hybrid"
    assert result["results"][0]["is_coarse_label"] is False
