from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.vector_pipeline import run_vector_pipeline
from src.core.vector_layouts import VECTOR_LAYOUT_BASE, VECTOR_LAYOUT_L3


class _StubExtractor:
    def __init__(self, vector):  # noqa: ANN001
        self._vector = list(vector)

    def flatten(self, features):  # noqa: ANN001, ANN201
        return list(self._vector)


@pytest.mark.asyncio
async def test_run_vector_pipeline_registers_local_vector_and_updates_memory_meta():
    captured: dict[str, object] = {}
    observed_materials: list[str] = []
    observed_dims: list[int] = []
    updated_meta: list[tuple[str, dict[str, str]]] = []
    doc = SimpleNamespace(format="dxf", complexity_bucket=lambda: "medium")

    def _metadata_builder(**kwargs):  # noqa: ANN003, ANN201
        captured["metadata_builder"] = kwargs
        return {"material": "steel", "part_type": "法兰"}

    def _register(vector_id, vector, meta):  # noqa: ANN001, ANN201
        captured["register"] = {
            "vector_id": vector_id,
            "vector": list(vector),
            "meta": dict(meta),
        }
        return True

    result = await run_vector_pipeline(
        analysis_id="vec-1",
        doc=doc,
        features={"geometric": [1.0], "semantic": [2.0]},
        features_3d=None,
        material="steel",
        classification_meta={"part_type": "法兰"},
        get_qdrant_store=lambda: None,
        feature_extractor_factory=lambda: _StubExtractor([0.1, 0.2, 0.3]),
        metadata_builder=_metadata_builder,
        register_local_vector=_register,
        vector_material_observer=observed_materials.append,
        feature_dimension_observer=observed_dims.append,
        memory_meta_updater=lambda vector_id, meta: updated_meta.append(
            (vector_id, dict(meta))
        ),
    )

    assert result == {
        "registered": True,
        "similarity": None,
        "vector_metadata": {"material": "steel", "part_type": "法兰"},
        "feature_vector_dim": 3,
    }
    assert captured["register"] == {
        "vector_id": "vec-1",
        "vector": [0.1, 0.2, 0.3],
        "meta": {"material": "steel", "part_type": "法兰"},
    }
    assert captured["metadata_builder"] == {
        "material": "steel",
        "doc": doc,
        "features": {"geometric": [1.0], "semantic": [2.0]},
        "feature_vector": [0.1, 0.2, 0.3],
        "feature_version": "v1",
        "vector_layout": VECTOR_LAYOUT_BASE,
        "classification_meta": {"part_type": "法兰"},
        "l3_dim": None,
    }
    assert observed_materials == ["steel"]
    assert observed_dims == [3]
    assert updated_meta == [("vec-1", {"material": "steel", "part_type": "法兰"})]


@pytest.mark.asyncio
async def test_run_vector_pipeline_uses_qdrant_registration_and_similarity():
    captured: dict[str, object] = {}
    doc = SimpleNamespace(format="step", complexity_bucket=lambda: "high")

    class _DummyQdrantStore:
        async def register_vector(self, vector_id, vector, metadata=None):  # noqa: ANN001, ANN201
            captured["register"] = {
                "vector_id": vector_id,
                "vector": list(vector),
                "metadata": dict(metadata or {}),
            }

        async def get_vector(self, vector_id):  # pragma: no cover - not used here
            raise AssertionError("get_vector should not be called in similarity mode")

    async def _compute_qdrant_similarity(store, reference_id, feature_vector):  # noqa: ANN001, ANN201
        captured["similarity"] = {
            "store": store,
            "reference_id": reference_id,
            "feature_vector": list(feature_vector),
        }
        return {"reference_id": reference_id, "score": 0.97}

    result = await run_vector_pipeline(
        analysis_id="vec-2",
        doc=doc,
        features={"geometric": [1.0], "semantic": [2.0]},
        features_3d={"embedding_vector": [3.0, 4.0]},
        material="aluminum",
        classification_meta={"part_type": "支架"},
        calculate_similarity=True,
        reference_id="ref-7",
        get_qdrant_store=lambda: _DummyQdrantStore(),
        compute_qdrant_similarity=_compute_qdrant_similarity,
        feature_extractor_factory=lambda: _StubExtractor([0.5, 0.6]),
        metadata_builder=lambda **kwargs: {  # noqa: ANN003
            "vector_layout": kwargs["vector_layout"],
            "l3_3d_dim": str(kwargs["l3_dim"]),
        },
        register_local_vector=lambda *args, **kwargs: False,
    )

    assert result == {
        "registered": True,
        "similarity": {"reference_id": "ref-7", "score": 0.97},
        "vector_metadata": {
            "vector_layout": VECTOR_LAYOUT_L3,
            "l3_3d_dim": "2",
        },
        "feature_vector_dim": 4,
    }
    assert captured["register"] == {
        "vector_id": "vec-2",
        "vector": [0.5, 0.6, 3.0, 4.0],
        "metadata": {
            "vector_layout": VECTOR_LAYOUT_L3,
            "l3_3d_dim": "2",
        },
    }
    assert captured["similarity"] == {
        "store": captured["similarity"]["store"],
        "reference_id": "ref-7",
        "feature_vector": [0.5, 0.6, 3.0, 4.0],
    }


@pytest.mark.asyncio
async def test_run_vector_pipeline_registration_failure_does_not_block_similarity():
    doc = SimpleNamespace(format="dxf", complexity_bucket=lambda: "low")

    result = await run_vector_pipeline(
        analysis_id="vec-3",
        doc=doc,
        features={"geometric": [1.0], "semantic": []},
        features_3d=None,
        material=None,
        calculate_similarity=True,
        reference_id="ref-8",
        get_qdrant_store=lambda: None,
        feature_extractor_factory=lambda: _StubExtractor([0.9, 0.8]),
        metadata_builder=lambda **kwargs: {"material": "unknown"},  # noqa: ANN003
        register_local_vector=lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("register failed")
        ),
        compute_local_similarity=lambda reference_id, feature_vector: {  # noqa: ANN001
            "reference_id": reference_id,
            "score": sum(feature_vector),
        },
    )

    assert result == {
        "registered": False,
        "similarity": {"reference_id": "ref-8", "score": pytest.approx(1.7)},
        "vector_metadata": {"material": "unknown"},
        "feature_vector_dim": 2,
    }


@pytest.mark.asyncio
async def test_run_vector_pipeline_reports_reference_not_found_without_similarity_compute():
    doc = SimpleNamespace(format="dxf", complexity_bucket=lambda: "simple")

    result = await run_vector_pipeline(
        analysis_id="vec-4",
        doc=doc,
        features={"geometric": [], "semantic": []},
        features_3d=None,
        material="steel",
        reference_id="missing-ref",
        get_qdrant_store=lambda: None,
        feature_extractor_factory=lambda: _StubExtractor([0.1]),
        metadata_builder=lambda **kwargs: {"material": "steel"},  # noqa: ANN003
        register_local_vector=lambda *args, **kwargs: True,
        has_local_vector=lambda vector_id: False,
    )

    assert result == {
        "registered": True,
        "similarity": {
            "reference_id": "missing-ref",
            "status": "reference_not_found",
        },
        "vector_metadata": {"material": "steel"},
        "feature_vector_dim": 1,
    }


@pytest.mark.asyncio
async def test_run_vector_pipeline_adds_faiss_entry_when_backend_enabled(
    monkeypatch: pytest.MonkeyPatch,
):
    added: list[tuple[str, list[float]]] = []
    doc = SimpleNamespace(format="step", complexity_bucket=lambda: "complex")

    class _DummyFaissStore:
        def add(self, vector_id, vector):  # noqa: ANN001, ANN201
            added.append((vector_id, list(vector)))

    monkeypatch.setenv("VECTOR_STORE_BACKEND", "faiss")
    result = await run_vector_pipeline(
        analysis_id="vec-5",
        doc=doc,
        features={"geometric": [1.0], "semantic": [2.0]},
        features_3d=None,
        material="steel",
        get_qdrant_store=lambda: None,
        feature_extractor_factory=lambda: _StubExtractor([0.2, 0.4]),
        metadata_builder=lambda **kwargs: {"material": "steel"},  # noqa: ANN003
        register_local_vector=lambda *args, **kwargs: True,
        faiss_store_factory=lambda: _DummyFaissStore(),
    )

    assert result["registered"] is True
    assert result["feature_vector_dim"] == 2
    assert added == [("vec-5", [0.2, 0.4])]


@pytest.mark.asyncio
async def test_run_vector_pipeline_reports_qdrant_reference_not_found_without_similarity():
    doc = SimpleNamespace(format="step", complexity_bucket=lambda: "medium")

    class _DummyQdrantStore:
        async def register_vector(self, vector_id, vector, metadata=None):  # noqa: ANN001, ANN201
            return None

        async def get_vector(self, vector_id):  # noqa: ANN001, ANN201
            return None

    result = await run_vector_pipeline(
        analysis_id="vec-6",
        doc=doc,
        features={"geometric": [1.0], "semantic": []},
        features_3d=None,
        material="steel",
        reference_id="missing-qdrant-ref",
        get_qdrant_store=lambda: _DummyQdrantStore(),
        feature_extractor_factory=lambda: _StubExtractor([0.3]),
        metadata_builder=lambda **kwargs: {"material": "steel"},  # noqa: ANN003
        compute_qdrant_similarity=None,
    )

    assert result == {
        "registered": True,
        "similarity": {
            "reference_id": "missing-qdrant-ref",
            "status": "reference_not_found",
        },
        "vector_metadata": {"material": "steel"},
        "feature_vector_dim": 1,
    }
