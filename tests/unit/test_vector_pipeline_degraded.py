"""F4 discriminator: a degraded/mock UV-Net embedding must NOT enter L3.

Defect (2) reproduction guard: ``run_vector_pipeline`` previously appended
``features_3d["embedding_vector"]`` to the feature vector with NO check of the
degrade marker, so a mock heuristic embedding was registered as
``base_sem_ext_v1+l3`` and participated in similarity (reviewer reproduced
``l3_dim=3``). These tests pin that:

  * a DEGRADED embedding (``embedding_degraded=True``) -> ``VECTOR_LAYOUT_BASE``,
    ``l3_dim`` None, and the embedding is NOT appended to the feature vector; and
  * a VERIFIED embedding (``embedding_degraded=False``) -> ``VECTOR_LAYOUT_L3``
    with the embedding appended (positive control, proves the guard is narrow).
"""

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


def _capturing_metadata_builder(captured):  # noqa: ANN001, ANN201
    def _builder(**kwargs):  # noqa: ANN003, ANN201
        captured["metadata_builder"] = kwargs
        return {
            "vector_layout": kwargs["vector_layout"],
            "l3_dim": str(kwargs["l3_dim"]),
        }

    return _builder


@pytest.mark.asyncio
async def test_degraded_embedding_stays_base_and_is_not_appended():
    """A mock/degraded UV-Net embedding keeps the vector at BASE layout with
    ``l3_dim`` None and does not extend the feature vector into L3."""
    captured: dict[str, object] = {}
    registered_vector: dict[str, object] = {}
    doc = SimpleNamespace(format="step", complexity_bucket=lambda: "high")

    def _register(vector_id, vector, meta):  # noqa: ANN001, ANN201
        registered_vector["vector"] = list(vector)
        return True

    result = await run_vector_pipeline(
        analysis_id="vec-degraded",
        doc=doc,
        features={"geometric": [1.0], "semantic": [2.0]},
        features_3d={
            "embedding_vector": [7.0, 8.0, 9.0],
            "embedding_degraded": True,
            "embedding_provenance": "mock_heuristic",
        },
        material="steel",
        get_qdrant_store=lambda: None,
        feature_extractor_factory=lambda: _StubExtractor([0.1, 0.2]),
        metadata_builder=_capturing_metadata_builder(captured),
        register_local_vector=_register,
    )

    # The base vector is registered untouched — the 3D embedding is NOT appended.
    assert registered_vector["vector"] == [0.1, 0.2]
    assert result["feature_vector_dim"] == 2
    # Layout stays BASE and l3_dim is None (NOT registered as L3).
    assert captured["metadata_builder"]["vector_layout"] == VECTOR_LAYOUT_BASE
    assert captured["metadata_builder"]["l3_dim"] is None
    assert result["vector_metadata"] == {
        "vector_layout": VECTOR_LAYOUT_BASE,
        "l3_dim": "None",
    }


@pytest.mark.asyncio
async def test_verified_embedding_is_appended_as_l3():
    """Positive control: a VERIFIED embedding (marker False) DOES register as L3,
    proving the degrade guard is narrow and does not suppress real embeddings."""
    captured: dict[str, object] = {}
    registered_vector: dict[str, object] = {}
    doc = SimpleNamespace(format="step", complexity_bucket=lambda: "high")

    def _register(vector_id, vector, meta):  # noqa: ANN001, ANN201
        registered_vector["vector"] = list(vector)
        return True

    result = await run_vector_pipeline(
        analysis_id="vec-verified",
        doc=doc,
        features={"geometric": [1.0], "semantic": [2.0]},
        features_3d={
            "embedding_vector": [7.0, 8.0, 9.0],
            "embedding_degraded": False,
            "embedding_provenance": "uvnet_model",
        },
        material="steel",
        get_qdrant_store=lambda: None,
        feature_extractor_factory=lambda: _StubExtractor([0.1, 0.2]),
        metadata_builder=_capturing_metadata_builder(captured),
        register_local_vector=_register,
    )

    # The verified 3D embedding IS appended after the base vector.
    assert registered_vector["vector"] == [0.1, 0.2, 7.0, 8.0, 9.0]
    assert result["feature_vector_dim"] == 5
    assert captured["metadata_builder"]["vector_layout"] == VECTOR_LAYOUT_L3
    assert captured["metadata_builder"]["l3_dim"] == 3
    assert result["vector_metadata"] == {
        "vector_layout": VECTOR_LAYOUT_L3,
        "l3_dim": "3",
    }
