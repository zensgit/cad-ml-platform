from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.core.qdrant_similarity_helper import compute_qdrant_cosine_similarity


class _DummyQdrantStore:
    def __init__(self, vector):  # noqa: ANN001
        self._vector = vector

    async def get_vector(self, reference_id):  # noqa: ANN001, ANN201
        return self._vector


@pytest.mark.asyncio
async def test_compute_qdrant_cosine_similarity_returns_reference_not_found():
    result = await compute_qdrant_cosine_similarity(
        _DummyQdrantStore(None),
        "ref-1",
        [1.0, 2.0],
    )

    assert result == {
        "reference_id": "ref-1",
        "status": "reference_not_found",
        "score": 0.0,
    }


@pytest.mark.asyncio
async def test_compute_qdrant_cosine_similarity_reports_dimension_mismatch():
    result = await compute_qdrant_cosine_similarity(
        _DummyQdrantStore(SimpleNamespace(vector=[1.0])),
        "ref-2",
        [1.0, 2.0],
    )

    assert result == {
        "reference_id": "ref-2",
        "status": "dimension_mismatch",
        "score": 0.0,
        "method": "cosine",
        "dimension": 1,
    }


@pytest.mark.asyncio
async def test_compute_qdrant_cosine_similarity_returns_cosine_payload():
    result = await compute_qdrant_cosine_similarity(
        _DummyQdrantStore(SimpleNamespace(vector=[1.0, 0.0])),
        "ref-3",
        [1.0, 0.0],
    )

    assert result == {
        "reference_id": "ref-3",
        "score": 1.0,
        "method": "cosine",
        "dimension": 2,
    }
