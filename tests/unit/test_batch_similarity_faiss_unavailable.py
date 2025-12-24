"""Tests: batch similarity when Faiss backend is unavailable (fallback / degraded mode).

Goals (TODO):
1. Force vector store factory to choose non-Faiss backend (e.g. env override or monkeypatch).
2. POST /api/v1/vectors/similarity/batch with sample ids.
3. Assert response contains degraded=true flag when fallback occurs.
4. Verify latency metric still emitted (optional future step).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.mark.skip(reason="TODO: implement Faiss unavailability + degraded assertion")
def test_batch_similarity_faiss_unavailable_degraded_flag() -> None:
    payload = {"ids": ["a", "b"], "top_k": 3}
    response = client.post("/api/v1/vectors/similarity/batch", json=payload)
    assert response.status_code in {200, 422}
