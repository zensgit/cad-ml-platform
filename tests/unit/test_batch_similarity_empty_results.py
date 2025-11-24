"""Tests: batch similarity empty results triggers rejection metric.

Goals (TODO):
1. Provide IDs unlikely to return matches or set high min_score.
2. Assert empty results list and potentially presence of rejection metric (future metrics scrape).
3. Response should be 200 with results=[] rather than error.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


@pytest.mark.skip(reason="TODO: implement high min_score empty results assertion")
def test_batch_similarity_empty_results() -> None:
    payload = {"ids": ["x", "y"], "top_k": 5, "min_score": 0.9999}
    response = client.post("/api/v1/vectors/similarity/batch", json=payload)
    assert response.status_code in {200, 422}

