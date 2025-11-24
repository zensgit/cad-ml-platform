"""Tests: vector store backend reload failure returns structured error without altering backend.

Goals (TODO):
1. Call /api/v1/vectors/backend/reload with invalid backend parameter or missing admin header.
2. Assert structured error (e.g. code=INPUT_VALIDATION_FAILED or UNAUTHORIZED depending design).
3. Confirm subsequent similarity call still uses original backend (follow-up metric scrape optional).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


@pytest.mark.skip(reason="TODO: implement backend reload failure scenario and assertions")
def test_vector_backend_reload_failure_keeps_original_backend() -> None:
    response = client.post("/api/v1/vectors/backend/reload", json={"backend": "nonexistent"})
    assert response.status_code in {200, 400, 401, 404}

