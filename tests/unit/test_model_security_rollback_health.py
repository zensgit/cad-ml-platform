"""Tests: model security failure leads to rollback & health reflects failure state.

Goals (TODO):
1. Craft malicious model file (disallowed pickle opcode) and attempt /api/v1/model/reload.
2. Assert structured error with reason opcode_blocked and rollback status.
3. Call /api/v1/health/model and verify last_error / rollback_level fields.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.mark.skip(reason="TODO: implement malicious model upload & health checks")
def test_model_security_rollback_health_fields() -> None:
    response = client.get("/api/v1/health/model")
    assert response.status_code == 200
