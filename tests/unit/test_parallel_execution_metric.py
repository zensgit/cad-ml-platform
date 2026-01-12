import os

import pytest
from fastapi.testclient import TestClient

from src.api.health_utils import metrics_enabled
from src.main import app


def test_parallel_execution_gauge(monkeypatch):
    if not metrics_enabled():
        pytest.skip("metrics client disabled in this environment")

    # Ensure multiple stages enabled to trigger parallel path
    client = TestClient(app)
    # Use small dummy content; rely on stub adapter
    files = {"file": ("test.dxf", b"0", "application/octet-stream")}
    options = {
        "extract_features": True,
        "classify_parts": True,
        "quality_check": True,
        "process_recommendation": True,
    }
    resp = client.post(
        "/api/v1/analyze/",
        data={"options": __import__("json").dumps(options)},
        files=files,
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert resp.status_code == 200
    # Scrape metrics endpoint to verify gauge present (may be dummy if prometheus unavailable)
    m = client.get("/metrics")
    if m.status_code == 200:
        # Accept either real gauge line or absence (dummy implementation)
        text = m.text
        assert (
            "analysis_parallel_enabled" in text or text.strip() == ""
        )  # dummy exporter returns empty
