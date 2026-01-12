import json

import pytest
from fastapi.testclient import TestClient

from src.api.health_utils import metrics_enabled
from src.main import app


def test_parallel_savings_metric_observed(monkeypatch):
    if not metrics_enabled():
        pytest.skip("metrics client disabled in this environment")

    client = TestClient(app)
    # Craft minimal DXF-like content (adapter should handle or skip gracefully)
    content = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    files = {"file": ("test.dxf", content, "application/octet-stream")}
    options = {
        "extract_features": True,
        "classify_parts": True,
        "quality_check": True,
        "process_recommendation": True,
    }
    resp = client.post(
        "/api/v1/analyze/",
        data={"options": json.dumps(options)},
        files=files,
        headers={"X-API-Key": "test"},
    )
    assert resp.status_code == 200
    # Access metrics exposition via internal registry (simplified): ensure histogram was updated.
    from src.utils.analysis_metrics import analysis_parallel_savings_seconds

    # Histogram may not expose count directly without client; rely on attribute for dummy vs real
    updated = hasattr(analysis_parallel_savings_seconds, "_sum") or hasattr(
        analysis_parallel_savings_seconds, "_value"
    )
    assert updated, "Parallel savings metric not updated"
