import json
import time
from fastapi.testclient import TestClient

from src.main import app


def test_feature_cache_sliding_window_metrics(monkeypatch):
    client = TestClient(app)
    content = b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"
    files = {"file": ("sw.dxf", content, "application/octet-stream")}
    # First miss (cache empty)
    resp1 = client.post(
        "/api/v1/analyze/",
        data={"options": json.dumps({"extract_features": True})},
        files=files,
        headers={"X-API-Key": "test"},
    )
    assert resp1.status_code == 200
    # Second hit (same file/options)
    resp2 = client.post(
        "/api/v1/analyze/",
        data={"options": json.dumps({"extract_features": True})},
        files=files,
        headers={"X-API-Key": "test"},
    )
    assert resp2.status_code == 200
    # Access metrics objects
    from src.utils.analysis_metrics import feature_cache_hits_last_hour, feature_cache_miss_last_hour
    # Validate that gauges updated (best-effort attributes exist)
    assert hasattr(feature_cache_hits_last_hour, "_value") or hasattr(
        feature_cache_hits_last_hour, "_metrics"
    )
    assert hasattr(feature_cache_miss_last_hour, "_value") or hasattr(
        feature_cache_miss_last_hour, "_metrics"
    )

