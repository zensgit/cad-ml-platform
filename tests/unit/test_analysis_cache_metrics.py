import os
from fastapi.testclient import TestClient
from src.main import app


def test_analysis_cache_hit_miss_metrics(monkeypatch):
    client = TestClient(app)
    payload_options = {"extract_features": True, "classify_parts": True}
    files = {"file": ("cache_test.dxf", b"DATA123", "application/octet-stream")}
    # First request should be a miss
    r1 = client.post(
        "/api/v1/analyze/",
        data={"options": __import__("json").dumps(payload_options)},
        files=files,
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert r1.status_code == 200
    assert r1.json().get("cache_hit") is False
    # Second request same file/options should hit cache
    files2 = {"file": ("cache_test.dxf", b"DATA123", "application/octet-stream")}
    r2 = client.post(
        "/api/v1/analyze/",
        data={"options": __import__("json").dumps(payload_options)},
        files=files2,
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert r2.status_code == 200
    assert r2.json().get("cache_hit") is True
    # Metrics scrape (best-effort; prometheus_client may be absent)
    m = client.get("/metrics")
    if m.status_code == 200 and m.text:
        assert "analysis_cache_hits_total" in m.text
        assert "analysis_cache_miss_total" in m.text

