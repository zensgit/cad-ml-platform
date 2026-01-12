import os
import uuid

from fastapi.testclient import TestClient

from src.main import app


def test_analysis_cache_hit_miss_metrics(monkeypatch, metrics_text):
    """Test analysis cache hit/miss metrics with unique keys per test run."""
    client = TestClient(app)

    # Use unique file name and content to ensure cache isolation from other tests
    unique_id = uuid.uuid4().hex[:8]
    unique_filename = f"cache_test_{unique_id}.dxf"
    unique_content = f"DATA_{unique_id}".encode()

    payload_options = {"extract_features": True, "classify_parts": True}
    files = {"file": (unique_filename, unique_content, "application/octet-stream")}

    # First request should be a miss (unique content)
    r1 = client.post(
        "/api/v1/analyze/",
        data={"options": __import__("json").dumps(payload_options)},
        files=files,
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert r1.status_code == 200
    assert r1.json().get("cache_hit") is False

    # Second request same file/options should hit cache
    files2 = {"file": (unique_filename, unique_content, "application/octet-stream")}
    r2 = client.post(
        "/api/v1/analyze/",
        data={"options": __import__("json").dumps(payload_options)},
        files=files2,
        headers={"x-api-key": os.getenv("API_KEY", "test")},
    )
    assert r2.status_code == 200
    assert r2.json().get("cache_hit") is True

    # Metrics scrape (best-effort; prometheus_client may be absent)
    text = metrics_text(client)
    if text:
        assert "analysis_cache_hits_total" in text
        assert "analysis_cache_miss_total" in text
