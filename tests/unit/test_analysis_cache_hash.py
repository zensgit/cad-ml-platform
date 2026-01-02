from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_analysis_cache_hash_different_content_same_name():
    options = '{"extract_features": true, "classify_parts": false}'
    # First request with content A
    r1 = client.post(
        "/api/v1/analyze",
        files={"file": ("sample_same_name.dxf", b"AAAA", "application/octet-stream")},
        data={"options": options},
        headers={"X-API-Key": "test"},
    )
    assert r1.status_code == 200
    # Second request same filename different content
    r2 = client.post(
        "/api/v1/analyze",
        files={"file": ("sample_same_name.dxf", b"BBBB", "application/octet-stream")},
        data={"options": options},
        headers={"X-API-Key": "test"},
    )
    assert r2.status_code == 200
    # IDs should differ; second should not be cache hit because content hash differs
    assert r1.json()["id"] != r2.json()["id"]
    assert r1.json()["cache_hit"] is False
    assert r2.json()["cache_hit"] is False
