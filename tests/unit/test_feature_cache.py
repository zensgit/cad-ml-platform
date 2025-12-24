import io
from pathlib import Path

from fastapi.testclient import TestClient

from src.main import app


def test_feature_cache_hit(tmp_path):
    client = TestClient(app)
    content = b"FAKE_DXF_DATA_1"
    fileobj = io.BytesIO(content)
    # First request (miss)
    r1 = client.post(
        "/api/v1/analyze/",
        files={"file": ("sample.dxf", fileobj, "application/octet-stream")},
        data={"options": '{"extract_features": true}'},
        headers={"x-api-key": "test"},
    )
    assert r1.status_code == 200
    fileobj.seek(0)
    # Second request (should hit cache)
    r2 = client.post(
        "/api/v1/analyze/",
        files={"file": ("sample.dxf", fileobj, "application/octet-stream")},
        data={"options": '{"extract_features": true}'},
        headers={"x-api-key": "test"},
    )
    assert r2.status_code == 200
    data2 = r2.json()
    # Check top-level cache_hit for full analysis cache
    # or feature-level cache_hit in results.features
    assert (
        data2.get("cache_hit") is True
        or data2.get("results", {}).get("features", {}).get("cache_hit") is True
    )


def test_orphan_cleanup_endpoint(tmp_path):
    client = TestClient(app)
    # Create a vector by analyzing a file (will store vector with analysis_result id different from vector id; simplistic)
    # Use DXF format which has lenient validation
    content = b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF" + b"X" * 100
    fileobj = io.BytesIO(content)
    r = client.post(
        "/api/v1/analyze/",
        files={"file": ("sample.dxf", fileobj, "application/octet-stream")},
        data={"options": '{"extract_features": true}'},
        headers={"x-api-key": "test"},
    )
    assert r.status_code == 200
    # Invoke cleanup with force to ensure execution
    # Dry run first - endpoint may return 200 or 410 (deprecated)
    dry = client.delete(
        "/api/v1/analyze/vectors/orphans?threshold=0&dry_run=true",
        headers={"x-api-key": "test"},
    )
    assert dry.status_code in (200, 410)
    if dry.status_code == 200:
        assert dry.json()["status"] == "dry_run"
    cleanup = client.delete(
        "/api/v1/analyze/vectors/orphans?threshold=0&force=true",
        headers={"x-api-key": "test"},
    )
    assert cleanup.status_code in (200, 410)
    if cleanup.status_code == 200:
        cdata = cleanup.json()
        assert "status" in cdata
        assert cdata["status"] in ("cleaned", "skipped")
