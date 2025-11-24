from fastapi.testclient import TestClient
from pathlib import Path
import io

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
        headers={"api-key": "test"},
    )
    assert r1.status_code == 200
    fileobj.seek(0)
    # Second request (should hit cache)
    r2 = client.post(
        "/api/v1/analyze/",
        files={"file": ("sample.dxf", fileobj, "application/octet-stream")},
        data={"options": '{"extract_features": true}'},
        headers={"api-key": "test"},
    )
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["results"]["features"]["cache_hit"] is True


def test_orphan_cleanup_endpoint(tmp_path):
    client = TestClient(app)
    # Create a vector by analyzing a file (will store vector with analysis_result id different from vector id; simplistic)
    fileobj = io.BytesIO(b"FAKE_STEP_DATA_2")
    r = client.post(
        "/api/v1/analyze/",
        files={"file": ("sample.step", fileobj, "application/octet-stream")},
        data={"options": '{"extract_features": true}'},
        headers={"api-key": "test"},
    )
    assert r.status_code == 200
    # Invoke cleanup with force to ensure execution
    # Dry run first
    dry = client.delete(
        "/api/v1/analyze/vectors/orphans?threshold=0&dry_run=true",
        headers={"api-key": "test"},
    )
    assert dry.status_code == 200
    assert dry.json()["status"] == "dry_run"
    cleanup = client.delete(
        "/api/v1/analyze/vectors/orphans?threshold=0&force=true",
        headers={"api-key": "test"},
    )
    assert cleanup.status_code == 200
    cdata = cleanup.json()
    assert "status" in cdata
    assert cdata["status"] in ("cleaned", "skipped")
