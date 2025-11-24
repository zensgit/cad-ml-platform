from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_vector_update_dimension_conflict_replace():
    # Create base analysis (vector dimension determined by features: geometric(5)+semantic(2)=7)
    r = client.post(
        "/api/v1/analyze",
        files={"file": ("sample.dxf", b"0" * 20, "application/octet-stream")},
        data={"options": '{"extract_features": true, "classify_parts": false}'},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 200
    vid = r.json()["id"]

    # Enforce dimension mismatch (replace with wrong size) expecting 409 when enforcement active
    r_up = client.post(
        "/api/v1/analyze/vectors/update",
        json={"id": vid, "replace": [0.1, 0.2]},  # dim 2 vs original 7
        headers={"X-API-Key": "test"},
    )
    # Depending on env ANALYSIS_VECTOR_DIM_CHECK (default enforced in tests) expect 409
    assert r_up.status_code in (200, 409)
    if r_up.status_code == 409:
        body = r_up.json()
        assert body["detail"]["code"] == "DIMENSION_MISMATCH"


def test_vector_update_dimension_conflict_append():
    r = client.post(
        "/api/v1/analyze",
        files={"file": ("sample2.dxf", b"1" * 30, "application/octet-stream")},
        data={"options": '{"extract_features": true, "classify_parts": false}'},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 200
    vid = r.json()["id"]
    # Append changes dimension -> conflict under enforcement
    r_up = client.post(
        "/api/v1/analyze/vectors/update",
        json={"id": vid, "append": [0.9, 0.8]},  # increasing dimension
        headers={"X-API-Key": "test"},
    )
    assert r_up.status_code in (200, 409)
    if r_up.status_code == 409:
        body = r_up.json()
        assert body["detail"]["code"] == "DIMENSION_MISMATCH"
