from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_similarity_reference_not_found():
    r = client.post(
        "/api/v1/analyze/similarity",
        json={"reference_id": "nope", "target_id": "also_no"},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "reference_not_found"
    assert data["error"]["code"] == "DATA_NOT_FOUND"


def test_similarity_dimension_mismatch():
    # Create two analyses with different feature dimensions by toggling options (semantic length same now; simulate by manual vector injection)
    r1 = client.post(
        "/api/v1/analyze",
        files={"file": ("a.dxf", b"1", "application/octet-stream")},
        data={"options": '{"extract_features": true, "classify_parts": false}'},
        headers={"X-API-Key": "test"},
    )
    r2 = client.post(
        "/api/v1/analyze",
        files={"file": ("b.dxf", b"2", "application/octet-stream")},
        data={"options": '{"extract_features": true, "classify_parts": false}'},
        headers={"X-API-Key": "test"},
    )
    assert r1.status_code == 200 and r2.status_code == 200
    id1 = r1.json()["id"]
    id2 = r2.json()["id"]
    # Directly mutate global store to force mismatch
    from src.core import similarity
    similarity._VECTOR_STORE[id2] = similarity._VECTOR_STORE[id2] + [999.0]  # extend vector
    rq = client.post(
        "/api/v1/analyze/similarity",
        json={"reference_id": id1, "target_id": id2},
        headers={"X-API-Key": "test"},
    )
    assert rq.status_code == 200
    data = rq.json()
    assert data["status"] == "dimension_mismatch"
    assert data["error"]["code"] == "VALIDATION_FAILED"
