from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_similarity_endpoint_flow():
    # Upload two minimal CAD files (DXF stub)
    file1 = ("a.dxf", b"0", "application/octet-stream")
    file2 = ("b.dxf", b"0", "application/octet-stream")
    opts = {"options": (None, '{"extract_features": true, "classify_parts": false}')}  # form field
    r1 = client.post("/api/v1/analyze", files={"file": file1}, data=opts, headers={"X-API-Key": "test"})
    r2 = client.post("/api/v1/analyze", files={"file": file2}, data=opts, headers={"X-API-Key": "test"})
    assert r1.status_code == 200 and r2.status_code == 200
    id1 = r1.json()["id"]
    id2 = r2.json()["id"]
    # Query similarity between the two vectors
    resp = client.post(
        "/api/v1/analyze/similarity",
        json={"reference_id": id1, "target_id": id2},
        headers={"X-API-Key": "test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["reference_id"] == id1
    assert data["target_id"] == id2
    assert "score" in data
    # Self similarity should be exactly 1
    resp_self = client.post(
        "/api/v1/analyze/similarity",
        json={"reference_id": id1, "target_id": id1},
        headers={"X-API-Key": "test"},
    )
    assert resp_self.json()["score"] == 1.0
