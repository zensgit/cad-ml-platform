from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def _analyze(name: str, material: str) -> str:
    file = (name, b"stub", "application/octet-stream")
    data = {
        "options": '{"extract_features": true, "classify_parts": false}',
        "material": material,
    }
    r = client.post("/api/v1/analyze", files={"file": file}, data=data, headers={"X-API-Key": "test"})
    assert r.status_code == 200
    return r.json()["id"]


def test_similarity_filter_material():
    steel_id = _analyze("steel_part.dxf", "steel")
    alu_id = _analyze("alu_part.dxf", "aluminum")
    # Query topk with material filter 'steel'
    resp = client.post(
        "/api/v1/analyze/similarity/topk",
        json={"target_id": steel_id, "k": 5, "material_filter": "steel"},
        headers={"X-API-Key": "test"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert all(item["material"] == "steel" for item in data["results"])
    # Filter that excludes all
    resp_none = client.post(
        "/api/v1/analyze/similarity/topk",
        json={"target_id": steel_id, "k": 5, "material_filter": "titanium"},
        headers={"X-API-Key": "test"},
    )
    assert resp_none.status_code == 200
    assert resp_none.json()["results"] == []

