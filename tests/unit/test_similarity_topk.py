from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def _run_analysis(name: str) -> str:
    file = (name, b"stub", "application/octet-stream")
    opts = {"options": (None, '{"extract_features": true, "classify_parts": false}')}
    r = client.post("/api/v1/analyze", files={"file": file}, data=opts, headers={"X-API-Key": "test"})
    assert r.status_code == 200
    return r.json()["id"]


def test_similarity_topk_flow():
    ids = [_run_analysis(f"f{i}.dxf") for i in range(3)]
    # Include self
    resp_inc = client.post(
        "/api/v1/analyze/similarity/topk",
        json={"target_id": ids[0], "k": 3, "exclude_self": False},
        headers={"X-API-Key": "test"},
    )
    assert resp_inc.status_code == 200
    data_inc = resp_inc.json()
    assert any(r["id"] == ids[0] for r in data_inc["results"])  # self present
    # Exclude self
    resp_exc = client.post(
        "/api/v1/analyze/similarity/topk",
        json={"target_id": ids[0], "k": 3, "exclude_self": True},
        headers={"X-API-Key": "test"},
    )
    assert resp_exc.status_code == 200
    data_exc = resp_exc.json()
    assert all(r["id"] != ids[0] for r in data_exc["results"])  # self excluded
    # Score bounds
    for item in data_inc["results"]:
        assert 0.0 <= item["score"] <= 1.0
