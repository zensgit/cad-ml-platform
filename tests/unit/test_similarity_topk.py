from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _run_analysis(name: str) -> str:
    file = (name, b"stub", "application/octet-stream")
    opts = {"options": (None, '{"extract_features": true, "classify_parts": false}')}
    r = client.post(
        "/api/v1/analyze", files={"file": file}, data=opts, headers={"X-API-Key": "test"}
    )
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


def test_similarity_topk_exposes_coarse_contract_metadata():
    ids = {"topk-contract-a": [0.1] * 7, "topk-contract-b": [0.11] * 7}
    try:
        for vid, vector in ids.items():
            register = client.post(
                "/api/v1/vectors/register",
                json={
                    "id": vid,
                    "vector": vector,
                    "meta": {
                        "part_type": "人孔",
                        "fine_part_type": "人孔",
                        "coarse_part_type": "开孔件",
                        "final_decision_source": "hybrid",
                        "is_coarse_label": "false",
                    },
                },
                headers={"X-API-Key": "test"},
            )
            assert register.status_code == 200

        resp = client.post(
            "/api/v1/analyze/similarity/topk",
            json={"target_id": "topk-contract-a", "k": 2, "exclude_self": False},
            headers={"X-API-Key": "test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["results"]
        first = next(item for item in data["results"] if item["id"] == "topk-contract-a")
        assert first["part_type"] == "人孔"
        assert first["fine_part_type"] == "人孔"
        assert first["coarse_part_type"] == "开孔件"
        assert first["decision_source"] == "hybrid"
        assert first["is_coarse_label"] is False
    finally:
        for vid in ids:
            client.post("/api/v1/vectors/delete", json={"id": vid}, headers={"X-API-Key": "test"})
