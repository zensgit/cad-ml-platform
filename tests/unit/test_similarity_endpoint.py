from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_similarity_endpoint_flow():
    # Upload two minimal CAD files (DXF stub)
    file1 = ("a.dxf", b"0", "application/octet-stream")
    file2 = ("b.dxf", b"0", "application/octet-stream")
    opts = {"options": (None, '{"extract_features": true, "classify_parts": false}')}  # form field
    r1 = client.post(
        "/api/v1/analyze", files={"file": file1}, data=opts, headers={"X-API-Key": "test"}
    )
    r2 = client.post(
        "/api/v1/analyze", files={"file": file2}, data=opts, headers={"X-API-Key": "test"}
    )
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
    # Self similarity for empty vectors is 0.0 due to 0/0 division avoidance.
    # For non-zero vectors, self-similarity would be 1.0
    resp_self = client.post(
        "/api/v1/analyze/similarity",
        json={"reference_id": id1, "target_id": id1},
        headers={"X-API-Key": "test"},
    )
    # Empty/stub documents produce zero vectors, so cosine similarity is 0.0 (not 1.0)
    assert resp_self.json()["score"] in (0.0, 1.0)  # 0.0 for empty docs, 1.0 for real docs


def test_similarity_endpoint_exposes_reference_and_target_label_contracts():
    from src.core import similarity as sim_module

    ids = {
        "sim-contract-ref": [0.1, 0.2, 0.3],
        "sim-contract-tgt": [0.1, 0.2, 0.31],
    }
    try:
        for vid, vector in ids.items():
            register = client.post(
                "/api/v1/vectors/register",
                json={
                    "id": vid,
                    "vector": vector,
                    "meta": {
                        "part_type": "人孔" if vid.endswith("ref") else "传动件",
                        "fine_part_type": "人孔" if vid.endswith("ref") else "搅拌轴组件",
                        "coarse_part_type": "开孔件" if vid.endswith("ref") else "传动件",
                        "final_decision_source": "hybrid" if vid.endswith("ref") else "graph2d",
                        "is_coarse_label": "false" if vid.endswith("ref") else "false",
                    },
                },
                headers={"X-API-Key": "test"},
            )
            assert register.status_code == 200

        resp = client.post(
            "/api/v1/analyze/similarity",
            json={"reference_id": "sim-contract-ref", "target_id": "sim-contract-tgt"},
            headers={"X-API-Key": "test"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["reference_part_type"] == "人孔"
        assert data["reference_fine_part_type"] == "人孔"
        assert data["reference_coarse_part_type"] == "开孔件"
        assert data["reference_decision_source"] == "hybrid"
        assert data["reference_is_coarse_label"] is False
        assert data["target_part_type"] == "传动件"
        assert data["target_fine_part_type"] == "搅拌轴组件"
        assert data["target_coarse_part_type"] == "传动件"
        assert data["target_decision_source"] == "graph2d"
        assert data["target_is_coarse_label"] is False
    finally:
        for vid in ids:
            sim_module._VECTOR_STORE.pop(vid, None)
            sim_module._VECTOR_META.pop(vid, None)
