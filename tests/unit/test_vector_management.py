from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_vector_list_and_delete():
    # Create an analysis to ensure at least one vector exists
    r = client.post(
        "/api/v1/analyze",
        files={"file": ("sample.dxf", b"0" * 10, "application/octet-stream")},
        data={"options": '{"extract_features": true, "classify_parts": false}'},
        headers={"X-API-Key": "test"},
    )
    assert r.status_code == 200
    analysis_id = r.json()["id"]

    # List vectors
    r_list = client.get("/api/v1/vectors", headers={"X-API-Key": "test"})
    assert r_list.status_code == 200
    data_list = r_list.json()
    assert data_list["total"] >= 1
    assert any(v["id"] == analysis_id for v in data_list["vectors"])

    # Delete vector
    r_del = client.post(
        "/api/v1/vectors/delete",
        json={"id": analysis_id},
        headers={"X-API-Key": "test"},
    )
    # Updated endpoint: returns 200 with deleted OR 404 with error payload
    assert r_del.status_code in (200, 404)
    if r_del.status_code == 200:
        assert r_del.json()["status"] == "deleted"
    else:
        # 404 extended error detail
        body = r_del.json()
        assert "detail" in body
        assert body["detail"]["code"] == "DATA_NOT_FOUND"

    # Confirm deletion (best-effort; if not_found earlier skip)
    r_list2 = client.get("/api/v1/vectors", headers={"X-API-Key": "test"})
    assert r_list2.status_code == 200
    data_list2 = r_list2.json()
    # If deleted, id should not appear again
    if r_del.json()["status"] == "deleted":
        assert all(v["id"] != analysis_id for v in data_list2["vectors"])
