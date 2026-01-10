from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def _run(name: str) -> str:
    file = (name, b"stub", "application/octet-stream")
    opts = {"options": (None, '{"extract_features": true, "classify_parts": false}')}
    r = client.post(
        "/api/v1/analyze", files={"file": file}, data=opts, headers={"X-API-Key": "test"}
    )
    assert r.status_code == 200
    return r.json()["id"]


def test_topk_pagination():
    ids = [_run(f"p{i}.dxf") for i in range(6)]
    first_page = client.post(
        "/api/v1/analyze/similarity/topk",
        json={"target_id": ids[0], "k": 3, "offset": 0},
        headers={"X-API-Key": "test"},
    ).json()
    second_page = client.post(
        "/api/v1/analyze/similarity/topk",
        json={"target_id": ids[0], "k": 3, "offset": 3},
        headers={"X-API-Key": "test"},
    ).json()
    assert first_page["results"]
    assert second_page["results"]
    # Ensure pagination slices differ (may overlap if scores identical but ids should differ)
    first_ids = {r["id"] for r in first_page["results"]}
    second_ids = {r["id"] for r in second_page["results"]}
    assert not first_ids.issubset(second_ids) or not second_ids.issubset(first_ids)
