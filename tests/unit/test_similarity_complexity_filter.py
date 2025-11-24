from fastapi.testclient import TestClient
from src.main import app
import uuid

client = TestClient(app)


def _analyze_stub(name: str, entity_count: int, material: str):
    # Craft minimal DXF-like content; current parser may fallback to stub.
    fake_bytes = b"0" * 10  # small content
    options = '{"extract_features": true, "classify_parts": false}'
    resp = client.post(
        "/api/v1/analyze",
        files={"file": (name, fake_bytes, "application/octet-stream")},
        data={"options": options, "material": material},
        headers={"X-API-Key": "test"},
    )
    assert resp.status_code == 200
    return resp.json()["id"], resp.json()["results"]["features"]["geometric"]


def test_similarity_topk_complexity_filter():
    # Create several analyses with different synthetic complexity via entity counts.
    # For stub documents entity_count may be 0; we simulate diversity by reusing returned vectors.
    ids = []
    materials = ["steel", "steel", "aluminum"]
    for i, m in enumerate(materials):
        aid, geom = _analyze_stub(f"part_{i}.dxf", entity_count=i * 30 + 5, material=m)
        ids.append(aid)

    target_id = ids[0]
    # Query top-k without filter
    r_all = client.post(
        "/api/v1/analyze/similarity/topk",
        json={"target_id": target_id, "k": 5},
        headers={"X-API-Key": "test"},
    )
    assert r_all.status_code == 200
    data_all = r_all.json()
    assert "results" in data_all

    # Apply complexity filter expecting only matching complexity bucket items
    # Determine target complexity from stored cad_document
    # NOTE: cad_document added in AnalysisResult; fetch analysis result for target
    r_target = client.get(f"/api/v1/analyze/{target_id}", headers={"X-API-Key": "test"})
    assert r_target.status_code == 200
    complexity = r_target.json().get("complexity") or r_target.json()["statistics"]["complexity"]

    r_filtered = client.post(
        "/api/v1/analyze/similarity/topk",
        json={"target_id": target_id, "k": 5, "complexity_filter": complexity},
        headers={"X-API-Key": "test"},
    )
    assert r_filtered.status_code == 200
    data_filtered = r_filtered.json()
    for item in data_filtered["results"]:
        if item["complexity"] is not None:
            assert item["complexity"] == complexity
