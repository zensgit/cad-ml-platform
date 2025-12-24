from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_feature_slots_presence(monkeypatch):
    # Minimal fake DXF content satisfying lenient checks
    content = b"0\nSECTION\n" + b"A" * 120
    files = {"file": ("fs_test.dxf", content, "application/octet-stream")}
    r = client.post(
        "/api/v1/analyze",
        data={
            "options": '{"extract_features": true, "classify_parts": false, "process_recommendation": false}'
        },
        files=files,
        headers={"x-api-key": "test"},
    )
    assert r.status_code == 200
    body = r.json()
    features = body.get("results", {}).get("features", {})
    slots = features.get("feature_slots")
    assert isinstance(slots, list)
    combined = features.get("combined")
    assert isinstance(combined, list)
    assert len(combined) == features.get("dimension")
    # Basic expectations: at least base v1 slots present
    names = {s.get("name") for s in slots}
    for required in [
        "entity_count",
        "bbox_width",
        "bbox_height",
        "bbox_depth",
        "bbox_volume_estimate",
        "layer_count",
    ]:
        assert required in names
