from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_feature_slots_v4():
    r = client.get("/api/v1/features/slots?version=v4", headers={"x-api-key": "test"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "ok"
    slots = data["slots"]
    # Expected dimension count: v1(7)+v2(5)+v3(10)+v4(2)=24
    assert len(slots) == 24, f"Unexpected slot count {len(slots)}"
    # Last two should be v4 experimental slots
    assert slots[-1]["version"] == "v4"
    assert any(s["name"] == "surface_count" for s in slots)
    assert any(s["name"] == "shape_entropy" for s in slots)

