from fastapi.testclient import TestClient
from src.main import app


client = TestClient(app)


def test_feature_slots_v1():
    r = client.get("/api/v1/features/slots?version=v1", headers={"x-api-key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["version"] == "v1"
    assert data["status"] == "ok"
    assert len(data["slots"]) >= 5


def test_feature_slots_v2():
    r = client.get("/api/v1/features/slots?version=v2", headers={"x-api-key": "test"})
    assert r.status_code == 200
    data = r.json()
    names = [s["name"] for s in data["slots"]]
    # v2 should include v1 + v2 extensions
    assert any(n == "norm_width" for n in names)


def test_feature_slots_invalid_version():
    r = client.get("/api/v1/features/slots?version=unknown", headers={"x-api-key": "test"})
    assert r.status_code == 422
    data = r.json()
    assert data["code"] == "INPUT_VALIDATION_FAILED"
    assert data["stage"] == "feature_slots"
