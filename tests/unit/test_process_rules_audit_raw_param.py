from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_process_rules_audit_raw_disabled():
    r = client.get("/api/v1/analyze/process/rules/audit?raw=0", headers={"X-API-Key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data["raw"] == {}  # raw omitted
    assert isinstance(data["materials"], list)
    assert "version" in data
    assert "complexities" in data
