from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)


def test_process_rules_audit_endpoint():
    r = client.get("/api/v1/analyze/process/rules/audit", headers={"X-API-Key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert "version" in data
    assert "materials" in data and isinstance(data["materials"], list)
    assert "raw" in data and isinstance(data["raw"], dict)
    # each material in complexities keys
    for m in data["materials"]:
        assert m in data["complexities"]
