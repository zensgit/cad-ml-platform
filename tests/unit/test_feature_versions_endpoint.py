import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


def test_feature_versions_endpoint_structure():
    r = client.get("/api/v1/features/versions", headers={"x-api-key": "test"})
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "ok"
    versions = {v["version"]: v for v in data["versions"]}
    # Basic presence
    for v in ["v1", "v2", "v3", "v4"]:
        assert v in versions
        assert "dimension" in versions[v]
    # Stability flags
    assert versions["v1"]["stable"] is True and versions["v1"]["experimental"] is False
    assert versions["v4"]["experimental"] is True
