from fastapi.testclient import TestClient
from src.main import app
from src.core.similarity import register_vector, _VECTOR_STORE  # type: ignore
import os

client = TestClient(app)


def test_vector_migrate_dry_run():
    os.environ["FEATURE_VERSION"] = "v1"
    register_vector("migrate_a", [0.1] * 7)
    payload = {"ids": ["migrate_a"], "to_version": "v2", "dry_run": True}
    resp = client.post("/api/v1/vectors/migrate", json=payload, headers={"x-api-key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"][0]["status"] == "dry_run"
    # original vector unchanged
    assert len(_VECTOR_STORE["migrate_a"]) == 7


def test_vector_migrate_execute():
    os.environ["FEATURE_VERSION"] = "v1"
    register_vector("migrate_b", [0.2] * 7)
    payload = {"ids": ["migrate_b"], "to_version": "v2", "dry_run": False}
    resp = client.post("/api/v1/vectors/migrate", json=payload, headers={"x-api-key": "test"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["items"][0]["status"] == "migrated"
    # Expect v2 geometric extension (dimension > original)
    assert len(_VECTOR_STORE["migrate_b"]) > 7
