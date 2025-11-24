from fastapi.testclient import TestClient
from src.main import app
from src.core.similarity import register_vector  # type: ignore

client = TestClient(app)


def test_vector_migrate_response_contains_batch_fields(monkeypatch):
    register_vector("mig_resp_a", [0.1, 0.2, 0.3])
    payload = {"ids": ["mig_resp_a"], "to_version": "v2", "dry_run": True}
    r = client.post("/api/v1/vectors/migrate", json=payload, headers={"api-key": "test"})
    assert r.status_code == 200
    data = r.json()
    assert data.get("migration_id") is not None
    assert data.get("started_at") is not None
    assert data.get("finished_at") is not None
    assert data.get("total") == 1
    assert "items" in data
