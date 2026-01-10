from fastapi.testclient import TestClient

from src.core import similarity
from src.core.similarity import register_vector  # type: ignore
from src.main import app

client = TestClient(app)


def test_vector_migration_status_flow(monkeypatch):
    # Reset faiss state from prior tests
    similarity._FAISS_INDEX = None
    similarity._FAISS_DIM = None
    similarity._FAISS_ID_MAP = {}
    similarity._FAISS_REVERSE_MAP = {}

    register_vector("mig_a", [0.1, 0.2, 0.3])
    register_vector("mig_b", [0.2, 0.3, 0.4])
    # initial status (no migration yet)
    r0 = client.get("/api/v1/vectors/migrate/status", headers={"x-api-key": "test"})
    assert r0.status_code == 200
    data0 = r0.json()
    assert data0.get("last_migration_id") is None
    # perform dry-run migration
    payload = {"ids": ["mig_a", "mig_b"], "to_version": "v2", "dry_run": True}
    r1 = client.post("/api/v1/vectors/migrate", json=payload, headers={"x-api-key": "test"})
    assert r1.status_code == 200
    mig = r1.json()
    assert mig.get("migration_id") is not None
    # status should reflect migration batch
    r2 = client.get("/api/v1/vectors/migrate/status", headers={"x-api-key": "test"})
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2.get("last_migration_id") == mig.get("migration_id")
    assert data2.get("last_total") == 2
    assert data2.get("last_skipped") >= 1  # dry-run counts as skipped
