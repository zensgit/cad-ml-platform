from fastapi.testclient import TestClient
from src.main import app
from src.core.similarity import register_vector  # type: ignore

client = TestClient(app)


def test_vector_migration_history_ring(monkeypatch):
    # Prepare several vectors
    for i in range(12):
        register_vector(f"mh_{i}", [0.1, 0.2, 0.3])
    # Perform >10 dry-run migrations to test ring buffer truncation
    for i in range(12):
        payload = {"ids": [f"mh_{i}"], "to_version": "v2", "dry_run": True}
        r = client.post("/api/v1/vectors/migrate", json=payload, headers={"api-key": "test"})
        assert r.status_code == 200
    status = client.get("/api/v1/vectors/migrate/status", headers={"api-key": "test"})
    assert status.status_code == 200
    data = status.json()
    hist = data.get("history")
    assert isinstance(hist, list)
    # ring buffer should cap at 10
    assert len(hist) == 10
    # last entry should correspond to last migration id
    assert hist[-1]["migration_id"] == data.get("last_migration_id")

