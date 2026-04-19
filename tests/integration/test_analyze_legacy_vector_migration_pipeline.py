from __future__ import annotations

from fastapi.testclient import TestClient

from src.main import app


def test_analyze_legacy_vector_migrate_route_delegates_to_shared_pipeline(monkeypatch) -> None:
    client = TestClient(app)
    captured = {}

    async def _stub_run_legacy_vector_migrate_pipeline(**kwargs):  # noqa: ANN003, ANN201
        captured.update(kwargs)
        return {
            "total": 1,
            "migrated": 0,
            "skipped": 1,
            "items": [{"id": "vec-1", "status": "dry_run", "to_version": "v2"}],
            "migration_id": "mig-1",
            "started_at": "2026-04-17T00:00:00+00:00",
            "finished_at": "2026-04-17T00:00:01+00:00",
            "dry_run_total": 1,
        }

    monkeypatch.setattr(
        "src.api.v1.analyze_vector_compat.run_legacy_vector_migrate_pipeline",
        _stub_run_legacy_vector_migrate_pipeline,
    )

    response = client.post(
        "/api/v1/analyze/vectors/migrate",
        json={"ids": ["vec-1"], "to_version": "v2", "dry_run": True},
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["migration_id"] == "mig-1"
    assert payload["dry_run_total"] == 1
    assert captured["payload"].ids == ["vec-1"]


def test_analyze_legacy_vector_migration_status_route_delegates_to_shared_pipeline(
    monkeypatch,
) -> None:
    client = TestClient(app)

    monkeypatch.setattr(
        "src.api.v1.analyze_vector_compat.run_legacy_vector_migration_status_pipeline",
        lambda: {
            "last_migration_id": "mig-1",
            "last_total": 2,
            "last_migrated": 1,
            "last_skipped": 1,
            "pending_vectors": 5,
            "feature_versions": {"v1": 2, "v2": 3},
            "history": [{"migration_id": "mig-1"}],
        },
    )

    response = client.get(
        "/api/v1/analyze/vectors/migrate/status",
        headers={"X-API-Key": "test-key"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["last_migration_id"] == "mig-1"
    assert payload["pending_vectors"] == 5
    assert payload["feature_versions"] == {"v1": 2, "v2": 3}
