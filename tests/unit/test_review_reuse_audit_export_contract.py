"""Contract tests for GET /api/v1/review-reuse/tasks/{id}/audit-export.

Locks the quarantined audit-bundle shape and R2 HOLD: export is not a training path.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

# Ensure local/test identity posture before app import side-effects.
os.environ.setdefault("ENVIRONMENT", "development")
os.environ.setdefault("API_KEY", "test")


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("API_KEY", "test")
    monkeypatch.delenv("REVIEW_REUSE_DECISIONS_ENABLED", raising=False)
    from src.core.review_reuse import service as svc_mod
    from src.core.review_reuse.store import InMemoryReviewReuseStore

    svc_mod.reset_review_reuse_store_for_tests(InMemoryReviewReuseStore())

    from src.main import app

    with TestClient(app, headers={"X-API-Key": "test"}) as c:
        yield c


def _create_task(client: TestClient, *, name: str = "sample.dxf", body: bytes = b"0\nSECTION\n") -> str:
    r = client.post(
        "/api/v1/review-reuse/tasks",
        files={"file": (name, body, "application/octet-stream")},
    )
    assert r.status_code == 200, r.text
    return r.json()["task_id"]


def test_audit_export_contract_schema_and_fields(client: TestClient) -> None:
    """Bundle returns schema_version, export_kind=audit_quarantine, evidence_pack, events."""
    task_id = _create_task(client)

    r = client.get(f"/api/v1/review-reuse/tasks/{task_id}/audit-export")
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["schema_version"] == "review-reuse-audit-bundle-v1"
    assert body["export_kind"] == "audit_quarantine"
    assert "evidence_pack" in body
    assert body["evidence_pack"] is not None
    assert body["evidence_pack"].get("schema_version") == "evidence-pack-v1"
    assert body["evidence_pack"].get("task_id") == task_id

    assert "events" in body
    assert isinstance(body["events"], list)
    assert len(body["events"]) >= 1
    event_types = {e["event_type"] for e in body["events"]}
    assert "submitted" in event_types
    assert "evidence_pack_ready" in event_types

    # Full contract surface also includes task snapshot + markdown (pilot checklist §3).
    assert body["task"]["task_id"] == task_id
    assert isinstance(body.get("evidence_pack_markdown"), str)
    assert "EvidencePack" in body["evidence_pack_markdown"]


def test_audit_export_tenant_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Other API key must not audit-export a task owned by a different tenant."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("API_KEY", "test")
    monkeypatch.setenv("API_KEYS", "tenant-a-key,tenant-b-key,test")
    from src.core.review_reuse import service as svc_mod
    from src.core.review_reuse.store import InMemoryReviewReuseStore

    svc_mod.reset_review_reuse_store_for_tests(InMemoryReviewReuseStore())
    from src.main import app

    with TestClient(app) as client:
        ra = client.post(
            "/api/v1/review-reuse/tasks",
            headers={"X-API-Key": "tenant-a-key"},
            files={"file": ("a.dxf", b"a", "application/octet-stream")},
        )
        assert ra.status_code == 200, ra.text
        task_id = ra.json()["task_id"]

        denied = client.get(
            f"/api/v1/review-reuse/tasks/{task_id}/audit-export",
            headers={"X-API-Key": "tenant-b-key"},
        )
        assert denied.status_code == 404

        owned = client.get(
            f"/api/v1/review-reuse/tasks/{task_id}/audit-export",
            headers={"X-API-Key": "tenant-a-key"},
        )
        assert owned.status_code == 200, owned.text
        body = owned.json()
        assert body["export_kind"] == "audit_quarantine"
        assert body["schema_version"] == "review-reuse-audit-bundle-v1"
        assert body["task"]["task_id"] == task_id


def test_audit_export_does_not_write_feedback_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """R2 HOLD: export_kind=audit_quarantine is not a training path — no FEEDBACK_LOG_PATH write."""
    log_path = tmp_path / "feedback_log.jsonl"
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("API_KEY", "test")
    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(log_path))
    monkeypatch.delenv("REVIEW_REUSE_DECISIONS_ENABLED", raising=False)
    from src.core.review_reuse import service as svc_mod
    from src.core.review_reuse.store import InMemoryReviewReuseStore

    svc_mod.reset_review_reuse_store_for_tests(InMemoryReviewReuseStore())
    from src.main import app

    with TestClient(app, headers={"X-API-Key": "test"}) as client:
        task_id = _create_task(client, name="r2.dxf", body=b"r2-hold")
        assert not log_path.exists(), "create_task must not write feedback JSONL"

        r = client.get(f"/api/v1/review-reuse/tasks/{task_id}/audit-export")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["export_kind"] == "audit_quarantine"
        assert body["export_kind"] != "training"
        assert "training" not in body["export_kind"]
        assert body["schema_version"] == "review-reuse-audit-bundle-v1"
        assert "evidence_pack" in body
        assert "events" in body

        assert not log_path.exists(), (
            "audit-export must not create/write FEEDBACK_LOG_PATH (R2 HOLD — not a training path)"
        )
