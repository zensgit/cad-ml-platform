"""API tests for /api/v1/review-reuse/*."""

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
    monkeypatch.delenv("REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER", raising=False)
    # Fresh process-local store for isolation: rebind service singleton store.
    from src.core.review_reuse import service as svc_mod
    from src.core.review_reuse.store import InMemoryReviewReuseStore

    svc_mod.reset_review_reuse_store_for_tests(InMemoryReviewReuseStore())

    from src.main import app

    with TestClient(app, headers={"X-API-Key": "test"}) as c:
        yield c


def test_create_list_get_events_evidence(client: TestClient) -> None:
    r = client.post(
        "/api/v1/review-reuse/tasks",
        files={"file": ("sample.dxf", b"0\nSECTION\n", "application/octet-stream")},
        data={"idempotency_key": "api-idem-1"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "evidence_ready"
    task_id = body["task_id"]
    assert body["evidence_pack"] is not None

    r2 = client.post(
        "/api/v1/review-reuse/tasks",
        files={"file": ("sample.dxf", b"0\nSECTION\n", "application/octet-stream")},
        data={"idempotency_key": "api-idem-1"},
    )
    assert r2.json()["task_id"] == task_id

    listed = client.get("/api/v1/review-reuse/tasks")
    assert listed.status_code == 200
    assert any(t["task_id"] == task_id for t in listed.json())

    got = client.get(f"/api/v1/review-reuse/tasks/{task_id}")
    assert got.status_code == 200
    assert got.json()["task_id"] == task_id

    events = client.get(f"/api/v1/review-reuse/tasks/{task_id}/events")
    assert events.status_code == 200
    types = {e["event_type"] for e in events.json()}
    assert "submitted" in types
    assert "evidence_pack_ready" in types

    pack = client.get(f"/api/v1/review-reuse/tasks/{task_id}/evidence-pack")
    assert pack.status_code == 200
    assert pack.json()["schema_version"] == "evidence-pack-v1"

    md = client.get(
        f"/api/v1/review-reuse/tasks/{task_id}/evidence-pack",
        params={"format": "markdown"},
    )
    assert md.status_code == 200
    assert "markdown" in md.headers.get("content-type", "")
    assert "EvidencePack" in md.text
    assert task_id in md.text


def test_evidence_pack_format_markdown(client: TestClient) -> None:
    """GET .../evidence-pack?format=markdown → text/markdown with EvidencePack + task_id."""
    r = client.post(
        "/api/v1/review-reuse/tasks",
        files={"file": ("md-pack.dxf", b"0\nSECTION\n", "application/octet-stream")},
        data={"idempotency_key": "api-md-evidence-1"},
    )
    assert r.status_code == 200, r.text
    task_id = r.json()["task_id"]

    md = client.get(
        f"/api/v1/review-reuse/tasks/{task_id}/evidence-pack",
        params={"format": "markdown"},
    )
    assert md.status_code == 200, md.text
    content_type = md.headers.get("content-type", "")
    assert "markdown" in content_type
    assert "EvidencePack" in md.text
    assert task_id in md.text


def test_decision_default_off_403(client: TestClient) -> None:
    r = client.post(
        "/api/v1/review-reuse/tasks",
        files={"file": ("a.dxf", b"x", "application/octet-stream")},
    )
    task = r.json()
    task_id = task["task_id"]
    dec = client.post(
        f"/api/v1/review-reuse/tasks/{task_id}/decision",
        json={
            "state": "new",
            "reason_codes": ["new_part_required"],
            "reason_text": "nope",
            "expected_revision": task["revision"],
            "evidence_pack_sha256": task["evidence_pack"]["evidence_pack_sha256"],
        },
    )
    assert dec.status_code == 403
    assert dec.json()["detail"]["code"] == "decisions_disabled"


def test_decision_enabled_still_rejects_api_key_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("API_KEY", "test")
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    monkeypatch.delenv("REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER", raising=False)
    from src.core.review_reuse import service as svc_mod
    from src.core.review_reuse.store import InMemoryReviewReuseStore

    svc_mod.reset_review_reuse_store_for_tests(InMemoryReviewReuseStore())
    from src.main import app

    with TestClient(app, headers={"X-API-Key": "test"}) as client:
        r = client.post(
            "/api/v1/review-reuse/tasks",
            files={"file": ("a.dxf", b"x", "application/octet-stream")},
        )
        task = r.json()
        task_id = task["task_id"]
        dec = client.post(
            f"/api/v1/review-reuse/tasks/{task_id}/decision",
            json={
                "state": "new",
                "reason_codes": ["new_part_required"],
                "idempotency_key": "d1",
                "expected_revision": task["revision"],
                "evidence_pack_sha256": task["evidence_pack"]["evidence_pack_sha256"],
            },
        )
        assert dec.status_code == 403, dec.text
        assert dec.json()["detail"]["code"] == "tenant_not_validated"


def test_decision_require_validated_reviewer_api_key_403(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decisions on + require validated reviewer: API-key-only → 403.

    With a validated tenant but no JWT subject, reviewer identity is fallback
    validated=False (R2 HOLD: pilot must not accept key-only identity).
    """
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("API_KEY", "test")
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    monkeypatch.setenv("REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER", "true")
    from src.core.review_reuse import service as svc_mod
    from src.core.review_reuse.store import InMemoryReviewReuseStore

    svc_mod.reset_review_reuse_store_for_tests(InMemoryReviewReuseStore())
    import src.api.v1.review_reuse as rr_api
    from src.main import app

    monkeypatch.setattr(
        rr_api, "_tenant_identity", lambda request, api_key: ("tenant-jwt", True)
    )

    with TestClient(app, headers={"X-API-Key": "test"}) as client:
        r = client.post(
            "/api/v1/review-reuse/tasks",
            files={"file": ("a.dxf", b"x", "application/octet-stream")},
        )
        assert r.status_code == 200, r.text
        task = r.json()
        task_id = task["task_id"]
        dec = client.post(
            f"/api/v1/review-reuse/tasks/{task_id}/decision",
            json={
                "state": "new",
                "reason_codes": ["new_part_required"],
                "reason_text": "no jwt subject",
                "expected_revision": task["revision"],
                "evidence_pack_sha256": task["evidence_pack"]["evidence_pack_sha256"],
            },
        )
        assert dec.status_code == 403, dec.text
        detail = dec.json()["detail"]
        assert detail["code"] == "reviewer_not_validated"


def test_decision_require_validated_reviewer_with_jwt_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Decisions on + validated tenant/reviewer principals → 200."""
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("API_KEY", "test")
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    monkeypatch.setenv("REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER", "true")
    from src.core.review_reuse import service as svc_mod
    from src.core.review_reuse.store import InMemoryReviewReuseStore

    svc_mod.reset_review_reuse_store_for_tests(InMemoryReviewReuseStore())
    import src.api.v1.review_reuse as rr_api
    from src.main import app

    reviewer_id = "principal-v1-" + "a" * 64
    monkeypatch.setattr(
        rr_api, "_tenant_identity", lambda request, api_key: ("tenant-jwt", True)
    )
    monkeypatch.setattr(
        rr_api,
        "_reviewer_identity",
        lambda request, api_key: (reviewer_id, "validated_principal", True),
    )

    with TestClient(app, headers={"X-API-Key": "test"}) as client:
        r = client.post(
            "/api/v1/review-reuse/tasks",
            files={"file": ("a.dxf", b"x", "application/octet-stream")},
        )
        assert r.status_code == 200, r.text
        task = r.json()
        task_id = task["task_id"]
        dec = client.post(
            f"/api/v1/review-reuse/tasks/{task_id}/decision",
            json={
                "state": "new",
                "reason_codes": ["new_part_required"],
                "idempotency_key": "validated-d1",
                "expected_revision": task["revision"],
                "evidence_pack_sha256": task["evidence_pack"]["evidence_pack_sha256"],
            },
        )
        assert dec.status_code == 200, dec.text
        body = dec.json()
        assert body["status"] == "decided"
        assert body["human_decision"]["state"] == "new"
        assert body["human_decision"]["reviewer_id"] == reviewer_id


def test_tenant_isolation_different_api_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("API_KEY", "test")
    # Allow alternate keys via API_KEYS if supported; otherwise use same key
    # with explicit request.state is hard — hash of different keys must differ.
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
        # Other tenant cannot see it
        rb = client.get(
            f"/api/v1/review-reuse/tasks/{task_id}",
            headers={"X-API-Key": "tenant-b-key"},
        )
        assert rb.status_code == 404
        # Owner can
        ra2 = client.get(
            f"/api/v1/review-reuse/tasks/{task_id}",
            headers={"X-API-Key": "tenant-a-key"},
        )
        assert ra2.status_code == 200


def test_cancel(client: TestClient) -> None:
    r = client.post(
        "/api/v1/review-reuse/tasks",
        files={"file": ("a.dxf", b"x", "application/octet-stream")},
    )
    task_id = r.json()["task_id"]
    c = client.post(f"/api/v1/review-reuse/tasks/{task_id}/cancel")
    assert c.status_code == 200
    assert c.json()["status"] == "canceled"


def test_not_found(client: TestClient) -> None:
    r = client.get("/api/v1/review-reuse/tasks/does-not-exist")
    assert r.status_code == 404


def test_metrics_endpoint(client: TestClient) -> None:
    client.post(
        "/api/v1/review-reuse/tasks",
        files={"file": ("a.dxf", b"metrics", "application/octet-stream")},
    )
    r = client.get("/api/v1/review-reuse/metrics")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metric_family"] == "review_workflow"
    assert body["task_count"] >= 1

    # default remains json
    r_default = client.get("/api/v1/review-reuse/metrics", params={"format": "json"})
    assert r_default.status_code == 200
    assert r_default.json()["metric_family"] == "review_workflow"

    md = client.get("/api/v1/review-reuse/metrics", params={"format": "markdown"})
    assert md.status_code == 200, md.text
    content_type = md.headers.get("content-type", "")
    assert "text/markdown" in content_type
    assert "ReviewReuse workflow metrics" in md.text
    assert "metric_family: `review_workflow`" in md.text
    assert "task_count:" in md.text
