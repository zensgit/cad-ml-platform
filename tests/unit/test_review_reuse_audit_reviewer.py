"""Audit export + validated reviewer gate tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.core.review_reuse.models import HumanDecisionState, TaskStatus
from src.core.review_reuse.service import (
    ENV_DECISIONS_ENABLED,
    ENV_REQUIRE_VALIDATED_REVIEWER,
    ReviewReuseError,
    ReviewReuseService,
)
from src.core.review_reuse.store import InMemoryReviewReuseStore


def test_export_audit_bundle() -> None:
    svc = ReviewReuseService(InMemoryReviewReuseStore())
    task = svc.create_task(
        tenant_id="t",
        file_name="a.dxf",
        file_bytes=b"x",
        seed_candidates=[
            {
                "candidate_id": "c1",
                "state": "similar",
                "scores": {"geometric": 0.9, "semantic": 0.8},
            }
        ],
    )
    bundle = svc.export_audit_bundle("t", task.task_id)
    assert bundle["schema_version"] == "review-reuse-audit-bundle-v1"
    assert bundle["export_kind"] == "audit_quarantine"
    assert bundle["task"]["task_id"] == task.task_id
    assert bundle["evidence_pack"]["task_id"] == task.task_id
    assert "EvidencePack" in bundle["evidence_pack_markdown"]
    assert len(bundle["events"]) >= 1


def test_require_validated_reviewer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_DECISIONS_ENABLED, "true")
    monkeypatch.setenv(ENV_REQUIRE_VALIDATED_REVIEWER, "true")
    svc = ReviewReuseService(InMemoryReviewReuseStore())
    task = svc.create_task(tenant_id="t", file_name="a.dxf", file_bytes=b"x")
    with pytest.raises(ReviewReuseError) as ei:
        svc.submit_decision(
            tenant_id="t",
            task_id=task.task_id,
            state=HumanDecisionState.reuse,
            reviewer_id="ak-user-deadbeef",
            reviewer_validated=False,
        )
    assert ei.value.code == "reviewer_not_validated"

    ok = svc.submit_decision(
        tenant_id="t",
        task_id=task.task_id,
        state=HumanDecisionState.revise,
        reviewer_id="jwt-sub-123",
        reviewer_validated=True,
    )
    assert ok.status == TaskStatus.decided


def test_audit_export_api(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("API_KEY", "test")
    from src.core.review_reuse import service as svc_mod

    svc_mod.reset_review_reuse_store_for_tests(InMemoryReviewReuseStore())
    from src.main import app

    with TestClient(app, headers={"X-API-Key": "test"}) as client:
        r = client.post(
            "/api/v1/review-reuse/tasks",
            files={"file": ("a.dxf", b"bytes", "application/octet-stream")},
        )
        assert r.status_code == 200, r.text
        tid = r.json()["task_id"]
        ex = client.get(f"/api/v1/review-reuse/tasks/{tid}/audit-export")
        assert ex.status_code == 200, ex.text
        body = ex.json()
        assert body["export_kind"] == "audit_quarantine"
        assert body["task"]["task_id"] == tid


def test_isolated_archive_script(tmp_path: Path) -> None:
    from scripts.review_reuse_isolated_archive_run import main

    out = tmp_path / "exports"
    rc = main(
        [
            "--out",
            str(out),
            "--seed-similar",
            "--tenant",
            "script-tenant",
            "--idempotency-key",
            "script-1",
        ]
    )
    assert rc == 0
    assert (out / "evidence.json").exists()
    assert (out / "evidence.md").exists()
    assert (out / "audit_bundle.json").exists()
    pack = (out / "evidence.json").read_text(encoding="utf-8")
    assert "synthetic-archive-001" in pack
