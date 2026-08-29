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

_REVIEWER = "principal-v1-" + "a" * 64


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


def test_export_audit_bundle_uses_one_task_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_DECISIONS_ENABLED, "true")
    svc = ReviewReuseService(InMemoryReviewReuseStore())
    ready = svc.create_task(
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
    decided = svc.submit_decision(
        tenant_id="t",
        task_id=ready.task_id,
        state=HumanDecisionState.revise,
        reviewer_id=_REVIEWER,
        reviewer_kind="validated_principal",
        tenant_validated=True,
        reviewer_validated=True,
        expected_revision=ready.revision,
        evidence_pack_sha256=ready.evidence_pack["evidence_pack_sha256"],
        candidate_id="c1",
        reason_codes=["needs_modification"],
        reason_text="Reviewed evidence.",
    )
    snapshots = iter([ready, decided])
    calls = 0

    def staged_get(_tenant_id: str, _task_id: str):
        nonlocal calls
        calls += 1
        return next(snapshots)

    monkeypatch.setattr(svc, "get_task", staged_get)
    bundle = svc.export_audit_bundle("t", ready.task_id)

    assert calls == 1
    assert bundle["task"]["revision"] == ready.revision
    assert bundle["evidence_pack"]["task_revision"] == ready.revision
    assert bundle["evidence_pack"] == ready.evidence_pack


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
            reviewer_kind="api_key_fallback",
            tenant_validated=True,
            reviewer_validated=False,
            expected_revision=task.revision,
            evidence_pack_sha256=task.evidence_pack["evidence_pack_sha256"],
        )
    assert ei.value.code == "reviewer_not_validated"

    ok = svc.submit_decision(
        tenant_id="t",
        task_id=task.task_id,
        state=HumanDecisionState.revise,
        reviewer_id=_REVIEWER,
        reviewer_kind="validated_principal",
        tenant_validated=True,
        reviewer_validated=True,
        expected_revision=task.revision,
        evidence_pack_sha256=task.evidence_pack["evidence_pack_sha256"],
        candidate_id=task.candidates[0].candidate_id,
        reason_codes=["needs_modification"],
        reason_text="Reviewed evidence.",
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


def _assert_isolated_exports(out: Path) -> None:
    assert (out / "task.json").exists()
    assert (out / "evidence.json").exists()
    assert (out / "evidence.md").exists()
    assert (out / "audit_bundle.json").exists()


def test_isolated_archive_script_seed_similar(tmp_path: Path) -> None:
    """main() with --seed-similar writes EvidencePack / audit exports."""
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
    _assert_isolated_exports(out)
    pack = (out / "evidence.json").read_text(encoding="utf-8")
    assert "synthetic-archive-001" in pack
    task = (out / "task.json").read_text(encoding="utf-8")
    assert "similar" in task


def test_isolated_archive_script_offline_insufficient_evidence(tmp_path: Path) -> None:
    """main() without seed still writes exports (offline insufficient_evidence)."""
    from scripts.review_reuse_isolated_archive_run import main

    out = tmp_path / "exports"
    rc = main(
        [
            "--out",
            str(out),
            "--tenant",
            "script-tenant-offline",
            "--idempotency-key",
            "script-offline-1",
        ]
    )
    assert rc == 0
    _assert_isolated_exports(out)
    task_raw = (out / "task.json").read_text(encoding="utf-8")
    assert "insufficient_evidence" in task_raw
    pack = (out / "evidence.json").read_text(encoding="utf-8")
    assert "insufficient_evidence" in pack


def test_isolated_archive_script_does_not_enable_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Script pops/never sets REVIEW_REUSE_DECISIONS_ENABLED; env not left enabled."""
    from scripts.review_reuse_isolated_archive_run import main

    # Pre-set as if an operator had decisions on; script must clear it.
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    out = tmp_path / "exports"
    rc = main(
        [
            "--out",
            str(out),
            "--seed-similar",
            "--tenant",
            "script-tenant-decisions",
            "--idempotency-key",
            "script-decisions-1",
        ]
    )
    assert rc == 0
    _assert_isolated_exports(out)
    # Script pops the var; must not remain true after run.
    assert os.environ.get("REVIEW_REUSE_DECISIONS_ENABLED") is None
    assert os.environ.get("REVIEW_REUSE_DECISIONS_ENABLED") != "true"

    # Second path: absent before run, still absent after (never sets true).
    monkeypatch.delenv("REVIEW_REUSE_DECISIONS_ENABLED", raising=False)
    out2 = tmp_path / "exports2"
    rc2 = main(
        [
            "--out",
            str(out2),
            "--tenant",
            "script-tenant-decisions-2",
            "--idempotency-key",
            "script-decisions-2",
        ]
    )
    assert rc2 == 0
    _assert_isolated_exports(out2)
    assert os.environ.get("REVIEW_REUSE_DECISIONS_ENABLED") is None
