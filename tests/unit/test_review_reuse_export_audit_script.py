"""CLI tests for scripts/review_reuse_export_audit.py.

Filesystem store: create task → export by task_id → audit_quarantine files.
Decisions stay off; R2 HOLD (not training-readable).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.core.review_reuse.service import ReviewReuseService
from src.core.review_reuse.store import ENV_STORE, ENV_STORE_DIR, create_review_reuse_store


def test_export_audit_script_filesystem_create_then_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Create task on filesystem store, then export by --tenant/--task-id."""
    store_root = tmp_path / "review_reuse_tasks"
    out = tmp_path / "export"
    monkeypatch.setenv(ENV_STORE, "filesystem")
    monkeypatch.setenv(ENV_STORE_DIR, str(store_root))
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("REVIEW_REUSE_DECISIONS_ENABLED", raising=False)

    store = create_review_reuse_store()
    svc = ReviewReuseService(store)
    task = svc.create_task(
        tenant_id="export-tenant",
        file_name="export.dxf",
        file_bytes=b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF\n",
        idempotency_key="export-audit-1",
        seed_candidates=[
            {
                "candidate_id": "archive-c1",
                "candidate_source": "archive",
                "state": "similar",
                "scores": {"geometric": 0.87, "semantic": 0.71},
            }
        ],
    )

    from scripts.review_reuse_export_audit import main

    rc = main(
        [
            "--tenant",
            "export-tenant",
            "--task-id",
            task.task_id,
            "--out",
            str(out),
        ]
    )
    assert rc == 0

    audit_path = out / "audit_bundle.json"
    md_path = out / "evidence.md"
    assert audit_path.is_file()
    assert md_path.is_file()

    bundle = json.loads(audit_path.read_text(encoding="utf-8"))
    assert bundle["schema_version"] == "review-reuse-audit-bundle-v1"
    assert bundle["export_kind"] == "audit_quarantine"
    assert "training" not in bundle["export_kind"]
    assert bundle["task"]["task_id"] == task.task_id
    assert bundle["task"]["tenant_id"] == "export-tenant"
    assert bundle["evidence_pack"]["task_id"] == task.task_id
    assert len(bundle["events"]) >= 1
    assert "EvidencePack" in bundle["evidence_pack_markdown"]
    assert "EvidencePack" in md_path.read_text(encoding="utf-8")
    assert "archive-c1" in json.dumps(bundle["evidence_pack"])

    # Script must leave decisions disabled (R2 / pilot fail-closed).
    assert os.environ.get("REVIEW_REUSE_DECISIONS_ENABLED") is None


def test_export_audit_script_missing_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unknown task_id returns non-zero and writes no outputs."""
    store_root = tmp_path / "review_reuse_tasks"
    out = tmp_path / "export"
    monkeypatch.setenv(ENV_STORE, "filesystem")
    monkeypatch.setenv(ENV_STORE_DIR, str(store_root))
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.delenv("REVIEW_REUSE_DECISIONS_ENABLED", raising=False)

    from scripts.review_reuse_export_audit import main

    rc = main(
        [
            "--tenant",
            "export-tenant",
            "--task-id",
            "does-not-exist",
            "--out",
            str(out),
        ]
    )
    assert rc == 1
    assert not (out / "audit_bundle.json").exists()
    assert not (out / "evidence.md").exists()


def test_export_audit_script_does_not_enable_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Script pops REVIEW_REUSE_DECISIONS_ENABLED even if pre-set true."""
    store_root = tmp_path / "review_reuse_tasks"
    out = tmp_path / "export"
    monkeypatch.setenv(ENV_STORE, "filesystem")
    monkeypatch.setenv(ENV_STORE_DIR, str(store_root))
    monkeypatch.setenv("ENVIRONMENT", "development")
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")

    store = create_review_reuse_store()
    svc = ReviewReuseService(store)
    task = svc.create_task(
        tenant_id="export-tenant-dec",
        file_name="d.dxf",
        file_bytes=b"x",
        idempotency_key="export-dec-1",
    )

    from scripts.review_reuse_export_audit import main

    rc = main(
        [
            "--tenant",
            "export-tenant-dec",
            "--task-id",
            task.task_id,
            "--out",
            str(out),
        ]
    )
    assert rc == 0
    assert os.environ.get("REVIEW_REUSE_DECISIONS_ENABLED") is None
    bundle = json.loads((out / "audit_bundle.json").read_text(encoding="utf-8"))
    assert bundle["export_kind"] == "audit_quarantine"
