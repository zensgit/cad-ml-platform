"""R2 HOLD boundary tests — design-lock invariants R1–R8 (no false external claims)."""

from __future__ import annotations

import ast
import os
from pathlib import Path

import pytest

from src.core.review_reuse import dedup_adapter as da
from src.core.review_reuse.models import HumanDecisionState, TaskStatus
from src.core.review_reuse.service import (
    ENV_DECISIONS_ENABLED,
    ReviewReuseError,
    ReviewReuseService,
)
from src.core.review_reuse.store import ReviewReuseStore

ROOT = Path(__file__).resolve().parents[2]
REVIEW_REUSE_DIR = ROOT / "src" / "core" / "review_reuse"
API_FILE = ROOT / "src" / "api" / "v1" / "review_reuse.py"


def _py_files():
    files = list(REVIEW_REUSE_DIR.glob("*.py"))
    files.append(API_FILE)
    return files


def test_r2_no_import_of_feedback_training_path() -> None:
    """R2: decision ledger must not import feedback JSONL training surface."""
    banned = {
        "src.api.v1.feedback",
        "src.ml.learning.feedback_loop",
        "src.core.assistant.cost_cap",
    }
    for path in _py_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in banned, f"{path} imports {alias.name}"
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in banned, f"{path} imports from {node.module}"
                # also ban feedback_log.jsonl string writes via module name fragments
                assert "feedback" not in (node.module or "").split(".")[-1] or "review" in path.name


def test_r2_decision_does_not_write_feedback_jsonl(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """R2: submitting a decision must not create/write FEEDBACK_LOG_PATH."""
    log_path = tmp_path / "feedback_log.jsonl"
    monkeypatch.setenv("FEEDBACK_LOG_PATH", str(log_path))
    monkeypatch.setenv(ENV_DECISIONS_ENABLED, "true")
    da.set_live_recall_hook(None)
    monkeypatch.delenv(da.ENV_LIVE_DEDUP, raising=False)

    svc = ReviewReuseService(ReviewReuseStore())
    task = svc.create_task(tenant_id="r2", file_name="a.dxf", file_bytes=b"x")
    decided = svc.submit_decision(
        tenant_id="r2",
        task_id=task.task_id,
        state=HumanDecisionState.revise,
        reviewer_id="reviewer-r2",
        reason_codes=["r2-hold"],
        idempotency_key="r2-1",
    )
    assert decided.status == TaskStatus.decided
    assert not log_path.exists(), "decision must not create feedback JSONL (R2 HOLD)"


def test_r1_decision_default_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ENV_DECISIONS_ENABLED, raising=False)
    svc = ReviewReuseService(ReviewReuseStore())
    task = svc.create_task(tenant_id="r2", file_name="a.dxf", file_bytes=b"x")
    with pytest.raises(ReviewReuseError) as ei:
        svc.submit_decision(
            tenant_id="r2",
            task_id=task.task_id,
            state=HumanDecisionState.reuse,
            reviewer_id="r",
        )
    assert ei.value.code == "decisions_disabled"


def test_r5_r6_r7_boundary_files_absent() -> None:
    """R5/R7: no cost_cap revive; workbench does not import/call eval integrity gate."""
    assert not (ROOT / "src/core/assistant/cost_cap.py").exists()
    banned_imports = {
        "scripts.eval_integrity_gate",
        "src.core.assistant.cost_cap",
    }
    for path in list(REVIEW_REUSE_DIR.glob("*.py")) + [API_FILE]:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in banned_imports
            if isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in banned_imports
                assert not (node.module or "").endswith("eval_integrity_gate")
            if isinstance(node, ast.Call):
                # ban direct call name eval_integrity_gate.check
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "check":
                    if isinstance(func.value, ast.Name) and "eval_integrity" in func.value.id:
                        raise AssertionError(f"banned call in {path}")



def test_metrics_family_is_review_workflow_not_track_e() -> None:
    svc = ReviewReuseService(ReviewReuseStore())
    svc.create_task(tenant_id="r2", file_name="a.dxf", file_bytes=b"m")
    m = svc.metrics("r2")
    assert m["metric_family"] == "review_workflow"
    assert "model_release" not in m["metric_family"]
