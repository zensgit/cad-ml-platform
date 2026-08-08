"""Unit tests for ReviewReuse workbench (Track R MVP)."""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pytest

from src.core.review_reuse.evidence import build_evidence_pack, evidence_pack_markdown
from src.core.review_reuse.models import (
    CandidateState,
    HumanDecisionState,
    TaskEventType,
    TaskStatus,
)
from src.core.review_reuse.service import (
    ENV_DECISIONS_ENABLED,
    ReviewReuseError,
    ReviewReuseService,
    decisions_enabled,
)
from src.core.review_reuse.store import ReviewReuseStore


def _svc() -> ReviewReuseService:
    return ReviewReuseService(ReviewReuseStore())


def _seed_similar() -> List[Dict[str, Any]]:
    return [
        {
            "candidate_id": "arch-001",
            "candidate_source": "archive",
            "state": "similar",
            "scores": {"geometric": 0.91, "semantic": 0.72},
            "verification": {
                "verdict": "similar",
                "level": 2,
                "methods": ["dedup2d-adapter", "precision-l4"],
            },
            "rejection_reasons": [],
            "decision_source": "synthetic-fixture",
        }
    ]


class TestDecisionsEnabled:
    def test_default_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv(ENV_DECISIONS_ENABLED, raising=False)
        assert decisions_enabled() is False

    def test_true_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for v in ("1", "true", "YES", "on"):
            monkeypatch.setenv(ENV_DECISIONS_ENABLED, v)
            assert decisions_enabled() is True


class TestCreateAndEvidence:
    def test_offline_insufficient_evidence(self) -> None:
        svc = _svc()
        task = svc.create_task(
            tenant_id="t-a",
            file_name="part.dxf",
            file_bytes=b"0\nSECTION\n",
        )
        assert task.status == TaskStatus.evidence_ready
        assert len(task.candidates) == 1
        assert task.candidates[0].state == CandidateState.insufficient_evidence
        assert "tool_unavailable" in task.candidates[0].rejection_reasons
        assert task.evidence_pack is not None
        pack = task.evidence_pack
        assert pack["schema_version"] == "evidence-pack-v1"
        assert pack["task_id"] == task.task_id
        assert pack["trace_id"] == task.trace_id
        assert "geometric" in pack["candidates"][0]["scores"]
        assert "semantic" in pack["candidates"][0]["scores"]
        assert pack["calibration"]["version"]
        assert pack["human_decision"]["allowed_actions"] == ["reuse", "revise", "new"]
        types = [e.event_type for e in task.events]
        assert TaskEventType.submitted in types
        assert TaskEventType.evidence_pack_ready in types
        assert TaskEventType.recall_completed in types
        assert TaskEventType.precision_completed in types

    def test_seeded_archive_candidate(self) -> None:
        svc = _svc()
        task = svc.create_task(
            tenant_id="t-a",
            file_name="part.dxf",
            file_bytes=b"DXF",
            seed_candidates=_seed_similar(),
        )
        assert task.candidates[0].candidate_id == "arch-001"
        assert task.candidates[0].state == CandidateState.similar
        assert task.evidence_pack["candidates"][0]["scores"]["geometric"] == 0.91
        md = evidence_pack_markdown(task.evidence_pack)
        assert "arch-001" in md
        assert "EvidencePack" in md

    def test_idempotent_create(self) -> None:
        svc = _svc()
        a = svc.create_task(
            tenant_id="t-a",
            file_name="a.dxf",
            file_bytes=b"1",
            idempotency_key="idem-1",
        )
        b = svc.create_task(
            tenant_id="t-a",
            file_name="a.dxf",
            file_bytes=b"1",
            idempotency_key="idem-1",
        )
        assert a.task_id == b.task_id

    def test_tenant_isolation(self) -> None:
        svc = _svc()
        task = svc.create_task(
            tenant_id="t-a",
            file_name="a.dxf",
            file_bytes=b"1",
        )
        with pytest.raises(ReviewReuseError) as ei:
            svc.get_task("t-b", task.task_id)
        assert ei.value.code == "not_found"
        assert svc.list_tasks("t-b") == []
        assert len(svc.list_tasks("t-a")) == 1


class TestDecisionGate:
    def test_decision_disabled_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(ENV_DECISIONS_ENABLED, raising=False)
        svc = _svc()
        task = svc.create_task(
            tenant_id="t-a",
            file_name="a.dxf",
            file_bytes=b"1",
            seed_candidates=_seed_similar(),
        )
        with pytest.raises(ReviewReuseError) as ei:
            svc.submit_decision(
                tenant_id="t-a",
                task_id=task.task_id,
                state=HumanDecisionState.reuse,
                reviewer_id="reviewer-1",
            )
        assert ei.value.code == "decisions_disabled"

    def test_decision_enabled_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(ENV_DECISIONS_ENABLED, "true")
        svc = _svc()
        task = svc.create_task(
            tenant_id="t-a",
            file_name="a.dxf",
            file_bytes=b"1",
            seed_candidates=_seed_similar(),
        )
        d1 = svc.submit_decision(
            tenant_id="t-a",
            task_id=task.task_id,
            state=HumanDecisionState.revise,
            reviewer_id="reviewer-1",
            reason_codes=["needs_check"],
            candidate_id="arch-001",
            idempotency_key="dec-1",
        )
        assert d1.status == TaskStatus.decided
        assert d1.human_decision is not None
        assert d1.human_decision.state == HumanDecisionState.revise
        assert d1.evidence_pack["human_decision"]["state"] == "revise"

        d2 = svc.submit_decision(
            tenant_id="t-a",
            task_id=task.task_id,
            state=HumanDecisionState.revise,
            reviewer_id="reviewer-1",
            idempotency_key="dec-1",
        )
        assert d2.task_id == d1.task_id

        with pytest.raises(ReviewReuseError) as ei:
            svc.submit_decision(
                tenant_id="t-a",
                task_id=task.task_id,
                state=HumanDecisionState.new,
                reviewer_id="reviewer-1",
                idempotency_key="dec-2",
            )
        assert ei.value.code == "already_decided"

    def test_cancel_blocks_later_decision(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(ENV_DECISIONS_ENABLED, "1")
        svc = _svc()
        task = svc.create_task(
            tenant_id="t-a",
            file_name="a.dxf",
            file_bytes=b"1",
        )
        canceled = svc.cancel("t-a", task.task_id)
        assert canceled.status == TaskStatus.canceled
        with pytest.raises(ReviewReuseError) as ei:
            svc.submit_decision(
                tenant_id="t-a",
                task_id=task.task_id,
                state=HumanDecisionState.new,
                reviewer_id="r1",
            )
        assert ei.value.code == "canceled"


class TestEvidenceBuilder:
    def test_section_33_minimum_fields(self) -> None:
        svc = _svc()
        task = svc.create_task(
            tenant_id="t-a",
            file_name="x.dxf",
            file_bytes=b"bytes",
            seed_candidates=_seed_similar(),
        )
        pack = build_evidence_pack(task)
        required_top = [
            "schema_version",
            "task_id",
            "trace_id",
            "candidates",
            "confidence",
            "calibration",
            "evidence",
            "rejection_reasons",
            "provenance",
            "human_decision",
        ]
        for k in required_top:
            assert k in pack
        c0 = pack["candidates"][0]
        assert "candidate_id" in c0
        assert "candidate_source" in c0
        assert "scores" in c0
        assert "verification" in c0
        assert "provenance" in c0


class TestDedupAdapterAndMetrics:
    def test_live_dedup_off_offline_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.core.review_reuse import dedup_adapter as da

        monkeypatch.delenv(da.ENV_LIVE_DEDUP, raising=False)
        da.set_live_recall_hook(None)
        svc = _svc()
        task = svc.create_task(tenant_id="t-m", file_name="a.dxf", file_bytes=b"x")
        assert task.candidates[0].state == CandidateState.insufficient_evidence

    def test_live_dedup_hook(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from src.core.review_reuse import dedup_adapter as da

        monkeypatch.setenv(da.ENV_LIVE_DEDUP, "true")

        def _hook(fn: str, fb: bytes, sha: str):
            return [
                {
                    "candidate_id": "live-1",
                    "state": "duplicate",
                    "scores": {"geometric": 0.99, "semantic": 0.5},
                    "match_level": 3,
                }
            ]

        da.set_live_recall_hook(_hook)
        try:
            svc = _svc()
            task = svc.create_task(tenant_id="t-m", file_name="a.dxf", file_bytes=b"x")
            assert task.candidates[0].candidate_id == "live-1"
            assert task.candidates[0].state == CandidateState.duplicate
        finally:
            da.set_live_recall_hook(None)
            monkeypatch.delenv(da.ENV_LIVE_DEDUP, raising=False)

    def test_live_dedup_hook_failure_fail_closed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from src.core.review_reuse import dedup_adapter as da

        monkeypatch.setenv(da.ENV_LIVE_DEDUP, "1")

        def _boom(fn: str, fb: bytes, sha: str):
            raise RuntimeError("dedup down")

        da.set_live_recall_hook(_boom)
        try:
            svc = _svc()
            task = svc.create_task(tenant_id="t-m", file_name="a.dxf", file_bytes=b"x")
            assert task.candidates[0].state == CandidateState.insufficient_evidence
            assert (
                "external_service_unavailable" in task.candidates[0].rejection_reasons
            )
        finally:
            da.set_live_recall_hook(None)

    def test_metrics(self) -> None:
        svc = _svc()
        svc.create_task(
            tenant_id="t-m",
            file_name="a.dxf",
            file_bytes=b"1",
            seed_candidates=_seed_similar(),
        )
        m = svc.metrics("t-m")
        assert m["metric_family"] == "review_workflow"
        assert m["task_count"] == 1
        assert m["by_status"].get("evidence_ready") == 1
        assert "track_e" not in m["metric_family"]

    def test_format_metrics_markdown(self) -> None:
        from src.core.review_reuse.metrics import format_metrics_markdown

        payload = {
            "schema_version": "review-reuse-metrics-v1",
            "metric_family": "review_workflow",
            "tenant_id": "t-md",
            "task_count": 2,
            "by_status": {"evidence_ready": 1, "decided": 1},
            "by_decision": {"reuse": 1},
            "accepted_reuse": 1,
            "candidate_total": 3,
            "insufficient_evidence_count": 1,
            "insufficient_evidence_rate": 1 / 3,
            "median_time_to_evidence_seconds": 1.5,
            "reviewer_coverage": 1,
            "notes": ["pilot labels needed"],
        }
        md = format_metrics_markdown(payload)
        assert md.startswith("# ReviewReuse workflow metrics\n")
        assert "metric_family: `review_workflow`" in md
        assert "task_count: 2" in md
        assert "accepted_reuse: 1" in md
        assert "- evidence_ready: 1" in md
        assert "- reuse: 1" in md
        assert "pilot labels needed" in md
        assert "track_e" not in md

        empty_md = format_metrics_markdown(
            {
                "schema_version": "review-reuse-metrics-v1",
                "metric_family": "review_workflow",
                "tenant_id": "empty",
                "task_count": 0,
                "by_status": {},
                "by_decision": {},
                "accepted_reuse": 0,
                "candidate_total": 0,
                "insufficient_evidence_count": 0,
                "insufficient_evidence_rate": 0.0,
                "median_time_to_evidence_seconds": None,
                "reviewer_coverage": 0,
                "notes": [],
            }
        )
        assert "_(none)_" in empty_md
        assert empty_md.endswith("\n")
