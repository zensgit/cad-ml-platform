"""Precision pass + filename gate tests (honest geometric / rejection reasons)."""

from __future__ import annotations

import json
import math
from typing import List

import pytest

from src.core.review_reuse.dedup_adapter import map_raw_hits_to_candidates
from src.core.review_reuse.files import is_allowed_review_reuse_filename
from src.core.review_reuse.models import (
    CandidateDecision,
    CandidateState,
    HumanDecisionState,
    RejectionReason,
    TaskEventType,
)
from src.core.review_reuse.precision import apply_precision, set_precision_hook
from src.core.review_reuse.service import (
    ENV_DECISIONS_ENABLED,
    ReviewReuseError,
    ReviewReuseService,
)
from src.core.review_reuse.store import ReviewReuseStore


def _svc() -> ReviewReuseService:
    return ReviewReuseService(ReviewReuseStore())


def test_filename_gate() -> None:
    assert is_allowed_review_reuse_filename("part.dxf")
    assert is_allowed_review_reuse_filename("PART.DWG")
    assert is_allowed_review_reuse_filename("query.json")
    assert is_allowed_review_reuse_filename("sheet.PDF")
    assert is_allowed_review_reuse_filename("scan.PNG")
    assert not is_allowed_review_reuse_filename("payload.exe")
    assert not is_allowed_review_reuse_filename("upload.bin")
    assert not is_allowed_review_reuse_filename("")


def test_create_rejects_unsupported_file_type() -> None:
    svc = _svc()
    with pytest.raises(ReviewReuseError) as ei:
        svc.create_task(
            tenant_id="t-a",
            file_name="payload.exe",
            file_bytes=b"MZ",
        )
    assert ei.value.code == RejectionReason.unsupported_file_type.value


def test_create_accepts_geom_json_filename() -> None:
    svc = _svc()
    geom = _line_geom()
    task = svc.create_task(
        tenant_id="t-json",
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
        seed_candidates=[
            {
                "candidate_id": "arch-geom",
                "state": "similar",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
    )
    assert task.source_file_name == "query.json"
    assert task.candidates
    assert "precision-l4" in (task.candidates[0].verification.get("methods") or [])


def test_concurrent_idempotent_creates_single_task() -> None:
    import threading

    svc = _svc()
    seed = [
        {
            "candidate_id": "c1",
            "state": "similar",
            "scores": {"geometric": 0.9, "semantic": 0.8},
            "methods": ["precision-l4"],
        }
    ]
    out: list = []

    def _run() -> None:
        out.append(
            svc.create_task(
                tenant_id="t-idem",
                file_name="a.dxf",
                file_bytes=b"x",
                idempotency_key="same-key",
                seed_candidates=seed,
            )
        )

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(out) == 2
    assert out[0].task_id == out[1].task_id
    assert len(svc.list_tasks("t-idem")) == 1


def test_pipeline_honors_mid_flight_cancel(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.review_reuse.models import TaskStatus

    svc = _svc()

    canceled_at = {"ts": 0.0}

    def _cancel_then_pass(cands, **_k):
        running = svc.list_tasks("t-cancel")
        assert running
        canceled = svc.cancel("t-cancel", running[0].task_id)
        canceled_at["ts"] = canceled.updated_at
        return cands

    monkeypatch.setattr(
        "src.core.review_reuse.service.apply_precision", _cancel_then_pass
    )
    task = svc.create_task(
        tenant_id="t-cancel",
        file_name="a.dxf",
        file_bytes=b"x",
        seed_candidates=[
            {
                "candidate_id": "c1",
                "state": "similar",
                "scores": {"geometric": 0.9, "semantic": 0.8},
                "methods": ["precision-l4"],
            }
        ],
    )
    assert task.status == TaskStatus.canceled
    assert svc.get_task("t-cancel", task.task_id).status == TaskStatus.canceled
    assert task.updated_at >= canceled_at["ts"]
    types = {e.event_type for e in task.events}
    assert TaskEventType.canceled in types
    assert TaskEventType.recall_completed in types
    assert TaskEventType.precision_completed in types
    assert TaskEventType.evidence_pack_ready in types


def test_pipeline_rebuilds_pack_after_mid_flight_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.review_reuse.models import TaskStatus

    monkeypatch.setenv(ENV_DECISIONS_ENABLED, "true")
    svc = _svc()
    seed = [
        {
            "candidate_id": "c1",
            "state": "similar",
            "scores": {"geometric": 0.9, "semantic": 0.8},
            "methods": ["precision-l4"],
        }
    ]

    def _decide_then_pass(cands, **_k):
        running = svc.list_tasks("t-dec")
        assert running
        decided = svc.submit_decision(
            tenant_id="t-dec",
            task_id=running[0].task_id,
            state=HumanDecisionState.new,
            reviewer_id="reviewer-1",
        )
        assert decided.status == TaskStatus.decided
        assert decided.candidates == []
        assert decided.evidence_pack is not None
        assert decided.evidence_pack.get("candidates") == []
        return cands

    monkeypatch.setattr(
        "src.core.review_reuse.service.apply_precision", _decide_then_pass
    )
    task = svc.create_task(
        tenant_id="t-dec",
        file_name="a.dxf",
        file_bytes=b"x",
        seed_candidates=seed,
    )
    assert task.status == TaskStatus.decided
    assert [c.candidate_id for c in task.candidates] == ["c1"]
    pack = task.evidence_pack or {}
    assert [row.get("candidate_id") for row in pack.get("candidates") or []] == ["c1"]
    assert (pack.get("human_decision") or {}).get("state") == "new"
    stored = svc.get_task("t-dec", task.task_id)
    assert stored.status == TaskStatus.decided
    assert stored.evidence_pack == pack
    bundle = svc.export_audit_bundle("t-dec", task.task_id)
    assert bundle["evidence_pack"]["candidates"][0]["candidate_id"] == "c1"
    assert bundle["task"]["candidates"][0]["candidate_id"] == "c1"
    types = {e["event_type"] for e in bundle["events"]}
    assert "decision_submitted" in types
    assert "recall_completed" in types
    assert "precision_completed" in types
    assert "evidence_pack_ready" in types
    metrics = svc.metrics("t-dec")
    assert metrics["median_review_time_seconds"] is None


def test_idempotency_replay_skips_file_gate() -> None:
    import time

    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus

    svc = _svc()
    now = time.time()
    prior = ReviewReuseTask(
        task_id="pre-gate",
        tenant_id="t-a",
        status=TaskStatus.evidence_ready,
        created_at=now,
        updated_at=now,
        source_file_name="legacy.bin",
        source_content_sha256="ab",
        idempotency_key="idem-legacy",
        trace_id="tr-legacy",
    )
    svc.store.put(prior)
    again = svc.create_task(
        tenant_id="t-a",
        file_name="legacy.bin",
        file_bytes=b"x",
        idempotency_key="idem-legacy",
    )
    assert again.task_id == "pre-gate"


def test_precision_labels_vision_only_without_copying_visual() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "v1",
                "state": "similar",
                "scores": {"semantic": 0.9, "visual": 0.9},
                "methods": ["dedup2d-vision"],
                "decision_source": "dedup2d-vision",
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    assert cands[0].scores.get("geometric") is None
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"not-json")
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.vision_only_unverified.value in out[0].rejection_reasons


def _line_geom() -> dict:
    return {
        "layers": {"0": {"color": 7, "linetype": "CONTINUOUS"}},
        "entities": [
            {
                "type": "LINE",
                "layer": "0",
                "start": [0.0, 0.0],
                "end": [100.0, 0.0],
            }
        ],
    }


def test_precision_missing_geom_json_for_non_vision() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "s1",
                "state": "similar",
                "candidate_source": "archive",
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"dxf")
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_precision_keeps_seeded_l4_geometric() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "arch-001",
                "state": "similar",
                "scores": {"geometric": 0.91, "semantic": 0.72},
                "methods": ["dedup2d-adapter", "precision-l4"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"dxf")
    assert out[0].scores.get("geometric") == 0.91
    assert RejectionReason.vision_only_unverified.value not in out[0].rejection_reasons
    assert int(out[0].verification.get("level") or 0) >= 4


def test_seeded_l4_without_match_level_exports_level_4() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "x",
                "state": "similar",
                "scores": {"geometric": 0.9},
                "methods": ["precision-l4"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    assert int(cands[0].verification.get("level") or 0) == 0
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"x")
    assert int(out[0].verification.get("level") or 0) >= 4
    assert "precision-l4" in (out[0].verification.get("methods") or [])


def test_local_l4_clears_stale_vision_only_reason() -> None:
    from src.core.review_reuse.evidence import build_evidence_pack
    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus

    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "x",
                "state": "similar",
                "geom_json": geom,
                "methods": ["dedup2d-vision"],
                "decision_source": "dedup2d-vision",
                "rejection_reasons": ["vision_only_unverified"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert out[0].scores.get("geometric") == 1.0
    assert int(out[0].verification.get("level") or 0) >= 4
    assert RejectionReason.vision_only_unverified.value not in out[0].rejection_reasons
    assert RejectionReason.missing_geom_json.value not in out[0].rejection_reasons
    now = 0.0
    task = ReviewReuseTask(
        task_id="t-l4",
        tenant_id="t",
        status=TaskStatus.evidence_ready,
        created_at=now,
        updated_at=now,
        source_file_name="query.json",
        source_content_sha256="ab",
        trace_id="tr",
        candidates=out,
    )
    pack = build_evidence_pack(task)
    assert pack["confidence"]["score"] == 1.0
    assert pack["confidence"]["band"] == "high"


def test_precision_hook_runs_in_create_task() -> None:
    def _hook(
        file_name: str, file_bytes: bytes, cands: List[CandidateDecision]
    ) -> List[CandidateDecision]:
        del file_name, file_bytes
        out = []
        for c in cands:
            copied = c.model_copy(deep=True)
            copied.scores = dict(copied.scores)
            copied.scores["geometric"] = 0.77
            copied.verification = {
                "verdict": copied.state.value,
                "level": 4,
                "methods": ["precision-l4"],
            }
            out.append(copied)
        return out

    set_precision_hook(_hook)
    try:
        svc = _svc()
        task = svc.create_task(
            tenant_id="t-p",
            file_name="a.dxf",
            file_bytes=b"x",
            seed_candidates=[
                {
                    "candidate_id": "h1",
                    "state": "similar",
                    "scores": {"semantic": 0.5},
                    "methods": ["dedup2d-vision"],
                    "decision_source": "dedup2d-vision",
                }
            ],
        )
        assert task.candidates[0].scores.get("geometric") == 0.77
        prec = [
            e
            for e in task.events
            if e.event_type == TaskEventType.precision_completed
        ]
        assert prec
        assert prec[0].detail.get("precision_l4") == 1
    finally:
        set_precision_hook(None)


def test_unknown_candidate_on_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_DECISIONS_ENABLED, "true")
    svc = _svc()
    task = svc.create_task(
        tenant_id="t-a",
        file_name="a.dxf",
        file_bytes=b"x",
        seed_candidates=[
            {
                "candidate_id": "arch-001",
                "state": "similar",
                "scores": {"geometric": 0.91, "semantic": 0.7},
                "methods": ["precision-l4"],
            }
        ],
    )
    with pytest.raises(ReviewReuseError) as ei:
        svc.submit_decision(
            tenant_id="t-a",
            task_id=task.task_id,
            state=HumanDecisionState.reuse,
            reviewer_id="r1",
            candidate_id="not-on-task",
        )
    assert ei.value.code == "unknown_candidate"


def test_pipeline_failed_persists_failed_task(monkeypatch: pytest.MonkeyPatch) -> None:
    from src.core.review_reuse.models import TaskEventType, TaskStatus

    secret = "precision exploded /secret/path token=abc"

    def _boom(*_a, **_k):
        raise RuntimeError(secret)

    monkeypatch.setattr("src.core.review_reuse.service.apply_precision", _boom)
    svc = _svc()
    with pytest.raises(ReviewReuseError) as ei:
        svc.create_task(
            tenant_id="t-fail",
            file_name="a.dxf",
            file_bytes=b"x",
            seed_candidates=[
                {
                    "candidate_id": "c1",
                    "state": "similar",
                    "scores": {"geometric": 0.9, "semantic": 0.8},
                    "methods": ["precision-l4"],
                }
            ],
        )
    assert ei.value.code == "pipeline_failed"
    assert ei.value.message == "review-reuse pipeline failed"
    listed = svc.list_tasks("t-fail")
    assert len(listed) == 1
    task = listed[0]
    assert task.status == TaskStatus.failed
    assert task.error == "review-reuse pipeline failed"
    assert secret not in (task.error or "")
    dumped = task.model_dump_json()
    assert secret not in dumped
    assert "/secret/path" not in dumped
    assert any(e.event_type == TaskEventType.failed for e in task.events)
    for event in task.events:
        assert secret not in str(event.detail)


def test_low_precision_confidence_ignores_visual() -> None:
    from src.core.review_reuse.models import TaskStatus

    svc = _svc()
    task = svc.create_task(
        tenant_id="t-low-prec",
        file_name="a.dxf",
        file_bytes=b"x",
        seed_candidates=[
            {
                "candidate_id": "c1",
                "state": "duplicate",
                "scores": {"geometric": 0.1, "visual": 0.99, "semantic": 0.99},
                "methods": ["precision-l4"],
            }
        ],
    )
    pack = task.evidence_pack or {}
    reasons = task.candidates[0].rejection_reasons
    assert RejectionReason.low_precision_score.value in reasons
    assert pack["confidence"]["score"] == 0.1
    assert pack["confidence"]["band"] == "low"
    assert task.status == TaskStatus.evidence_ready


def test_verified_l4_confidence_uses_geometric_not_visual() -> None:
    from src.core.review_reuse.models import TaskStatus

    svc = _svc()
    task = svc.create_task(
        tenant_id="t-geo-conf",
        file_name="a.dxf",
        file_bytes=b"x",
        seed_candidates=[
            {
                "candidate_id": "c1",
                "state": "similar",
                "scores": {"geometric": 0.60, "visual": 0.99, "semantic": 0.99},
                "methods": ["precision-l4"],
                "match_level": 4,
            }
        ],
    )
    pack = task.evidence_pack or {}
    assert task.status == TaskStatus.evidence_ready
    assert pack["confidence"]["score"] == 0.60
    assert pack["confidence"]["band"] == "medium"


def test_vision_only_confidence_stays_low() -> None:
    from src.core.review_reuse.models import TaskStatus

    svc = _svc()
    task = svc.create_task(
        tenant_id="t-conf",
        file_name="a.dxf",
        file_bytes=b"x",
        seed_candidates=[
            {
                "candidate_id": "v1",
                "state": "similar",
                "scores": {"semantic": 0.99, "visual": 0.99},
                "methods": ["dedup2d-vision"],
                "decision_source": "dedup2d-vision",
            }
        ],
    )
    pack = task.evidence_pack or {}
    assert pack["confidence"]["band"] == "low"
    assert pack["confidence"]["score"] == 0.0
    assert task.status == TaskStatus.evidence_ready
    assert RejectionReason.vision_only_unverified.value in task.candidates[0].rejection_reasons


def test_insufficient_evidence_does_not_raise_confidence() -> None:
    from src.core.review_reuse.models import TaskStatus

    svc = _svc()
    task = svc.create_task(
        tenant_id="t-insuf",
        file_name="a.dxf",
        file_bytes=b"x",
        seed_candidates=[
            {
                "candidate_id": "x",
                "state": "insufficient_evidence",
                "scores": {"geometric": 0.99},
                "methods": ["precision-l4"],
            }
        ],
    )
    pack = task.evidence_pack or {}
    assert task.candidates[0].state.value == "insufficient_evidence"
    assert pack["confidence"]["score"] == 0.0
    assert pack["confidence"]["band"] == "low"
    assert task.status == TaskStatus.evidence_ready


def test_adapter_preserves_candidate_geom_json() -> None:
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "arch-geom",
                "state": "similar",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    assert cands[0].provenance.get("geom_json") == geom


def test_precision_scores_json_query_and_candidate_geom() -> None:
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "arch-geom",
                "state": "similar",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert out[0].scores.get("geometric") == 1.0
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert int(out[0].verification.get("level") or 0) >= 4
    assert RejectionReason.missing_geom_json.value not in out[0].rejection_reasons


def test_rescore_clears_stale_low_precision_reason() -> None:
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "stale-low",
                "state": "different",
                "geom_json": geom,
                "methods": ["seed-adapter"],
                "rejection_reasons": [RejectionReason.low_precision_score.value],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert out[0].scores.get("geometric") == 1.0
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert RejectionReason.low_precision_score.value not in out[0].rejection_reasons
    assert out[0].state != CandidateState.different
    assert out[0].verification.get("verdict") != CandidateState.different.value


def test_high_l4_preserves_version_gate_different() -> None:
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "gated",
                "state": "different",
                "geom_json": geom,
                "methods": ["seed-adapter"],
                "rejection_reasons": [RejectionReason.version_gate_filtered.value],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert out[0].scores.get("geometric") == 1.0
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert out[0].state == CandidateState.different
    assert out[0].verification.get("verdict") == CandidateState.different.value
    assert RejectionReason.version_gate_filtered.value in out[0].rejection_reasons
    assert RejectionReason.low_precision_score.value not in out[0].rejection_reasons
    from src.core.review_reuse.evidence import build_evidence_pack
    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus

    now = 0.0
    pack = build_evidence_pack(
        ReviewReuseTask(
            task_id="t-gate",
            tenant_id="t",
            status=TaskStatus.evidence_ready,
            created_at=now,
            updated_at=now,
            source_file_name="query.json",
            source_content_sha256="ab",
            trace_id="tr",
            candidates=out,
        )
    )
    assert pack["confidence"]["score"] == 0.0
    assert pack["confidence"]["band"] == "low"


def test_high_prescored_l4_preserves_version_gate_different() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "gated-pre",
                "state": "different",
                "scores": {"geometric": 1.0},
                "methods": ["precision-l4"],
                "rejection_reasons": [
                    RejectionReason.low_precision_score.value,
                    RejectionReason.version_gate_filtered.value,
                ],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"x")
    assert out[0].scores.get("geometric") == 1.0
    assert out[0].state == CandidateState.different
    assert out[0].verification.get("verdict") == CandidateState.different.value
    assert RejectionReason.version_gate_filtered.value in out[0].rejection_reasons
    assert RejectionReason.low_precision_score.value not in out[0].rejection_reasons


def test_high_l4_preserves_reasonless_different() -> None:
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "reasonless",
                "state": "different",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert out[0].scores.get("geometric") == 1.0
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert out[0].state == CandidateState.different
    assert out[0].verification.get("verdict") == CandidateState.different.value
    assert RejectionReason.low_precision_score.value not in out[0].rejection_reasons


def test_high_prescored_l4_preserves_reasonless_different() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "reasonless-pre",
                "state": "different",
                "scores": {"geometric": 1.0},
                "methods": ["precision-l4"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"x")
    assert out[0].scores.get("geometric") == 1.0
    assert out[0].state == CandidateState.different
    assert out[0].verification.get("verdict") == CandidateState.different.value
    assert RejectionReason.low_precision_score.value not in out[0].rejection_reasons


def test_lowercase_line_type_still_scores_l4() -> None:
    geom = {
        "entities": [
            {"type": "line", "start": [0.0, 0.0], "end": [100.0, 0.0]}
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "lc",
                "state": "similar",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert out[0].scores.get("geometric") == 1.0
    assert "precision-l4" in (out[0].verification.get("methods") or [])


def test_boolean_circle_coords_are_not_l4_geometry() -> None:
    bogus = {
        "entities": [
            {"type": "CIRCLE", "center": [True, False], "radius": True}
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "bool-circ",
                "state": "duplicate",
                "geom_json": bogus,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(bogus).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons
    assert out[0].scores.get("geometric") is None


def test_insert_block_geom_is_l4_geometry() -> None:
    geom = {
        "entities": [
            {
                "type": "INSERT",
                "block": "DOOR",
                "insert": [10.0, 20.0],
                "block_hash": "blk-hash-1",
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "blk",
                "state": "similar",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value not in out[0].rejection_reasons
    assert out[0].scores.get("geometric") is not None


def test_zero_ratio_ellipse_is_not_l4_geometry() -> None:
    bogus = {
        "entities": [
            {
                "type": "ELLIPSE",
                "center": [0.0, 0.0],
                "major": [10.0, 0.0],
                "ratio": 0,
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "ell",
                "state": "duplicate",
                "geom_json": bogus,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(bogus).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_nonfinite_prescored_l4_is_not_trusted() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "nan-l4",
                "state": "duplicate",
                "scores": {"geometric": float("nan")},
                "methods": ["precision-l4"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"x")
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_out_of_range_prescored_l4_is_not_trusted() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "hi-l4",
                "state": "duplicate",
                "scores": {"geometric": 2.0},
                "methods": ["precision-l4"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"x")
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons
    assert out[0].scores.get("geometric") is None


def test_zero_scale_insert_is_not_l4_geometry() -> None:
    bogus = {
        "entities": [
            {
                "type": "INSERT",
                "block": "DOOR",
                "insert": [0.0, 0.0],
                "block_hash": "h",
                "scale": [0.0, 0.0],
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "zscale",
                "state": "similar",
                "geom_json": bogus,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(bogus).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_omitted_ellipse_params_match_full_span() -> None:
    omitted = {
        "entities": [
            {
                "type": "ELLIPSE",
                "center": [0.0, 0.0],
                "major": [1.0, 0.0],
                "ratio": 0.5,
            }
        ]
    }
    full = {
        "entities": [
            {
                "type": "ELLIPSE",
                "center": [0.0, 0.0],
                "major": [1.0, 0.0],
                "ratio": 0.5,
                "start_param": 0.0,
                "end_param": 2.0 * math.pi,
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "ell-full",
                "state": "similar",
                "geom_json": full,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(omitted).encode("utf-8"),
    )
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") == 1.0
    assert RejectionReason.low_precision_score.value not in out[0].rejection_reasons


def test_nonfinite_insert_rotation_is_not_l4_geometry() -> None:
    bogus = {
        "entities": [
            {
                "type": "INSERT",
                "block": "DOOR",
                "insert": [10.0, 20.0],
                "rotation": float("nan"),
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "nanrot",
                "state": "similar",
                "geom_json": bogus,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(bogus).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_zero_sweep_ellipse_is_not_l4_geometry() -> None:
    bogus = {
        "entities": [
            {
                "type": "ELLIPSE",
                "center": [0.0, 0.0],
                "major": [1.0, 0.0],
                "ratio": 0.5,
                "start_param": 0.0,
                "end_param": 0.0,
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "zell",
                "state": "similar",
                "geom_json": bogus,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(bogus).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_zero_sweep_arc_is_not_l4_geometry() -> None:
    bogus = {
        "entities": [
            {
                "type": "ARC",
                "center": [0.0, 0.0],
                "radius": 1.0,
                "start_angle": 10.0,
                "end_angle": 10.0,
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "zarc",
                "state": "similar",
                "geom_json": bogus,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(bogus).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_zero_length_polyline_is_not_l4_geometry() -> None:
    bogus = {
        "entities": [{"type": "POLYLINE", "points": [[0.0, 0.0], [0.0, 0.0]]}]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "zero-poly",
                "state": "duplicate",
                "geom_json": bogus,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(bogus).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons
    assert out[0].scores.get("geometric") is None


def test_negative_radius_circle_is_not_l4_geometry() -> None:
    bogus = {"entities": [{"type": "CIRCLE", "center": [0.0, 0.0], "radius": -1}]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "neg-r",
                "state": "duplicate",
                "geom_json": bogus,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(bogus).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons
    assert out[0].scores.get("geometric") is None


def test_mixed_valid_line_and_zero_radius_circle_is_not_l4() -> None:
    """A valid LINE must not admit a degenerate CIRCLE into L4 scoring."""
    mixed = {
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [100.0, 0.0]},
            {"type": "CIRCLE", "center": [0.0, 0.0], "radius": 0.0},
        ]
    }
    other = {
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [0.0, 100.0]},
            {"type": "CIRCLE", "center": [0.0, 0.0], "radius": 0.0},
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "mixed-junk",
                "state": "similar",
                "geom_json": other,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(mixed).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons
    assert out[0].scores.get("geometric") is None


def test_valid_line_with_unknown_type_still_l4() -> None:
    geom = {
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [100.0, 0.0]},
            {"type": "TEXT", "insert": [1.0, 1.0], "text": "note"},
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "line-text",
                "state": "similar",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") == 1.0
    assert RejectionReason.missing_geom_json.value not in out[0].rejection_reasons


def test_malformed_entity_dict_is_not_l4_geometry() -> None:
    bogus = {"entities": [{}]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "bogus",
                "state": "similar",
                "geom_json": bogus,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(bogus).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons
    assert out[0].scores.get("geometric") is None


def test_empty_dxf_extract_is_not_l4_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.core.dedupcad_precision.cad_pipeline.extract_geom_json_from_dxf",
        lambda _path: {"schema": "geom-json/v2", "entities": []},
    )
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "arch-geom",
                "state": "similar",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(
        cands,
        file_name="empty.dxf",
        file_bytes=b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nEOF\n",
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons
    assert out[0].scores.get("geometric") is None


def test_empty_json_object_is_not_l4_geometry() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "empty",
                "state": "similar",
                "geom_json": {},
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(cands, file_name="query.json", file_bytes=b"{}")
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons
    assert out[0].scores.get("geometric") is None


def test_store_loaded_empty_geom_is_not_l4(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Store:
        def load(self, _cid: str) -> dict:
            return {}

    monkeypatch.setattr(
        "src.core.dedupcad_precision.create_geom_store", lambda: _Store()
    )
    geom = _line_geom()
    file_hash = "a" * 64
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": file_hash,
                "state": "similar",
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_invalid_inline_geom_falls_back_to_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geom = _line_geom()

    class _Store:
        def load(self, _cid: str) -> dict:
            return geom

    monkeypatch.setattr(
        "src.core.dedupcad_precision.create_geom_store", lambda: _Store()
    )
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "a" * 64,
                "state": "similar",
                "geom_json": {},
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") == 1.0
    assert RejectionReason.missing_geom_json.value not in out[0].rejection_reasons


def test_json_bytes_on_dxf_filename_are_not_query_geom() -> None:
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "arch-geom",
                "state": "similar",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(
        cands,
        file_name="a.dxf",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_precision_skips_dxf_extract_when_no_candidate_geom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(_path: object) -> dict:
        raise AssertionError("dxf extract should not run")

    monkeypatch.setattr(
        "src.core.dedupcad_precision.cad_pipeline.extract_geom_json_from_dxf",
        _boom,
    )
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "s1",
                "state": "similar",
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"0\nSECTION\n")
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons

    insuff = map_raw_hits_to_candidates(
        [{"candidate_id": "none", "state": "insufficient_evidence"}],
        content_sha="ab",
        file_name="a.dxf",
    )
    apply_precision(insuff, file_name="a.dxf", file_bytes=b"0\nSECTION\n")


def test_precision_extracts_dxf_query_geom(monkeypatch: pytest.MonkeyPatch) -> None:
    geom = _line_geom()
    monkeypatch.setattr(
        "src.core.dedupcad_precision.cad_pipeline.extract_geom_json_from_dxf",
        lambda _path: geom,
    )
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "arch-geom",
                "state": "similar",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="part.dxf", file_bytes=b"0\nSECTION\n")
    assert out[0].scores.get("geometric") == 1.0
    assert "precision-l4" in (out[0].verification.get("methods") or [])


def test_precision_loads_candidate_geom_from_store(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.core.dedupcad_precision.store import (
        GeomJsonStore,
        GeomJsonStoreConfig,
        create_geom_store,
    )

    geom = _line_geom()
    sha = "a" * 64
    store_dir = tmp_path / "geom"
    GeomJsonStore(GeomJsonStoreConfig(base_dir=store_dir)).save(sha, geom)
    monkeypatch.setenv("DEDUPCAD_GEOM_STORE_DIR", str(store_dir))
    create_geom_store.cache_clear()
    try:
        cands = map_raw_hits_to_candidates(
            [
                {
                    "candidate_id": sha,
                    "state": "similar",
                    "methods": ["seed-adapter"],
                }
            ],
            content_sha="ab",
            file_name="a.dxf",
        )
        out = apply_precision(
            cands,
            file_name="query.json",
            file_bytes=json.dumps(geom).encode("utf-8"),
        )
        assert out[0].scores.get("geometric") == 1.0
        assert "precision-l4" in (out[0].verification.get("methods") or [])
    finally:
        create_geom_store.cache_clear()


def test_low_precision_reason() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "low",
                "state": "similar",
                "scores": {"geometric": 0.2, "semantic": 0.1},
                "methods": ["precision-l4"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"x")
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
    assert out[0].state == CandidateState.different
    assert out[0].verification.get("verdict") == CandidateState.different.value


def test_unverified_numeric_geom_does_not_raise_confidence() -> None:
    from src.core.review_reuse.models import TaskStatus

    svc = _svc()
    task = svc.create_task(
        tenant_id="t-unverified-geom",
        file_name="a.dxf",
        file_bytes=b"x",
        seed_candidates=[
            {
                "candidate_id": "c1",
                "state": "duplicate",
                "scores": {"geometric": 0.99, "visual": 0.99, "semantic": 0.99},
                "methods": ["seed-adapter"],
            }
        ],
    )
    pack = task.evidence_pack or {}
    cand = task.candidates[0]
    assert RejectionReason.missing_geom_json.value in cand.rejection_reasons
    assert pack["confidence"]["score"] == 0.0
    assert pack["confidence"]["band"] == "low"
    assert task.status == TaskStatus.evidence_ready


def test_stale_l4_method_stripped_on_vision_only() -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "v1",
                "state": "duplicate",
                "scores": {"semantic": 0.99, "visual": 0.99},
                "methods": ["dedup2d-vision", "precision-l4"],
                "match_level": 4,
                "decision_source": "dedup2d-vision",
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(cands, file_name="a.dxf", file_bytes=b"x")
    assert RejectionReason.vision_only_unverified.value in out[0].rejection_reasons
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert int(out[0].verification.get("level") or 0) < 4


def test_local_l4_reject_downgrades_duplicate_verdict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geom = _line_geom()

    class _Low:
        score = 0.1

    monkeypatch.setattr(
        "src.core.dedupcad_precision.PrecisionVerifier.score_pair",
        lambda self, _left, _right, **_k: _Low(),
    )
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "dup",
                "state": "duplicate",
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="a.dxf",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert out[0].scores.get("geometric") == 0.1
    assert out[0].state == CandidateState.different
    assert out[0].verification.get("verdict") == "different"
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
    assert int(out[0].verification.get("level") or 0) >= 4
