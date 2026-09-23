"""Precision pass + filename gate tests (honest geometric / rejection reasons)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

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
from src.core.review_reuse.precision import (
    LOW_PRECISION_THRESHOLD,
    apply_precision,
    set_precision_hook,
)
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


def test_idempotent_replay_rejects_mismatched_input() -> None:
    svc = _svc()
    first = svc.create_task(
        tenant_id="t-idem-mis",
        file_name="a.dxf",
        file_bytes=b"orig-bytes",
        idempotency_key="same-key",
        seed_candidates=[
            {
                "candidate_id": "c1",
                "state": "similar",
                "scores": {"geometric": 0.9, "semantic": 0.8},
                "methods": ["precision-l4"],
            }
        ],
    )
    with pytest.raises(ReviewReuseError) as exc:
        svc.create_task(
            tenant_id="t-idem-mis",
            file_name="a.dxf",
            file_bytes=b"other-bytes",
            idempotency_key="same-key",
        )
    assert exc.value.code == "idempotency_conflict"
    stored = svc.get_task("t-idem-mis", first.task_id)
    assert stored.source_content_sha256 == hashlib.sha256(b"orig-bytes").hexdigest()


def test_fresh_running_idempotency_rejects_mismatched_input() -> None:
    import time

    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus

    svc = _svc()
    now = time.time()
    prior = ReviewReuseTask(
        task_id="fresh-run",
        tenant_id="t-fresh-mis",
        status=TaskStatus.running,
        created_at=now,
        updated_at=now,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(b"orig").hexdigest(),
        idempotency_key="idem-fresh-mis",
        trace_id="tr-fresh-mis",
    )
    svc.store.put(prior)
    with pytest.raises(ReviewReuseError) as exc:
        svc.create_task(
            tenant_id="t-fresh-mis",
            file_name="part.dxf",
            file_bytes=b"other",
            idempotency_key="idem-fresh-mis",
        )
    assert exc.value.code == "idempotency_conflict"
    stuck = svc.get_task("t-fresh-mis", "fresh-run")
    assert stuck.status == TaskStatus.running


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
    assert TaskEventType.precision_completed not in types
    assert TaskEventType.evidence_pack_ready not in types
    assert task.evidence_pack is None


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


def test_idempotency_replay_does_not_skip_file_gate() -> None:
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
        source_content_sha256=hashlib.sha256(b"x").hexdigest(),
        idempotency_key="idem-legacy",
        trace_id="tr-legacy",
    )
    svc.store.put(prior)
    with pytest.raises(ReviewReuseError) as ei:
        svc.create_task(
            tenant_id="t-a",
            file_name="legacy.bin",
            file_bytes=b"x",
            idempotency_key="idem-legacy",
        )
    assert ei.value.code == RejectionReason.unsupported_file_type.value
    stuck = svc.get_task("t-a", "pre-gate")
    assert stuck.status == TaskStatus.evidence_ready


def test_stale_running_unsupported_filename_does_not_resume() -> None:
    import time

    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus
    from src.core.review_reuse.service import STALE_RUNNING_SECONDS

    svc = _svc()
    now = time.time()
    prior = ReviewReuseTask(
        task_id="pre-gate-run",
        tenant_id="t-stale-bin",
        status=TaskStatus.running,
        created_at=now - STALE_RUNNING_SECONDS - 10.0,
        updated_at=now - STALE_RUNNING_SECONDS - 10.0,
        source_file_name="legacy.bin",
        source_content_sha256=hashlib.sha256(b"x").hexdigest(),
        idempotency_key="idem-stale-bin",
        trace_id="tr-stale-bin",
    )
    svc.store.put(prior)
    with pytest.raises(ReviewReuseError) as ei:
        svc.create_task(
            tenant_id="t-stale-bin",
            file_name="legacy.bin",
            file_bytes=b"x",
            idempotency_key="idem-stale-bin",
        )
    assert ei.value.code == RejectionReason.unsupported_file_type.value
    stuck = svc.get_task("t-stale-bin", "pre-gate-run")
    assert stuck.status == TaskStatus.running


def test_idempotent_replay_of_failed_task_raises_pipeline_failed() -> None:
    import time

    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus
    from src.core.review_reuse.service import PIPELINE_FAILED_PUBLIC

    svc = _svc()
    now = time.time()
    prior = ReviewReuseTask(
        task_id="failed-run",
        tenant_id="t-fail",
        status=TaskStatus.failed,
        created_at=now,
        updated_at=now,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(b"x").hexdigest(),
        idempotency_key="idem-failed",
        trace_id="tr-failed",
        error=PIPELINE_FAILED_PUBLIC,
    )
    svc.store.put(prior)
    with pytest.raises(ReviewReuseError) as exc:
        svc.create_task(
            tenant_id="t-fail",
            file_name="part.dxf",
            file_bytes=b"x",
            idempotency_key="idem-failed",
        )
    assert exc.value.code == "pipeline_failed"
    assert exc.value.message == PIPELINE_FAILED_PUBLIC
    stuck = svc.get_task("t-fail", "failed-run")
    assert stuck.status == TaskStatus.failed


def test_cancel_does_not_overwrite_failed_task() -> None:
    import time

    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus
    from src.core.review_reuse.service import PIPELINE_FAILED_PUBLIC

    svc = _svc()
    now = time.time()
    prior = ReviewReuseTask(
        task_id="failed-cancel",
        tenant_id="t-fail-cancel",
        status=TaskStatus.failed,
        created_at=now,
        updated_at=now,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(b"x").hexdigest(),
        idempotency_key="idem-failed-cancel",
        trace_id="tr-failed-cancel",
        error=PIPELINE_FAILED_PUBLIC,
    )
    svc.store.put(prior)
    after = svc.cancel("t-fail-cancel", "failed-cancel")
    assert after.status == TaskStatus.failed
    assert after.error == PIPELINE_FAILED_PUBLIC
    with pytest.raises(ReviewReuseError) as exc:
        svc.create_task(
            tenant_id="t-fail-cancel",
            file_name="part.dxf",
            file_bytes=b"x",
            idempotency_key="idem-failed-cancel",
        )
    assert exc.value.code == "pipeline_failed"
    stuck = svc.get_task("t-fail-cancel", "failed-cancel")
    assert stuck.status == TaskStatus.failed


def test_submit_decision_does_not_overwrite_failed_task(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus
    from src.core.review_reuse.service import PIPELINE_FAILED_PUBLIC

    monkeypatch.setenv(ENV_DECISIONS_ENABLED, "true")
    svc = _svc()
    now = time.time()
    prior = ReviewReuseTask(
        task_id="failed-decide",
        tenant_id="t-fail-decide",
        status=TaskStatus.failed,
        created_at=now,
        updated_at=now,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(b"x").hexdigest(),
        idempotency_key="idem-failed-decide",
        trace_id="tr-failed-decide",
        error=PIPELINE_FAILED_PUBLIC,
    )
    svc.store.put(prior)
    with pytest.raises(ReviewReuseError) as ei:
        svc.submit_decision(
            tenant_id="t-fail-decide",
            task_id="failed-decide",
            state=HumanDecisionState.new,
            reviewer_id="r1",
        )
    assert ei.value.code == "failed"
    stuck = svc.get_task("t-fail-decide", "failed-decide")
    assert stuck.status == TaskStatus.failed
    assert stuck.human_decision is None
    with pytest.raises(ReviewReuseError) as replay:
        svc.create_task(
            tenant_id="t-fail-decide",
            file_name="part.dxf",
            file_bytes=b"x",
            idempotency_key="idem-failed-decide",
        )
    assert replay.value.code == "pipeline_failed"


def test_idempotent_replay_of_decided_task_with_pipeline_error_raises() -> None:
    """A decided snapshot that still carries pipeline_failed must not 200."""
    import time

    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus
    from src.core.review_reuse.service import PIPELINE_FAILED_PUBLIC

    svc = _svc()
    now = time.time()
    prior = ReviewReuseTask(
        task_id="decided-fail",
        tenant_id="t-dec-fail-replay",
        status=TaskStatus.decided,
        created_at=now,
        updated_at=now,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(b"x").hexdigest(),
        idempotency_key="idem-decided-fail",
        trace_id="tr-decided-fail",
        error=PIPELINE_FAILED_PUBLIC,
    )
    svc.store.put(prior)
    with pytest.raises(ReviewReuseError) as exc:
        svc.create_task(
            tenant_id="t-dec-fail-replay",
            file_name="part.dxf",
            file_bytes=b"x",
            idempotency_key="idem-decided-fail",
        )
    assert exc.value.code == "pipeline_failed"
    assert exc.value.message == PIPELINE_FAILED_PUBLIC
    stuck = svc.get_task("t-dec-fail-replay", "decided-fail")
    assert stuck.status == TaskStatus.decided
    assert stuck.error == PIPELINE_FAILED_PUBLIC


def test_idempotent_replay_of_canceled_task_with_pipeline_error_raises() -> None:
    """A canceled snapshot that still carries pipeline_failed must not 200."""
    import time

    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus
    from src.core.review_reuse.service import PIPELINE_FAILED_PUBLIC

    svc = _svc()
    now = time.time()
    prior = ReviewReuseTask(
        task_id="canceled-fail",
        tenant_id="t-can-fail-replay",
        status=TaskStatus.canceled,
        created_at=now,
        updated_at=now,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(b"x").hexdigest(),
        idempotency_key="idem-canceled-fail",
        trace_id="tr-canceled-fail",
        error=PIPELINE_FAILED_PUBLIC,
    )
    svc.store.put(prior)
    with pytest.raises(ReviewReuseError) as exc:
        svc.create_task(
            tenant_id="t-can-fail-replay",
            file_name="part.dxf",
            file_bytes=b"x",
            idempotency_key="idem-canceled-fail",
        )
    assert exc.value.code == "pipeline_failed"
    assert exc.value.message == PIPELINE_FAILED_PUBLIC
    stuck = svc.get_task("t-can-fail-replay", "canceled-fail")
    assert stuck.status == TaskStatus.canceled
    assert stuck.error == PIPELINE_FAILED_PUBLIC


def test_stale_midflight_decision_resumes_until_evidence_ready() -> None:
    """A decided snapshot with no evidence_pack_ready must still resume."""
    import time

    from src.core.review_reuse.models import HumanDecision, ReviewReuseTask, TaskStatus
    from src.core.review_reuse.service import STALE_RUNNING_SECONDS

    svc = _svc()
    now = time.time()
    payload = b"midflight"
    prior = ReviewReuseTask(
        task_id="stuck-decided",
        tenant_id="t-midflight",
        status=TaskStatus.decided,
        created_at=now - STALE_RUNNING_SECONDS - 10.0,
        updated_at=now - STALE_RUNNING_SECONDS - 10.0,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(payload).hexdigest(),
        idempotency_key="idem-midflight",
        trace_id="tr-midflight",
        human_decision=HumanDecision(
            state=HumanDecisionState.reuse,
            reviewer_id="reviewer-1",
            ts=now - STALE_RUNNING_SECONDS - 10.0,
        ),
        evidence_pack={"schema_version": "empty-midflight"},
    )
    svc.store.put(prior)
    again = svc.create_task(
        tenant_id="t-midflight",
        file_name="part.dxf",
        file_bytes=payload,
        idempotency_key="idem-midflight",
    )
    assert again.task_id == "stuck-decided"
    assert again.status == TaskStatus.decided
    assert again.human_decision is not None
    assert again.human_decision.reviewer_id == "reviewer-1"
    assert again.evidence_pack is not None
    assert any(
        event.event_type == TaskEventType.evidence_pack_ready for event in again.events
    )


def test_fresh_midflight_decision_is_not_resumed() -> None:
    import time

    from src.core.review_reuse.models import HumanDecision, ReviewReuseTask, TaskStatus
    from src.core.review_reuse.service import STALE_RUNNING_SECONDS

    svc = _svc()
    now = time.time()
    payload = b"fresh-decision"
    prior = ReviewReuseTask(
        task_id="fresh-decided",
        tenant_id="t-fresh-decided",
        status=TaskStatus.decided,
        created_at=now,
        updated_at=now,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(payload).hexdigest(),
        idempotency_key="idem-fresh-decided",
        trace_id="tr-fresh-decided",
        human_decision=HumanDecision(
            state=HumanDecisionState.reuse,
            reviewer_id="reviewer-1",
            ts=now,
        ),
    )
    svc.store.put(prior)
    again = svc.create_task(
        tenant_id="t-fresh-decided",
        file_name="part.dxf",
        file_bytes=payload,
        idempotency_key="idem-fresh-decided",
    )
    assert again.status == TaskStatus.decided
    assert again.evidence_pack is None
    assert not any(
        event.event_type == TaskEventType.evidence_pack_ready for event in again.events
    )
    assert STALE_RUNNING_SECONDS > 0


def test_decided_with_evidence_ready_is_not_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import time

    from src.core.review_reuse.models import (
        HumanDecision,
        ReviewReuseTask,
        TaskEvent,
        TaskStatus,
    )
    from src.core.review_reuse.service import STALE_RUNNING_SECONDS
    import src.core.review_reuse.service as service_mod

    svc = _svc()
    now = time.time()
    payload = b"already-ready"
    prior = ReviewReuseTask(
        task_id="done-decided",
        tenant_id="t-done-decided",
        status=TaskStatus.decided,
        created_at=now - STALE_RUNNING_SECONDS - 10.0,
        updated_at=now - STALE_RUNNING_SECONDS - 10.0,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(payload).hexdigest(),
        idempotency_key="idem-done-decided",
        trace_id="tr-done-decided",
        human_decision=HumanDecision(
            state=HumanDecisionState.new,
            reviewer_id="reviewer-1",
            ts=now - STALE_RUNNING_SECONDS - 10.0,
        ),
        evidence_pack={"schema_version": "kept"},
        events=[
            TaskEvent(
                event_type=TaskEventType.evidence_pack_ready,
                ts=now - STALE_RUNNING_SECONDS - 20.0,
                detail={},
            )
        ],
    )
    svc.store.put(prior)
    calls = {"n": 0}
    orig = service_mod.recall_candidates

    def _count(*args: object, **kwargs: object):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(service_mod, "recall_candidates", _count)
    again = svc.create_task(
        tenant_id="t-done-decided",
        file_name="part.dxf",
        file_bytes=payload,
        idempotency_key="idem-done-decided",
    )
    assert calls["n"] == 0
    assert again.status == TaskStatus.decided
    assert again.evidence_pack == {"schema_version": "kept"}


def test_hash_candidate_skips_geom_store_without_query_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.core.dedupcad_precision as dedup_mod

    def _boom() -> None:
        raise AssertionError("geom store must not load without query geometry")

    monkeypatch.setattr(dedup_mod, "create_geom_store", _boom)
    digest = "ab" * 32
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": digest,
                "state": "similar",
                "scores": {"geometric": 0.99},
                "methods": ["precision-l4"],
            }
        ],
        content_sha="cd",
        file_name="query.png",
    )
    out = apply_precision(
        cands, file_name="query.png", file_bytes=b"\x89PNG\r\n"
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_stale_running_idempotency_resumes_pipeline() -> None:
    import hashlib
    import time

    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus
    from src.core.review_reuse.service import STALE_RUNNING_SECONDS

    svc = _svc()
    now = time.time()
    payload = b"x"
    prior = ReviewReuseTask(
        task_id="stuck-run",
        tenant_id="t-stale",
        status=TaskStatus.running,
        created_at=now - STALE_RUNNING_SECONDS - 10.0,
        updated_at=now - STALE_RUNNING_SECONDS - 10.0,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(payload).hexdigest(),
        idempotency_key="idem-stale",
        trace_id="tr-stale",
    )
    svc.store.put(prior)
    again = svc.create_task(
        tenant_id="t-stale",
        file_name="part.dxf",
        file_bytes=payload,
        idempotency_key="idem-stale",
    )
    assert again.task_id == "stuck-run"
    assert again.status == TaskStatus.evidence_ready
    assert again.evidence_pack is not None


def test_stale_running_idempotency_rejects_mismatched_input() -> None:
    import hashlib
    import time

    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus
    from src.core.review_reuse.service import STALE_RUNNING_SECONDS

    svc = _svc()
    now = time.time()
    prior = ReviewReuseTask(
        task_id="stuck-mismatch",
        tenant_id="t-stale-mis",
        status=TaskStatus.running,
        created_at=now - STALE_RUNNING_SECONDS - 10.0,
        updated_at=now - STALE_RUNNING_SECONDS - 10.0,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(b"orig").hexdigest(),
        idempotency_key="idem-stale-mis",
        trace_id="tr-stale-mis",
    )
    svc.store.put(prior)
    with pytest.raises(ReviewReuseError) as exc:
        svc.create_task(
            tenant_id="t-stale-mis",
            file_name="other.dxf",
            file_bytes=b"other",
            idempotency_key="idem-stale-mis",
        )
    assert exc.value.code == "idempotency_conflict"
    stuck = svc.get_task("t-stale-mis", "stuck-mismatch")
    assert stuck.status == TaskStatus.running
    assert stuck.evidence_pack is None


def test_stale_running_idempotency_claim_is_single_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hashlib
    import threading
    import time

    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus
    from src.core.review_reuse.service import STALE_RUNNING_SECONDS
    import src.core.review_reuse.service as service_mod

    svc = _svc()
    now = time.time()
    payload = b"same-bytes"
    prior = ReviewReuseTask(
        task_id="stuck-race",
        tenant_id="t-stale-race",
        status=TaskStatus.running,
        created_at=now - STALE_RUNNING_SECONDS - 10.0,
        updated_at=now - STALE_RUNNING_SECONDS - 10.0,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(payload).hexdigest(),
        idempotency_key="idem-stale-race",
        trace_id="tr-stale-race",
    )
    svc.store.put(prior)
    calls = {"n": 0}
    orig = service_mod.recall_candidates

    def _count(*args: object, **kwargs: object):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(service_mod, "recall_candidates", _count)
    out: list = []

    def _run() -> None:
        out.append(
            svc.create_task(
                tenant_id="t-stale-race",
                file_name="part.dxf",
                file_bytes=payload,
                idempotency_key="idem-stale-race",
            )
        )

    threads = [threading.Thread(target=_run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(out) == 2
    assert {task.task_id for task in out} == {"stuck-race"}
    assert calls["n"] == 1
    final = svc.get_task("t-stale-race", "stuck-race")
    assert final.status == TaskStatus.evidence_ready
    assert final.evidence_pack is not None


def test_stale_claim_blocks_previous_owner_commit() -> None:
    import hashlib
    import time

    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus
    from src.core.review_reuse.service import STALE_RUNNING_SECONDS

    svc = _svc()
    now = time.time()
    payload = b"claim-fence"
    original = ReviewReuseTask(
        task_id="stuck-fence",
        tenant_id="t-claim-fence",
        status=TaskStatus.running,
        created_at=now - STALE_RUNNING_SECONDS - 10.0,
        updated_at=now - STALE_RUNNING_SECONDS - 10.0,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(payload).hexdigest(),
        idempotency_key="idem-claim-fence",
        trace_id="tr-claim-fence",
        pipeline_claim_id="claim-original",
    )
    svc.store.put(original)
    claimed, won = svc._claim_stale_running(original)
    assert won
    assert claimed.pipeline_claim_id != "claim-original"
    original.status = TaskStatus.evidence_ready
    original.evidence_pack = {"schema_version": "test"}
    kept = svc._commit_pipeline_result(original)
    assert kept.pipeline_claim_id == claimed.pipeline_claim_id
    assert kept.status == TaskStatus.running
    assert kept.evidence_pack is None


def test_fresh_running_idempotency_is_not_resumed() -> None:
    import time

    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus

    svc = _svc()
    now = time.time()
    prior = ReviewReuseTask(
        task_id="live-run",
        tenant_id="t-fresh",
        status=TaskStatus.running,
        created_at=now,
        updated_at=now,
        source_file_name="part.dxf",
        source_content_sha256=hashlib.sha256(b"x").hexdigest(),
        idempotency_key="idem-fresh",
        trace_id="tr-fresh",
    )
    svc.store.put(prior)
    again = svc.create_task(
        tenant_id="t-fresh",
        file_name="part.dxf",
        file_bytes=b"x",
        idempotency_key="idem-fresh",
    )
    assert again.task_id == "live-run"
    assert again.status == TaskStatus.running
    assert again.evidence_pack is None


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
        "file_info": {"insunits": 4},
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


def test_pipeline_failed_after_mid_flight_decision_keeps_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent decision must not hide a later pipeline failure."""
    from src.core.review_reuse.models import TaskEventType, TaskStatus

    monkeypatch.setenv(ENV_DECISIONS_ENABLED, "true")
    svc = _svc()

    def _decide_then_boom(*_a, **_k):
        running = svc.list_tasks("t-dec-fail")
        assert running
        decided = svc.submit_decision(
            tenant_id="t-dec-fail",
            task_id=running[0].task_id,
            state=HumanDecisionState.new,
            reviewer_id="reviewer-1",
        )
        assert decided.status == TaskStatus.decided
        raise RuntimeError("precision exploded")

    monkeypatch.setattr(
        "src.core.review_reuse.service.apply_precision", _decide_then_boom
    )
    with pytest.raises(ReviewReuseError) as ei:
        svc.create_task(
            tenant_id="t-dec-fail",
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
    stored = svc.list_tasks("t-dec-fail")[0]
    assert stored.status == TaskStatus.decided
    assert stored.error == "review-reuse pipeline failed"
    assert any(e.event_type == TaskEventType.failed for e in stored.events)
    assert any(e.event_type == TaskEventType.decision_submitted for e in stored.events)


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


def test_similar_version_gate_does_not_raise_confidence() -> None:
    """Independent rejection must suppress confidence even if state is similar."""
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "gated-similar",
                "state": "similar",
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
    assert RejectionReason.version_gate_filtered.value in out[0].rejection_reasons
    from src.core.review_reuse.evidence import build_evidence_pack
    from src.core.review_reuse.models import ReviewReuseTask, TaskStatus

    now = 0.0
    pack = build_evidence_pack(
        ReviewReuseTask(
            task_id="t-gate-sim",
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
        "file_info": {"insunits": 4},
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
        "file_info": {"insunits": 4},
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
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons
    assert out[0].scores.get("geometric") is None


def test_ratio_just_above_one_ellipse_is_not_l4_geometry() -> None:
    """1.0004 rounds to 1.0 at L4 quant; the raw ratio must still fail closed."""
    bogus = {
        "entities": [
            {
                "type": "ELLIPSE",
                "center": [0.0, 0.0],
                "major": [10.0, 0.0],
                "ratio": 1.0004,
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "ell-round",
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


def test_ratio_above_one_ellipse_is_not_l4_geometry() -> None:
    bogus = {
        "entities": [
            {
                "type": "ELLIPSE",
                "center": [0.0, 0.0],
                "major": [10.0, 0.0],
                "ratio": 1.5,
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "ell-hi",
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


def test_nonfinite_local_l4_score_is_not_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Scored:
        score = float("nan")

    class _Verifier:
        def score_pair(self, *_args: object, **_kwargs: object) -> _Scored:
            return _Scored()

    monkeypatch.setattr(
        "src.core.dedupcad_precision.PrecisionVerifier", _Verifier
    )
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "nan-local",
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
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_out_of_range_local_l4_score_is_not_trusted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Scored:
        score = 2.0

    class _Verifier:
        def score_pair(self, *_args: object, **_kwargs: object) -> _Scored:
            return _Scored()

    monkeypatch.setattr(
        "src.core.dedupcad_precision.PrecisionVerifier", _Verifier
    )
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "hi-local",
                "state": "duplicate",
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
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
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


def test_line_plus_insert_is_not_certified_as_l4() -> None:
    """Dropping INSERT would let an incidental LINE fake an L4 match."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
            {
                "type": "INSERT",
                "block": "DOOR",
                "insert": [1.0, 1.0],
                "block_hash": "hash-a",
            },
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
            {
                "type": "INSERT",
                "block": "WINDOW",
                "insert": [50.0, 50.0],
                "block_hash": "hash-b",
            },
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "line-insert",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


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
        "file_info": {"insunits": 4},
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
        "file_info": {"insunits": 4},
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
        "file_info": {"insunits": 4},
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
        "file_info": {"insunits": 4},
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


def test_line_plus_point_is_not_certified_as_l4() -> None:
    """Unsupported POINT must not be dropped so an incidental LINE can L4-match."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
            {"type": "POINT", "location": [1.0, 1.0]},
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
            {"type": "POINT", "location": [99.0, 99.0]},
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "line-point",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_layer_mismatch_does_not_reject_identical_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Per-entity layer_mismatch_penalty must not change geometry-only L4."""
    monkeypatch.setenv("DEDUPCAD2_LAYER_MISMATCH_PENALTY", "1")
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LINE",
                "layer": "A",
                "start": [0.0, 0.0],
                "end": [10.0, 0.0],
            }
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LINE",
                "layer": "B",
                "start": [0.0, 0.0],
                "end": [10.0, 0.0],
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "layer-mismatch",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") == 1.0
    assert RejectionReason.low_precision_score.value not in out[0].rejection_reasons
    assert out[0].state == CandidateState.similar


def test_shared_text_does_not_certify_different_geometry() -> None:
    """Orthogonal LINEs plus identical TEXT must not pass the L4 threshold."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [100.0, 0.0]},
            {"type": "TEXT", "insert": [1.0, 1.0], "text": "note"},
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [0.0, 100.0]},
            {"type": "TEXT", "insert": [1.0, 1.0], "text": "note"},
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "text-inflate",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo < LOW_PRECISION_THRESHOLD
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
    assert out[0].state == CandidateState.different


def test_shared_hatch_does_not_certify_different_geometry() -> None:
    """HATCH is unscored geometry; mixed LINE+HATCH must not be L4."""
    hatch = {"type": "HATCH", "pattern": "ANSI31", "color": 1, "loops": 1}
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [100.0, 0.0]},
            hatch,
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [0.0, 100.0]},
            hatch,
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "hatch-inflate",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_layout_shifted_same_primitives_are_not_certified() -> None:
    """Bag-of-features fallback must not treat translated clones as L4 matches."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
            {"type": "LINE", "start": [0.0, 10.0], "end": [10.0, 10.0]},
        ]
    }
    shifted = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
            {"type": "LINE", "start": [100.0, 100.0], "end": [110.0, 100.0]},
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "shifted",
                "state": "similar",
                "geom_json": shifted,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo < LOW_PRECISION_THRESHOLD
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
    assert out[0].state == CandidateState.different


def test_identical_polyline_still_l4() -> None:
    geom = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                "closed": True,
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "same-poly",
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
    assert out[0].state == CandidateState.similar


def test_string_false_polyline_closed_is_not_certified() -> None:
    """``closed: "false"`` must not explode a closer and L4-match a closed poly."""
    points = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]]
    query = {"entities": [{"type": "LWPOLYLINE", "points": points, "closed": "false"}]}
    other = {"entities": [{"type": "LWPOLYLINE", "points": points, "closed": True}]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "str-closed",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None


def test_layout_shifted_polylines_are_not_certified() -> None:
    """Canonical polyline matching must not treat translated clones as L4."""
    for poly_type in ("LWPOLYLINE", "POLYLINE"):
        query = {
            "file_info": {"insunits": 4},
            "entities": [
                {
                    "type": poly_type,
                    "points": [
                        [0.0, 0.0],
                        [10.0, 0.0],
                        [10.0, 10.0],
                        [0.0, 10.0],
                    ],
                    "closed": True,
                }
            ]
        }
        shifted = {
            "file_info": {"insunits": 4},
            "entities": [
                {
                    "type": poly_type,
                    "points": [
                        [100.0, 100.0],
                        [110.0, 100.0],
                        [110.0, 110.0],
                        [100.0, 110.0],
                    ],
                    "closed": True,
                }
            ]
        }
        cands = map_raw_hits_to_candidates(
            [
                {
                    "candidate_id": f"shifted-{poly_type.lower()}",
                    "state": "similar",
                    "geom_json": shifted,
                    "methods": ["seed-adapter"],
                }
            ],
            content_sha="ab",
            file_name="query.json",
        )
        out = apply_precision(
            cands,
            file_name="query.json",
            file_bytes=json.dumps(query).encode("utf-8"),
        )
        geo = out[0].scores.get("geometric")
        assert "precision-l4" in (out[0].verification.get("methods") or [])
        assert geo is not None
        assert geo < LOW_PRECISION_THRESHOLD
        assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
        assert out[0].state == CandidateState.different


def test_long_polylines_diverging_after_matcher_cap_are_not_certified() -> None:
    """Exploded segments past max_match_entities=64 must still affect L4."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[float(i), 0.0] for i in range(101)],
            }
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[float(i), 0.0] for i in range(10)]
                + [[float(i), 1000.0] for i in range(10, 101)],
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "long-poly",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo < LOW_PRECISION_THRESHOLD
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
    assert out[0].state == CandidateState.different


def test_bulged_polyline_is_not_certified_as_straight_l4() -> None:
    """A chord LINE must not match a bulged LWPOLYLINE with the same ends."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0, 0.0, 0.0, 1.0], [10.0, 0.0]],
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "bulge",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_overflow_integer_polyline_bulge_is_not_pipeline_failed() -> None:
    """JSON ints too large for float must not crash bulge explode."""
    huge = 10**400
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ],
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0]],
                "bulges": [huge],
            }
        ],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "overflow-bulge",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_overflow_integer_polyline_width_is_not_pipeline_failed() -> None:
    """JSON ints too large for float must not crash width explode."""
    huge = 10**400
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ],
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0]],
                "const_width": huge,
            }
        ],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "overflow-width",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_overflow_integer_coords_are_not_l4_geometry() -> None:
    """JSON ints too large for float must be missing_geom_json, not 500."""
    huge = 10**400
    geom = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [huge, 0], "end": [10, 0]},
        ],
    }
    payload = json.dumps(geom).encode("utf-8")
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "overflow-int",
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
        file_bytes=payload,
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_deeply_nested_json_geom_is_not_pipeline_failed() -> None:
    """RecursionError from json.loads must be missing_geom_json, not 500."""
    payload = ("[" * 2000 + "0" + "]" * 2000).encode("utf-8")
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "nested-json",
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
        file_bytes=payload,
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_overflow_integer_adapter_score_is_not_a_unit_score() -> None:
    """JSON ints too large for float must not crash score mapping."""
    huge = 10**400
    geom = _line_geom()
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "overflow-score",
                "state": "similar",
                "geometric": huge,
                "semantic": huge,
                "visual": huge,
                "geom_json": geom,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    assert cands[0].scores.get("geometric") is None
    assert cands[0].scores.get("semantic") is None
    assert cands[0].scores.get("visual") is None
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert out[0].scores.get("geometric") is not None


def test_non_list_polyline_bulges_is_not_certified_as_straight_l4() -> None:
    """Present non-list bulges must not explode into a chord LINE."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ],
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0]],
                "bulges": {"0": 1.0},
            }
        ],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "bulge-dict",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_sub_quantum_bulge_is_not_certified_as_straight_l4() -> None:
    """Bulge 0.0004 must not round to 0 and explode into a chord."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0, 0.0, 0.0, 0.0004], [10.0, 0.0]],
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "tiny-bulge",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None


def test_dxf_extracted_bulge_is_not_certified_as_straight_l4() -> None:
    """DXF extract must keep bulge so a chord LINE cannot match a semicircle."""
    import ezdxf

    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4
    doc.modelspace().add_lwpolyline(
        [(0.0, 0.0, 1.0), (10.0, 0.0, 0.0)],
        format="xyb",
    )
    from io import StringIO

    buf = StringIO()
    doc.write(buf)
    chord = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "dxf-bulge",
                "state": "similar",
                "geom_json": chord,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="bulge.dxf",
    )
    out = apply_precision(
        cands,
        file_name="bulge.dxf",
        file_bytes=buf.getvalue().encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_long_splines_truncated_by_matcher_are_not_l4() -> None:
    """Matcher only compares 16 spline controls; longer tails must not be L4."""
    head = [[float(i), 0.0] for i in range(16)]
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "SPLINE", "control_points": head + [[16.0, 0.0], [17.0, 0.0]]},
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "SPLINE",
                "control_points": head + [[16.0, 1000.0], [17.0, 1000.0]],
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "long-spline",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_spline_prefix_with_unequal_controls_is_not_certified() -> None:
    """Incomplete spline identity must not be certified as L4."""
    head = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    query = {"entities": [{"type": "SPLINE", "control_points": head}]}
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "SPLINE",
                "control_points": head + [[float(i), 1000.0] for i in range(3, 16)],
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "spline-prefix",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_spline_degree_mismatch_is_not_certified() -> None:
    """Splines lack knots/weights in L4; fail closed rather than score 1.0."""
    cps = [[0.0, 0.0], [1.0, 2.0], [2.0, 0.0], [3.0, 3.0]]
    query = {
        "file_info": {"insunits": 4},
        "entities": [{"type": "SPLINE", "control_points": cps, "degree": 1}]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [{"type": "SPLINE", "control_points": cps, "degree": 3}]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "deg",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_malformed_polyline_vertex_is_not_joined_as_l4() -> None:
    """Invalid intermediate vertices must not be skipped into a chord LINE."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], ["bad", 1], [10.0, 0.0]],
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "bad-vert",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None


def test_polyline_tail_malformed_vertex_fails_closed() -> None:
    """A valid prefix plus a later bad vertex must not leave a shared LINE as L4."""
    shared = {"type": "LINE", "start": [0.0, 0.0], "end": [1.0, 0.0]}
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            shared,
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0], ["bad", 1]],
            },
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            shared,
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0], ["worse", 2]],
            },
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "tail-bad",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None


def test_leader_plus_line_is_not_l4() -> None:
    """Different LEADERs plus the same LINE must not score precision-l4 1.0."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
            {"type": "LEADER", "vertices": [[0.0, 0.0], [1.0, 1.0]]},
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
            {"type": "LEADER", "vertices": [[9.0, 9.0], [8.0, 8.0]]},
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "leader",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_opposite_half_ellipses_are_not_l4_geometry() -> None:
    """Vendor span-only ELLIPSE cost would treat 0..π and π..2π as equal."""
    left = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "ELLIPSE",
                "center": [0.0, 0.0],
                "major": [1.0, 0.0],
                "ratio": 0.5,
                "start_param": 0.0,
                "end_param": math.pi,
            }
        ]
    }
    right = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "ELLIPSE",
                "center": [0.0, 0.0],
                "major": [1.0, 0.0],
                "ratio": 0.5,
                "start_param": math.pi,
                "end_param": 2.0 * math.pi,
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "half-ell",
                "state": "similar",
                "geom_json": right,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(left).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_swapped_insert_block_hashes_are_not_certified() -> None:
    """Hash sets can match while per-INSERT contents are swapped."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "INSERT",
                "block": "A",
                "insert": [0.0, 0.0],
                "block_hash": "hA",
            },
            {
                "type": "INSERT",
                "block": "B",
                "insert": [10.0, 0.0],
                "block_hash": "hB",
            },
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "INSERT",
                "block": "A",
                "insert": [0.0, 0.0],
                "block_hash": "hB",
            },
            {
                "type": "INSERT",
                "block": "B",
                "insert": [10.0, 0.0],
                "block_hash": "hA",
            },
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "swapped-hash",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_mismatched_insert_block_hash_is_not_certified() -> None:
    """Same INSERT pose with different block_hash must not pass L4."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "INSERT",
                "block": "DOOR",
                "insert": [0.0, 0.0],
                "block_hash": "aaa",
            }
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "INSERT",
                "block": "DOOR",
                "insert": [0.0, 0.0],
                "block_hash": "bbb",
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "blk-hash",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_insert_without_block_hash_is_not_l4_geometry() -> None:
    """Name+pose INSERT cannot prove block contents."""
    geom = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "INSERT", "block": "DOOR", "insert": [0.0, 0.0]},
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "name-only",
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
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_legacy_dxf_extract_cache_without_version_is_reextracted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-bulge extract_sig cache must not be reused as a cache hit."""
    import hashlib

    import ezdxf

    from src.core.dedupcad_precision.vendor import dxf_extract as dxf_mod

    cache = tmp_path / "cache"

    class _Settings:
        cache_dir = str(cache)

    monkeypatch.setattr(dxf_mod, "get_settings", lambda: _Settings())
    doc = ezdxf.new("R2010")
    doc.modelspace().add_lwpolyline(
        [(0.0, 0.0, 1.0), (10.0, 0.0, 0.0)],
        format="xyb",
    )
    from io import StringIO

    buf = StringIO()
    doc.write(buf)
    dxf_path = tmp_path / "bulge.dxf"
    dxf_path.write_text(buf.getvalue(), encoding="utf-8")
    digest = hashlib.sha256(dxf_path.read_bytes()).hexdigest()
    cache_file = cache / "extract_sig" / f"{digest}.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        json.dumps(
            {
                "entities": [
                    {
                        "type": "LWPOLYLINE",
                        "points": [[0.0, 0.0], [10.0, 0.0]],
                    }
                ],
                "blocks": {},
            }
        ),
        encoding="utf-8",
    )
    extracted = dxf_mod.extract_dxf(str(dxf_path))
    assert extracted.get("file_info", {}).get("cache_hit") is not True
    poly = next(
        e
        for e in (extracted.get("entities") or [])
        if e.get("type") == "LWPOLYLINE"
    )
    assert poly.get("bulges")


def test_review_reuse_dxf_extract_does_not_write_shared_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Tenant DXF L4 extract must not persist into DEDUPCAD2_CACHE_DIR."""
    import ezdxf

    from src.core.dedupcad_precision.vendor import dxf_extract as dxf_mod

    cache = tmp_path / "shared-cache"
    cache.mkdir()

    class _Settings:
        cache_dir = str(cache)

    monkeypatch.setattr(dxf_mod, "get_settings", lambda: _Settings())
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4
    doc.modelspace().add_line((0.0, 0.0), (10.0, 0.0))
    from io import StringIO

    buf = StringIO()
    doc.write(buf)
    dxf_bytes = buf.getvalue().encode("utf-8")
    geom = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ],
    }
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
        file_name="query.dxf",
    )
    apply_precision(
        cands,
        file_name="query.dxf",
        file_bytes=dxf_bytes,
    )
    extract_sig = cache / "extract_sig"
    assert not extract_sig.exists() or not any(extract_sig.iterdir())


def test_extra_unmatched_line_is_not_certified() -> None:
    """A subset LINE match plus extra geometry must not score L4 1.0."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
            {"type": "LINE", "start": [100.0, 100.0], "end": [110.0, 100.0]},
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "extra-line",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo < LOW_PRECISION_THRESHOLD
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
    assert out[0].state == CandidateState.different


def test_over_cap_geometry_is_not_certified_as_l4() -> None:
    """Drawings larger than the matcher cap must not be prefix-certified."""
    geom = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[float(i), 0.0] for i in range(200)],
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "over-cap",
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
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_layout_shifted_clones_not_certified_when_geom_hash_env_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CAD_ML_PLATFORM_L4_ENTITIES_GEOM_HASH=1 must not re-enable bag-of-features."""
    monkeypatch.setenv("CAD_ML_PLATFORM_L4_ENTITIES_GEOM_HASH", "1")
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
            {"type": "LINE", "start": [0.0, 10.0], "end": [10.0, 10.0]},
        ]
    }
    shifted = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
            {"type": "LINE", "start": [100.0, 100.0], "end": [110.0, 100.0]},
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "shifted-env",
                "state": "similar",
                "geom_json": shifted,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo < LOW_PRECISION_THRESHOLD
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
    assert out[0].state == CandidateState.different


def test_sub_millimeter_orthogonal_lines_are_not_l4_geometry() -> None:
    """Verifier rounds to 3 decimals; collapsed LINEs must not score L4 1.0."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [0.0001, 0.0]},
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [0.0, 0.0001]},
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "tiny-ortho",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons
    assert out[0].scores.get("geometric") is None


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


def test_oversized_dxf_extract_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip local DXF parse before write when over DEDUPCAD2_MAX_FILE_MB."""
    monkeypatch.setattr(
        "src.core.review_reuse.precision._dxf_extract_limit_bytes",
        lambda: 16,
    )
    called = {"n": 0}

    def _boom(_path: Path, **_kwargs: object) -> dict:
        called["n"] += 1
        raise AssertionError("must not parse oversized dxf")

    monkeypatch.setattr(
        "src.core.dedupcad_precision.cad_pipeline.extract_geom_json_from_dxf",
        _boom,
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
        file_name="huge.dxf",
        file_bytes=b"0\nSECTION\n" + b"x" * 32,
    )
    assert called["n"] == 0
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons
    assert out[0].scores.get("geometric") is None


def test_oversized_json_geom_is_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip JSON decode when over DEDUPCAD2_MAX_FILE_MB."""
    monkeypatch.setattr(
        "src.core.review_reuse.precision._dxf_extract_limit_bytes",
        lambda: 16,
    )
    geom = _line_geom()
    payload = json.dumps(geom).encode("utf-8") + b"x" * 32
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
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="huge.json",
        file_bytes=payload,
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons
    assert out[0].scores.get("geometric") is None


def test_empty_dxf_extract_is_not_l4_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "src.core.dedupcad_precision.cad_pipeline.extract_geom_json_from_dxf",
        lambda _path, **_kwargs: {"schema": "geom-json/v2", "entities": []},
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


def test_hashed_inline_geom_does_not_bypass_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A 64-hex id must score the store geom, not a query-mirroring live payload."""
    query = _line_geom()
    stored = {
        "file_info": {"insunits": 4},
        "layers": {"0": {"color": 7, "linetype": "CONTINUOUS"}},
        "entities": [
            {
                "type": "LINE",
                "layer": "0",
                "start": [0.0, 0.0],
                "end": [0.0, 100.0],
            }
        ],
    }

    class _Store:
        def load(self, _cid: str) -> dict:
            return stored

    monkeypatch.setattr(
        "src.core.dedupcad_precision.create_geom_store", lambda: _Store()
    )
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "a" * 64,
                "state": "similar",
                "geom_json": query,
                "methods": ["dedup2d-vision"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert out[0].scores.get("geometric") == 0.0
    assert out[0].state == CandidateState.different
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons


def test_mixed_case_hash_id_uses_store_not_inline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Uppercase 64-hex ids must still load the store, not inline geom_json."""
    query = _line_geom()
    stored = {
        "file_info": {"insunits": 4},
        "layers": {"0": {"color": 7, "linetype": "CONTINUOUS"}},
        "entities": [
            {
                "type": "LINE",
                "layer": "0",
                "start": [0.0, 0.0],
                "end": [0.0, 100.0],
            }
        ],
    }
    seen: List[str] = []

    class _Store:
        def load(self, cid: str) -> dict:
            seen.append(cid)
            return stored

    monkeypatch.setattr(
        "src.core.dedupcad_precision.create_geom_store", lambda: _Store()
    )
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "A" * 64,
                "state": "similar",
                "geom_json": query,
                "methods": ["dedup2d-vision"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert seen == ["a" * 64]
    assert out[0].scores.get("geometric") == 0.0
    assert out[0].state == CandidateState.different
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons


def test_mixed_case_hash_store_miss_does_not_revive_prescored_l4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _line_geom()

    class _EmptyStore:
        def load(self, cid: str) -> None:
            assert cid == "a" * 64
            return None

    monkeypatch.setattr(
        "src.core.dedupcad_precision.create_geom_store", lambda: _EmptyStore()
    )
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "Ab" * 32,
                "state": "similar",
                "geom_json": query,
                "scores": {"geometric": 1.0},
                "methods": ["precision-l4", "dedup2d-vision"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_hashed_store_miss_does_not_revive_prescored_l4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hashed ids must not fall through to adapter L4 when the store is empty."""
    query = _line_geom()

    class _EmptyStore:
        def load(self, _cid: str) -> None:
            return None

    monkeypatch.setattr(
        "src.core.dedupcad_precision.create_geom_store", lambda: _EmptyStore()
    )
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "a" * 64,
                "state": "similar",
                "geom_json": query,
                "scores": {"geometric": 1.0},
                "methods": ["precision-l4", "dedup2d-vision"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


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
    def _boom(_path: object, **_kwargs: object) -> dict:
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
        lambda _path, **_kwargs: geom,
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


def test_precision_strips_inline_geom_json_after_l4() -> None:
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
        file_name="query.json",
    )
    assert cands[0].provenance.get("geom_json") == geom
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(geom).encode("utf-8"),
    )
    assert out[0].scores.get("geometric") == 1.0
    assert "geom_json" not in (out[0].provenance or {})


def test_task_export_does_not_include_candidate_geom_json() -> None:
    """Live/seed geom_json must not appear on GET, EvidencePack, or audit."""
    svc = _svc()
    geom = _line_geom()
    task = svc.create_task(
        tenant_id="t-no-leak",
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
    stored = svc.get_task("t-no-leak", task.task_id)
    pack, _ = svc.get_evidence_pack("t-no-leak", task.task_id)
    bundle = svc.export_audit_bundle("t-no-leak", task.task_id)

    def _no_inline_geom(obj: object) -> None:
        if isinstance(obj, dict):
            assert "geom_json" not in obj
            for value in obj.values():
                _no_inline_geom(value)
        elif isinstance(obj, list):
            for item in obj:
                _no_inline_geom(item)

    _no_inline_geom(stored.model_dump(mode="json"))
    _no_inline_geom(pack)
    _no_inline_geom(bundle)
    assert stored.candidates[0].scores.get("geometric") == 1.0


def test_wide_polyline_is_not_certified_as_zero_width_l4() -> None:
    """Exploding a thick LWPOLYLINE into LINEs would drop stroke width."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0]],
                "const_width": 2.5,
            }
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0]],
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "thin-poly",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_integer_has_width_marker_is_not_certified_as_thin_l4() -> None:
    """has_width: 1 must not explode into a zero-width chord LINE."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ],
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0]],
                "has_width": 1,
            }
        ],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "has-width-int",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_non_list_polyline_widths_is_not_certified_as_thin_l4() -> None:
    """Present non-list widths must not explode into a zero-width chord LINE."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ],
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0]],
                "widths": "2.5",
            }
        ],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "widths-str",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_vertex_polyline_width_is_not_certified_as_l4() -> None:
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0, 1.0, 1.0, 0.0], [10.0, 0.0]],
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "wide-vertex",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None


def test_insunits_mismatch_is_not_certified() -> None:
    line = {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]}
    query = {"file_info": {"insunits": 1}, "entities": [line]}
    other = {"file_info": {"insunits": 4}, "entities": [line]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "mm",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_prescored_l4_does_not_override_after_store_geom_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hash-backed units refuse must stay fail-closed if the store later misses."""
    from src.core.dedupcad_precision import create_geom_store

    line = {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]}
    query = {"file_info": {"insunits": 1}, "entities": [line]}
    other = {"file_info": {"insunits": 4}, "entities": [line]}
    sha = "b" * 64

    class _OnceStore:
        def __init__(self) -> None:
            self.loads = 0

        def load(self, file_hash: str) -> Optional[Dict[str, Any]]:
            del file_hash
            self.loads += 1
            return other if self.loads == 1 else None

    store = _OnceStore()
    create_geom_store.cache_clear()
    monkeypatch.setattr(
        "src.core.dedupcad_precision.create_geom_store", lambda: store
    )
    try:
        cands = map_raw_hits_to_candidates(
            [
                {
                    "candidate_id": sha,
                    "state": "similar",
                    "scores": {"geometric": 1.0},
                    "methods": ["precision-l4"],
                }
            ],
            content_sha="ab",
            file_name="query.json",
        )
        out = apply_precision(
            cands,
            file_name="query.json",
            file_bytes=json.dumps(query).encode("utf-8"),
        )
    finally:
        create_geom_store.cache_clear()
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_prescored_l4_does_not_override_units_conflict() -> None:
    """Adapter precision-l4 1.0 must not certify inch vs mm identical coords."""
    line = {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]}
    query = {"file_info": {"insunits": 1}, "entities": [line]}
    other = {"file_info": {"insunits": 4}, "entities": [line]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "prescored-units",
                "state": "similar",
                "scores": {"geometric": 1.0},
                "geom_json": other,
                "methods": ["precision-l4"],
            }
        ],
        content_sha="ab",
        file_name="query.json",
    )
    out = apply_precision(
        cands,
        file_name="query.json",
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert int(out[0].verification.get("level") or 0) < 4
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_matching_insunits_still_l4() -> None:
    line = {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]}
    geom = {"file_info": {"insunits": 4}, "entities": [line]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "same-mm",
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


def test_conflicting_insunits_aliases_are_not_certified() -> None:
    """Top-level insunits 4 vs file_info inches must not match a mm drawing."""
    line = {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]}
    query = {
        "insunits": 4,
        "file_info": {"insunits": 1},
        "entities": [line],
    }
    other = {"file_info": {"insunits": 4}, "entities": [line]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "alias-units",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_fractional_insunits_is_not_certified() -> None:
    """4.9 must not truncate to 4 and match a millimeter payload."""
    line = {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]}
    query = {"file_info": {"insunits": 4.9}, "entities": [line]}
    other = {"file_info": {"insunits": 4}, "entities": [line]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "frac-units",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_wrapped_arc_sweeps_are_not_certified() -> None:
    """0→359.9 vs 0→0.1 are not the same ARC; modulo endpoints would score ~0.9."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "ARC",
                "center": [0.0, 0.0],
                "radius": 10.0,
                "start_angle": 0.0,
                "end_angle": 359.9,
            }
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "ARC",
                "center": [0.0, 0.0],
                "radius": 10.0,
                "start_angle": 0.0,
                "end_angle": 0.1,
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "wrap-arc",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo < LOW_PRECISION_THRESHOLD
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
    assert out[0].state == CandidateState.different


def test_swapped_multi_arc_sweeps_are_not_certified() -> None:
    """Sweeps exchanged across centers must not L4-match via sorted sweep bags."""

    def _arc(x: float, end: float) -> dict:
        return {
            "type": "ARC",
            "center": [x, 0.0],
            "radius": 10.0,
            "start_angle": 0.0,
            "end_angle": end,
        }

    query = {"file_info": {"insunits": 4}, "entities": [_arc(0.0, 359.9), _arc(100.0, 0.1)]}
    other = {"file_info": {"insunits": 4}, "entities": [_arc(0.0, 0.1), _arc(100.0, 359.9)]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "swap-arc",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo < LOW_PRECISION_THRESHOLD
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
    assert out[0].state == CandidateState.different


def test_swapped_concentric_arc_sweeps_are_not_certified() -> None:
    """Sweeps exchanged across concentric radii must not L4-match by sweep bags."""

    def _arc(radius: float, end: float) -> dict:
        return {
            "type": "ARC",
            "center": [0.0, 0.0],
            "radius": radius,
            "start_angle": 0.0,
            "end_angle": end,
        }

    query = {"file_info": {"insunits": 4}, "entities": [_arc(10.0, 359.9), _arc(20.0, 0.1)]}
    other = {"file_info": {"insunits": 4}, "entities": [_arc(10.0, 0.1), _arc(20.0, 359.9)]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "swap-concentric",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo < LOW_PRECISION_THRESHOLD
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
    assert out[0].state == CandidateState.different


def test_near_similar_arc_radius_is_not_forced_to_zero() -> None:
    """Radius 10.0 vs 10.1 is inside tol_circle_radius; do not zero the score."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "ARC",
                "center": [0.0, 0.0],
                "radius": 10.0,
                "start_angle": 0.0,
                "end_angle": 90.0,
            }
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "ARC",
                "center": [0.0, 0.0],
                "radius": 10.1,
                "start_angle": 0.0,
                "end_angle": 90.0,
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "near-arc",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo > LOW_PRECISION_THRESHOLD


def test_swapped_nearby_arc_sweeps_are_not_certified() -> None:
    """Sweeps swapped between centers inside tol_circle_center must not L4."""

    def _arc(x: float, end: float) -> dict:
        return {
            "type": "ARC",
            "center": [x, 0.0],
            "radius": 10.0,
            "start_angle": 0.0,
            "end_angle": end,
        }

    query = {"file_info": {"insunits": 4}, "entities": [_arc(0.0, 359.9), _arc(0.3, 0.1)]}
    other = {"file_info": {"insunits": 4}, "entities": [_arc(0.0, 0.1), _arc(0.3, 359.9)]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "nearby-swap-arc",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo < LOW_PRECISION_THRESHOLD
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
    assert out[0].state == CandidateState.different


def test_swapped_quantized_step_arc_sweeps_are_not_certified() -> None:
    """A 0.001 center step must outrank angle tie-break on swapped sweeps."""

    def _arc(x: float, end: float) -> dict:
        return {
            "type": "ARC",
            "center": [x, 0.0],
            "radius": 10.0,
            "start_angle": 0.0,
            "end_angle": end,
        }

    query = {"file_info": {"insunits": 4}, "entities": [_arc(0.0, 359.9), _arc(0.001, 0.1)]}
    other = {"file_info": {"insunits": 4}, "entities": [_arc(0.0, 0.1), _arc(0.001, 359.9)]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "quant-step-swap-arc",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo < LOW_PRECISION_THRESHOLD
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
    assert out[0].state == CandidateState.different


def test_near_equal_spatial_arc_sweep_swap_is_not_certified() -> None:
    """A 5e-7 spatial gap must outrank angle on swapped 3-decimal centers."""

    def _arc(x: float, y: float, end: float) -> dict:
        return {
            "type": "ARC",
            "center": [x, y],
            "radius": 10.0,
            "start_angle": 0.0,
            "end_angle": end,
        }

    query = {
        "file_info": {"insunits": 4},
        "entities": [
            _arc(0.266, 0.185, 359.9),
            _arc(0.087, 0.026, 0.1),
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            _arc(0.254, 0.298, 0.1),
            _arc(0.245, 0.252, 359.9),
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "near-equal-spatial-swap",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo < LOW_PRECISION_THRESHOLD
    assert RejectionReason.low_precision_score.value in out[0].rejection_reasons
    assert out[0].state == CandidateState.different


def test_shifted_colinear_arcs_matching_sweeps_still_l4() -> None:
    """Equal-total spatial assignments must still pair by matching sweeps."""

    def _arc(x: float, end: float) -> dict:
        return {
            "type": "ARC",
            "center": [x, 0.0],
            "radius": 10.0,
            "start_angle": 0.0,
            "end_angle": end,
        }

    # Identity 0.1+0.1 and shift 0.0+0.2 both total 0.2; matching sweeps
    # live on the shift. Row/column equality would keep only identity.
    query = {"file_info": {"insunits": 4}, "entities": [_arc(0.0, 359.9), _arc(0.1, 0.1)]}
    other = {"file_info": {"insunits": 4}, "entities": [_arc(0.1, 0.1), _arc(0.2, 359.9)]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "colinear-shift-arcs",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo > LOW_PRECISION_THRESHOLD


def test_reordered_same_center_arcs_still_l4() -> None:
    """Same-center/radius ARCs with different spans must L4 after reorder."""

    def _arc(start: float, end: float) -> dict:
        return {
            "type": "ARC",
            "center": [0.0, 0.0],
            "radius": 10.0,
            "start_angle": start,
            "end_angle": end,
        }

    query = {"file_info": {"insunits": 4}, "entities": [_arc(0.0, 90.0), _arc(180.0, 0.0)]}
    other = {"file_info": {"insunits": 4}, "entities": [_arc(180.0, 0.0), _arc(0.0, 90.0)]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "reorder-same-center",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo > LOW_PRECISION_THRESHOLD


def test_reordered_uniform_offset_same_center_arcs_still_l4() -> None:
    """Uniform center offset must still pair by sweep after reorder."""

    def _arc(x: float, start: float, end: float) -> dict:
        return {
            "type": "ARC",
            "center": [x, 0.0],
            "radius": 10.0,
            "start_angle": start,
            "end_angle": end,
        }

    query = {
        "file_info": {"insunits": 4},
        "entities": [_arc(0.0, 0.0, 90.0), _arc(0.0, 180.0, 0.0)],
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [_arc(0.001, 180.0, 0.0), _arc(0.001, 0.0, 90.0)],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "reorder-uniform-offset",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo > LOW_PRECISION_THRESHOLD


def test_reordered_concentric_near_radii_still_l4() -> None:
    """Identical concentric ARCs within radius_tol must L4 after reorder."""

    def _arc(radius: float, end: float) -> dict:
        return {
            "type": "ARC",
            "center": [0.0, 0.0],
            "radius": radius,
            "start_angle": 0.0,
            "end_angle": end,
        }

    query = {"file_info": {"insunits": 4}, "entities": [_arc(10.0, 90.0), _arc(10.3, 180.0)]}
    other = {"file_info": {"insunits": 4}, "entities": [_arc(10.3, 180.0), _arc(10.0, 90.0)]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "reorder-concentric",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo > LOW_PRECISION_THRESHOLD


def test_reordered_nearby_arcs_still_l4() -> None:
    """Greedy sweep pairing must not zero identical ARCs listed in another order."""

    def _arc(x: float, end: float) -> dict:
        return {
            "type": "ARC",
            "center": [x, 0.0],
            "radius": 10.0,
            "start_angle": 0.0,
            "end_angle": end,
        }

    query = {
        "file_info": {"insunits": 4},
        "entities": [_arc(0.0, 90.0), _arc(0.3, 90.0), _arc(0.6, 90.0)],
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [_arc(0.6, 90.0), _arc(0.3, 90.0), _arc(0.0, 90.0)],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "reorder-arc",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo > LOW_PRECISION_THRESHOLD


def test_near_similar_arc_center_is_not_forced_to_zero() -> None:
    """Center offset 0.1 is inside tol_circle_center; do not zero the score."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "ARC",
                "center": [0.0, 0.0],
                "radius": 10.0,
                "start_angle": 0.0,
                "end_angle": 90.0,
            }
        ]
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "ARC",
                "center": [0.1, 0.0],
                "radius": 10.0,
                "start_angle": 0.0,
                "end_angle": 90.0,
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "near-center",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    geo = out[0].scores.get("geometric")
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert geo is not None
    assert geo > LOW_PRECISION_THRESHOLD


def test_out_of_range_insunits_is_not_certified() -> None:
    line = {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]}
    geom = {"file_info": {"insunits": 999}, "entities": [line]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "bad-units",
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
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None


def test_omitted_insunits_on_both_sides_is_not_certified() -> None:
    line = {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]}
    geom = {"entities": [line]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "no-units",
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
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None


def test_unknown_insunits_is_not_certified() -> None:
    line = {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]}
    geom = {"file_info": {"insunits": 0}, "entities": [line]}
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "unitless",
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
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None


def test_dxf_extracted_closed_polyline_is_not_open_l4(tmp_path: Path) -> None:
    """Classic POLYLINE is_closed must not L4-match an open vertex list."""
    import ezdxf
    from io import StringIO

    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf

    doc = ezdxf.new("R2000")
    doc.header["$INSUNITS"] = 4
    doc.modelspace().add_polyline2d(
        [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)],
        close=True,
    )
    buf = StringIO()
    doc.write(buf)
    dxf_bytes = buf.getvalue().encode("utf-8")
    path = tmp_path / "closed.dxf"
    path.write_bytes(dxf_bytes)
    extracted = extract_dxf(str(path))
    poly = next(
        e
        for e in (extracted.get("entities") or [])
        if e.get("type") in ("POLYLINE", "LWPOLYLINE")
    )
    assert poly.get("closed") is True
    open_clone = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "POLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
                "closed": False,
            }
        ],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "open-poly",
                "state": "similar",
                "geom_json": open_clone,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.dxf",
    )
    out = apply_precision(cands, file_name="query.dxf", file_bytes=dxf_bytes)
    geo = out[0].scores.get("geometric")
    assert geo is None or geo < 0.999


def test_dxf_extracted_polyline_default_width_is_not_certified_as_l4(
    tmp_path: Path,
) -> None:
    """Classic POLYLINE parent default widths must not explode into thin LINEs."""
    import ezdxf
    from io import StringIO

    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf

    doc = ezdxf.new("R2000")
    doc.header["$INSUNITS"] = 4
    doc.modelspace().add_polyline2d(
        [(0.0, 0.0), (10.0, 0.0)],
        dxfattribs={"default_start_width": 2.5},
    )
    buf = StringIO()
    doc.write(buf)
    dxf_bytes = buf.getvalue().encode("utf-8")
    path = tmp_path / "wide_poly.dxf"
    path.write_bytes(dxf_bytes)
    extracted = extract_dxf(str(path))
    poly = next(
        e
        for e in (extracted.get("entities") or [])
        if e.get("type") in ("POLYLINE", "LWPOLYLINE")
    )
    assert poly.get("has_width") is True
    thin = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "thin-line",
                "state": "similar",
                "geom_json": thin,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.dxf",
    )
    out = apply_precision(cands, file_name="query.dxf", file_bytes=dxf_bytes)
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_three_coordinate_line_is_not_certified_as_l4() -> None:
    """A discarded Z must not make two different 3D lines precision-l4."""
    query = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0, 0.0], "end": [10.0, 0.0, 0.0]},
        ],
    }
    other = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0, 100.0], "end": [10.0, 0.0, 100.0]},
        ],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "z-line",
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
        file_bytes=json.dumps(query).encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_classic_3d_polyline_is_not_flattened_to_l4(tmp_path: Path) -> None:
    """3D POLYLINE (flag 8) must not score as a 2D XY polyline."""
    import ezdxf
    from io import StringIO

    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf

    doc = ezdxf.new("R2000")
    doc.header["$INSUNITS"] = 4
    doc.modelspace().add_polyline3d([(0.0, 0.0, 0.0), (10.0, 0.0, 50.0)])
    buf = StringIO()
    doc.write(buf)
    dxf_bytes = buf.getvalue().encode("utf-8")
    path = tmp_path / "poly3d.dxf"
    path.write_bytes(dxf_bytes)
    extracted = extract_dxf(str(path), use_cache=False)
    types = [e.get("type") for e in extracted.get("entities") or []]
    assert "POLYLINE" not in types
    assert "POLYLINE3D" in types
    flat = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0]],
                "closed": False,
            }
        ],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "flat-poly",
                "state": "similar",
                "geom_json": flat,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.dxf",
    )
    out = apply_precision(cands, file_name="query.dxf", file_bytes=dxf_bytes)
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None


def _assert_not_planar_l4(dxf_bytes: bytes, planar: dict) -> None:
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "planar-xy",
                "state": "similar",
                "geom_json": planar,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.dxf",
    )
    out = apply_precision(cands, file_name="query.dxf", file_bytes=dxf_bytes)
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None


def test_nonzero_thickness_is_not_certified_as_planar_l4(tmp_path: Path) -> None:
    """Group-39 thickness must not collapse to a zero-thickness primitive."""
    import ezdxf
    from io import StringIO

    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf

    doc = ezdxf.new("R2000")
    doc.header["$INSUNITS"] = 4
    line = doc.modelspace().add_line((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
    line.dxf.thickness = 5.0
    buf = StringIO()
    doc.write(buf)
    dxf_bytes = buf.getvalue().encode("utf-8")
    path = tmp_path / "thick.dxf"
    path.write_bytes(dxf_bytes)
    extracted = extract_dxf(str(path), use_cache=False)
    types = [e.get("type") for e in extracted.get("entities") or []]
    assert "LINE" not in types
    assert "NONPLANAR" in types
    planar = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ],
    }
    _assert_not_planar_l4(dxf_bytes, planar)


def test_fitted_polyline_is_not_flattened_to_l4(tmp_path: Path) -> None:
    """Curve-fit and spline-fit POLYLINE flags are not straight chords."""
    import ezdxf
    from io import StringIO

    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf

    for flag in (2, 4):
        doc = ezdxf.new("R2000")
        doc.header["$INSUNITS"] = 4
        poly = doc.modelspace().add_polyline2d([(0.0, 0.0), (5.0, 5.0), (10.0, 0.0)])
        poly.dxf.flags = int(poly.dxf.flags or 0) | flag
        buf = StringIO()
        doc.write(buf)
        dxf_bytes = buf.getvalue().encode("utf-8")
        path = tmp_path / f"fit_{flag}.dxf"
        path.write_bytes(dxf_bytes)
        extracted = extract_dxf(str(path), use_cache=False)
        types = [e.get("type") for e in extracted.get("entities") or []]
        assert "POLYLINE" not in types
        assert "POLYLINE3D" in types
        flat = {
            "file_info": {"insunits": 4},
            "entities": [
                {
                    "type": "LWPOLYLINE",
                    "points": [[0.0, 0.0], [5.0, 5.0], [10.0, 0.0]],
                    "closed": False,
                }
            ],
        }
        _assert_not_planar_l4(dxf_bytes, flat)


def test_polyface_flag_64_is_not_flattened_to_l4(tmp_path: Path) -> None:
    """Polyface (group-70 bit 64) must not become a 2D polyline."""
    import ezdxf
    from io import StringIO

    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf

    doc = ezdxf.new("R2000")
    doc.header["$INSUNITS"] = 4
    face = doc.modelspace().add_polyface()
    face.append_face([(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (10.0, 10.0, 5.0)])
    buf = StringIO()
    doc.write(buf)
    dxf_bytes = buf.getvalue().encode("utf-8")
    path = tmp_path / "polyface.dxf"
    path.write_bytes(dxf_bytes)
    extracted = extract_dxf(str(path), use_cache=False)
    types = [e.get("type") for e in extracted.get("entities") or []]
    assert "POLYLINE" not in types
    assert "POLYLINE3D" in types
    planar = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]],
                "closed": True,
            }
        ],
    }
    _assert_not_planar_l4(dxf_bytes, planar)


def test_nonplanar_dxf_primitives_are_not_certified_as_l4(tmp_path: Path) -> None:
    """Nonzero Z/elevation or a non-default extrusion must not score as XY."""
    import ezdxf
    from io import StringIO

    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf

    def _bytes(build) -> bytes:  # type: ignore[no-untyped-def]
        doc = ezdxf.new("R2000")
        doc.header["$INSUNITS"] = 4
        build(doc.modelspace())
        buf = StringIO()
        doc.write(buf)
        return buf.getvalue().encode("utf-8")

    def _line_z(msp) -> None:  # type: ignore[no-untyped-def]
        msp.add_line((0.0, 0.0, 5.0), (10.0, 0.0, 5.0))

    def _circle_z(msp) -> None:  # type: ignore[no-untyped-def]
        msp.add_circle((0.0, 0.0, 5.0), radius=2.0)

    def _arc_z(msp) -> None:  # type: ignore[no-untyped-def]
        msp.add_arc(
            (0.0, 0.0, 5.0), radius=2.0, start_angle=0.0, end_angle=90.0
        )

    def _ellipse_z(msp) -> None:  # type: ignore[no-untyped-def]
        msp.add_ellipse(
            center=(0.0, 0.0, 5.0), major_axis=(4.0, 0.0, 0.0), ratio=0.5
        )

    def _lw_elev(msp) -> None:  # type: ignore[no-untyped-def]
        msp.add_lwpolyline(
            [(0.0, 0.0), (10.0, 0.0)], dxfattribs={"elevation": 5.0}
        )

    def _line_extrusion(msp) -> None:  # type: ignore[no-untyped-def]
        line = msp.add_line((0.0, 0.0, 0.0), (10.0, 0.0, 0.0))
        line.dxf.extrusion = (0.0, 0.0, -1.0)

    planar_line = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ],
    }
    builders = (
        _line_z,
        _circle_z,
        _arc_z,
        _ellipse_z,
        _lw_elev,
        _line_extrusion,
    )
    for index, build in enumerate(builders):
        dxf_bytes = _bytes(build)
        path = tmp_path / f"nonplanar_{index}.dxf"
        path.write_bytes(dxf_bytes)
        extracted = extract_dxf(str(path), use_cache=False)
        types = [e.get("type") for e in extracted.get("entities") or []]
        assert "NONPLANAR" in types, types
        assert "LINE" not in types
        assert "CIRCLE" not in types
        assert "ARC" not in types
        assert "ELLIPSE" not in types
        assert "LWPOLYLINE" not in types
        _assert_not_planar_l4(dxf_bytes, planar_line)


def test_insunits_comment_does_not_hide_fractional_value(tmp_path: Path) -> None:
    """Group 999 between $INSUNITS and group 70 must not drop the token."""
    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf

    dxf = (
        "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n999\ncomment\n70\n4.9\n"
        "0\nENDSEC\n0\nSECTION\n2\nENTITIES\n0\nLINE\n8\n0\n10\n0.0\n"
        "20\n0.0\n11\n10.0\n21\n0.0\n0\nENDSEC\n0\nEOF\n"
    )
    path = tmp_path / "comment_units.dxf"
    path.write_text(dxf, encoding="utf-8")
    extracted = extract_dxf(str(path), use_cache=False)
    assert extracted.get("file_info", {}).get("insunits") == 0


def test_late_cancel_does_not_receive_pipeline_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel between the last abort check and commit stays evidence-free."""
    from src.core.review_reuse.models import TaskStatus
    from src.core.review_reuse.service import ReviewReuseService

    svc = _svc()
    orig = ReviewReuseService._commit_pipeline_result

    def _cancel_then_commit(self: ReviewReuseService, task):  # type: ignore[no-untyped-def]
        current = self.store.get(task.tenant_id, task.task_id)
        assert current is not None
        current.status = TaskStatus.canceled
        self.store.put(current)
        return orig(self, task)

    monkeypatch.setattr(
        ReviewReuseService, "_commit_pipeline_result", _cancel_then_commit
    )
    task = svc.create_task(
        tenant_id="t-late-cancel",
        file_name="a.dxf",
        file_bytes=b"x",
        seed_candidates=[
            {
                "candidate_id": "c1",
                "state": "similar",
                "scores": {"geometric": 0.9},
                "methods": ["seed-adapter"],
            }
        ],
    )
    assert task.status == TaskStatus.canceled
    assert not task.candidates
    assert task.evidence_pack is None
    stored = svc.get_task("t-late-cancel", task.task_id)
    assert stored.status == TaskStatus.canceled
    assert not stored.candidates
    assert stored.evidence_pack is None


def test_dxf_fractional_insunits_header_is_not_certified(tmp_path: Path) -> None:
    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf

    dxf = (
        "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4.9\n0\nENDSEC\n"
        "0\nSECTION\n2\nENTITIES\n0\nLINE\n8\n0\n10\n0.0\n20\n0.0\n"
        "11\n10.0\n21\n0.0\n0\nENDSEC\n0\nEOF\n"
    )
    path = tmp_path / "frac_units.dxf"
    path.write_text(dxf, encoding="utf-8")
    extracted = extract_dxf(str(path))
    assert extracted.get("file_info", {}).get("insunits") == 0
    mm = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "mm-line",
                "state": "similar",
                "geom_json": mm,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.dxf",
    )
    out = apply_precision(
        cands, file_name="query.dxf", file_bytes=path.read_bytes()
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])


def test_dxf_crlf_fractional_insunits_header_is_not_certified(tmp_path: Path) -> None:
    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf

    dxf = (
        "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4.9\n0\nENDSEC\n"
        "0\nSECTION\n2\nENTITIES\n0\nLINE\n8\n0\n10\n0.0\n20\n0.0\n"
        "11\n10.0\n21\n0.0\n0\nENDSEC\n0\nEOF\n"
    ).replace("\n", "\r\n")
    path = tmp_path / "frac_units_crlf.dxf"
    path.write_bytes(dxf.encode("ascii"))
    extracted = extract_dxf(str(path))
    assert extracted.get("file_info", {}).get("insunits") == 0


def test_raw_insunits_scan_does_not_slurp_whole_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from src.core.dedupcad_precision.vendor import dxf_extract as mod

    def _forbid_slurp(self: Path) -> bytes:
        raise AssertionError("INSUNITS scan must not Path.read_bytes the DXF")

    monkeypatch.setattr(mod.Path, "read_bytes", _forbid_slurp)
    path = tmp_path / "frac_units_noslurp.dxf"
    path.write_text(
        "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4.9\n0\nENDSEC\n",
        encoding="utf-8",
    )
    assert mod._raw_insunits_non_integral(str(path)) is True


def test_raw_insunits_ignores_tokens_after_header(tmp_path: Path) -> None:
    from src.core.dedupcad_precision.vendor import dxf_extract as mod

    dxf = (
        "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4\n0\nENDSEC\n"
        "0\nSECTION\n2\nENTITIES\n9\n$INSUNITS\n70\n4.9\n"
        "0\nENDSEC\n0\nEOF\n"
    )
    path = tmp_path / "units_after_header.dxf"
    path.write_text(dxf, encoding="utf-8")
    assert mod._raw_insunits_non_integral(str(path)) is False


def test_dxf_duplicate_fractional_insunits_header_is_not_certified(
    tmp_path: Path,
) -> None:
    """Integral $INSUNITS then a later 4.9 must not L4-match a mm drawing."""
    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf

    dxf = (
        "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4\n"
        "9\n$INSUNITS\n70\n4.9\n0\nENDSEC\n"
        "0\nSECTION\n2\nENTITIES\n0\nLINE\n8\n0\n10\n0.0\n20\n0.0\n"
        "11\n10.0\n21\n0.0\n0\nENDSEC\n0\nEOF\n"
    )
    path = tmp_path / "dup_frac_units.dxf"
    path.write_text(dxf, encoding="utf-8")
    extracted = extract_dxf(str(path))
    assert extracted.get("file_info", {}).get("insunits") == 0
    mm = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "mm-line",
                "state": "similar",
                "geom_json": mm,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.dxf",
    )
    out = apply_precision(
        cands, file_name="query.dxf", file_bytes=path.read_bytes()
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])


def test_raw_insunits_duplicate_conflict_is_unsafe(tmp_path: Path) -> None:
    from src.core.dedupcad_precision.vendor import dxf_extract as mod

    path = tmp_path / "dup_conflict_units.dxf"
    path.write_text(
        "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4\n"
        "9\n$INSUNITS\n70\n1\n0\nENDSEC\n",
        encoding="utf-8",
    )
    assert mod._raw_insunits_non_integral(str(path)) is True


def test_raw_insunits_duplicate_same_integral_is_ok(tmp_path: Path) -> None:
    from src.core.dedupcad_precision.vendor import dxf_extract as mod

    path = tmp_path / "dup_same_units.dxf"
    path.write_text(
        "0\nSECTION\n2\nHEADER\n9\n$INSUNITS\n70\n4\n"
        "9\n$INSUNITS\n70\n4\n0\nENDSEC\n",
        encoding="utf-8",
    )
    assert mod._raw_insunits_non_integral(str(path)) is False


def test_binary_dxf_insunits_is_certified(tmp_path: Path) -> None:
    """ASCII HEADER scan must not wipe units from a valid binary DXF."""
    import ezdxf
    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf
    from src.core.dedupcad_precision.vendor import dxf_extract as mod

    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4
    doc.modelspace().add_line((0.0, 0.0), (10.0, 0.0))
    path = tmp_path / "bin_units.dxf"
    doc.saveas(str(path), fmt="bin")
    assert path.read_bytes().startswith(b"AutoCAD Binary DXF")
    assert mod._raw_insunits_non_integral(str(path)) is False
    extracted = extract_dxf(str(path))
    assert extracted.get("file_info", {}).get("insunits") == 4
    mm = {
        "file_info": {"insunits": 4},
        "entities": [
            {"type": "LINE", "start": [0.0, 0.0], "end": [10.0, 0.0]},
        ],
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "mm-line",
                "state": "similar",
                "geom_json": mm,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="query.dxf",
    )
    out = apply_precision(
        cands, file_name="query.dxf", file_bytes=path.read_bytes()
    )
    assert "precision-l4" in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") == 1.0


def test_isolated_file_run_uses_content_idempotency_key() -> None:
    from scripts.review_reuse_isolated_archive_run import resolve_idempotency_key

    left = resolve_idempotency_key(None, b"drawing-a", from_file=True)
    right = resolve_idempotency_key(None, b"drawing-b", from_file=True)
    assert left != right
    assert left.startswith("isolated-file-")
    assert (
        resolve_idempotency_key(None, b"x", from_file=False)
        == "isolated-archive-demo-unseeded-offline"
    )


def test_isolated_file_idempotency_key_binds_seed_and_live_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.review_reuse_isolated_archive_run import resolve_idempotency_key

    monkeypatch.delenv("REVIEW_REUSE_LIVE_DEDUP", raising=False)
    body = b"drawing-a"
    unseeded = resolve_idempotency_key(None, body, from_file=True, file_name="a.dxf")
    seeded = resolve_idempotency_key(
        None, body, from_file=True, seed_similar=True, file_name="a.dxf"
    )
    assert unseeded != seeded
    assert "unseeded-offline" in unseeded
    assert "seed-offline" in seeded
    monkeypatch.setenv("REVIEW_REUSE_LIVE_DEDUP", "true")
    live = resolve_idempotency_key(None, body, from_file=True, file_name="a.dxf")
    assert live != unseeded
    assert "unseeded-live" in live
    assert (
        resolve_idempotency_key(
            "explicit", body, from_file=True, seed_similar=True, file_name="a.dxf"
        )
        == "explicit"
    )


def test_isolated_file_idempotency_key_binds_basename() -> None:
    from scripts.review_reuse_isolated_archive_run import resolve_idempotency_key

    body = b"same-bytes"
    left = resolve_idempotency_key(None, body, from_file=True, file_name="a.dxf")
    right = resolve_idempotency_key(None, body, from_file=True, file_name="b.dxf")
    again = resolve_idempotency_key(
        None, body, from_file=True, file_name="/tmp/a.dxf"
    )
    assert left != right
    assert left == again


def test_dxf_extracted_polyline_width_is_not_certified_as_l4() -> None:
    """DXF extract must keep const_width so a thin clone cannot match."""
    import ezdxf
    from io import StringIO

    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 4
    doc.modelspace().add_lwpolyline(
        [(0.0, 0.0), (10.0, 0.0)],
        dxfattribs={"const_width": 2.5},
    )
    buf = StringIO()
    doc.write(buf)
    thin = {
        "file_info": {"insunits": 4},
        "entities": [
            {
                "type": "LWPOLYLINE",
                "points": [[0.0, 0.0], [10.0, 0.0]],
            }
        ]
    }
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "dxf-width",
                "state": "similar",
                "geom_json": thin,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="wide.dxf",
    )
    out = apply_precision(
        cands,
        file_name="wide.dxf",
        file_bytes=buf.getvalue().encode("utf-8"),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_dxf_inch_vs_mm_same_numbers_are_not_l4(tmp_path: Path) -> None:
    """$INSUNITS 1 vs 4 with identical coordinates must not score L4 1.0."""
    import ezdxf
    from io import StringIO

    from src.core.dedupcad_precision.vendor.dxf_extract import extract_dxf

    def _dxf_bytes(units: int) -> bytes:
        doc = ezdxf.new("R2010")
        doc.header["$INSUNITS"] = units
        doc.modelspace().add_line((0.0, 0.0), (10.0, 0.0))
        buf = StringIO()
        doc.write(buf)
        return buf.getvalue().encode("utf-8")

    mm_path = tmp_path / "mm.dxf"
    mm_path.write_bytes(_dxf_bytes(4))
    other = extract_dxf(str(mm_path))
    assert other.get("file_info", {}).get("insunits") == 4
    cands = map_raw_hits_to_candidates(
        [
            {
                "candidate_id": "mm-dxf",
                "state": "similar",
                "geom_json": other,
                "methods": ["seed-adapter"],
            }
        ],
        content_sha="ab",
        file_name="inch.dxf",
    )
    out = apply_precision(
        cands,
        file_name="inch.dxf",
        file_bytes=_dxf_bytes(1),
    )
    assert "precision-l4" not in (out[0].verification.get("methods") or [])
    assert out[0].scores.get("geometric") is None
    assert RejectionReason.missing_geom_json.value in out[0].rejection_reasons


def test_legacy_v2_extract_cache_without_width_is_reextracted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """v2 extract_sig cache dropped polyline widths and must not be reused."""
    import hashlib

    import ezdxf

    from src.core.dedupcad_precision.vendor import dxf_extract as dxf_mod

    cache = tmp_path / "cache"

    class _Settings:
        cache_dir = str(cache)

    monkeypatch.setattr(dxf_mod, "get_settings", lambda: _Settings())
    doc = ezdxf.new("R2010")
    doc.modelspace().add_lwpolyline(
        [(0.0, 0.0), (10.0, 0.0)],
        dxfattribs={"const_width": 2.5},
    )
    from io import StringIO

    buf = StringIO()
    doc.write(buf)
    dxf_path = tmp_path / "wide.dxf"
    dxf_path.write_text(buf.getvalue(), encoding="utf-8")
    digest = hashlib.sha256(dxf_path.read_bytes()).hexdigest()
    cache_file = cache / "extract_sig" / f"{digest}.json"
    cache_file.parent.mkdir(parents=True)
    cache_file.write_text(
        json.dumps(
            {
                "extract_cache_version": 2,
                "entities": [
                    {
                        "type": "LWPOLYLINE",
                        "points": [[0.0, 0.0], [10.0, 0.0]],
                    }
                ],
                "blocks": {},
            }
        ),
        encoding="utf-8",
    )
    extracted = dxf_mod.extract_dxf(str(dxf_path))
    assert extracted.get("file_info", {}).get("cache_hit") is not True
    poly = next(
        e
        for e in (extracted.get("entities") or [])
        if e.get("type") == "LWPOLYLINE"
    )
    assert poly.get("has_width") is True
