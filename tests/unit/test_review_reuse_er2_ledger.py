"""Fail-first state, idempotency, identity, and CAS contract for ER2."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import pytest

from src.core.review_reuse.evidence import build_evidence_pack
from src.core.review_reuse.models import (
    CandidateDecision,
    CandidateState,
    HumanDecision,
    HumanDecisionState,
    TaskEvent,
    TaskEventType,
    TaskStatus,
)
from src.core.review_reuse.service import (
    ReviewReuseError,
    ReviewReuseService,
    canonical_reviewer_principal,
)
from src.core.review_reuse.store import (
    FilesystemReviewReuseStore,
    InMemoryReviewReuseStore,
)

_REVIEWER_A = "principal-v1-" + "a" * 64
_REVIEWER_B = "principal-v1-" + "b" * 64


def _seed_candidate() -> List[Dict[str, Any]]:
    return [
        {
            "candidate_id": "archive-1",
            "candidate_source": "archive",
            "state": "similar",
            "scores": {"geometric": 0.91, "semantic": 0.73},
            "verification": {
                "verdict": "similar",
                "level": 2,
                "methods": ["geometry-check"],
            },
            "rejection_reasons": [],
            "provenance": {"path": "seed_fixture", "synthetic": True},
        }
    ]


def _service() -> ReviewReuseService:
    return ReviewReuseService(InMemoryReviewReuseStore())


def _create_ready(
    service: ReviewReuseService,
    *,
    tenant_id: str = "tenant-a",
    idempotency_key: str | None = None,
    content: bytes = b"0\nSECTION\n",
) -> Any:
    return service.create_task(
        tenant_id=tenant_id,
        file_name="part.dxf",
        file_bytes=content,
        idempotency_key=idempotency_key,
        seed_candidates=_seed_candidate(),
    )


def _pack_digest(task: Any) -> str:
    assert task.evidence_pack is not None
    digest = task.evidence_pack.get("evidence_pack_sha256")
    assert isinstance(digest, str) and len(digest) == 64
    return digest


def _submit(
    service: ReviewReuseService,
    task: Any,
    *,
    state: HumanDecisionState = HumanDecisionState.revise,
    candidate_id: str | None = "archive-1",
    reason_codes: List[str] | None = None,
    reason_text: str = "Reviewed evidence.",
    idempotency_key: str | None = None,
    reviewer_id: str = _REVIEWER_A,
    reviewer_kind: str = "validated_principal",
    tenant_validated: bool = True,
    reviewer_validated: bool = True,
    expected_revision: int | None = None,
    evidence_pack_sha256: str | None = None,
) -> Any:
    return service.submit_decision(
        tenant_id=task.tenant_id,
        task_id=task.task_id,
        state=state,
        reviewer_id=reviewer_id,
        reviewer_kind=reviewer_kind,
        reason_codes=(["needs_modification"] if reason_codes is None else reason_codes),
        reason_text=reason_text,
        candidate_id=candidate_id,
        idempotency_key=idempotency_key,
        tenant_validated=tenant_validated,
        reviewer_validated=reviewer_validated,
        expected_revision=(
            task.revision if expected_revision is None else expected_revision
        ),
        evidence_pack_sha256=(
            _pack_digest(task) if evidence_pack_sha256 is None else evidence_pack_sha256
        ),
    )


def test_empty_oversized_and_unsupported_input_do_not_persist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REVIEW_REUSE_MAX_UPLOAD_BYTES", "4")
    cases = [
        ("empty.dxf", b"", "empty_input"),
        ("large.dxf", b"12345", "input_too_large"),
        ("part.txt", b"1234", "unsupported_file_type"),
    ]

    for file_name, payload, expected_code in cases:
        store = InMemoryReviewReuseStore()
        service = ReviewReuseService(store)
        with pytest.raises(ReviewReuseError) as raised:
            service.create_task(
                tenant_id="tenant-a",
                file_name=file_name,
                file_bytes=payload,
            )
        assert raised.value.code == expected_code
        assert store.list_for_tenant("tenant-a") == []


def test_reviewer_principal_is_bound_to_identity_provider() -> None:
    first = canonical_reviewer_principal("issuer-a", "shared-subject")
    second = canonical_reviewer_principal("issuer-b", "shared-subject")
    assert first.startswith("principal-v1-")
    assert second.startswith("principal-v1-")
    assert first != second


def test_pipeline_failure_preserves_attempt_events_without_invalid_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.review_reuse import service as service_module

    store = InMemoryReviewReuseStore()
    service = ReviewReuseService(store)

    def fail_evidence(_task: Any) -> Dict[str, Any]:
        raise RuntimeError("synthetic evidence failure")

    monkeypatch.setattr(service_module, "build_evidence_pack", fail_evidence)
    with pytest.raises(ReviewReuseError) as raised:
        _create_ready(service)
    assert raised.value.code == "internal_error"

    [failed] = service.list_tasks("tenant-a")
    assert failed.status == TaskStatus.failed
    assert failed.revision == 2
    assert failed.candidates == []
    assert failed.evidence_pack is None
    assert [event.event_type.value for event in failed.events][-6:] == [
        "input_validated",
        "recall_started",
        "recall_completed",
        "precision_started",
        "precision_completed",
        "failed",
    ]


def test_pipeline_event_timestamps_survive_wall_clock_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.core.review_reuse import service as service_module

    wall_clock = 100.0

    def regressing_time() -> float:
        nonlocal wall_clock
        wall_clock -= 1.0
        return wall_clock

    monkeypatch.setattr(service_module.time, "time", regressing_time)
    task = _create_ready(_service())
    timestamps = [
        task.created_at,
        *(event.ts for event in task.events),
        task.updated_at,
    ]
    assert all(earlier <= later for earlier, later in zip(timestamps, timestamps[1:]))


def test_writer_restart_fails_interrupted_running_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SimulatedProcessDeath(BaseException):
        pass

    root = tmp_path / "store"
    file_bytes = b"0\nSECTION\n"
    key = "crash-recovery-key"
    first_store = FilesystemReviewReuseStore(root)
    create_if_absent = first_store.create_if_absent

    def crash_after_create(*args: Any, **kwargs: Any) -> Any:
        created = create_if_absent(*args, **kwargs)
        assert created.status == TaskStatus.running
        raise SimulatedProcessDeath

    monkeypatch.setattr(first_store, "create_if_absent", crash_after_create)
    try:
        with pytest.raises(SimulatedProcessDeath):
            ReviewReuseService(first_store).create_task(
                tenant_id="tenant-crash-recovery",
                file_name="part.dxf",
                file_bytes=file_bytes,
                idempotency_key=key,
            )
    finally:
        first_store.close()

    second_store = FilesystemReviewReuseStore(root)
    try:
        [recovered] = second_store.list_for_tenant("tenant-crash-recovery")
        assert recovered.status == TaskStatus.failed
        assert recovered.revision == 2
        assert recovered.error_code == "internal_error"
        assert recovered.events[-1].event_type == TaskEventType.failed

        replayed = ReviewReuseService(second_store).create_task(
            tenant_id="tenant-crash-recovery",
            file_name="part.dxf",
            file_bytes=file_bytes,
            idempotency_key=key,
        )
        assert replayed.task_id == recovered.task_id
        assert replayed.status == TaskStatus.failed
        assert [
            task.task_id
            for task in second_store.list_for_tenant("tenant-crash-recovery")
        ] == [recovered.task_id]
    finally:
        second_store.close()


@pytest.mark.parametrize(
    "adapter_case", ["unknown_reason", "duplicate_id", "noncanonical_id"]
)
def test_store_rejected_adapter_results_commit_failed_task(
    monkeypatch: pytest.MonkeyPatch,
    adapter_case: str,
) -> None:
    from src.core.review_reuse import service as service_module

    candidate = CandidateDecision(
        candidate_id=(
            " archive-1 " if adapter_case == "noncanonical_id" else "archive-1"
        ),
        candidate_source="archive",
        state=CandidateState.similar,
        scores={"geometric": 0.8, "semantic": 0.7},
        verification={"verdict": "review"},
        rejection_reasons=(
            ["not_in_vocabulary"] if adapter_case == "unknown_reason" else []
        ),
        provenance={"model": "dedup2d-live"},
    )
    candidates = (
        [candidate, candidate.model_copy(deep=True)]
        if adapter_case == "duplicate_id"
        else [candidate]
    )
    calls = 0

    def malformed_recall(**_: Any) -> List[CandidateDecision]:
        nonlocal calls
        calls += 1
        return candidates

    monkeypatch.setattr(service_module, "recall_candidates", malformed_recall)
    store = InMemoryReviewReuseStore()
    service = ReviewReuseService(store)

    with pytest.raises(ReviewReuseError) as raised:
        service.create_task(
            tenant_id="tenant-adapter",
            file_name="part.dxf",
            file_bytes=b"0\nSECTION\n",
            idempotency_key="adapter-key",
        )

    [failed] = service.list_tasks("tenant-adapter")
    assert failed.status == TaskStatus.failed
    assert failed.revision == 2
    assert failed.error_code == "internal_error"
    assert failed.candidates == []
    assert failed.evidence_pack is None
    event_types = [event.event_type.value for event in failed.events]
    assert event_types[-1] == "failed"
    assert "evidence_pack_ready" not in event_types
    assert raised.value.code == "internal_error"

    replayed = service.create_task(
        tenant_id="tenant-adapter",
        file_name="part.dxf",
        file_bytes=b"0\nSECTION\n",
        idempotency_key="adapter-key",
    )
    assert replayed.task_id == failed.task_id
    assert replayed.status == TaskStatus.failed
    assert calls == 1


def test_adapter_canonicalizes_candidate_id_before_persistence() -> None:
    seed = _seed_candidate()
    seed[0]["candidate_id"] = " archive-1 "

    task = ReviewReuseService(InMemoryReviewReuseStore()).create_task(
        tenant_id="tenant-canonical-candidate",
        file_name="part.dxf",
        file_bytes=b"0\nSECTION\n",
        seed_candidates=seed,
    )

    assert task.candidates[0].candidate_id == "archive-1"
    assert task.evidence_pack is not None
    assert task.evidence_pack["candidates"][0]["candidate_id"] == "archive-1"


def test_decision_requires_validated_tenant_and_reviewer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    monkeypatch.setenv("REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER", "false")
    service = _service()
    task = _create_ready(service)

    with pytest.raises(ReviewReuseError) as tenant_error:
        _submit(service, task, tenant_validated=False)
    assert tenant_error.value.code == "tenant_not_validated"

    with pytest.raises(ReviewReuseError) as reviewer_error:
        _submit(
            service,
            task,
            reviewer_id="ak-user-0123456789ab",
            reviewer_kind="api_key_fallback",
            reviewer_validated=False,
        )
    assert reviewer_error.value.code == "reviewer_not_validated"
    stored = service.get_task(task.tenant_id, task.task_id)
    assert stored.status == TaskStatus.evidence_ready
    assert stored.human_decision is None


def test_decision_requires_current_revision_and_evidence_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    service = _service()
    task = _create_ready(service)
    initial_revision = task.revision
    digest = _pack_digest(task)

    with pytest.raises(ReviewReuseError) as stale_revision:
        _submit(service, task, expected_revision=initial_revision + 1)
    assert stale_revision.value.code == "revision_conflict"

    with pytest.raises(ReviewReuseError) as stale_digest:
        _submit(service, task, evidence_pack_sha256="f" * 64)
    assert stale_digest.value.code == "revision_conflict"

    unchanged = service.get_task(task.tenant_id, task.task_id)
    assert unchanged.revision == initial_revision
    assert _pack_digest(unchanged) == digest
    assert unchanged.human_decision is None

    decided = _submit(service, unchanged, idempotency_key="decision-current")
    assert decided.revision == initial_revision + 1
    assert decided.human_decision is not None
    assert decided.human_decision.reviewed_revision == initial_revision
    assert decided.human_decision.evidence_pack_sha256 == digest


def test_store_cas_rejects_forged_reviewed_evidence_binding() -> None:
    store = InMemoryReviewReuseStore()
    service = ReviewReuseService(store)
    current = _create_ready(service)
    forged = current.model_copy(deep=True)
    forged.revision = current.revision + 1
    forged.status = TaskStatus.decided
    forged.human_decision = HumanDecision(
        state=HumanDecisionState.new,
        reviewer_id=_REVIEWER_A,
        reviewer_kind="validated_principal",
        reason_codes=["new_part_required"],
        reason_text="Forged binding.",
        candidate_id=None,
        ts=1.0,
        reviewed_revision=current.revision - 1,
        evidence_pack_sha256="f" * 64,
    )
    forged.events.append(
        TaskEvent(event_type=TaskEventType.decision_submitted, ts=1.0, detail={})
    )
    forged.evidence_pack = build_evidence_pack(forged)

    with pytest.raises(Exception) as raised:
        store.put(forged, expected_revision=current.revision)
    assert getattr(raised.value, "code", None) == "revision_conflict"
    unchanged = store.get(current.tenant_id, current.task_id)
    assert unchanged is not None
    assert unchanged.revision == current.revision
    assert unchanged.human_decision is None


def test_store_cas_rejects_unknown_decision_reason_code() -> None:
    store = InMemoryReviewReuseStore()
    service = ReviewReuseService(store)
    current = _create_ready(service)
    altered = current.model_copy(deep=True)
    altered.revision = current.revision + 1
    altered.status = TaskStatus.decided
    altered.human_decision = HumanDecision(
        state=HumanDecisionState.revise,
        reviewer_id=_REVIEWER_A,
        reviewer_kind="validated_principal",
        reason_codes=["not_in_vocabulary"],
        reason_text="Bypassed service validation.",
        candidate_id="archive-1",
        ts=1.0,
        reviewed_revision=current.revision,
        evidence_pack_sha256=_pack_digest(current),
    )
    altered.events.append(
        TaskEvent(event_type=TaskEventType.decision_submitted, ts=1.0, detail={})
    )
    altered.evidence_pack = build_evidence_pack(altered)

    with pytest.raises(Exception) as raised:
        store.put(altered, expected_revision=current.revision)
    assert getattr(raised.value, "code", None) == "store_record_corrupt"


@pytest.mark.parametrize(
    "event_case", ["wrong_type", "wrong_detail", "unrelated_prefix"]
)
def test_store_cas_binds_decision_event_to_transition(event_case: str) -> None:
    store = InMemoryReviewReuseStore()
    service = ReviewReuseService(store)
    current = _create_ready(service)
    altered = current.model_copy(deep=True)
    altered.revision = current.revision + 1
    altered.status = TaskStatus.decided
    altered.human_decision = HumanDecision(
        state=HumanDecisionState.revise,
        reviewer_id=_REVIEWER_A,
        reviewer_kind="validated_principal",
        reason_codes=["needs_modification"],
        reason_text="Reviewed evidence.",
        candidate_id="archive-1",
        ts=1.0,
        reviewed_revision=current.revision,
        evidence_pack_sha256=_pack_digest(current),
    )
    expected_detail = {
        "state": "revise",
        "reviewer_id": _REVIEWER_A,
        "candidate_id": "archive-1",
        "reviewed_revision": current.revision,
        "evidence_pack_sha256": _pack_digest(current),
    }
    if event_case == "wrong_type":
        event_type = TaskEventType.canceled
        detail = {}
    else:
        event_type = TaskEventType.decision_submitted
        detail = (
            {**expected_detail, "reviewer_id": _REVIEWER_B}
            if event_case == "wrong_detail"
            else expected_detail
        )
    if event_case == "unrelated_prefix":
        altered.events.append(
            TaskEvent(event_type=TaskEventType.submitted, ts=0.5, detail={})
        )
    altered.events.append(TaskEvent(event_type=event_type, ts=1.0, detail=detail))
    altered.evidence_pack = build_evidence_pack(altered)

    with pytest.raises(Exception) as raised:
        store.put(altered, expected_revision=current.revision)
    assert getattr(raised.value, "code", None) == "store_record_corrupt"


def test_store_cas_rejects_identity_or_candidate_rewrite() -> None:
    store = InMemoryReviewReuseStore()
    service = ReviewReuseService(store)
    current = _create_ready(service)

    for field, replacement in (
        ("trace_id", "00000000-0000-0000-0000-000000000000"),
        ("source_file_name", "renamed.dxf"),
        ("candidates", []),
    ):
        altered = current.model_copy(deep=True)
        altered.revision = current.revision + 1
        altered.status = TaskStatus.canceled
        setattr(altered, field, replacement)
        altered.events.append(
            TaskEvent(event_type=TaskEventType.canceled, ts=1.0, detail={})
        )
        altered.evidence_pack = build_evidence_pack(altered)

        with pytest.raises(Exception) as raised:
            store.put(altered, expected_revision=current.revision)
        assert getattr(raised.value, "code", None) == "store_record_corrupt"

    unchanged = store.get(current.tenant_id, current.task_id)
    assert unchanged is not None
    assert unchanged.revision == current.revision


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("calibration_version", "tampered-calibration"),
        ("calibration_status", "calibrated"),
    ],
)
def test_store_cas_freezes_reviewed_calibration(field: str, value: str) -> None:
    store = InMemoryReviewReuseStore()
    service = ReviewReuseService(store)
    current = _create_ready(service)
    altered = current.model_copy(deep=True)
    altered.revision = current.revision + 1
    altered.status = TaskStatus.decided
    setattr(altered, field, value)
    altered.human_decision = HumanDecision(
        state=HumanDecisionState.revise,
        reviewer_id=_REVIEWER_A,
        reviewer_kind="validated_principal",
        reason_codes=["needs_modification"],
        reason_text="Reviewed evidence.",
        candidate_id="archive-1",
        ts=1.0,
        reviewed_revision=current.revision,
        evidence_pack_sha256=_pack_digest(current),
    )
    altered.events.append(
        TaskEvent(event_type=TaskEventType.decision_submitted, ts=1.0, detail={})
    )
    altered.evidence_pack = build_evidence_pack(altered)

    with pytest.raises(Exception) as raised:
        store.put(altered, expected_revision=current.revision)
    assert getattr(raised.value, "code", None) == "store_record_corrupt"


def test_idempotency_key_conflicts_on_payload_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service = _service()
    original = _create_ready(service, idempotency_key="create-key", content=b"same")
    replay = service.create_task(
        tenant_id="tenant-a",
        file_name="renamed.dxf",
        file_bytes=b"same",
        idempotency_key="create-key",
        seed_candidates=_seed_candidate(),
    )
    assert replay.task_id == original.task_id
    assert replay.source_file_name == original.source_file_name

    with pytest.raises(ReviewReuseError) as create_conflict:
        _create_ready(service, idempotency_key="create-key", content=b"different")
    assert create_conflict.value.code == "idempotency_key_conflict"

    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    decision_task = _create_ready(service, tenant_id="tenant-b")
    decided = _submit(service, decision_task, idempotency_key="decision-key")
    replayed = _submit(service, decision_task, idempotency_key="decision-key")
    assert replayed.task_id == decided.task_id
    assert replayed.revision == decided.revision

    with pytest.raises(ReviewReuseError) as decision_conflict:
        _submit(
            service,
            decision_task,
            idempotency_key="decision-key",
            reason_text="Changed rationale.",
        )
    assert decision_conflict.value.code == "idempotency_key_conflict"

    with pytest.raises(ReviewReuseError) as actor_conflict:
        _submit(
            service,
            decision_task,
            idempotency_key="decision-key",
            reviewer_id=_REVIEWER_B,
        )
    assert actor_conflict.value.code == "idempotency_key_conflict"


def test_decision_matrix_and_reason_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    invalid_cases = [
        {
            "state": HumanDecisionState.reuse,
            "candidate_id": None,
            "reason_codes": ["geometry_match"],
        },
        {
            "state": HumanDecisionState.new,
            "candidate_id": "archive-1",
            "reason_codes": ["new_part_required"],
        },
        {
            "state": HumanDecisionState.revise,
            "candidate_id": "missing",
            "reason_codes": ["needs_modification"],
        },
        {
            "state": HumanDecisionState.revise,
            "candidate_id": "archive-1",
            "reason_codes": ["not_in_vocabulary"],
        },
        {
            "state": HumanDecisionState.revise,
            "candidate_id": "archive-1",
            "reason_codes": ["other"],
            "reason_text": "",
        },
        {
            "state": HumanDecisionState.revise,
            "candidate_id": "archive-1",
            "reason_codes": [],
            "reason_text": "",
        },
    ]

    for case in invalid_cases:
        service = _service()
        task = _create_ready(service)
        with pytest.raises(ReviewReuseError) as raised:
            _submit(service, task, **case)
        assert raised.value.code == "invalid_decision"
        stored = service.get_task(task.tenant_id, task.task_id)
        assert stored.revision == task.revision
        assert stored.human_decision is None

    valid_service = _service()
    valid_task = _create_ready(valid_service)
    decided = _submit(
        valid_service,
        valid_task,
        state=HumanDecisionState.new,
        candidate_id=None,
        reason_codes=["new_part_required"],
    )
    assert decided.status == TaskStatus.decided


def test_cancel_retry_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    service = _service()
    task = _create_ready(service)
    first = service.cancel(task.tenant_id, task.task_id)
    first_snapshot = first.model_dump(mode="json")
    second = service.cancel(task.tenant_id, task.task_id)
    assert second.model_dump(mode="json") == first_snapshot
    assert second.revision == task.revision + 1

    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    decided_task = _create_ready(service, tenant_id="tenant-b")
    decided = _submit(service, decided_task)
    with pytest.raises(ReviewReuseError) as raised:
        service.cancel(decided.tenant_id, decided.task_id)
    assert raised.value.code == "invalid_state_transition"


def test_concurrent_decision_and_cancel_commit_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("REVIEW_REUSE_DECISIONS_ENABLED", "true")
    service = _service()
    task = _create_ready(service)
    initial_revision = task.revision
    initial_events = [event.model_dump(mode="json") for event in task.events]
    barrier = threading.Barrier(3)

    def decide() -> Any:
        barrier.wait()
        return _submit(service, task, idempotency_key="concurrent-decision")

    def cancel() -> Any:
        barrier.wait()
        return service.cancel(task.tenant_id, task.task_id)

    def capture(callable_: Any) -> tuple[str, Any]:
        try:
            return "ok", callable_()
        except ReviewReuseError as exc:
            return "error", exc.code

    with ThreadPoolExecutor(max_workers=2) as pool:
        decision_future = pool.submit(capture, decide)
        cancel_future = pool.submit(capture, cancel)
        barrier.wait()
        outcomes = [decision_future.result(timeout=5), cancel_future.result(timeout=5)]

    assert [kind for kind, _ in outcomes].count("ok") == 1
    errors = [value for kind, value in outcomes if kind == "error"]
    assert errors[0] in {
        "revision_conflict",
        "already_decided",
        "invalid_state_transition",
    }

    final = service.get_task(task.tenant_id, task.task_id)
    assert final.status in {TaskStatus.decided, TaskStatus.canceled}
    assert final.revision == initial_revision + 1
    final_events = [event.model_dump(mode="json") for event in final.events]
    assert final_events[: len(initial_events)] == initial_events
    assert len(final_events) == len(initial_events) + 1
