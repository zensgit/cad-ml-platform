"""ReviewReuse task lifecycle and human decision ledger."""

from __future__ import annotations

import hashlib
import logging
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .canonical import CanonicalJSONError, canonical_sha256
from .dedup_adapter import recall_candidates
from .evidence import (
    build_evidence_pack,
    evidence_pack_digest_is_valid,
    evidence_pack_markdown,
)
from .metrics import compute_review_metrics
from .models import (
    DECISION_REASON_CODES,
    HumanDecision,
    HumanDecisionState,
    ReviewReuseTask,
    TaskEvent,
    TaskEventType,
    TaskStatus,
)
from .store import (
    ReviewReuseStoreError,
    ReviewReuseStoreProtocol,
    create_review_reuse_store,
    validate_review_reuse_task_payload,
    validate_tenant_id,
)

ENV_DECISIONS_ENABLED = "REVIEW_REUSE_DECISIONS_ENABLED"
ENV_REQUIRE_VALIDATED_REVIEWER = "REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER"
ENV_MAX_UPLOAD_BYTES = "REVIEW_REUSE_MAX_UPLOAD_BYTES"
DEFAULT_MAX_UPLOAD_BYTES = 52_428_800
_TRUE = frozenset({"1", "true", "yes", "on"})
_HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_PRINCIPAL_PATTERN = re.compile(r"^principal-v1-[0-9a-f]{64}$")
_CANDIDATE_REQUIRED = frozenset(
    {
        HumanDecisionState.reuse,
        HumanDecisionState.revise,
        HumanDecisionState.reject_candidate,
    }
)

_STORE: Optional[ReviewReuseStoreProtocol] = None
logger = logging.getLogger(__name__)


def decisions_enabled() -> bool:
    return os.getenv(ENV_DECISIONS_ENABLED, "").strip().lower() in _TRUE


def require_validated_reviewer() -> bool:
    """Legacy compatibility flag; it cannot weaken the mandatory ER2 rule."""

    return os.getenv(ENV_REQUIRE_VALIDATED_REVIEWER, "").strip().lower() in _TRUE


def is_api_key_fallback_reviewer(reviewer_id: str) -> bool:
    rid = (reviewer_id or "").strip()
    return rid.startswith("ak-user-") or rid in ("", "anonymous", "unknown")


def max_upload_bytes() -> int:
    raw = os.getenv(ENV_MAX_UPLOAD_BYTES, str(DEFAULT_MAX_UPLOAD_BYTES))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ReviewReuseError(
            "internal_error", "upload limit is misconfigured"
        ) from exc
    if value <= 0:
        raise ReviewReuseError("internal_error", "upload limit is misconfigured")
    return value


def canonical_reviewer_principal(identity_provider: str, subject: str) -> str:
    if (
        not isinstance(identity_provider, str)
        or not identity_provider
        or identity_provider.strip() != identity_provider
        or not isinstance(subject, str)
        or not subject
        or subject.strip() != subject
    ):
        raise ReviewReuseError(
            "reviewer_not_validated", "validated reviewer identity is unavailable"
        )
    try:
        digest = canonical_sha256(
            {"identity_provider": identity_provider, "subject": subject}
        )
    except CanonicalJSONError as exc:
        raise ReviewReuseError(
            "reviewer_not_validated", "validated reviewer identity is unavailable"
        ) from exc
    return f"principal-v1-{digest}"


def get_review_reuse_store() -> ReviewReuseStoreProtocol:
    global _STORE
    if _STORE is None:
        _STORE = create_review_reuse_store()
    return _STORE


def reset_review_reuse_store_for_tests(
    store: Optional[ReviewReuseStoreProtocol] = None,
) -> None:
    """Test helper: replace the process singleton and release any writer lease."""

    global _STORE
    previous = _STORE
    if previous is store:
        return
    _STORE = None
    if previous is not None:
        close = getattr(previous, "close", None)
        if close is not None:
            close()
    _STORE = store if store is not None else create_review_reuse_store()


def get_review_reuse_service() -> "ReviewReuseService":
    return ReviewReuseService(get_review_reuse_store())


def close_review_reuse_store() -> None:
    """Release the process-lifetime store lease during application shutdown."""

    global _STORE
    store = _STORE
    _STORE = None
    if store is not None:
        close = getattr(store, "close", None)
        if close is not None:
            close()


class ReviewReuseError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _translate_store_error(exc: ReviewReuseStoreError) -> ReviewReuseError:
    if exc.code in {
        "store_index_corrupt",
        "store_record_corrupt",
        "store_writer_conflict",
    }:
        logger.error("review_reuse_store_failure code=%s", exc.code)
    return ReviewReuseError(exc.code, exc.message)


def _normalized_idempotency_key(key: Optional[str]) -> Optional[str]:
    if key is None:
        return None
    if not isinstance(key, str):
        raise ReviewReuseError("invalid_request", "idempotency key is invalid")
    normalized = key.strip()
    if (
        not normalized
        or len(normalized) > 128
        or any(not char.isprintable() for char in normalized)
    ):
        raise ReviewReuseError("invalid_request", "idempotency key is invalid")
    return normalized


def _evidence_pack_from_task(
    task: ReviewReuseTask, *, as_markdown: bool
) -> Tuple[Dict[str, Any], Optional[str]]:
    if not task.evidence_pack:
        raise ReviewReuseError("not_ready", "evidence pack not ready")
    try:
        digest_valid = evidence_pack_digest_is_valid(task.evidence_pack)
    except (CanonicalJSONError, TypeError, ValueError) as exc:
        raise ReviewReuseError(
            "store_record_corrupt", "persisted EvidencePack digest is invalid"
        ) from exc
    if not digest_valid:
        raise ReviewReuseError(
            "store_record_corrupt", "persisted EvidencePack digest is invalid"
        )
    markdown = evidence_pack_markdown(task.evidence_pack) if as_markdown else None
    return task.evidence_pack, markdown


class ReviewReuseService:
    def __init__(self, store: ReviewReuseStoreProtocol) -> None:
        self.store = store

    def create_task(
        self,
        *,
        tenant_id: str,
        file_name: str,
        file_bytes: bytes,
        idempotency_key: Optional[str] = None,
        seed_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> ReviewReuseTask:
        try:
            validate_tenant_id(tenant_id)
        except ReviewReuseStoreError as exc:
            raise _translate_store_error(exc) from exc
        if not file_bytes:
            raise ReviewReuseError("empty_input", "DXF input must not be empty")
        if len(file_bytes) > max_upload_bytes():
            raise ReviewReuseError(
                "input_too_large", "DXF input exceeds the upload limit"
            )
        display_name = file_name or "upload.bin"
        if Path(display_name).suffix.lower() != ".dxf":
            raise ReviewReuseError(
                "unsupported_file_type", "ReviewReuse MVP accepts only .dxf input"
            )

        key = _normalized_idempotency_key(idempotency_key)
        content_sha = hashlib.sha256(file_bytes).hexdigest()
        create_digest = (
            canonical_sha256(
                {
                    "tenant_id": tenant_id,
                    "source_content_sha256": content_sha,
                }
            )
            if key is not None
            else None
        )
        now = time.time()
        task = ReviewReuseTask(
            task_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            status=TaskStatus.pending,
            created_at=now,
            updated_at=now,
            source_file_name=display_name,
            source_content_sha256=content_sha,
            idempotency_key=key,
            idempotency_digest=create_digest,
            revision=1,
            trace_id=str(uuid.uuid4()),
            events=[],
        )
        task = self._emit(task, TaskEventType.submitted, {"file_name": display_name})
        task = self._emit(
            task, TaskEventType.input_validated, {"bytes": len(file_bytes)}
        )
        task.status = TaskStatus.running
        task.updated_at = max(time.time(), task.updated_at)
        proposed_task_id = task.task_id
        try:
            task = self.store.create_if_absent(task, key=key, digest=create_digest)
        except ReviewReuseStoreError as exc:
            raise _translate_store_error(exc) from exc
        if task.task_id != proposed_task_id:
            return task

        base_revision = task.revision
        try:
            task = self._emit(task, TaskEventType.recall_started, {})
            candidates = recall_candidates(
                file_name=display_name,
                file_bytes=file_bytes,
                content_sha=content_sha,
                seed=seed_candidates,
            )
            task.candidates = candidates
            task = self._emit(
                task,
                TaskEventType.recall_completed,
                {"count": len(candidates)},
            )
            task = self._emit(task, TaskEventType.precision_started, {})
            task = self._emit(
                task,
                TaskEventType.precision_completed,
                {"count": len(candidates)},
            )
            task.revision = base_revision + 1
            task.status = TaskStatus.evidence_ready
            task.evidence_pack = build_evidence_pack(task)
            task = self._emit(
                task,
                TaskEventType.evidence_pack_ready,
                {"candidates": len(candidates)},
            )
            validate_review_reuse_task_payload(task)
        except Exception as exc:
            failed = task.model_copy(deep=True)
            if (
                failed.events
                and failed.events[-1].event_type == TaskEventType.evidence_pack_ready
            ):
                failed.events = failed.events[:-1]
            failed.candidates = []
            failed.evidence_pack = None
            failed.revision = base_revision + 1
            failed.status = TaskStatus.failed
            failed.error_code = "internal_error"
            failed.error = "ReviewReuse evidence generation failed"
            failed = self._emit(
                failed, TaskEventType.failed, {"code": "internal_error"}
            )
            try:
                self.store.put(failed, expected_revision=base_revision)
            except ReviewReuseStoreError as store_exc:
                raise _translate_store_error(store_exc) from store_exc
            raise ReviewReuseError(
                "internal_error", "ReviewReuse evidence generation failed"
            ) from exc

        try:
            return self.store.put(task, expected_revision=base_revision)
        except ReviewReuseStoreError as exc:
            raise _translate_store_error(exc) from exc

    def get_task(self, tenant_id: str, task_id: str) -> ReviewReuseTask:
        try:
            task = self.store.get(tenant_id, task_id)
        except ReviewReuseStoreError as exc:
            raise _translate_store_error(exc) from exc
        if task is None:
            raise ReviewReuseError(
                "not_found", f"task {task_id!r} not found for tenant"
            )
        return task

    def list_tasks(self, tenant_id: str) -> List[ReviewReuseTask]:
        try:
            tasks = self.store.list_for_tenant(tenant_id)
        except ReviewReuseStoreError as exc:
            raise _translate_store_error(exc) from exc
        return sorted(tasks, key=lambda task: task.created_at, reverse=True)

    def cancel(self, tenant_id: str, task_id: str) -> ReviewReuseTask:
        task = self.get_task(tenant_id, task_id)
        if task.status == TaskStatus.canceled:
            return task
        if task.status in (TaskStatus.decided, TaskStatus.failed):
            raise ReviewReuseError(
                "invalid_state_transition", "terminal task cannot be canceled"
            )
        expected_revision = task.revision
        task.revision = expected_revision + 1
        task.status = TaskStatus.canceled
        task = self._emit(task, TaskEventType.canceled, {})
        if task.evidence_pack is not None:
            task.evidence_pack = build_evidence_pack(task)
        try:
            return self.store.put(task, expected_revision=expected_revision)
        except ReviewReuseStoreError as exc:
            raise _translate_store_error(exc) from exc

    def get_events(self, tenant_id: str, task_id: str) -> List[TaskEvent]:
        return list(self.get_task(tenant_id, task_id).events)

    def get_evidence_pack(
        self, tenant_id: str, task_id: str, *, as_markdown: bool = False
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        task = self.get_task(tenant_id, task_id)
        return _evidence_pack_from_task(task, as_markdown=as_markdown)

    def metrics(self, tenant_id: str) -> Dict[str, Any]:
        try:
            return compute_review_metrics(self.store, tenant_id)
        except ReviewReuseStoreError as exc:
            raise _translate_store_error(exc) from exc

    def export_audit_bundle(self, tenant_id: str, task_id: str) -> Dict[str, Any]:
        """Return a quarantined audit export, never a training manifest."""

        task = self.get_task(tenant_id, task_id)
        pack: Optional[Dict[str, Any]] = None
        markdown = ""
        if task.evidence_pack is not None:
            pack, rendered = _evidence_pack_from_task(task, as_markdown=True)
            markdown = rendered or ""
        return {
            "schema_version": "review-reuse-audit-bundle-v1",
            "export_kind": "audit_quarantine",
            "task": task.model_dump(mode="json"),
            "events": [event.model_dump(mode="json") for event in task.events],
            "evidence_pack": pack,
            "evidence_pack_markdown": markdown,
        }

    def submit_decision(
        self,
        *,
        tenant_id: str,
        task_id: str,
        state: HumanDecisionState,
        reviewer_id: str,
        reviewer_kind: str,
        tenant_validated: bool,
        reviewer_validated: bool,
        expected_revision: int,
        evidence_pack_sha256: str,
        reason_codes: Optional[List[str]] = None,
        reason_text: str = "",
        candidate_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> ReviewReuseTask:
        if not decisions_enabled():
            raise ReviewReuseError(
                "decisions_disabled",
                "REVIEW_REUSE_DECISIONS_ENABLED is off; owner enable is required",
            )
        if not tenant_validated:
            raise ReviewReuseError(
                "tenant_not_validated", "decision requires a validated tenant claim"
            )
        if (
            not reviewer_validated
            or reviewer_kind != "validated_principal"
            or not isinstance(reviewer_id, str)
            or not _PRINCIPAL_PATTERN.fullmatch(reviewer_id)
        ):
            raise ReviewReuseError(
                "reviewer_not_validated",
                "decision requires a validated reviewer principal",
            )
        if (
            isinstance(expected_revision, bool)
            or not isinstance(expected_revision, int)
            or expected_revision < 1
            or not isinstance(evidence_pack_sha256, str)
            or not _HEX64_PATTERN.fullmatch(evidence_pack_sha256)
        ):
            raise ReviewReuseError(
                "invalid_request", "decision revision binding is invalid"
            )
        try:
            state = HumanDecisionState(state)
        except (TypeError, ValueError) as exc:
            raise ReviewReuseError(
                "invalid_decision", "decision state is invalid"
            ) from exc
        if candidate_id is not None and not isinstance(candidate_id, str):
            raise ReviewReuseError("invalid_decision", "candidate id is invalid")
        if reason_codes is not None and (
            not isinstance(reason_codes, list)
            or any(not isinstance(code, str) for code in reason_codes)
        ):
            raise ReviewReuseError("invalid_decision", "decision reasons are invalid")
        if not isinstance(reason_text, str):
            raise ReviewReuseError("invalid_decision", "decision rationale is invalid")

        key = _normalized_idempotency_key(idempotency_key)
        normalized_candidate = candidate_id.strip() if candidate_id else None
        normalized_reasons = sorted(
            {code.strip() for code in (reason_codes or []) if code.strip()}
        )
        normalized_text = (reason_text or "").strip()
        decision_digest = canonical_sha256(
            {
                "tenant_id": tenant_id,
                "task_id": task_id,
                "state": state.value,
                "candidate_id": normalized_candidate,
                "reason_codes": normalized_reasons,
                "reason_text": normalized_text,
                "reviewer_id": reviewer_id,
                "reviewer_kind": reviewer_kind,
                "expected_revision": expected_revision,
                "evidence_pack_sha256": evidence_pack_sha256,
            }
        )
        task = self.get_task(tenant_id, task_id)
        if task.human_decision is not None:
            existing = task.human_decision
            if key is not None and existing.idempotency_key == key:
                if existing.idempotency_digest == decision_digest:
                    return task
                raise ReviewReuseError(
                    "idempotency_key_conflict",
                    "idempotency key is bound to another decision payload",
                )
            raise ReviewReuseError(
                "already_decided", "task already has a human decision"
            )
        if task.status != TaskStatus.evidence_ready:
            raise ReviewReuseError(
                "invalid_state_transition", "only evidence-ready tasks may be decided"
            )
        try:
            digest_valid = (
                task.evidence_pack is not None
                and evidence_pack_digest_is_valid(task.evidence_pack)
            )
        except (CanonicalJSONError, TypeError, ValueError) as exc:
            raise ReviewReuseError(
                "store_record_corrupt", "persisted EvidencePack digest is invalid"
            ) from exc
        if task.evidence_pack is not None and not digest_valid:
            raise ReviewReuseError(
                "store_record_corrupt", "persisted EvidencePack digest is invalid"
            )
        if (
            task.revision != expected_revision
            or task.evidence_pack is None
            or task.evidence_pack.get("evidence_pack_sha256") != evidence_pack_sha256
        ):
            raise ReviewReuseError(
                "revision_conflict", "task revision or EvidencePack digest has changed"
            )
        self._validate_decision(
            task=task,
            state=state,
            candidate_id=normalized_candidate,
            reason_codes=normalized_reasons,
            reason_text=normalized_text,
        )

        decision = HumanDecision(
            state=state,
            reviewer_id=reviewer_id,
            reviewer_kind=reviewer_kind,
            reason_codes=normalized_reasons,
            reason_text=normalized_text,
            candidate_id=normalized_candidate,
            ts=max(time.time(), task.updated_at),
            idempotency_key=key,
            idempotency_digest=decision_digest if key is not None else None,
            reviewed_revision=expected_revision,
            evidence_pack_sha256=evidence_pack_sha256,
        )
        task.human_decision = decision
        task.status = TaskStatus.decided
        task.revision = expected_revision + 1
        try:
            task.evidence_pack = build_evidence_pack(task)
        except (CanonicalJSONError, ValueError, TypeError) as exc:
            raise ReviewReuseError(
                "internal_error", "decision evidence could not be materialized"
            ) from exc
        task = self._emit(
            task,
            TaskEventType.decision_submitted,
            {
                "state": state.value,
                "reviewer_id": decision.reviewer_id,
                "candidate_id": normalized_candidate,
                "reviewed_revision": expected_revision,
                "evidence_pack_sha256": evidence_pack_sha256,
            },
        )
        try:
            return self.store.put(task, expected_revision=expected_revision)
        except ReviewReuseStoreError as exc:
            raise _translate_store_error(exc) from exc

    def _validate_decision(
        self,
        *,
        task: ReviewReuseTask,
        state: HumanDecisionState,
        candidate_id: Optional[str],
        reason_codes: List[str],
        reason_text: str,
    ) -> None:
        if any(code not in DECISION_REASON_CODES for code in reason_codes):
            raise ReviewReuseError("invalid_decision", "unknown decision reason code")
        if not reason_codes and not reason_text:
            raise ReviewReuseError("invalid_decision", "decision rationale is required")
        if "other" in reason_codes and not reason_text:
            raise ReviewReuseError("invalid_decision", "other requires rationale text")
        candidate_ids = {candidate.candidate_id for candidate in task.candidates}
        if state in _CANDIDATE_REQUIRED:
            if candidate_id is None or candidate_id not in candidate_ids:
                raise ReviewReuseError(
                    "invalid_decision", "decision requires a task candidate"
                )
        elif state == HumanDecisionState.new:
            if candidate_id is not None:
                raise ReviewReuseError(
                    "invalid_decision", "new decision cannot select a candidate"
                )
        elif state == HumanDecisionState.need_more_info:
            if candidate_id is not None and candidate_id not in candidate_ids:
                raise ReviewReuseError(
                    "invalid_decision", "selected candidate does not exist"
                )

    def _emit(
        self, task: ReviewReuseTask, event_type: TaskEventType, detail: Dict[str, Any]
    ) -> ReviewReuseTask:
        event_time = max(time.time(), task.updated_at)
        task.events = list(task.events) + [
            TaskEvent(event_type=event_type, ts=event_time, detail=detail)
        ]
        task.updated_at = event_time
        return task
