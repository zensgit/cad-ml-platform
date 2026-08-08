"""ReviewReuseService — task lifecycle wrapping deterministic candidate evidence."""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

from .dedup_adapter import recall_candidates
from .evidence import build_evidence_pack, evidence_pack_markdown
from .metrics import compute_review_metrics
from .models import (
    HumanDecision,
    HumanDecisionState,
    ReviewReuseTask,
    TaskEvent,
    TaskEventType,
    TaskStatus,
)
from .store import ReviewReuseStoreProtocol, create_review_reuse_store

# Default-off human decision sink (plan §8).
ENV_DECISIONS_ENABLED = "REVIEW_REUSE_DECISIONS_ENABLED"
_TRUE = frozenset({"1", "true", "yes", "on"})

_STORE: Optional[ReviewReuseStoreProtocol] = None


def decisions_enabled() -> bool:
    return os.getenv(ENV_DECISIONS_ENABLED, "").strip().lower() in _TRUE


def get_review_reuse_store() -> ReviewReuseStoreProtocol:
    global _STORE
    if _STORE is None:
        _STORE = create_review_reuse_store()
    return _STORE


def reset_review_reuse_store_for_tests(
    store: Optional[ReviewReuseStoreProtocol] = None,
) -> None:
    """Test helper: replace process singleton store."""
    global _STORE
    _STORE = store if store is not None else create_review_reuse_store()


def get_review_reuse_service() -> "ReviewReuseService":
    return ReviewReuseService(get_review_reuse_store())


class ReviewReuseError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


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
        if not tenant_id or not str(tenant_id).strip():
            raise ReviewReuseError("tenant_required", "tenant_id is required")
        if idempotency_key:
            existing = self.store.get_by_idempotency(tenant_id, idempotency_key)
            if existing is not None:
                return existing

        now = time.time()
        task_id = str(uuid.uuid4())
        trace_id = str(uuid.uuid4())
        content_sha = hashlib.sha256(file_bytes).hexdigest()
        task = ReviewReuseTask(
            task_id=task_id,
            tenant_id=tenant_id,
            status=TaskStatus.pending,
            created_at=now,
            updated_at=now,
            source_file_name=file_name or "upload.bin",
            source_content_sha256=content_sha,
            idempotency_key=idempotency_key,
            trace_id=trace_id,
            events=[],
        )
        task = self._emit(task, TaskEventType.submitted, {"file_name": task.source_file_name})
        task = self._emit(task, TaskEventType.input_validated, {"bytes": len(file_bytes)})
        task.status = TaskStatus.running
        task.updated_at = time.time()
        self.store.put(task)

        # Pipeline: recall → precision → evidence (adapter; no training path).
        task = self._emit(task, TaskEventType.recall_started, {})
        candidates = recall_candidates(
            file_name=file_name,
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
        pack = build_evidence_pack(task)
        task.evidence_pack = pack
        task.status = TaskStatus.evidence_ready
        task = self._emit(task, TaskEventType.evidence_pack_ready, {"candidates": len(candidates)})
        self.store.put(task)
        return task

    def get_task(self, tenant_id: str, task_id: str) -> ReviewReuseTask:
        task = self.store.get(tenant_id, task_id)
        if task is None:
            raise ReviewReuseError("not_found", f"task {task_id!r} not found for tenant")
        return task

    def list_tasks(self, tenant_id: str) -> List[ReviewReuseTask]:
        return sorted(
            self.store.list_for_tenant(tenant_id),
            key=lambda t: t.created_at,
            reverse=True,
        )

    def cancel(self, tenant_id: str, task_id: str) -> ReviewReuseTask:
        task = self.get_task(tenant_id, task_id)
        if task.status in (TaskStatus.decided, TaskStatus.canceled):
            return task
        task.status = TaskStatus.canceled
        task = self._emit(task, TaskEventType.canceled, {})
        self.store.put(task)
        return task

    def get_events(self, tenant_id: str, task_id: str) -> List[TaskEvent]:
        return list(self.get_task(tenant_id, task_id).events)

    def get_evidence_pack(
        self, tenant_id: str, task_id: str, *, as_markdown: bool = False
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        task = self.get_task(tenant_id, task_id)
        if not task.evidence_pack:
            raise ReviewReuseError("not_ready", "evidence pack not ready")
        md = evidence_pack_markdown(task.evidence_pack) if as_markdown else None
        return task.evidence_pack, md

    def metrics(self, tenant_id: str) -> Dict[str, Any]:
        return compute_review_metrics(self.store, tenant_id)

    def submit_decision(
        self,
        *,
        tenant_id: str,
        task_id: str,
        state: HumanDecisionState,
        reviewer_id: str,
        reason_codes: Optional[List[str]] = None,
        reason_text: str = "",
        candidate_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> ReviewReuseTask:
        if not decisions_enabled():
            raise ReviewReuseError(
                "decisions_disabled",
                "REVIEW_REUSE_DECISIONS_ENABLED is off (default). Owner enable required for pilot.",
            )
        if not reviewer_id or not str(reviewer_id).strip():
            raise ReviewReuseError("reviewer_required", "reviewer_id must come from validated identity")

        task = self.get_task(tenant_id, task_id)
        if task.status == TaskStatus.canceled:
            raise ReviewReuseError("canceled", "cannot decide a canceled task")
        if task.human_decision is not None:
            # Idempotent: same key returns existing; different is conflict.
            if (
                idempotency_key
                and task.human_decision.idempotency_key == idempotency_key
            ):
                return task
            raise ReviewReuseError("already_decided", "task already has a human decision")

        # Strategy-center states always allowed; extensions allowed but labeled.
        decision = HumanDecision(
            state=state,
            reviewer_id=str(reviewer_id).strip(),
            reason_codes=list(reason_codes or []),
            reason_text=reason_text or "",
            candidate_id=candidate_id,
            ts=time.time(),
            idempotency_key=idempotency_key,
        )
        task.human_decision = decision
        task.status = TaskStatus.decided
        # Refresh evidence pack with decision.
        task.evidence_pack = build_evidence_pack(task)
        task = self._emit(
            task,
            TaskEventType.decision_submitted,
            {
                "state": state.value,
                "reviewer_id": decision.reviewer_id,
                "candidate_id": candidate_id,
            },
        )
        self.store.put(task)
        return task

    def _emit(
        self, task: ReviewReuseTask, event_type: TaskEventType, detail: Dict[str, Any]
    ) -> ReviewReuseTask:
        task.events = list(task.events) + [
            TaskEvent(event_type=event_type, ts=time.time(), detail=detail)
        ]
        task.updated_at = time.time()
        return task


