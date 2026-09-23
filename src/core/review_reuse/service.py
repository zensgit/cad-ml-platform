"""ReviewReuseService — task lifecycle wrapping deterministic candidate evidence."""

from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from .dedup_adapter import recall_candidates
from .evidence import build_evidence_pack, evidence_pack_markdown
from .files import is_allowed_review_reuse_filename
from .metrics import compute_review_metrics
from .models import (
    HumanDecision,
    HumanDecisionState,
    RejectionReason,
    ReviewReuseTask,
    TaskEvent,
    TaskEventType,
    TaskStatus,
)
from .precision import apply_precision, strip_transient_candidate_geom
from .store import (
    CorruptIdempotencyIndexError,
    OccupiedTenantDirError,
    ReviewReuseStoreProtocol,
    create_review_reuse_store,
)

logger = logging.getLogger(__name__)

# Default-off human decision sink (plan §8).
ENV_DECISIONS_ENABLED = "REVIEW_REUSE_DECISIONS_ENABLED"
# When on (pilot): reject ak-user-* / empty reviewer ids (require JWT subject).
ENV_REQUIRE_VALIDATED_REVIEWER = "REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER"
_TRUE = frozenset({"1", "true", "yes", "on"})
PIPELINE_FAILED_PUBLIC = "review-reuse pipeline failed"
# Live recall bound is 120s; a running snapshot older than this is a crash.
STALE_RUNNING_SECONDS = 180.0


def _pipeline_still_open(task: ReviewReuseTask) -> bool:
    """Running, or a mid-flight decision that never reached evidence."""
    if task.error:
        return False
    if task.status == TaskStatus.running:
        return True
    if task.status != TaskStatus.decided:
        return False
    return not any(
        event.event_type == TaskEventType.evidence_pack_ready for event in task.events
    )


_STORE: Optional[ReviewReuseStoreProtocol] = None


def decisions_enabled() -> bool:
    return os.getenv(ENV_DECISIONS_ENABLED, "").strip().lower() in _TRUE


def require_validated_reviewer() -> bool:
    return os.getenv(ENV_REQUIRE_VALIDATED_REVIEWER, "").strip().lower() in _TRUE


def is_api_key_fallback_reviewer(reviewer_id: str) -> bool:
    rid = (reviewer_id or "").strip()
    return rid.startswith("ak-user-") or rid in ("", "anonymous", "unknown")


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
        if not is_allowed_review_reuse_filename(file_name):
            raise ReviewReuseError(
                RejectionReason.unsupported_file_type.value,
                "file type is not a supported drawing or raster for ReviewReuse",
            )
        if idempotency_key:
            try:
                existing = self.store.get_by_idempotency(tenant_id, idempotency_key)
            except CorruptIdempotencyIndexError as exc:
                raise ReviewReuseError(
                    "store_conflict",
                    "hashed idempotency index is unreadable",
                ) from exc
            if existing is not None:
                return self._return_idempotent(
                    existing,
                    file_name=file_name,
                    file_bytes=file_bytes,
                    seed_candidates=seed_candidates,
                )

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
        task.pipeline_claim_id = str(uuid.uuid4())
        task.updated_at = time.time()
        try:
            stored = self.store.put_new_idempotent(task)
        except OccupiedTenantDirError as exc:
            raise ReviewReuseError(
                "store_conflict",
                "hashed tenant directory is occupied by another tenant",
            ) from exc
        except CorruptIdempotencyIndexError as exc:
            raise ReviewReuseError(
                "store_conflict",
                "hashed idempotency index is unreadable",
            ) from exc
        if stored.task_id != task.task_id:
            return self._return_idempotent(
                stored,
                file_name=file_name,
                file_bytes=file_bytes,
                seed_candidates=seed_candidates,
            )
        task = stored

        try:
            return self._run_pipeline(
                task,
                file_name=file_name,
                file_bytes=file_bytes,
                content_sha=content_sha,
                seed_candidates=seed_candidates,
            )
        except OccupiedTenantDirError as exc:
            raise ReviewReuseError(
                "store_conflict",
                "hashed tenant directory is occupied by another tenant",
            ) from exc
        except CorruptIdempotencyIndexError as exc:
            raise ReviewReuseError(
                "store_conflict",
                "hashed idempotency index is unreadable",
            ) from exc
        except ReviewReuseError:
            raise
        except Exception as exc:
            logger.warning("review_reuse_pipeline_failed", exc_info=True)
            task.status = TaskStatus.failed
            # Persist a public message only; GET/audit must not leak str(exc)
            # or live unscoped candidate geometry.
            strip_transient_candidate_geom(task.candidates)
            task.error = PIPELINE_FAILED_PUBLIC
            task = self._emit(
                task, TaskEventType.failed, {"error": PIPELINE_FAILED_PUBLIC}
            )
            try:
                self._commit_pipeline_result(task)
            except Exception:
                logger.warning("review_reuse_failed_task_persist_failed", exc_info=True)
            raise ReviewReuseError(
                "pipeline_failed",
                PIPELINE_FAILED_PUBLIC,
            ) from exc

    def _return_idempotent(
        self,
        existing: ReviewReuseTask,
        *,
        file_name: str,
        file_bytes: bytes,
        seed_candidates: Optional[List[Dict[str, Any]]],
    ) -> ReviewReuseTask:
        """Replay a stored idempotent task; resume stale ``running`` snapshots."""
        content_sha = self._require_idempotent_input(
            existing, file_name=file_name, file_bytes=file_bytes
        )
        if existing.status == TaskStatus.failed or (
            existing.status in (TaskStatus.decided, TaskStatus.canceled)
            and existing.error
        ):
            raise ReviewReuseError(
                "pipeline_failed",
                existing.error or PIPELINE_FAILED_PUBLIC,
            )
        if not _pipeline_still_open(existing):
            return existing
        age = time.time() - float(existing.updated_at or 0.0)
        if age < STALE_RUNNING_SECONDS:
            return existing
        existing, claimed = self._claim_stale_running(existing)
        if not claimed:
            return existing
        try:
            return self._run_pipeline(
                existing,
                file_name=file_name or existing.source_file_name,
                file_bytes=file_bytes,
                content_sha=content_sha,
                seed_candidates=seed_candidates,
            )
        except OccupiedTenantDirError as exc:
            raise ReviewReuseError(
                "store_conflict",
                "hashed tenant directory is occupied by another tenant",
            ) from exc
        except CorruptIdempotencyIndexError as exc:
            raise ReviewReuseError(
                "store_conflict",
                "hashed idempotency index is unreadable",
            ) from exc
        except ReviewReuseError:
            raise
        except Exception as exc:
            logger.warning("review_reuse_pipeline_failed", exc_info=True)
            existing.status = TaskStatus.failed
            strip_transient_candidate_geom(existing.candidates)
            existing.error = PIPELINE_FAILED_PUBLIC
            existing = self._emit(
                existing, TaskEventType.failed, {"error": PIPELINE_FAILED_PUBLIC}
            )
            try:
                self._commit_pipeline_result(existing)
            except Exception:
                logger.warning("review_reuse_failed_task_persist_failed", exc_info=True)
            raise ReviewReuseError(
                "pipeline_failed",
                PIPELINE_FAILED_PUBLIC,
            ) from exc

    def _require_idempotent_input(
        self,
        existing: ReviewReuseTask,
        *,
        file_name: str,
        file_bytes: bytes,
    ) -> str:
        """Resume only when retry bytes/name match the stored reservation."""
        content_sha = hashlib.sha256(file_bytes).hexdigest()
        stored_sha = (existing.source_content_sha256 or "").strip().lower()
        if stored_sha != content_sha.lower():
            raise ReviewReuseError(
                "idempotency_conflict",
                "idempotency key is bound to a different source file",
            )
        stored_name = existing.source_file_name or ""
        retry_name = file_name or stored_name
        if stored_name and retry_name != stored_name:
            raise ReviewReuseError(
                "idempotency_conflict",
                "idempotency key is bound to a different source file",
            )
        return content_sha

    def _claim_stale_running(
        self, existing: ReviewReuseTask
    ) -> Tuple[ReviewReuseTask, bool]:
        """CAS-lease a stale running snapshot so only one worker resumes."""
        claim_id = str(uuid.uuid4())
        claim_at = time.time()

        def updater(current: Optional[ReviewReuseTask]) -> ReviewReuseTask:
            task = current if current is not None else existing
            if not _pipeline_still_open(task):
                return task
            age = time.time() - float(task.updated_at or 0.0)
            if age < STALE_RUNNING_SECONDS:
                return task
            task.updated_at = claim_at
            task.pipeline_claim_id = claim_id
            return task

        updated = self._update_atomically(
            existing.tenant_id, existing.task_id, updater
        )
        won = (
            _pipeline_still_open(updated) and updated.pipeline_claim_id == claim_id
        )
        return updated, won

    def _owns_pipeline_claim(
        self, stored: ReviewReuseTask, claim_id: Optional[str]
    ) -> bool:
        return (
            bool(claim_id)
            and stored.status == TaskStatus.running
            and stored.pipeline_claim_id == claim_id
        )

    def _renew_pipeline_claim(self, task: ReviewReuseTask) -> ReviewReuseTask:
        """Refresh the running lease; return the stored snapshot if stolen."""
        claim_id = task.pipeline_claim_id
        if not claim_id:
            return task
        now = time.time()

        def updater(current: Optional[ReviewReuseTask]) -> ReviewReuseTask:
            stored = current if current is not None else task
            if stored.pipeline_claim_id != claim_id:
                return stored
            if not _pipeline_still_open(stored):
                return stored
            stored.updated_at = now
            return stored

        stored = self._update_atomically(task.tenant_id, task.task_id, updater)
        if stored.pipeline_claim_id != claim_id:
            return stored
        if stored.status == TaskStatus.running:
            task.updated_at = stored.updated_at
        return task

    def _abort_if_pipeline_lost(
        self, task: ReviewReuseTask, claim_id: Optional[str]
    ) -> Optional[ReviewReuseTask]:
        """Return the stored snapshot when the lease was stolen or canceled."""
        renewed = self._renew_pipeline_claim(task)
        stored = self.store.get(task.tenant_id, task.task_id)
        snapshot = stored if stored is not None else renewed
        if claim_id and not self._owns_pipeline_claim(snapshot, claim_id):
            if snapshot.status == TaskStatus.canceled:
                return snapshot
            if snapshot.pipeline_claim_id != claim_id:
                return snapshot
        return None

    def _run_pipeline(
        self,
        task: ReviewReuseTask,
        *,
        file_name: str,
        file_bytes: bytes,
        content_sha: str,
        seed_candidates: Optional[List[Dict[str, Any]]],
    ) -> ReviewReuseTask:
        # Pipeline: recall → precision → evidence (adapter; no training path).
        claim_id = task.pipeline_claim_id
        aborted = self._abort_if_pipeline_lost(task, claim_id)
        if aborted is not None:
            return aborted
        task = self._emit(task, TaskEventType.recall_started, {})
        candidates = recall_candidates(
            file_name=file_name,
            file_bytes=file_bytes,
            content_sha=content_sha,
            seed=seed_candidates,
        )
        aborted = self._abort_if_pipeline_lost(task, claim_id)
        if aborted is not None:
            return aborted
        task.candidates = candidates
        task = self._emit(
            task,
            TaskEventType.recall_completed,
            {"count": len(candidates)},
        )
        aborted = self._abort_if_pipeline_lost(task, claim_id)
        if aborted is not None:
            return aborted
        task = self._emit(task, TaskEventType.precision_started, {})
        candidates = apply_precision(
            candidates, file_name=file_name, file_bytes=file_bytes
        )
        aborted = self._abort_if_pipeline_lost(task, claim_id)
        if aborted is not None:
            return aborted
        task.candidates = candidates
        vision_only = sum(
            1
            for c in candidates
            if RejectionReason.vision_only_unverified.value in c.rejection_reasons
        )
        l4 = sum(
            1
            for c in candidates
            if "precision-l4" in list((c.verification or {}).get("methods") or [])
        )
        task = self._emit(
            task,
            TaskEventType.precision_completed,
            {
                "count": len(candidates),
                "precision_l4": l4,
                "vision_only_unverified": vision_only,
            },
        )
        aborted = self._abort_if_pipeline_lost(task, claim_id)
        if aborted is not None:
            return aborted
        pack = build_evidence_pack(task)
        task.evidence_pack = pack
        task.status = TaskStatus.evidence_ready
        task = self._emit(
            task, TaskEventType.evidence_pack_ready, {"candidates": len(candidates)}
        )
        return self._commit_pipeline_result(task)

    def _commit_pipeline_result(self, task: ReviewReuseTask) -> ReviewReuseTask:
        """Do not overwrite a concurrent cancel/decision with pipeline completion."""

        def updater(current: Optional[ReviewReuseTask]) -> ReviewReuseTask:
            if (
                current is not None
                and task.pipeline_claim_id
                and current.pipeline_claim_id != task.pipeline_claim_id
            ):
                # Another worker holds the lease; do not mask their result.
                return current
            if current is not None and current.status == TaskStatus.canceled:
                # A cancel that lands after the last abort check must not
                # receive candidates, an EvidencePack, or completion events.
                return current
            if current is not None and current.status == TaskStatus.decided:
                merged = False
                if task.candidates and not current.candidates:
                    current.candidates = task.candidates
                    merged = True
                if merged:
                    # Mid-flight decision built an empty pack; rebuild so
                    # audit export matches stored candidates + decision.
                    current.evidence_pack = build_evidence_pack(current)
                elif (
                    task.evidence_pack is not None and current.evidence_pack is None
                ):
                    current.evidence_pack = task.evidence_pack
                    merged = True
                before_events = len(current.events)
                current.events = _merge_append_only_events(
                    current.events, task.events
                )
                # Keep a concurrent cancel/decision, but do not drop the
                # pipeline failure: GET/audit must still show the error.
                if task.error and not current.error:
                    current.error = task.error
                    merged = True
                if merged or len(current.events) != before_events:
                    current.updated_at = time.time()
                return current
            return task

        return self._update_atomically(task.tenant_id, task.task_id, updater)

    def _update_atomically(
        self,
        tenant_id: str,
        task_id: str,
        updater: Callable[[Optional[ReviewReuseTask]], ReviewReuseTask],
    ) -> ReviewReuseTask:
        try:
            return self.store.update_atomically(tenant_id, task_id, updater)
        except OccupiedTenantDirError as exc:
            raise ReviewReuseError(
                "store_conflict",
                "hashed tenant directory is occupied by another tenant",
            ) from exc
        except CorruptIdempotencyIndexError as exc:
            raise ReviewReuseError(
                "store_conflict",
                "hashed idempotency index is unreadable",
            ) from exc

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
        def updater(current: Optional[ReviewReuseTask]) -> ReviewReuseTask:
            if current is None:
                raise ReviewReuseError(
                    "not_found", f"task {task_id!r} not found for tenant"
                )
            if current.status in (
                TaskStatus.decided,
                TaskStatus.canceled,
                TaskStatus.failed,
            ):
                return current
            current.status = TaskStatus.canceled
            return self._emit(current, TaskEventType.canceled, {})

        return self._update_atomically(tenant_id, task_id, updater)

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

    def export_audit_bundle(self, tenant_id: str, task_id: str) -> Dict[str, Any]:
        """Machine-readable audit export (not a training manifest).

        Contains task snapshot, events, EvidencePack JSON + markdown. Does not
        write feedback JSONL or any training-readable path (R2 HOLD).
        """
        task = self.get_task(tenant_id, task_id)
        pack, md = self.get_evidence_pack(tenant_id, task_id, as_markdown=True)
        return {
            "schema_version": "review-reuse-audit-bundle-v1",
            "export_kind": "audit_quarantine",  # not training-readable
            "task": task.model_dump(mode="json"),
            "events": [e.model_dump(mode="json") for e in task.events],
            "evidence_pack": pack,
            "evidence_pack_markdown": md or "",
        }

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
        reviewer_validated: bool = False,
    ) -> ReviewReuseTask:
        if not decisions_enabled():
            raise ReviewReuseError(
                "decisions_disabled",
                "REVIEW_REUSE_DECISIONS_ENABLED is off (default). Owner enable required for pilot.",
            )
        if not reviewer_id or not str(reviewer_id).strip():
            raise ReviewReuseError(
                "reviewer_required",
                "reviewer_id must come from validated identity",
            )
        if require_validated_reviewer() and (
            not reviewer_validated or is_api_key_fallback_reviewer(reviewer_id)
        ):
            raise ReviewReuseError(
                "reviewer_not_validated",
                "REVIEW_REUSE_REQUIRE_VALIDATED_REVIEWER requires JWT/integration subject "
                "(not API-key fallback ak-user-*).",
            )

        def updater(current: Optional[ReviewReuseTask]) -> ReviewReuseTask:
            if current is None:
                raise ReviewReuseError(
                    "not_found", f"task {task_id!r} not found for tenant"
                )
            if current.status == TaskStatus.canceled:
                raise ReviewReuseError("canceled", "cannot decide a canceled task")
            if current.status == TaskStatus.failed:
                raise ReviewReuseError("failed", "cannot decide a failed task")
            if candidate_id:
                known = {c.candidate_id for c in current.candidates}
                if candidate_id not in known:
                    raise ReviewReuseError(
                        "unknown_candidate",
                        f"candidate_id {candidate_id!r} is not on this task",
                    )
            if current.human_decision is not None:
                # Idempotent: same key returns existing; different is conflict.
                if (
                    idempotency_key
                    and current.human_decision.idempotency_key == idempotency_key
                ):
                    return current
                raise ReviewReuseError(
                    "already_decided", "task already has a human decision"
                )

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
            current.human_decision = decision
            current.status = TaskStatus.decided
            # Refresh evidence pack with decision (and any candidates already stored).
            current.evidence_pack = build_evidence_pack(current)
            return self._emit(
                current,
                TaskEventType.decision_submitted,
                {
                    "state": state.value,
                    "reviewer_id": decision.reviewer_id,
                    "candidate_id": candidate_id,
                },
            )

        return self._update_atomically(tenant_id, task_id, updater)

    def _emit(
        self, task: ReviewReuseTask, event_type: TaskEventType, detail: Dict[str, Any]
    ) -> ReviewReuseTask:
        task.events = list(task.events) + [
            TaskEvent(event_type=event_type, ts=time.time(), detail=detail)
        ]
        task.updated_at = time.time()
        return task


def _merge_append_only_events(
    base: List[TaskEvent], incoming: List[TaskEvent]
) -> List[TaskEvent]:
    """Keep stored terminal events and fill in pipeline steps by type + ts."""
    seen = {(e.event_type, e.ts) for e in base}
    have_types = {e.event_type for e in base}
    extra: List[TaskEvent] = []
    for event in incoming:
        if (event.event_type, event.ts) in seen:
            continue
        if event.event_type in have_types:
            continue
        extra.append(event)
        seen.add((event.event_type, event.ts))
        have_types.add(event.event_type)
    merged = list(base) + extra
    merged.sort(key=lambda e: e.ts)
    return merged


