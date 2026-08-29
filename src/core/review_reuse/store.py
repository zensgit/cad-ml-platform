"""Tenant-isolated ReviewReuse stores with atomic ledger mutations."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Tuple

from .canonical import (
    CanonicalJSONError,
    canonical_json_v1,
    canonical_sha256,
    strict_json_loads,
)
from .evidence import (
    build_evidence_pack,
    evidence_pack_digest,
    evidence_pack_digest_is_valid,
)
from .models import (
    DECISION_REASON_CODES,
    HumanDecisionState,
    RejectionReason,
    ReviewReuseTask,
    TaskEventType,
    TaskStatus,
)

ENV_STORE = "REVIEW_REUSE_STORE"
ENV_STORE_DIR = "REVIEW_REUSE_STORE_DIR"
_TRUE_BACKENDS_FS = frozenset({"fs", "file", "filesystem", "disk"})

_TENANT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_HEX64_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TENANT_SCHEMA = "review-reuse-tenant-v1"
_INDEX_SCHEMA = "review-reuse-idempotency-v1"
_TENANT_FIELDS = frozenset(
    {"schema_version", "tenant_id", "identity_source", "tenant_digest_sha256"}
)
_INDEX_FIELDS = frozenset(
    {"schema_version", "tenant_id", "tenant_digest_sha256", "entries"}
)
_INDEX_ENTRY_FIELDS = frozenset({"task_id", "payload_digest"})
_PRINCIPAL_PATTERN = re.compile(r"^principal-v1-[0-9a-f]{64}$")
_REJECTION_REASON_CODES = frozenset(reason.value for reason in RejectionReason)
_NEW_TENANT_DIR_PATTERN = re.compile(r"^tenant-v1-[0-9a-f]{64}$")
_TENANT_STAGE_PATTERN = re.compile(r"^\.tenant-v1-[0-9a-f]{64}\.stage-[0-9a-f]{32}$")
_METADATA_TEMP_PATTERN = re.compile(
    r"^\.(?:tenant|idempotency)\.json\.[a-z0-9_]{8}(?:\.tmp)?$"
)
_TASK_TEMP_PATTERN = re.compile(
    r"^\.[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{12}\.json\.[a-z0-9_]{8}(?:\.tmp)?$"
)
_RUNNING_PIPELINE_EVENTS = (
    TaskEventType.recall_started,
    TaskEventType.recall_completed,
    TaskEventType.precision_started,
    TaskEventType.precision_completed,
)
_TRANSITION_EVENT_SEQUENCES = {
    (TaskStatus.pending, TaskStatus.running): (TaskEventType.input_validated,),
    (TaskStatus.pending, TaskStatus.failed): (TaskEventType.failed,),
    (TaskStatus.pending, TaskStatus.canceled): (TaskEventType.canceled,),
    (TaskStatus.running, TaskStatus.evidence_ready): _RUNNING_PIPELINE_EVENTS
    + (TaskEventType.evidence_pack_ready,),
    (TaskStatus.running, TaskStatus.canceled): (TaskEventType.canceled,),
    (TaskStatus.evidence_ready, TaskStatus.decided): (
        TaskEventType.decision_submitted,
    ),
    (TaskStatus.evidence_ready, TaskStatus.canceled): (TaskEventType.canceled,),
}


class ReviewReuseStoreError(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _writer_lease_path(root: Path) -> Path:
    root_digest = hashlib.sha256(os.fsencode(str(root))).hexdigest()
    return root.parent / f".review-reuse-writer-{root_digest}.lock"


def _open_writer_lease(root: Path, *, message: str) -> Any:
    lease_path = _writer_lease_path(root)
    flags = os.O_RDWR | os.O_CREAT | os.O_NONBLOCK
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(lease_path, flags, 0o600)
    except OSError as exc:
        raise ReviewReuseStoreError("store_writer_conflict", message) from exc
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise OSError("writer lease is not a regular file")
        lease = os.fdopen(descriptor, "a+b")
    except Exception as exc:
        os.close(descriptor)
        raise ReviewReuseStoreError("store_writer_conflict", message) from exc
    try:
        fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (BlockingIOError, OSError) as exc:
        lease.close()
        raise ReviewReuseStoreError("store_writer_conflict", message) from exc
    return lease


def _validated_root_directories(store_root: Path) -> List[Path]:
    if not store_root.exists():
        return []
    if not store_root.is_dir():
        raise ReviewReuseStoreError(
            "store_record_corrupt", "filesystem store root is not a directory"
        )
    directories: List[Path] = []
    try:
        entries = sorted(store_root.iterdir())
    except OSError as exc:
        raise ReviewReuseStoreError(
            "store_record_corrupt", "filesystem store root is unreadable"
        ) from exc
    for entry in entries:
        if entry.is_symlink():
            raise ReviewReuseStoreError(
                "store_record_corrupt", "unexpected store root artifact"
            )
        if entry.is_dir():
            directories.append(entry)
            continue
        if entry.name == ".writer.lock" and entry.is_file():
            continue
        raise ReviewReuseStoreError(
            "store_record_corrupt", "unexpected store root artifact"
        )
    return directories


def _validate_tenant_directory_entries(store_root: Path, tenant_dir: Path) -> None:
    try:
        entries = sorted(tenant_dir.iterdir())
    except OSError as exc:
        raise ReviewReuseStoreError(
            "store_record_corrupt", "tenant ledger directory is unreadable"
        ) from exc
    for entry in entries:
        _assert_store_path(store_root, entry, code="store_record_corrupt")
        if entry.is_symlink():
            raise ReviewReuseStoreError(
                "store_record_corrupt", "unexpected tenant ledger artifact"
            )
        if entry.name == "tasks" and entry.is_dir():
            continue
        if entry.name in {"tenant.json", "idempotency.json"} and entry.is_file():
            continue
        if _METADATA_TEMP_PATTERN.fullmatch(entry.name) and entry.is_file():
            continue
        raise ReviewReuseStoreError(
            "store_record_corrupt", "unexpected tenant ledger artifact"
        )


def _validate_legacy_tenant_directory_entries(
    store_root: Path, tenant_dir: Path
) -> None:
    try:
        entries = sorted(tenant_dir.iterdir())
    except OSError as exc:
        raise ReviewReuseStoreError(
            "store_record_corrupt", "legacy tenant directory is unreadable"
        ) from exc
    for entry in entries:
        _assert_store_path(store_root, entry, code="store_record_corrupt")
        if entry.is_symlink():
            raise ReviewReuseStoreError(
                "store_record_corrupt", "unexpected legacy tenant artifact"
            )
        if entry.name == "tasks" and entry.is_dir():
            continue
        if entry.name == "idempotency.json" and entry.is_file():
            continue
        raise ReviewReuseStoreError(
            "store_record_corrupt", "unexpected legacy tenant artifact"
        )


class ReviewReuseStoreProtocol(Protocol):
    def create_if_absent(
        self,
        task: ReviewReuseTask,
        *,
        key: Optional[str],
        digest: Optional[str],
    ) -> ReviewReuseTask: ...

    def put(
        self, task: ReviewReuseTask, *, expected_revision: int
    ) -> ReviewReuseTask: ...

    def get(self, tenant_id: str, task_id: str) -> Optional[ReviewReuseTask]: ...

    def get_by_idempotency(
        self, tenant_id: str, key: str
    ) -> Optional[ReviewReuseTask]: ...

    def list_for_tenant(self, tenant_id: str) -> List[ReviewReuseTask]: ...


def validate_tenant_id(tenant_id: str) -> str:
    if not isinstance(tenant_id, str) or not _TENANT_PATTERN.fullmatch(tenant_id):
        raise ReviewReuseStoreError("tenant_invalid", "tenant identity is invalid")
    if tenant_id in {".", ".."}:
        raise ReviewReuseStoreError("tenant_invalid", "tenant identity is invalid")
    return tenant_id


def canonical_task_id(task_id: str) -> Optional[str]:
    try:
        parsed = uuid.UUID(str(task_id))
    except (ValueError, TypeError, AttributeError):
        return None
    canonical = str(parsed)
    return canonical if task_id == canonical else None


def tenant_identity_source(tenant_id: str) -> str:
    return "api_key_fallback" if tenant_id.startswith("ak-") else "validated_claim"


def _assert_store_path(root: Path, path: Path, *, code: str) -> None:
    """Reject lexical or symlink traversal outside a resolved store root."""

    try:
        relative = path.relative_to(root)
        path.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise ReviewReuseStoreError(
            code, "persistent ledger path escapes store root"
        ) from exc
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ReviewReuseStoreError(
                code, "persistent ledger path contains a symbolic link"
            )


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _mkdir_with_durable_parents(path: Path) -> None:
    missing: List[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        if current.parent == current:
            break
        current = current.parent
    path.mkdir(parents=True, exist_ok=True)
    for created in missing:
        _fsync_directory(created.parent)


def _clone(task: ReviewReuseTask) -> ReviewReuseTask:
    return ReviewReuseTask.model_validate(task.model_dump(mode="json"))


def _validate_digest(digest: Optional[str]) -> None:
    if digest is None or not _HEX64_PATTERN.fullmatch(digest):
        raise ReviewReuseStoreError(
            "store_record_corrupt", "ledger payload digest is invalid"
        )


def _validate_key(key: str) -> None:
    if not isinstance(key, str) or not key or key.strip() != key or len(key) > 128:
        raise ReviewReuseStoreError("invalid_request", "idempotency key is invalid")
    if any(not char.isprintable() for char in key):
        raise ReviewReuseStoreError("invalid_request", "idempotency key is invalid")


def _validate_create_binding(
    task: ReviewReuseTask, key: Optional[str], digest: Optional[str]
) -> None:
    validate_tenant_id(task.tenant_id)
    if canonical_task_id(task.task_id) is None:
        raise ReviewReuseStoreError("store_record_corrupt", "task identity is invalid")
    if task.revision != 1:
        raise ReviewReuseStoreError(
            "store_record_corrupt", "first persisted task revision must be 1"
        )
    validate_review_reuse_task_payload(task)
    if key is None:
        if (
            digest is not None
            or task.idempotency_key is not None
            or task.idempotency_digest is not None
        ):
            raise ReviewReuseStoreError(
                "store_record_corrupt", "unkeyed task has idempotency metadata"
            )
        return
    _validate_key(key)
    _validate_digest(digest)
    expected_digest = _create_payload_digest(task)
    if (
        task.idempotency_key != key
        or task.idempotency_digest != digest
        or digest != expected_digest
    ):
        raise ReviewReuseStoreError(
            "store_record_corrupt", "task idempotency ownership is inconsistent"
        )


def _create_payload_digest(task: ReviewReuseTask) -> str:
    return canonical_sha256(
        {
            "tenant_id": task.tenant_id,
            "source_content_sha256": task.source_content_sha256,
        }
    )


def _decision_payload_digest(task: ReviewReuseTask) -> str:
    decision = task.human_decision
    if decision is None:  # pragma: no cover - caller checks
        raise ReviewReuseStoreError(
            "store_record_corrupt", "decision attribution is missing"
        )
    return canonical_sha256(
        {
            "tenant_id": task.tenant_id,
            "task_id": task.task_id,
            "state": decision.state.value,
            "candidate_id": decision.candidate_id,
            "reason_codes": decision.reason_codes,
            "reason_text": decision.reason_text,
            "reviewer_id": decision.reviewer_id,
            "reviewer_kind": decision.reviewer_kind,
            "expected_revision": decision.reviewed_revision,
            "evidence_pack_sha256": decision.evidence_pack_sha256,
        }
    )


def _validate_stored_create_idempotency(task: ReviewReuseTask) -> None:
    if task.idempotency_key is None:
        if task.idempotency_digest is not None:
            raise ReviewReuseStoreError(
                "store_record_corrupt", "stored task idempotency is inconsistent"
            )
        return
    try:
        _validate_key(task.idempotency_key)
    except ReviewReuseStoreError as exc:
        raise ReviewReuseStoreError(
            "store_record_corrupt", "stored task idempotency key is invalid"
        ) from exc
    _validate_digest(task.idempotency_digest)
    if task.idempotency_digest != _create_payload_digest(task):
        raise ReviewReuseStoreError(
            "store_record_corrupt", "stored create idempotency is inconsistent"
        )


def _validate_decision_event_binding(task: ReviewReuseTask) -> None:
    if task.status != TaskStatus.decided or not task.events:
        return
    decision = task.human_decision
    if decision is None:
        raise ReviewReuseStoreError(
            "store_record_corrupt", "decision attribution is missing"
        )
    expected_detail = {
        "state": decision.state.value,
        "reviewer_id": decision.reviewer_id,
        "candidate_id": decision.candidate_id,
        "reviewed_revision": decision.reviewed_revision,
        "evidence_pack_sha256": decision.evidence_pack_sha256,
    }
    event = task.events[-1]
    if event.event_type != TaskEventType.decision_submitted or any(
        key not in event.detail
        or type(event.detail[key]) is not type(value)
        or event.detail[key] != value
        for key, value in expected_detail.items()
    ):
        raise ReviewReuseStoreError(
            "store_record_corrupt", "decision event binding is inconsistent"
        )


def validate_review_reuse_task_payload(task: ReviewReuseTask) -> None:
    """Validate one self-contained task snapshot before persistence."""

    try:
        canonical_json_v1(task.model_dump(mode="json"))
    except (CanonicalJSONError, TypeError, UnicodeError, ValueError) as exc:
        raise ReviewReuseStoreError(
            "store_record_corrupt", "stored task is not valid I-JSON"
        ) from exc
    if not _HEX64_PATTERN.fullmatch(task.source_content_sha256):
        raise ReviewReuseStoreError(
            "store_record_corrupt", "stored source digest is invalid"
        )
    if any(
        not candidate.candidate_id
        or candidate.candidate_id.strip() != candidate.candidate_id
        for candidate in task.candidates
    ):
        raise ReviewReuseStoreError(
            "store_record_corrupt", "stored candidate identity is invalid"
        )
    if any(
        reason not in _REJECTION_REASON_CODES
        for candidate in task.candidates
        for reason in candidate.rejection_reasons
    ):
        raise ReviewReuseStoreError(
            "store_record_corrupt", "stored candidate rejection reason is invalid"
        )
    pack = task.evidence_pack
    if task.status in {TaskStatus.evidence_ready, TaskStatus.decided} and pack is None:
        raise ReviewReuseStoreError(
            "store_record_corrupt", "stored task is missing its EvidencePack"
        )
    if task.status in {TaskStatus.pending, TaskStatus.running, TaskStatus.failed} and (
        pack is not None
    ):
        raise ReviewReuseStoreError(
            "store_record_corrupt", "stored task has an invalid EvidencePack state"
        )
    if pack is not None:
        try:
            digest_valid = evidence_pack_digest_is_valid(pack)
        except (CanonicalJSONError, TypeError, ValueError) as exc:
            raise ReviewReuseStoreError(
                "store_record_corrupt", "stored EvidencePack is invalid"
            ) from exc
        if (
            not digest_valid
            or isinstance(pack.get("task_revision"), bool)
            or not isinstance(pack.get("task_revision"), int)
            or pack.get("task_id") != task.task_id
            or pack.get("tenant_id") != task.tenant_id
            or pack.get("task_revision") != task.revision
            or pack.get("trace_id") != task.trace_id
            or pack.get("idempotency_key") != task.idempotency_key
            or pack.get("source_job_id") != task.task_id
            or (pack.get("source") or {}).get("content_sha256")
            != task.source_content_sha256
            or (pack.get("source") or {}).get("file_name") != task.source_file_name
        ):
            raise ReviewReuseStoreError(
                "store_record_corrupt", "stored EvidencePack identity is inconsistent"
            )
        calibration = pack.get("calibration")
        if (
            not isinstance(calibration, dict)
            or calibration.get("version") != task.calibration_version
            or calibration.get("status") != task.calibration_status
        ):
            raise ReviewReuseStoreError(
                "store_record_corrupt",
                "stored EvidencePack calibration is inconsistent",
            )
        task_candidate_ids = [candidate.candidate_id for candidate in task.candidates]
        pack_candidates = pack.get("candidates")
        expected_pack = build_evidence_pack(task)
        evidence_envelope_fields = (
            "schema_version",
            "source",
            "candidates",
            "confidence",
            "evidence",
            "rejection_reasons",
            "unsupported_states",
            "provenance",
        )
        if (
            set(pack) != set(expected_pack)
            or len(task_candidate_ids) != len(set(task_candidate_ids))
            or not isinstance(pack_candidates, list)
            or any(not isinstance(candidate, dict) for candidate in pack_candidates)
            or any(
                pack.get(field) != expected_pack[field]
                for field in evidence_envelope_fields
            )
        ):
            raise ReviewReuseStoreError(
                "store_record_corrupt",
                "stored EvidencePack envelope is inconsistent",
            )
        if pack.get("human_decision") != expected_pack["human_decision"]:
            raise ReviewReuseStoreError(
                "store_record_corrupt",
                "stored decision and EvidencePack are inconsistent",
            )
    decision = task.human_decision
    if decision is None:
        if task.status == TaskStatus.decided:
            raise ReviewReuseStoreError(
                "store_record_corrupt", "decided task lacks decision attribution"
            )
    else:
        if (
            task.status != TaskStatus.decided
            or decision.reviewer_kind != "validated_principal"
            or not _PRINCIPAL_PATTERN.fullmatch(decision.reviewer_id)
            or decision.reviewed_revision < 1
            or decision.reviewed_revision != task.revision - 1
            or not _HEX64_PATTERN.fullmatch(decision.evidence_pack_sha256)
            or decision.reason_codes != sorted(set(decision.reason_codes))
            or decision.reason_text.strip() != decision.reason_text
            or (
                decision.candidate_id is not None
                and decision.candidate_id.strip() != decision.candidate_id
            )
        ):
            raise ReviewReuseStoreError(
                "store_record_corrupt", "stored decision attribution is invalid"
            )
        candidate_ids = {candidate.candidate_id for candidate in task.candidates}
        candidate_required = {
            HumanDecisionState.reuse,
            HumanDecisionState.revise,
            HumanDecisionState.reject_candidate,
        }
        if (
            (
                decision.state in candidate_required
                and decision.candidate_id not in candidate_ids
            )
            or (
                decision.state == HumanDecisionState.new
                and decision.candidate_id is not None
            )
            or (
                decision.state == HumanDecisionState.need_more_info
                and decision.candidate_id is not None
                and decision.candidate_id not in candidate_ids
            )
            or (not decision.reason_codes and not decision.reason_text)
            or ("other" in decision.reason_codes and not decision.reason_text)
        ):
            raise ReviewReuseStoreError(
                "store_record_corrupt", "stored decision semantics are invalid"
            )
        reviewed = task.model_copy(deep=True)
        reviewed.status = TaskStatus.evidence_ready
        reviewed.revision = decision.reviewed_revision
        reviewed.human_decision = None
        reviewed.evidence_pack = None
        reviewed_pack = build_evidence_pack(reviewed)
        if decision.evidence_pack_sha256 != reviewed_pack["evidence_pack_sha256"]:
            raise ReviewReuseStoreError(
                "store_record_corrupt",
                "stored decision reviewed evidence is inconsistent",
            )
        if decision.idempotency_key is None:
            if decision.idempotency_digest is not None:
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "decision idempotency is inconsistent"
                )
        else:
            try:
                _validate_key(decision.idempotency_key)
            except ReviewReuseStoreError as exc:
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "decision idempotency key is invalid"
                ) from exc
            _validate_digest(decision.idempotency_digest)
            if decision.idempotency_digest != _decision_payload_digest(task):
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "decision idempotency is inconsistent"
                )
        if pack is None:  # guarded by decided state, kept explicit for type narrowing
            raise ReviewReuseStoreError(
                "store_record_corrupt", "decided task is missing its EvidencePack"
            )
        pack_decision = pack.get("human_decision")
        if (
            not isinstance(pack_decision, dict)
            or pack_decision.get("state") != decision.state.value
            or pack_decision.get("submitted") != decision.model_dump(mode="json")
        ):
            raise ReviewReuseStoreError(
                "store_record_corrupt",
                "stored decision and EvidencePack are inconsistent",
            )
    _validate_decision_event_binding(task)


def _assert_mutation(
    current: ReviewReuseTask, task: ReviewReuseTask, expected: int
) -> None:
    if current.revision != expected:
        raise ReviewReuseStoreError("revision_conflict", "task revision has changed")
    if task.revision != expected + 1:
        raise ReviewReuseStoreError(
            "revision_conflict", "next task revision must increment by one"
        )
    transitions = {
        TaskStatus.pending: {
            TaskStatus.running,
            TaskStatus.failed,
            TaskStatus.canceled,
        },
        TaskStatus.running: {
            TaskStatus.evidence_ready,
            TaskStatus.failed,
            TaskStatus.canceled,
        },
        TaskStatus.evidence_ready: {TaskStatus.decided, TaskStatus.canceled},
    }
    if task.status not in transitions.get(current.status, set()):
        raise ReviewReuseStoreError(
            "invalid_state_transition", "task state transition is invalid"
        )
    immutable = (
        "task_id",
        "tenant_id",
        "created_at",
        "source_file_name",
        "source_content_sha256",
        "idempotency_key",
        "idempotency_digest",
        "trace_id",
    )
    if any(getattr(current, name) != getattr(task, name) for name in immutable):
        raise ReviewReuseStoreError(
            "store_record_corrupt", "immutable task identity changed"
        )
    if (
        current.status == TaskStatus.evidence_ready
        and task.status == TaskStatus.decided
    ):
        decision = task.human_decision
        current_pack = current.evidence_pack
        if (
            decision is None
            or current_pack is None
            or decision.reviewed_revision != current.revision
            or decision.evidence_pack_sha256 != current_pack.get("evidence_pack_sha256")
        ):
            raise ReviewReuseStoreError(
                "revision_conflict",
                "decision is not bound to the current EvidencePack",
            )
        if any(code not in DECISION_REASON_CODES for code in decision.reason_codes):
            raise ReviewReuseStoreError(
                "store_record_corrupt", "decision reason code is invalid"
            )
    validate_review_reuse_task_payload(task)
    if current.status == TaskStatus.evidence_ready and (
        task.candidates != current.candidates
        or task.calibration_version != current.calibration_version
        or task.calibration_status != current.calibration_status
    ):
        raise ReviewReuseStoreError(
            "store_record_corrupt", "reviewed evidence cannot change at commit"
        )
    current_events = [event.model_dump(mode="json") for event in current.events]
    next_events = [event.model_dump(mode="json") for event in task.events]
    if (
        len(next_events) <= len(current_events)
        or next_events[: len(current_events)] != current_events
    ):
        raise ReviewReuseStoreError(
            "store_record_corrupt", "task mutation must append to its event ledger"
        )
    appended_events = task.events[len(current.events) :]
    appended_types = tuple(event.event_type for event in appended_events)
    transition = (current.status, task.status)
    if transition == (TaskStatus.running, TaskStatus.failed):
        pipeline_prefix = appended_types[:-1]
        sequence_valid = (
            appended_types[-1:] == (TaskEventType.failed,)
            and pipeline_prefix == _RUNNING_PIPELINE_EVENTS[: len(pipeline_prefix)]
        )
    else:
        sequence_valid = appended_types == _TRANSITION_EVENT_SEQUENCES[transition]
    if not sequence_valid:
        raise ReviewReuseStoreError(
            "store_record_corrupt", "task transition event is inconsistent"
        )


class InMemoryReviewReuseStore:
    """Process-local store with store-level create and mutation critical sections."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._tasks: Dict[str, Dict[str, ReviewReuseTask]] = {}
        self._idem: Dict[str, Dict[str, str]] = {}

    def create_if_absent(
        self,
        task: ReviewReuseTask,
        *,
        key: Optional[str],
        digest: Optional[str],
    ) -> ReviewReuseTask:
        _validate_create_binding(task, key, digest)
        with self._lock:
            bucket = self._tasks.setdefault(task.tenant_id, {})
            if key is not None:
                task_id = self._idem.setdefault(task.tenant_id, {}).get(key)
                if task_id is not None:
                    existing = bucket.get(task_id)
                    if existing is None:
                        raise ReviewReuseStoreError(
                            "store_index_corrupt", "idempotency owner is missing"
                        )
                    if existing.idempotency_digest != digest:
                        raise ReviewReuseStoreError(
                            "idempotency_key_conflict",
                            "idempotency key is bound to another payload",
                        )
                    return _clone(existing)
            if task.task_id in bucket:
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "task identity already exists"
                )
            stored = _clone(task)
            bucket[task.task_id] = stored
            if key is not None:
                self._idem[task.tenant_id][key] = task.task_id
            return _clone(stored)

    def put(self, task: ReviewReuseTask, *, expected_revision: int) -> ReviewReuseTask:
        validate_tenant_id(task.tenant_id)
        with self._lock:
            current = self._tasks.get(task.tenant_id, {}).get(task.task_id)
            if current is None:
                raise ReviewReuseStoreError("not_found", "task does not exist")
            _assert_mutation(current, task, expected_revision)
            stored = _clone(task)
            self._tasks[task.tenant_id][task.task_id] = stored
            return _clone(stored)

    def get(self, tenant_id: str, task_id: str) -> Optional[ReviewReuseTask]:
        validate_tenant_id(tenant_id)
        if canonical_task_id(task_id) is None:
            return None
        with self._lock:
            task = self._tasks.get(tenant_id, {}).get(task_id)
            return _clone(task) if task is not None else None

    def get_by_idempotency(self, tenant_id: str, key: str) -> Optional[ReviewReuseTask]:
        validate_tenant_id(tenant_id)
        _validate_key(key)
        with self._lock:
            task_id = self._idem.get(tenant_id, {}).get(key)
            if task_id is None:
                return None
            task = self._tasks.get(tenant_id, {}).get(task_id)
            if task is None:
                raise ReviewReuseStoreError(
                    "store_index_corrupt", "idempotency owner is missing"
                )
            return _clone(task)

    def list_for_tenant(self, tenant_id: str) -> List[ReviewReuseTask]:
        validate_tenant_id(tenant_id)
        with self._lock:
            return [_clone(task) for task in self._tasks.get(tenant_id, {}).values()]


class FilesystemReviewReuseStore:
    """Single-writer JSON store with hashed tenant paths and checked records."""

    def __init__(self, root: Path | str, *, read_only: bool = False) -> None:
        raw_root = Path(root).expanduser()
        self._read_only = read_only
        self._lock = threading.RLock()
        self._healthy = True
        self._lease = None
        self._lease_pid: Optional[int] = None
        if read_only:
            self._root = raw_root.resolve(strict=False)
        else:
            _mkdir_with_durable_parents(raw_root.parent)
            self._root = raw_root.resolve(strict=False)
            self._acquire_writer_lease()
            try:
                _mkdir_with_durable_parents(raw_root)
                if raw_root.resolve() != self._root:
                    raise ReviewReuseStoreError(
                        "store_writer_conflict",
                        "filesystem store root changed while acquiring writer lease",
                    )
                self._cleanup_stale_artifacts()
                validated_filesystem_tenants(self._root)
            except Exception:
                self.close()
                raise

    def _acquire_writer_lease(self) -> None:
        self._lease = _open_writer_lease(
            self._root, message="filesystem store already has a writer"
        )
        self._lease_pid = os.getpid()

    def close(self) -> None:
        with self._lock:
            if self._lease is not None:
                try:
                    if self._lease_pid == os.getpid():
                        fcntl.flock(self._lease.fileno(), fcntl.LOCK_UN)
                finally:
                    self._lease.close()
                    self._lease = None
                    self._lease_pid = None

    def __enter__(self) -> "FilesystemReviewReuseStore":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()

    def __del__(self) -> None:  # pragma: no cover - defensive cleanup
        try:
            self.close()
        except Exception:
            pass

    def _ensure_writer(self) -> None:
        self._ensure_healthy()
        if self._read_only or self._lease is None or self._lease_pid != os.getpid():
            raise ReviewReuseStoreError(
                "store_writer_conflict", "filesystem writer lease is unavailable"
            )

    def _ensure_healthy(self) -> None:
        if not self._healthy:
            raise ReviewReuseStoreError(
                "store_writer_conflict", "filesystem store is quarantined"
            )

    def _cleanup_stale_artifacts(self) -> None:
        """Remove only crash leftovers created by this store's write protocol."""

        self._ensure_writer()
        touched: set[Path] = {self._root}
        try:
            for entry in sorted(self._root.iterdir()):
                _assert_store_path(self._root, entry, code="store_record_corrupt")
                if entry.is_symlink():
                    raise ReviewReuseStoreError(
                        "store_record_corrupt", "store artifact is a symbolic link"
                    )
                if entry.is_dir() and _TENANT_STAGE_PATTERN.fullmatch(entry.name):
                    shutil.rmtree(entry)
                    touched.add(self._root)
                    continue
                if not entry.is_dir() or not _NEW_TENANT_DIR_PATTERN.fullmatch(
                    entry.name
                ):
                    continue
                touched.add(entry)

                for candidate in sorted(entry.iterdir()):
                    if not _METADATA_TEMP_PATTERN.fullmatch(candidate.name):
                        continue
                    _assert_store_path(
                        self._root, candidate, code="store_record_corrupt"
                    )
                    if candidate.is_symlink() or not candidate.is_file():
                        raise ReviewReuseStoreError(
                            "store_record_corrupt",
                            "atomic-write temporary artifact is invalid",
                        )
                    candidate.unlink()
                    touched.add(entry)

                tasks_dir = entry / "tasks"
                if not tasks_dir.is_dir() or tasks_dir.is_symlink():
                    continue
                touched.add(tasks_dir)
                for candidate in sorted(tasks_dir.iterdir()):
                    if not _TASK_TEMP_PATTERN.fullmatch(candidate.name):
                        continue
                    _assert_store_path(
                        self._root, candidate, code="store_record_corrupt"
                    )
                    if candidate.is_symlink() or not candidate.is_file():
                        raise ReviewReuseStoreError(
                            "store_record_corrupt",
                            "atomic-write temporary artifact is invalid",
                        )
                    candidate.unlink()
                    touched.add(tasks_dir)
            for directory in sorted(
                touched, key=lambda item: len(item.parts), reverse=True
            ):
                _fsync_directory(directory)
        except ReviewReuseStoreError:
            raise
        except OSError as exc:
            raise ReviewReuseStoreError(
                "store_record_corrupt", "stale store artifact cleanup failed"
            ) from exc

    def _tenant_digest(self, tenant_id: str) -> str:
        validate_tenant_id(tenant_id)
        return hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()

    def _tenant_dir(self, tenant_id: str) -> Path:
        digest = self._tenant_digest(tenant_id)
        path = self._root / f"tenant-v1-{digest}"
        _assert_store_path(self._root, path, code="store_record_corrupt")
        return path

    def _task_path(self, tenant_id: str, task_id: str) -> Path:
        if canonical_task_id(task_id) is None:
            raise ReviewReuseStoreError("not_found", "task does not exist")
        path = self._tenant_dir(tenant_id) / "tasks" / f"{task_id}.json"
        _assert_store_path(self._root, path, code="store_record_corrupt")
        return path

    def _sidecar_payload(self, tenant_id: str) -> Dict[str, str]:
        return {
            "schema_version": _TENANT_SCHEMA,
            "tenant_id": tenant_id,
            "identity_source": tenant_identity_source(tenant_id),
            "tenant_digest_sha256": self._tenant_digest(tenant_id),
        }

    def _load_json(self, path: Path, *, code: str) -> Any:
        _assert_store_path(self._root, path, code=code)
        try:
            return strict_json_loads(path.read_text(encoding="utf-8"))
        except (CanonicalJSONError, OSError, UnicodeError, ValueError) as exc:
            raise ReviewReuseStoreError(
                code, "persistent ledger data is corrupt"
            ) from exc

    def _load_sidecar(self, tenant_id: str) -> Optional[Dict[str, str]]:
        tenant_dir = self._tenant_dir(tenant_id)
        if not tenant_dir.exists():
            return None
        if not tenant_dir.is_dir():
            raise ReviewReuseStoreError(
                "store_record_corrupt", "tenant ledger path is not a directory"
            )
        sidecar_path = tenant_dir / "tenant.json"
        if not sidecar_path.is_file():
            raise ReviewReuseStoreError(
                "store_record_corrupt", "tenant identity sidecar is missing"
            )
        payload = self._load_json(sidecar_path, code="store_record_corrupt")
        expected = self._sidecar_payload(tenant_id)
        if (
            not isinstance(payload, dict)
            or set(payload) != _TENANT_FIELDS
            or payload != expected
        ):
            raise ReviewReuseStoreError(
                "store_record_corrupt", "tenant identity sidecar is inconsistent"
            )
        return payload

    def _ensure_tenant(self, tenant_id: str) -> Path:
        self._ensure_writer()
        tenant_dir = self._tenant_dir(tenant_id)
        if tenant_dir.exists():
            self._load_sidecar(tenant_id)
            tasks_dir = tenant_dir / "tasks"
            _assert_store_path(self._root, tasks_dir, code="store_record_corrupt")
            if not tasks_dir.is_dir():
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "tenant tasks directory is missing"
                )
            return tenant_dir
        staging = self._root / (f".{tenant_dir.name}.stage-{uuid.uuid4().hex}")
        _assert_store_path(self._root, staging, code="store_record_corrupt")
        published = False
        try:
            staging.mkdir(exist_ok=False)
            (staging / "tasks").mkdir(exist_ok=False)
            self._atomic_write_json(
                staging / "tenant.json", self._sidecar_payload(tenant_id)
            )
            os.replace(staging, tenant_dir)
            published = True
            _fsync_directory(self._root)
        except ReviewReuseStoreError:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        except OSError as exc:
            if published:
                self._healthy = False
            shutil.rmtree(staging, ignore_errors=True)
            raise ReviewReuseStoreError(
                "store_record_corrupt", "tenant ledger initialization failed"
            ) from exc
        return tenant_dir

    def _validate_loaded_task(
        self, task: ReviewReuseTask, *, tenant_id: str, task_id: str
    ) -> None:
        if task.tenant_id != tenant_id or task.task_id != task_id:
            raise ReviewReuseStoreError(
                "store_record_corrupt", "stored task identity is inconsistent"
            )
        if canonical_task_id(task.task_id) is None or task.revision < 1:
            raise ReviewReuseStoreError(
                "store_record_corrupt", "stored task metadata is invalid"
            )
        validate_review_reuse_task_payload(task)
        _validate_stored_create_idempotency(task)

    def _load_task_path(
        self, path: Path, *, tenant_id: str, task_id: str
    ) -> ReviewReuseTask:
        raw = self._load_json(path, code="store_record_corrupt")
        try:
            task = ReviewReuseTask.model_validate(raw)
        except ValueError as exc:
            raise ReviewReuseStoreError(
                "store_record_corrupt", "stored task schema is invalid"
            ) from exc
        self._validate_loaded_task(task, tenant_id=tenant_id, task_id=task_id)
        return task

    def _load_index(self, tenant_id: str) -> Optional[Dict[str, Any]]:
        sidecar = self._load_sidecar(tenant_id)
        if sidecar is None:
            return None
        path = self._tenant_dir(tenant_id) / "idempotency.json"
        if not path.exists():
            return None
        payload = self._load_json(path, code="store_index_corrupt")
        if not isinstance(payload, dict) or set(payload) != _INDEX_FIELDS:
            raise ReviewReuseStoreError(
                "store_index_corrupt", "idempotency index schema is invalid"
            )
        if (
            payload.get("schema_version") != _INDEX_SCHEMA
            or payload.get("tenant_id") != tenant_id
            or payload.get("tenant_digest_sha256") != sidecar["tenant_digest_sha256"]
            or not isinstance(payload.get("entries"), dict)
        ):
            raise ReviewReuseStoreError(
                "store_index_corrupt", "idempotency index identity is invalid"
            )
        for key, entry in payload["entries"].items():
            try:
                _validate_key(key)
            except ReviewReuseStoreError as exc:
                raise ReviewReuseStoreError(
                    "store_index_corrupt", "idempotency index key is invalid"
                ) from exc
            if not isinstance(entry, dict) or set(entry) != _INDEX_ENTRY_FIELDS:
                raise ReviewReuseStoreError(
                    "store_index_corrupt", "idempotency index entry is invalid"
                )
            task_id = entry.get("task_id")
            digest = entry.get("payload_digest")
            if (
                canonical_task_id(task_id) is None
                or not isinstance(digest, str)
                or not _HEX64_PATTERN.fullmatch(digest)
            ):
                raise ReviewReuseStoreError(
                    "store_index_corrupt", "idempotency index entry is invalid"
                )
            task_path = self._task_path(tenant_id, task_id)
            if not task_path.is_file():
                raise ReviewReuseStoreError(
                    "store_index_corrupt", "idempotency owner is missing"
                )
            try:
                task = self._load_task_path(
                    task_path, tenant_id=tenant_id, task_id=task_id
                )
            except ReviewReuseStoreError as exc:
                raise ReviewReuseStoreError(
                    "store_index_corrupt", "idempotency owner is corrupt"
                ) from exc
            if task.idempotency_key != key or task.idempotency_digest != digest:
                raise ReviewReuseStoreError(
                    "store_index_corrupt", "idempotency ownership is inconsistent"
                )
        return payload

    def _empty_index(self, tenant_id: str) -> Dict[str, Any]:
        return {
            "schema_version": _INDEX_SCHEMA,
            "tenant_id": tenant_id,
            "tenant_digest_sha256": self._tenant_digest(tenant_id),
            "entries": {},
        }

    def _atomic_write_json(self, path: Path, payload: Any) -> None:
        _assert_store_path(self._root, path, code="store_record_corrupt")
        try:
            encoded = json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        except (TypeError, ValueError, UnicodeError) as exc:
            raise ReviewReuseStoreError(
                "store_record_corrupt", "persistent ledger payload is invalid"
            ) from exc
        temporary: Optional[Path] = None
        published = False
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            published = True
            _fsync_directory(path.parent)
        except OSError as exc:
            if published:
                self._healthy = False
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise ReviewReuseStoreError(
                "store_record_corrupt", "persistent ledger write failed"
            ) from exc

    def _write_task(self, task: ReviewReuseTask) -> None:
        path = self._task_path(task.tenant_id, task.task_id)
        self._atomic_write_json(path, task.model_dump(mode="json"))

    def _write_index(self, tenant_id: str, payload: Dict[str, Any]) -> None:
        try:
            self._atomic_write_json(
                self._tenant_dir(tenant_id) / "idempotency.json", payload
            )
        except ReviewReuseStoreError as exc:
            self._healthy = False
            raise ReviewReuseStoreError(
                "store_index_corrupt", "idempotency index update failed"
            ) from exc

    def _scan_key(self, tenant_id: str, key: str) -> List[ReviewReuseTask]:
        return [
            task
            for task in self.list_for_tenant(tenant_id)
            if task.idempotency_key == key
        ]

    def create_if_absent(
        self,
        task: ReviewReuseTask,
        *,
        key: Optional[str],
        digest: Optional[str],
    ) -> ReviewReuseTask:
        _validate_create_binding(task, key, digest)
        with self._lock:
            self._ensure_writer()
            self._ensure_tenant(task.tenant_id)
            index = self._load_index(task.tenant_id)
            if key is not None:
                matches = self._scan_key(task.tenant_id, key)
                entry = (index or {"entries": {}})["entries"].get(key)
                if entry is not None:
                    if entry["payload_digest"] != digest:
                        raise ReviewReuseStoreError(
                            "idempotency_key_conflict",
                            "idempotency key is bound to another payload",
                        )
                    if len(matches) != 1 or matches[0].task_id != entry["task_id"]:
                        raise ReviewReuseStoreError(
                            "store_index_corrupt",
                            "idempotency ownership is inconsistent",
                        )
                    return matches[0]

                if len(matches) > 1:
                    raise ReviewReuseStoreError(
                        "store_index_corrupt", "duplicate idempotency owners found"
                    )
                if matches:
                    existing = matches[0]
                    if existing.idempotency_digest != digest:
                        raise ReviewReuseStoreError(
                            "idempotency_key_conflict",
                            "idempotency key is bound to another payload",
                        )
                    repaired = index or self._empty_index(task.tenant_id)
                    repaired["entries"][key] = {
                        "task_id": existing.task_id,
                        "payload_digest": digest,
                    }
                    self._write_index(task.tenant_id, repaired)
                    return existing

            path = self._task_path(task.tenant_id, task.task_id)
            if path.exists():
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "task identity already exists"
                )
            self._write_task(task)
            if key is not None:
                updated = index or self._empty_index(task.tenant_id)
                updated["entries"][key] = {
                    "task_id": task.task_id,
                    "payload_digest": digest,
                }
                self._write_index(task.tenant_id, updated)
            return _clone(task)

    def _import_migrated(self, task: ReviewReuseTask) -> ReviewReuseTask:
        """Install one preflighted historical snapshot into an empty staging tree."""

        validate_tenant_id(task.tenant_id)
        if canonical_task_id(task.task_id) is None or task.revision < 1:
            raise ReviewReuseStoreError(
                "store_record_corrupt", "migrated task identity is invalid"
            )
        validate_review_reuse_task_payload(task)
        with self._lock:
            self._ensure_writer()
            self._ensure_tenant(task.tenant_id)
            path = self._task_path(task.tenant_id, task.task_id)
            if path.exists():
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "migrated task identity collides"
                )
            index = self._load_index(task.tenant_id)
            if task.idempotency_key is not None:
                _validate_key(task.idempotency_key)
                _validate_digest(task.idempotency_digest)
                if self._scan_key(task.tenant_id, task.idempotency_key):
                    raise ReviewReuseStoreError(
                        "store_index_corrupt", "migrated idempotency key collides"
                    )
            self._write_task(task)
            if task.idempotency_key is not None:
                updated = index or self._empty_index(task.tenant_id)
                updated["entries"][task.idempotency_key] = {
                    "task_id": task.task_id,
                    "payload_digest": task.idempotency_digest,
                }
                self._write_index(task.tenant_id, updated)
            return _clone(task)

    def put(self, task: ReviewReuseTask, *, expected_revision: int) -> ReviewReuseTask:
        validate_tenant_id(task.tenant_id)
        with self._lock:
            self._ensure_writer()
            current = self.get(task.tenant_id, task.task_id)
            if current is None:
                raise ReviewReuseStoreError("not_found", "task does not exist")
            _assert_mutation(current, task, expected_revision)
            self._write_task(task)
            return _clone(task)

    def get(self, tenant_id: str, task_id: str) -> Optional[ReviewReuseTask]:
        validate_tenant_id(tenant_id)
        if canonical_task_id(task_id) is None:
            return None
        with self._lock:
            self._ensure_healthy()
            if self._load_sidecar(tenant_id) is None:
                return None
            self._load_index(tenant_id)
            path = self._task_path(tenant_id, task_id)
            if not path.exists():
                return None
            if not path.is_file():
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "task record path is invalid"
                )
            return _clone(
                self._load_task_path(path, tenant_id=tenant_id, task_id=task_id)
            )

    def get_by_idempotency(self, tenant_id: str, key: str) -> Optional[ReviewReuseTask]:
        validate_tenant_id(tenant_id)
        _validate_key(key)
        with self._lock:
            self._ensure_healthy()
            index = self._load_index(tenant_id)
            if index is None:
                return None
            entry = index["entries"].get(key)
            if entry is None:
                return None
            matches = self._scan_key(tenant_id, key)
            if len(matches) != 1 or matches[0].task_id != entry["task_id"]:
                raise ReviewReuseStoreError(
                    "store_index_corrupt", "idempotency ownership is inconsistent"
                )
            return matches[0]

    def list_for_tenant(self, tenant_id: str) -> List[ReviewReuseTask]:
        validate_tenant_id(tenant_id)
        with self._lock:
            self._ensure_healthy()
            if self._load_sidecar(tenant_id) is None:
                return []
            self._load_index(tenant_id)
            tasks_dir = self._tenant_dir(tenant_id) / "tasks"
            _assert_store_path(self._root, tasks_dir, code="store_record_corrupt")
            if not tasks_dir.is_dir():
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "tenant tasks directory is missing"
                )
            tasks: List[ReviewReuseTask] = []
            for path in sorted(tasks_dir.iterdir()):
                if path.is_symlink():
                    raise ReviewReuseStoreError(
                        "store_record_corrupt", "unexpected task record entry"
                    )
                if _TASK_TEMP_PATTERN.fullmatch(path.name) and path.is_file():
                    continue
                if not path.is_file() or path.suffix != ".json":
                    raise ReviewReuseStoreError(
                        "store_record_corrupt", "unexpected task record entry"
                    )
                task_id = path.stem
                if canonical_task_id(task_id) is None:
                    raise ReviewReuseStoreError(
                        "store_record_corrupt", "task record name is invalid"
                    )
                tasks.append(
                    _clone(
                        self._load_task_path(path, tenant_id=tenant_id, task_id=task_id)
                    )
                )
            idempotency_keys = [
                task.idempotency_key
                for task in tasks
                if task.idempotency_key is not None
            ]
            if len(idempotency_keys) != len(set(idempotency_keys)):
                raise ReviewReuseStoreError(
                    "store_index_corrupt", "duplicate idempotency owners found"
                )
            return tasks


def validated_filesystem_tenants(root: Path | str) -> List[Tuple[str, Path]]:
    """Resolve literal tenant identities and validate every persisted artifact."""

    store_root = Path(root).expanduser().resolve(strict=False)
    directories = _validated_root_directories(store_root)
    if not directories:
        return []
    reader = FilesystemReviewReuseStore(store_root, read_only=True)
    tenants: List[Tuple[str, Path]] = []
    try:
        for tenant_dir in directories:
            if _TENANT_STAGE_PATTERN.fullmatch(tenant_dir.name):
                continue
            if not _NEW_TENANT_DIR_PATTERN.fullmatch(tenant_dir.name):
                raise ReviewReuseStoreError(
                    "store_record_corrupt",
                    "legacy or unexpected tenant directory found",
                )
            _validate_tenant_directory_entries(store_root, tenant_dir)
            sidecar_path = tenant_dir / "tenant.json"
            sidecar = reader._load_json(sidecar_path, code="store_record_corrupt")
            tenant_id = sidecar.get("tenant_id") if isinstance(sidecar, dict) else None
            try:
                validate_tenant_id(tenant_id)
            except ReviewReuseStoreError as exc:
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "tenant sidecar identity is invalid"
                ) from exc
            if reader._tenant_dir(tenant_id) != tenant_dir:
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "tenant sidecar path is inconsistent"
                )
            reader._load_sidecar(tenant_id)
            reader.list_for_tenant(tenant_id)
            reader._load_index(tenant_id)
            tenants.append((tenant_id, tenant_dir))
    finally:
        reader.close()
    return tenants


def migrate_legacy_store(root: Path | str, *, apply: bool = False) -> Dict[str, Any]:
    """Preflight and optionally replace a legacy lossy-layout store."""

    store_root = Path(root).expanduser().resolve(strict=False)
    if not apply:
        return _migrate_legacy_store(store_root, apply=False)
    _mkdir_with_durable_parents(store_root.parent)
    lease = _open_writer_lease(
        store_root, message="legacy migration requires the writer lease"
    )
    try:
        _mkdir_with_durable_parents(store_root)
        if store_root.resolve() != store_root:
            raise ReviewReuseStoreError(
                "store_writer_conflict",
                "filesystem store root changed while acquiring migration lease",
            )
        return _migrate_legacy_store(store_root, apply=True)
    finally:
        try:
            fcntl.flock(lease.fileno(), fcntl.LOCK_UN)
        finally:
            lease.close()


def _migrate_legacy_store(root: Path | str, *, apply: bool) -> Dict[str, Any]:
    """Run migration after the public entry point establishes write exclusion."""

    store_root = Path(root).expanduser().resolve(strict=False)
    records: List[ReviewReuseTask] = []
    directories = _validated_root_directories(store_root)
    new_layout_dirs = [
        path for path in directories if _NEW_TENANT_DIR_PATTERN.fullmatch(path.name)
    ]
    legacy_dirs = [path for path in directories if path not in new_layout_dirs]
    if new_layout_dirs:
        if legacy_dirs:
            raise ReviewReuseStoreError(
                "store_record_corrupt", "mixed legacy and new store layouts"
            )
        validated_filesystem_tenants(store_root)
        return {
            "apply": apply,
            "legacy_directories": 0,
            "tasks": 0,
            "tenants": 0,
            "already_migrated": True,
        }

    tenant_origins: Dict[str, Path] = {}
    for legacy_dir in legacy_dirs:
        _assert_store_path(store_root, legacy_dir, code="store_record_corrupt")
        _validate_legacy_tenant_directory_entries(store_root, legacy_dir)
        tasks_dir = legacy_dir / "tasks"
        _assert_store_path(store_root, tasks_dir, code="store_record_corrupt")
        if not tasks_dir.is_dir():
            raise ReviewReuseStoreError(
                "store_record_corrupt", "legacy tenant tasks directory is missing"
            )
        task_paths = sorted(tasks_dir.iterdir())
        if not task_paths or any(
            not path.is_file() or path.suffix != ".json" for path in task_paths
        ):
            raise ReviewReuseStoreError(
                "store_record_corrupt", "legacy task record layout is invalid"
            )
        raw_records: List[tuple[Path, Dict[str, Any]]] = []
        for path in task_paths:
            _assert_store_path(store_root, path, code="store_record_corrupt")
            try:
                raw = strict_json_loads(path.read_text(encoding="utf-8"))
            except (CanonicalJSONError, OSError, UnicodeError, ValueError) as exc:
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "legacy task record is unreadable"
                ) from exc
            if not isinstance(raw, dict):
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "legacy task record is invalid"
                )
            raw_records.append((path, raw))

        identities = [raw.get("tenant_id") for _, raw in raw_records]
        first_identity = identities[0]
        if any(identity != first_identity for identity in identities[1:]):
            raise ReviewReuseStoreError(
                "store_record_corrupt", "legacy tenant collision detected"
            )
        directory_records: List[ReviewReuseTask] = []
        for path, raw in raw_records:
            tenant_id = raw.get("tenant_id")
            try:
                validate_tenant_id(tenant_id)
                task = ReviewReuseTask.model_validate(raw)
            except (ReviewReuseStoreError, ValueError) as exc:
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "legacy task identity is invalid"
                ) from exc
            if canonical_task_id(task.task_id) is None:
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "legacy task id is invalid"
                )
            if path.stem != task.task_id or legacy_dir.name != tenant_id:
                raise ReviewReuseStoreError(
                    "store_record_corrupt",
                    "legacy record path identity is inconsistent",
                )
            previous_origin = tenant_origins.setdefault(tenant_id, legacy_dir)
            if previous_origin != legacy_dir:
                raise ReviewReuseStoreError(
                    "store_record_corrupt", "legacy tenant collision detected"
                )
            task.revision = 1
            task.error_code = None
            if task.idempotency_key:
                task.idempotency_digest = canonical_sha256(
                    {
                        "tenant_id": task.tenant_id,
                        "source_content_sha256": task.source_content_sha256,
                    }
                )
            else:
                task.idempotency_digest = None
            if task.human_decision is not None:
                decision_raw = raw.get("human_decision") or {}
                required = {
                    "reviewer_kind",
                    "idempotency_digest",
                    "reviewed_revision",
                    "evidence_pack_sha256",
                }
                if not required <= set(decision_raw):
                    raise ReviewReuseStoreError(
                        "store_record_corrupt",
                        "legacy decision attribution cannot be proven",
                    )
                task.revision = max(
                    task.revision, task.human_decision.reviewed_revision + 1
                )
                decision = task.human_decision
                decision_digest = canonical_sha256(
                    {
                        "tenant_id": task.tenant_id,
                        "task_id": task.task_id,
                        "state": decision.state.value,
                        "candidate_id": decision.candidate_id,
                        "reason_codes": sorted(set(decision.reason_codes)),
                        "reason_text": decision.reason_text,
                        "reviewer_id": decision.reviewer_id,
                        "reviewer_kind": decision.reviewer_kind,
                        "expected_revision": decision.reviewed_revision,
                        "evidence_pack_sha256": decision.evidence_pack_sha256,
                    }
                )
                if decision.idempotency_key is None:
                    if decision.idempotency_digest is not None:
                        raise ReviewReuseStoreError(
                            "store_record_corrupt",
                            "legacy decision idempotency is inconsistent",
                        )
                elif decision.idempotency_digest != decision_digest:
                    raise ReviewReuseStoreError(
                        "store_record_corrupt",
                        "legacy decision idempotency is inconsistent",
                    )
                decision.idempotency_digest = (
                    decision_digest if decision.idempotency_key is not None else None
                )
            if task.evidence_pack is not None:
                pack = dict(task.evidence_pack)
                pack["task_revision"] = task.revision
                raw_calibration = pack.get("calibration")
                if "calibration" in pack and not isinstance(raw_calibration, dict):
                    raise ReviewReuseStoreError(
                        "store_record_corrupt",
                        "legacy EvidencePack calibration is invalid",
                    )
                calibration = dict(raw_calibration or {})
                if (
                    "version" in calibration
                    and calibration["version"] != task.calibration_version
                ):
                    raise ReviewReuseStoreError(
                        "store_record_corrupt",
                        "legacy EvidencePack calibration is inconsistent",
                    )
                if (
                    "status" in calibration
                    and calibration["status"] != task.calibration_status
                ):
                    raise ReviewReuseStoreError(
                        "store_record_corrupt",
                        "legacy EvidencePack calibration is inconsistent",
                    )
                calibration["version"] = task.calibration_version
                calibration["status"] = task.calibration_status
                pack["calibration"] = calibration
                pack["evidence_pack_sha256"] = evidence_pack_digest(pack)
                task.evidence_pack = pack
            validate_review_reuse_task_payload(task)
            _validate_stored_create_idempotency(task)
            directory_records.append(task)
            records.append(task)

        keyed: Dict[str, str] = {}
        for task in directory_records:
            if task.idempotency_key is None:
                continue
            if task.idempotency_key in keyed:
                raise ReviewReuseStoreError(
                    "store_index_corrupt", "legacy idempotency collision detected"
                )
            keyed[task.idempotency_key] = task.task_id
        index_path = legacy_dir / "idempotency.json"
        if keyed or index_path.exists():
            _assert_store_path(store_root, index_path, code="store_index_corrupt")
            try:
                legacy_index = strict_json_loads(index_path.read_text(encoding="utf-8"))
            except (CanonicalJSONError, OSError, UnicodeError, ValueError) as exc:
                raise ReviewReuseStoreError(
                    "store_index_corrupt", "legacy idempotency mapping is unprovable"
                ) from exc
            if legacy_index != keyed:
                raise ReviewReuseStoreError(
                    "store_index_corrupt", "legacy idempotency mapping is inconsistent"
                )

    tenant_ids = [task.tenant_id for task in records]
    report: Dict[str, Any] = {
        "apply": apply,
        "legacy_directories": len(legacy_dirs),
        "tasks": len(records),
        "tenants": len(set(tenant_ids)),
    }
    if not apply:
        return report

    staging = store_root.with_name(f".{store_root.name}.migration-{uuid.uuid4().hex}")
    backup = store_root.with_name(f"{store_root.name}.legacy-backup-{uuid.uuid4().hex}")
    staging_store = FilesystemReviewReuseStore(staging)
    cleanup_staging = True
    try:
        for task in records:
            staging_store._import_migrated(task)
        backup_created = False
        published = False
        try:
            if store_root.exists():
                os.replace(store_root, backup)
                backup_created = True
                cleanup_staging = False
                _fsync_directory(store_root.parent)
            os.replace(staging, store_root)
            published = True
            cleanup_staging = False
            _fsync_directory(store_root.parent)
        except OSError:
            if published and store_root.exists():
                os.replace(store_root, staging)
            try:
                if backup_created and backup.exists() and not store_root.exists():
                    os.replace(backup, store_root)
                elif staging.exists() and not store_root.exists():
                    os.replace(staging, store_root)
            except OSError:
                if staging.exists() and not store_root.exists():
                    os.replace(staging, store_root)
                    _fsync_directory(store_root.parent)
                    cleanup_staging = True
                raise
            if store_root.exists():
                _fsync_directory(store_root.parent)
                cleanup_staging = True
            raise
    except Exception:
        if cleanup_staging:
            shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        staging_store.close()
    report["backup"] = str(backup)
    return report


# Back-compat alias used by existing tests/service imports.
ReviewReuseStore = InMemoryReviewReuseStore


def create_review_reuse_store(*, read_only: bool = False) -> ReviewReuseStoreProtocol:
    backend = os.getenv(ENV_STORE, "memory").strip().lower()
    if backend in _TRUE_BACKENDS_FS:
        root = os.getenv(ENV_STORE_DIR, "data/review_reuse_tasks")
        return FilesystemReviewReuseStore(root, read_only=read_only)
    return InMemoryReviewReuseStore()
