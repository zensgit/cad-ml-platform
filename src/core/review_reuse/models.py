"""Domain models for ReviewReuseTask (strategy §3.3 / plan §4–§8)."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    pending = "pending"
    running = "running"
    evidence_ready = "evidence_ready"
    decided = "decided"
    failed = "failed"
    canceled = "canceled"


class TaskEventType(str, Enum):
    submitted = "submitted"
    input_validated = "input_validated"
    recall_started = "recall_started"
    recall_completed = "recall_completed"
    precision_started = "precision_started"
    precision_completed = "precision_completed"
    evidence_pack_ready = "evidence_pack_ready"
    decision_submitted = "decision_submitted"
    failed = "failed"
    canceled = "canceled"


class CandidateState(str, Enum):
    duplicate = "duplicate"
    similar = "similar"
    different = "different"
    insufficient_evidence = "insufficient_evidence"


class HumanDecisionState(str, Enum):
    """Strategy-center: reuse / revise / new. Extensions marked in design-lock."""

    reuse = "reuse"
    revise = "revise"
    new = "new"
    reject_candidate = "reject_candidate"  # implementation extension
    need_more_info = "need_more_info"  # implementation extension


DECISION_REASON_CODES = frozenset(
    {
        "geometry_match",
        "visual_similarity_only",
        "needs_modification",
        "new_part_required",
        "insufficient_evidence",
        "incorrect_candidate",
        "other",
    }
)


class RejectionReason(str, Enum):
    missing_geom_json = "missing_geom_json"
    version_gate_filtered = "version_gate_filtered"
    low_precision_score = "low_precision_score"
    vision_only_unverified = "vision_only_unverified"
    unsupported_file_type = "unsupported_file_type"
    external_service_unavailable = "external_service_unavailable"
    tool_unavailable = "tool_unavailable"


class TaskEvent(BaseModel):
    event_type: TaskEventType
    ts: float
    detail: Dict[str, Any] = Field(default_factory=dict)


class CandidateDecision(BaseModel):
    candidate_id: str
    candidate_source: str = "archive"
    state: CandidateState
    scores: Dict[str, Optional[float]] = Field(default_factory=dict)
    verification: Dict[str, Any] = Field(default_factory=dict)
    rejection_reasons: List[str] = Field(default_factory=list)
    provenance: Dict[str, Any] = Field(default_factory=dict)


class HumanDecision(BaseModel):
    state: HumanDecisionState
    reviewer_id: str
    reviewer_kind: str = "validated_principal"
    reason_codes: List[str] = Field(default_factory=list)
    reason_text: str = ""
    candidate_id: Optional[str] = None
    ts: float
    idempotency_key: Optional[str] = None
    idempotency_digest: Optional[str] = None
    reviewed_revision: int = Field(default=1, ge=1, strict=True)
    evidence_pack_sha256: str = ""


class ReviewReuseTask(BaseModel):
    task_id: str
    tenant_id: str
    status: TaskStatus
    created_at: float
    updated_at: float
    source_file_name: str = ""
    source_content_sha256: str = ""
    idempotency_key: Optional[str] = None
    idempotency_digest: Optional[str] = None
    revision: int = Field(default=1, ge=1, strict=True)
    trace_id: str
    candidates: List[CandidateDecision] = Field(default_factory=list)
    events: List[TaskEvent] = Field(default_factory=list)
    evidence_pack: Optional[Dict[str, Any]] = None
    human_decision: Optional[HumanDecision] = None
    error: Optional[str] = None
    error_code: Optional[str] = None
    calibration_version: str = "workbench-mvp-0"
    calibration_status: Literal["uncalibrated", "calibrated"] = "uncalibrated"
