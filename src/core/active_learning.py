"""
Active Learning module for human-in-the-loop model improvement.

Provides sample flagging, feedback collection, and training data export.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
import csv
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from src.core.classification.coarse_labels import normalize_coarse_label

logger = logging.getLogger(__name__)


class SampleStatus(str, Enum):
    """Status of a sample in the active learning pipeline."""

    PENDING = "pending"
    LABELED = "labeled"
    EXPORTED = "exported"
    SKIPPED = "skipped"


class ActiveLearningSample(BaseModel):
    """A sample flagged for human review."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    doc_id: str
    predicted_type: str
    predicted_fine_type: Optional[str] = None
    predicted_coarse_type: Optional[str] = None
    predicted_is_coarse_label: Optional[bool] = None
    confidence: float
    sample_type: Optional[str] = None
    feedback_priority: Optional[str] = None
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)
    evidence_count: int = 0
    evidence_sources: List[str] = Field(default_factory=list)
    evidence_summary: Optional[str] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    uncertainty_reason: str
    status: SampleStatus = SampleStatus.PENDING
    true_type: Optional[str] = None
    true_fine_type: Optional[str] = None
    true_coarse_type: Optional[str] = None
    true_is_coarse_label: Optional[bool] = None
    reviewer_id: Optional[str] = None
    feedback_time: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Training data governance (B6.4 Phase 1)
    sample_source: str = "unknown"  # analysis_review_queue | feedback_api | legacy_low_conf_queue | imported_manifest
    label_source: str = "unknown"   # human_feedback | human_review | claude_suggestion | model_auto | rule_auto | synthetic_demo
    review_source: Optional[str] = None  # human | claude_assisted | mixed
    human_verified: bool = False
    verified_by: Optional[str] = None
    verified_at: Optional[datetime] = None
    eligible_for_training: bool = False
    training_block_reason: Optional[str] = "missing_provenance"


def _normalize_predicted_contract(
    predicted_type: Optional[str],
) -> tuple[Optional[str], Optional[str], Optional[bool]]:
    fine_type = str(predicted_type or "").strip() or None
    coarse_type = normalize_coarse_label(fine_type)
    is_coarse_label = None
    if fine_type and coarse_type:
        is_coarse_label = fine_type == coarse_type
    return fine_type, coarse_type, is_coarse_label


def _normalize_sample(sample: ActiveLearningSample) -> ActiveLearningSample:
    fine_type, coarse_type, is_coarse_label = _normalize_predicted_contract(sample.predicted_type)
    sample.predicted_fine_type = sample.predicted_fine_type or fine_type
    sample.predicted_coarse_type = sample.predicted_coarse_type or coarse_type
    if sample.predicted_is_coarse_label is None:
        sample.predicted_is_coarse_label = is_coarse_label
    evidence_payload = _derive_evidence_payload(sample.score_breakdown)
    sample.evidence_count = int(evidence_payload["evidence_count"])
    sample.evidence_sources = list(evidence_payload["evidence_sources"])
    sample.evidence_summary = evidence_payload["evidence_summary"]
    sample.evidence = list(evidence_payload["evidence"])
    sample.sample_type = _derive_sample_type(sample)
    sample.feedback_priority = _derive_feedback_priority(sample)
    return sample


def _derive_evidence_payload(score_breakdown: Dict[str, Any]) -> Dict[str, Any]:
    evidence: List[Dict[str, Any]] = []
    evidence_sources: List[str] = []
    summary_parts: List[str] = []

    def add_source(source: Optional[str]) -> None:
        token = str(source or "").strip()
        if token and token not in evidence_sources:
            evidence_sources.append(token)

    def add_item(
        *,
        item_type: str,
        source: str,
        value: Any = None,
        text: Optional[str] = None,
    ) -> None:
        add_source(source)
        payload: Dict[str, Any] = {"type": item_type, "source": source}
        if value is not None:
            payload["value"] = value
        if text:
            payload["text"] = text
        evidence.append(payload)

    source_contributions = score_breakdown.get("source_contributions") or {}
    if isinstance(source_contributions, dict):
        ranked_sources = sorted(
            (
                (str(source).strip(), float(weight))
                for source, weight in source_contributions.items()
                if str(source).strip()
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        for source, weight in ranked_sources[:5]:
            add_item(
                item_type="source_contribution",
                source=source,
                value=round(weight, 6),
                text=f"{source}={round(weight, 3)}",
            )
        if ranked_sources:
            summary_parts.append(
                "sources="
                + ", ".join(
                    f"{source}:{round(weight, 2)}"
                    for source, weight in ranked_sources[:3]
                )
            )

    hybrid_explanation = score_breakdown.get("hybrid_explanation") or {}
    if isinstance(hybrid_explanation, dict):
        summary = str(hybrid_explanation.get("summary") or "").strip()
        if summary:
            add_item(item_type="summary", source="hybrid_explanation", text=summary)
            summary_parts.append(summary)

    hybrid_rejection = score_breakdown.get("hybrid_rejection") or {}
    if isinstance(hybrid_rejection, dict) and hybrid_rejection:
        rejection_reason = (
            str(hybrid_rejection.get("reason") or "").strip()
            or str(hybrid_rejection.get("status") or "").strip()
        )
        add_item(
            item_type="rejection",
            source="hybrid_rejection",
            text=rejection_reason or "hybrid_rejection",
            value={key: value for key, value in hybrid_rejection.items() if value is not None},
        )
        if rejection_reason:
            summary_parts.append(f"rejection={rejection_reason}")

    decision_path = score_breakdown.get("decision_path") or []
    if isinstance(decision_path, list) and decision_path:
        compact_path = [str(item).strip() for item in decision_path if str(item).strip()]
        if compact_path:
            add_item(
                item_type="decision_path",
                source="decision_path",
                value=compact_path,
                text=" -> ".join(compact_path[:4]),
            )
            summary_parts.append("path=" + "->".join(compact_path[:3]))

    fusion_metadata = score_breakdown.get("fusion_metadata") or {}
    if isinstance(fusion_metadata, dict) and fusion_metadata:
        fusion_strategy = str(
            fusion_metadata.get("strategy")
            or fusion_metadata.get("fusion_strategy")
            or fusion_metadata.get("selected_strategy")
            or ""
        ).strip()
        if fusion_strategy:
            add_item(
                item_type="fusion_strategy",
                source="fusion_metadata",
                text=fusion_strategy,
            )
            summary_parts.append(f"fusion={fusion_strategy}")

    evidence_summary = " | ".join(summary_parts[:3]) or None
    return {
        "evidence_count": len(evidence),
        "evidence_sources": evidence_sources,
        "evidence_summary": evidence_summary,
        "evidence": evidence,
    }


def _has_hybrid_rejection(sample: ActiveLearningSample) -> bool:
    reason = str(sample.uncertainty_reason or "").strip()
    if reason.startswith("hybrid_rejected:"):
        return True
    hybrid_rejection = sample.score_breakdown.get("hybrid_rejection")
    return isinstance(hybrid_rejection, dict) and bool(hybrid_rejection)


def _is_low_confidence_sample(sample: ActiveLearningSample) -> bool:
    reason = str(sample.uncertainty_reason or "").strip()
    return "low_confidence" in reason


def _derive_sample_type(sample: ActiveLearningSample) -> str:
    if sample.sample_type:
        return str(sample.sample_type)
    if sample.score_breakdown.get("review_has_knowledge_conflict"):
        return "knowledge_conflict"
    if sample.score_breakdown.get("review_has_branch_conflict"):
        return "branch_conflict"
    if sample.score_breakdown.get("review_has_hybrid_rejection"):
        return "hybrid_rejection"
    if sample.score_breakdown.get("review_is_low_confidence"):
        return "low_confidence"
    if _has_hybrid_rejection(sample):
        return "hybrid_rejection"
    if sample.score_breakdown.get("violations"):
        return "knowledge_conflict"
    if sample.score_breakdown.get("branch_conflicts"):
        return "branch_conflict"
    if _is_low_confidence_sample(sample):
        return "low_confidence"
    return "review"


def _derive_feedback_priority(sample: ActiveLearningSample) -> str:
    if sample.feedback_priority:
        return str(sample.feedback_priority)
    review_priority = str(sample.score_breakdown.get("review_priority") or "").strip()
    if review_priority:
        return review_priority
    is_correction = (
        sample.true_type is not None
        and str(sample.true_type).strip() != ""
        and sample.predicted_type != sample.true_type
    )
    if sample.score_breakdown.get("violations"):
        return "critical"
    if _has_hybrid_rejection(sample) or is_correction:
        return "high"
    if _is_low_confidence_sample(sample):
        return "medium"
    return "normal"


def _priority_rank(priority: Optional[str]) -> int:
    order = {
        "normal": 0,
        "medium": 1,
        "high": 2,
        "critical": 3,
    }
    return order.get(str(priority or "").strip().lower(), -1)


def _sample_type_rank(sample_type: Optional[str]) -> int:
    order = {
        "review": 0,
        "low_confidence": 1,
        "hybrid_rejection": 2,
        "branch_conflict": 3,
        "knowledge_conflict": 4,
    }
    return order.get(str(sample_type or "").strip().lower(), -1)


def _automation_ready(sample: ActiveLearningSample) -> bool:
    return bool(
        sample.score_breakdown.get("automation_ready")
        or sample.score_breakdown.get("review_automation_ready")
    )


class ActiveLearner:
    """Manages active learning samples and feedback collection."""

    def __init__(self) -> None:
        self._samples: Dict[str, ActiveLearningSample] = {}
        data_dir_env = os.environ.get("ACTIVE_LEARNING_DATA_DIR")
        if data_dir_env:
            self._data_dir = Path(data_dir_env)
        else:
            self._data_dir = Path(tempfile.gettempdir()) / "active_learning"
        self._store_type = os.environ.get("ACTIVE_LEARNING_STORE", "memory")
        self._retrain_threshold = int(os.environ.get("ACTIVE_LEARNING_RETRAIN_THRESHOLD", "10"))

        if self._store_type == "file":
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._load_samples()

    def _load_samples(self) -> None:
        """Load existing samples from file."""
        samples_file = self._data_dir / "samples.jsonl"
        if samples_file.exists():
            with open(samples_file, "r") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        sample = _normalize_sample(ActiveLearningSample(**data))
                        self._samples[sample.id] = sample

    def _persist_sample(self, sample: ActiveLearningSample) -> None:
        """Persist a sample to file."""
        if self._store_type != "file":
            return
        samples_file = self._data_dir / "samples.jsonl"
        with open(samples_file, "a") as f:
            f.write(sample.model_dump_json() + "\n")

    def _update_sample_file(self) -> None:
        """Rewrite the samples file with current state."""
        if self._store_type != "file":
            return
        samples_file = self._data_dir / "samples.jsonl"
        with open(samples_file, "w") as f:
            for sample in self._samples.values():
                f.write(sample.model_dump_json() + "\n")

    def flag_for_review(
        self,
        doc_id: str,
        predicted_type: str,
        confidence: float,
        alternatives: List[Dict[str, Any]],
        score_breakdown: Dict[str, Any],
        uncertainty_reason: str,
        sample_type: Optional[str] = None,
        feedback_priority: Optional[str] = None,
    ) -> ActiveLearningSample:
        """Flag a sample for human review."""
        fine_type, coarse_type, is_coarse_label = _normalize_predicted_contract(predicted_type)
        sample = ActiveLearningSample(
            doc_id=doc_id,
            predicted_type=predicted_type,
            predicted_fine_type=fine_type,
            predicted_coarse_type=coarse_type,
            predicted_is_coarse_label=is_coarse_label,
            confidence=confidence,
            sample_type=sample_type,
            feedback_priority=feedback_priority,
            alternatives=alternatives,
            score_breakdown=score_breakdown,
            uncertainty_reason=uncertainty_reason,
            sample_source="analysis_review_queue",
            label_source="model_auto",
        )
        sample = _normalize_sample(sample)
        self._samples[sample.id] = sample
        self._persist_sample(sample)
        return sample

    def submit_feedback(
        self,
        sample_id: str,
        true_type: str,
        true_fine_type: Optional[str] = None,
        true_coarse_type: Optional[str] = None,
        reviewer_id: Optional[str] = None,
        label_source: Optional[str] = None,
        review_source: Optional[str] = None,
        verified_by: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit feedback for a sample."""
        if sample_id not in self._samples:
            return {"status": "error", "message": "Sample not found"}

        sample = self._samples[sample_id]
        fine_type = str(true_fine_type or true_type or "").strip()
        coarse_type = normalize_coarse_label(true_coarse_type or fine_type)
        is_correction = sample.predicted_type != fine_type

        sample.true_type = fine_type or None
        sample.true_fine_type = fine_type or None
        sample.true_coarse_type = coarse_type
        sample.true_is_coarse_label = bool(fine_type and fine_type == coarse_type)
        sample.reviewer_id = reviewer_id
        sample.feedback_time = datetime.utcnow()
        sample.status = SampleStatus.LABELED
        derived_priority = _derive_feedback_priority(
            sample.model_copy(update={"feedback_priority": None})
        )
        if _priority_rank(derived_priority) > _priority_rank(sample.feedback_priority):
            sample.feedback_priority = derived_priority

        # Provenance tracking
        if label_source:
            sample.label_source = label_source
        if review_source:
            sample.review_source = review_source
        if label_source in ("human_feedback", "human_review"):
            sample.human_verified = True
            sample.verified_by = verified_by or reviewer_id
            sample.verified_at = datetime.utcnow()
            sample.eligible_for_training = True
            sample.training_block_reason = None

        self._update_sample_file()

        return {
            "status": "ok",
            "is_correction": is_correction,
            "sample_id": sample_id,
        }

    def check_retrain_threshold(self) -> Dict[str, Any]:
        """Check if retrain threshold is reached.

        Uses eligible_for_training (not just labeled status) to align with
        auto_retrain.sh provenance gate — only human-verified, eligible
        samples count toward the threshold.
        """
        labeled_count = sum(1 for s in self._samples.values() if s.status == SampleStatus.LABELED)
        eligible_count = sum(
            1 for s in self._samples.values()
            if s.status == SampleStatus.LABELED and s.eligible_for_training
        )
        remaining = max(self._retrain_threshold - eligible_count, 0)
        if eligible_count >= self._retrain_threshold:
            recommendation = "threshold_met"
        elif remaining == 1:
            recommendation = "need_1_more_eligible_sample"
        else:
            recommendation = f"need_{remaining}_more_eligible_samples"
        return {
            "ready": eligible_count >= self._retrain_threshold,
            "labeled_samples": labeled_count,
            "eligible_samples": eligible_count,
            "threshold": self._retrain_threshold,
            "remaining_samples": remaining,
            "recommendation": recommendation,
        }

    def export_training_data(
        self,
        format: str = "jsonl",  # noqa: A002
        only_labeled: bool = True,
        include_unverified: bool = False,
    ) -> Dict[str, Any]:
        """Export labeled samples for retraining."""
        samples_to_export = [
            s
            for s in self._samples.values()
            if (not only_labeled or s.status == SampleStatus.LABELED)
            and (include_unverified or s.eligible_for_training)
        ]

        if not samples_to_export:
            return {"status": "error", "message": "No samples to export"}

        export_dir = self._data_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        export_file = export_dir / f"training_data_{timestamp}.{format}"

        with open(export_file, "w") as f:
            for sample in samples_to_export:
                export_data = {
                    "doc_id": sample.doc_id,
                    "analysis_id": sample.doc_id,
                    "predicted_type": sample.predicted_type,
                    "predicted_fine_type": sample.predicted_fine_type or sample.predicted_type,
                    "predicted_coarse_type": sample.predicted_coarse_type
                    or normalize_coarse_label(sample.predicted_type),
                    "predicted_is_coarse_label": sample.predicted_is_coarse_label,
                    "true_type": sample.true_type,
                    "true_fine_type": sample.true_fine_type or sample.true_type,
                    "true_coarse_type": sample.true_coarse_type,
                    "true_is_coarse_label": sample.true_is_coarse_label,
                    "confidence": sample.confidence,
                    "sample_type": _derive_sample_type(sample),
                    "feedback_priority": _derive_feedback_priority(sample),
                    "alternatives": sample.alternatives,
                    "score_breakdown": sample.score_breakdown,
                    "evidence_count": sample.evidence_count,
                    "evidence_sources": sample.evidence_sources,
                    "evidence_summary": sample.evidence_summary,
                    "evidence": sample.evidence,
                    "uncertainty_reason": sample.uncertainty_reason,
                }
                export_data["correct_label"] = (
                    export_data["true_fine_type"] or export_data["true_type"]
                )
                export_data["correct_fine_label"] = export_data["true_fine_type"]
                export_data["correct_coarse_label"] = export_data["true_coarse_type"]
                export_data["original_label"] = export_data["predicted_fine_type"]
                export_data["original_fine_label"] = export_data["predicted_fine_type"]
                export_data["original_coarse_label"] = export_data["predicted_coarse_type"]
                f.write(json.dumps(export_data) + "\n")
                sample.status = SampleStatus.EXPORTED

        self._update_sample_file()

        return {
            "status": "ok",
            "count": len(samples_to_export),
            "file": str(export_file),
        }

    def get_pending_samples(self, limit: int = 10) -> List[ActiveLearningSample]:
        """Get pending samples for review."""
        pending = [s for s in self._samples.values() if s.status == SampleStatus.PENDING]
        return [_normalize_sample(sample) for sample in pending[:limit]]

    def _rank_review_queue_samples(
        self,
        samples: List[ActiveLearningSample],
        *,
        sort_by: str,
    ) -> tuple[List[ActiveLearningSample], str]:
        normalized_sort_by = str(sort_by or "priority").strip().lower() or "priority"

        def _priority_sort_key(sample: ActiveLearningSample) -> tuple[int, int, float, datetime]:
            return (
                _priority_rank(_derive_feedback_priority(sample)),
                _sample_type_rank(_derive_sample_type(sample)),
                -float(sample.confidence),
                -sample.created_at.timestamp(),
            )

        def _confidence_sort_key(sample: ActiveLearningSample) -> tuple[float, int, float]:
            return (
                float(sample.confidence),
                -_priority_rank(_derive_feedback_priority(sample)),
                sample.created_at.timestamp(),
            )

        def _created_at_sort_key(sample: ActiveLearningSample) -> tuple[float, int, float]:
            return (
                sample.created_at.timestamp(),
                -_priority_rank(_derive_feedback_priority(sample)),
                float(sample.confidence),
            )

        sort_key = _priority_sort_key
        reverse = True
        if normalized_sort_by == "confidence":
            sort_key = _confidence_sort_key
            reverse = False
        elif normalized_sort_by == "created_at":
            sort_key = _created_at_sort_key
            reverse = False

        return sorted(samples, key=sort_key, reverse=reverse), normalized_sort_by

    def _filter_review_queue_samples(
        self,
        *,
        status: str = "pending",
        sample_type: Optional[str] = None,
        feedback_priority: Optional[str] = None,
    ) -> List[ActiveLearningSample]:
        normalized_status = str(status or "pending").strip().lower() or "pending"
        normalized_sample_type = str(sample_type or "").strip() or None
        normalized_priority = str(feedback_priority or "").strip() or None

        samples = [_normalize_sample(sample) for sample in self._samples.values()]
        if normalized_status != "all":
            samples = [
                sample
                for sample in samples
                if sample.status.value == normalized_status
            ]
        if normalized_sample_type:
            samples = [
                sample
                for sample in samples
                if _derive_sample_type(sample) == normalized_sample_type
            ]
        if normalized_priority:
            samples = [
                sample
                for sample in samples
                if _derive_feedback_priority(sample) == normalized_priority
            ]
        return samples

    def _build_review_queue_summary(
        self,
        samples: List[ActiveLearningSample],
        *,
        status: str,
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "status": status,
            "total": len(samples),
            "by_sample_type": {},
            "by_feedback_priority": {},
            "by_decision_source": {},
            "by_uncertainty_reason": {},
            "by_review_reason": {},
            "critical_count": 0,
            "high_count": 0,
            "automation_ready_count": 0,
        }
        for sample in samples:
            derived_sample_type = _derive_sample_type(sample)
            derived_priority = _derive_feedback_priority(sample)
            decision_source = str(
                sample.score_breakdown.get("final_decision_source")
                or sample.score_breakdown.get("decision_source")
                or "unknown"
            )
            uncertainty_reason = str(sample.uncertainty_reason or "unknown")
            review_reasons = sample.score_breakdown.get("review_reasons") or []
            if not review_reasons and uncertainty_reason:
                review_reasons = [uncertainty_reason]

            summary["by_sample_type"][derived_sample_type] = (
                summary["by_sample_type"].get(derived_sample_type, 0) + 1
            )
            summary["by_feedback_priority"][derived_priority] = (
                summary["by_feedback_priority"].get(derived_priority, 0) + 1
            )
            if derived_priority == "critical":
                summary["critical_count"] += 1
            if derived_priority == "high":
                summary["high_count"] += 1
            summary["by_decision_source"][decision_source] = (
                summary["by_decision_source"].get(decision_source, 0) + 1
            )
            summary["by_uncertainty_reason"][uncertainty_reason] = (
                summary["by_uncertainty_reason"].get(uncertainty_reason, 0) + 1
            )
            if _automation_ready(sample):
                summary["automation_ready_count"] += 1
            for reason in review_reasons:
                reason_key = str(reason or "").strip() or "unknown"
                summary["by_review_reason"][reason_key] = (
                    summary["by_review_reason"].get(reason_key, 0) + 1
                )
        total = max(int(summary["total"]), 1)
        critical_count = int(summary["critical_count"])
        high_count = int(summary["high_count"])
        automation_ready_count = int(summary["automation_ready_count"])
        summary["critical_ratio"] = round(critical_count / total, 6)
        summary["high_ratio"] = round(high_count / total, 6)
        summary["automation_ready_ratio"] = round(automation_ready_count / total, 6)
        if int(summary["total"]) <= 0:
            summary["operational_status"] = "under_control"
        elif critical_count > 0:
            summary["operational_status"] = "critical_backlog"
        elif high_count > 0:
            summary["operational_status"] = "managed_backlog"
        else:
            summary["operational_status"] = "routine_backlog"
        return summary

    def get_review_queue(
        self,
        limit: int = 20,
        offset: int = 0,
        status: str = "pending",
        sample_type: Optional[str] = None,
        feedback_priority: Optional[str] = None,
        sort_by: str = "priority",
    ) -> Dict[str, Any]:
        """Return a ranked review queue with filtering and lightweight summary."""
        normalized_status = str(status or "pending").strip().lower() or "pending"
        normalized_sort_by = str(sort_by or "priority").strip().lower() or "priority"

        samples = self._filter_review_queue_samples(
            status=normalized_status,
            sample_type=sample_type,
            feedback_priority=feedback_priority,
        )
        summary = self._build_review_queue_summary(samples, status=normalized_status)
        ranked, normalized_sort_by = self._rank_review_queue_samples(
            samples,
            sort_by=normalized_sort_by,
        )
        window = ranked[offset : offset + limit]
        return {
            "total": len(ranked),
            "returned": len(window),
            "limit": limit,
            "offset": offset,
            "has_more": offset + len(window) < len(ranked),
            "sort_by": normalized_sort_by,
            "summary": summary,
            "items": window,
        }

    def get_review_queue_stats(
        self,
        *,
        status: str = "pending",
        sample_type: Optional[str] = None,
        feedback_priority: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized_status = str(status or "pending").strip().lower() or "pending"
        samples = self._filter_review_queue_samples(
            status=normalized_status,
            sample_type=sample_type,
            feedback_priority=feedback_priority,
        )
        return self._build_review_queue_summary(samples, status=normalized_status)

    def export_review_queue(
        self,
        *,
        status: str = "pending",
        sample_type: Optional[str] = None,
        feedback_priority: Optional[str] = None,
        sort_by: str = "priority",
        format: str = "csv",  # noqa: A002
    ) -> Dict[str, Any]:
        normalized_status = str(status or "pending").strip().lower() or "pending"
        normalized_format = str(format or "csv").strip().lower() or "csv"
        if normalized_format not in {"csv", "jsonl"}:
            return {"status": "error", "message": f"Unsupported export format: {normalized_format}"}

        samples = self._filter_review_queue_samples(
            status=normalized_status,
            sample_type=sample_type,
            feedback_priority=feedback_priority,
        )
        if not samples:
            return {"status": "error", "message": "No review queue samples to export"}

        summary = self._build_review_queue_summary(samples, status=normalized_status)
        ranked, normalized_sort_by = self._rank_review_queue_samples(samples, sort_by=sort_by)

        export_dir = self._data_dir / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        export_file = export_dir / f"review_queue_{timestamp}.{normalized_format}"

        rows: List[Dict[str, Any]] = []
        for sample in ranked:
            review_reasons = sample.score_breakdown.get("review_reasons")
            if not isinstance(review_reasons, list):
                review_reasons = []
            rows.append(
                {
                    "id": sample.id,
                    "doc_id": sample.doc_id,
                    "status": sample.status.value,
                    "confidence": sample.confidence,
                    "predicted_type": sample.predicted_type,
                    "predicted_fine_type": sample.predicted_fine_type,
                    "predicted_coarse_type": sample.predicted_coarse_type,
                    "predicted_is_coarse_label": sample.predicted_is_coarse_label,
                    "sample_type": _derive_sample_type(sample),
                    "feedback_priority": _derive_feedback_priority(sample),
                    "uncertainty_reason": sample.uncertainty_reason,
                    "decision_source": str(
                        sample.score_breakdown.get("final_decision_source")
                        or sample.score_breakdown.get("decision_source")
                        or "unknown"
                    ),
                    "evidence_count": sample.evidence_count,
                    "evidence_sources": sample.evidence_sources,
                    "evidence_summary": sample.evidence_summary,
                    "evidence": sample.evidence,
                    "review_reasons": review_reasons,
                    "true_type": sample.true_type,
                    "true_fine_type": sample.true_fine_type,
                    "true_coarse_type": sample.true_coarse_type,
                    "reviewer_id": sample.reviewer_id,
                    "created_at": sample.created_at.isoformat(),
                    "feedback_time": sample.feedback_time.isoformat()
                    if sample.feedback_time
                    else None,
                    "score_breakdown": sample.score_breakdown,
                }
            )

        if normalized_format == "jsonl":
            with open(export_file, "w") as handle:
                for row in rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            fieldnames = [
                "id",
                "doc_id",
                "status",
                "confidence",
                "predicted_type",
                "predicted_fine_type",
                "predicted_coarse_type",
                "predicted_is_coarse_label",
                "sample_type",
                "feedback_priority",
                "uncertainty_reason",
                "decision_source",
                "evidence_count",
                "evidence_sources",
                "evidence_summary",
                "evidence",
                "review_reasons",
                "true_type",
                "true_fine_type",
                "true_coarse_type",
                "reviewer_id",
                "created_at",
                "feedback_time",
                "score_breakdown",
            ]
            with open(export_file, "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            **row,
                            "evidence_sources": json.dumps(
                                row["evidence_sources"], ensure_ascii=False
                            ),
                            "evidence": json.dumps(row["evidence"], ensure_ascii=False),
                            "review_reasons": json.dumps(row["review_reasons"], ensure_ascii=False),
                            "score_breakdown": json.dumps(
                                row["score_breakdown"], ensure_ascii=False
                            ),
                        }
                    )

        return {
            "status": "ok",
            "count": len(rows),
            "file": str(export_file),
            "format": normalized_format,
            "sort_by": normalized_sort_by,
            "summary": summary,
        }

    def get_sample(self, sample_id: str) -> Optional[ActiveLearningSample]:
        """Get a sample by ID."""
        sample = self._samples.get(sample_id)
        if sample is None:
            return None
        return _normalize_sample(sample)

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about samples."""
        stats = {status.value: 0 for status in SampleStatus}
        for sample in self._samples.values():
            stats[sample.status.value] += 1
        stats["total"] = len(self._samples)
        return stats

    def get_stats_summary(self) -> Dict[str, Any]:
        """Get richer observability counters for review and retraining."""
        summary: Dict[str, Any] = {
            "status": self.get_stats(),
            "sample_type": {},
            "feedback_priority": {},
            "predicted_coarse_type": {},
            "predicted_fine_type": {},
            "labeled_true_coarse_type": {},
            "labeled_true_fine_type": {},
            "correction_count": 0,
        }
        for raw_sample in self._samples.values():
            sample = _normalize_sample(raw_sample)
            sample_type = _derive_sample_type(sample)
            feedback_priority = _derive_feedback_priority(sample)
            predicted_coarse = sample.predicted_coarse_type or "unknown"
            predicted_fine = sample.predicted_fine_type or sample.predicted_type or "unknown"

            summary["sample_type"][sample_type] = summary["sample_type"].get(sample_type, 0) + 1
            summary["feedback_priority"][feedback_priority] = (
                summary["feedback_priority"].get(feedback_priority, 0) + 1
            )
            summary["predicted_coarse_type"][predicted_coarse] = (
                summary["predicted_coarse_type"].get(predicted_coarse, 0) + 1
            )
            summary["predicted_fine_type"][predicted_fine] = (
                summary["predicted_fine_type"].get(predicted_fine, 0) + 1
            )

            if sample.status == SampleStatus.LABELED:
                true_coarse = sample.true_coarse_type or "unknown"
                true_fine = sample.true_fine_type or sample.true_type or "unknown"
                summary["labeled_true_coarse_type"][true_coarse] = (
                    summary["labeled_true_coarse_type"].get(true_coarse, 0) + 1
                )
                summary["labeled_true_fine_type"][true_fine] = (
                    summary["labeled_true_fine_type"].get(true_fine, 0) + 1
                )
                if predicted_fine != true_fine:
                    summary["correction_count"] += 1
        return summary


# Singleton instance
_active_learner: Optional[ActiveLearner] = None


def get_active_learner() -> ActiveLearner:
    """Get the global ActiveLearner instance."""
    global _active_learner
    if _active_learner is None:
        _active_learner = ActiveLearner()
    return _active_learner


def reset_active_learner() -> None:
    """Reset the global ActiveLearner instance."""
    global _active_learner
    _active_learner = None
