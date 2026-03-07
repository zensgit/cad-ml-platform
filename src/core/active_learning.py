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
    uncertainty_reason: str
    status: SampleStatus = SampleStatus.PENDING
    true_type: Optional[str] = None
    true_fine_type: Optional[str] = None
    true_coarse_type: Optional[str] = None
    true_is_coarse_label: Optional[bool] = None
    reviewer_id: Optional[str] = None
    feedback_time: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


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
    sample.sample_type = _derive_sample_type(sample)
    sample.feedback_priority = _derive_feedback_priority(sample)
    return sample


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

        self._update_sample_file()

        return {
            "status": "ok",
            "is_correction": is_correction,
            "sample_id": sample_id,
        }

    def check_retrain_threshold(self) -> Dict[str, Any]:
        """Check if retrain threshold is reached."""
        labeled_count = sum(1 for s in self._samples.values() if s.status == SampleStatus.LABELED)
        remaining = max(self._retrain_threshold - labeled_count, 0)
        if labeled_count >= self._retrain_threshold:
            recommendation = "threshold_met"
        elif remaining == 1:
            recommendation = "need_1_more_labeled_sample"
        else:
            recommendation = f"need_{remaining}_more_labeled_samples"
        return {
            "ready": labeled_count >= self._retrain_threshold,
            "labeled_samples": labeled_count,
            "threshold": self._retrain_threshold,
            "remaining_samples": remaining,
            "recommendation": recommendation,
        }

    def export_training_data(
        self,
        format: str = "jsonl",  # noqa: A002
        only_labeled: bool = True,
    ) -> Dict[str, Any]:
        """Export labeled samples for retraining."""
        samples_to_export = [
            s
            for s in self._samples.values()
            if not only_labeled or s.status == SampleStatus.LABELED
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
        normalized_sample_type = str(sample_type or "").strip() or None
        normalized_priority = str(feedback_priority or "").strip() or None
        normalized_sort_by = str(sort_by or "priority").strip().lower() or "priority"

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

        summary: Dict[str, Any] = {
            "status": normalized_status,
            "total": len(samples),
            "by_sample_type": {},
            "by_feedback_priority": {},
        }
        for sample in samples:
            derived_sample_type = _derive_sample_type(sample)
            derived_priority = _derive_feedback_priority(sample)
            summary["by_sample_type"][derived_sample_type] = (
                summary["by_sample_type"].get(derived_sample_type, 0) + 1
            )
            summary["by_feedback_priority"][derived_priority] = (
                summary["by_feedback_priority"].get(derived_priority, 0) + 1
            )

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

        ranked = sorted(samples, key=sort_key, reverse=reverse)
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
