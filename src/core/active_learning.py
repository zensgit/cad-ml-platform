"""
Active Learning module for human-in-the-loop model improvement.

Provides sample flagging, feedback collection, and training data export.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

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
    confidence: float
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    score_breakdown: Dict[str, Any] = Field(default_factory=dict)
    uncertainty_reason: str
    status: SampleStatus = SampleStatus.PENDING
    true_type: Optional[str] = None
    reviewer_id: Optional[str] = None
    feedback_time: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ActiveLearner:
    """Manages active learning samples and feedback collection."""

    def __init__(self) -> None:
        self._samples: Dict[str, ActiveLearningSample] = {}
        self._data_dir = Path(os.environ.get("ACTIVE_LEARNING_DATA_DIR", "/tmp/active_learning"))
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
                        sample = ActiveLearningSample(**data)
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
    ) -> ActiveLearningSample:
        """Flag a sample for human review."""
        sample = ActiveLearningSample(
            doc_id=doc_id,
            predicted_type=predicted_type,
            confidence=confidence,
            alternatives=alternatives,
            score_breakdown=score_breakdown,
            uncertainty_reason=uncertainty_reason,
        )
        self._samples[sample.id] = sample
        self._persist_sample(sample)
        return sample

    def submit_feedback(
        self,
        sample_id: str,
        true_type: str,
        reviewer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Submit feedback for a sample."""
        if sample_id not in self._samples:
            return {"status": "error", "message": "Sample not found"}

        sample = self._samples[sample_id]
        is_correction = sample.predicted_type != true_type

        sample.true_type = true_type
        sample.reviewer_id = reviewer_id
        sample.feedback_time = datetime.utcnow()
        sample.status = SampleStatus.LABELED

        self._update_sample_file()

        return {
            "status": "ok",
            "is_correction": is_correction,
            "sample_id": sample_id,
        }

    def check_retrain_threshold(self) -> Dict[str, Any]:
        """Check if retrain threshold is reached."""
        labeled_count = sum(1 for s in self._samples.values() if s.status == SampleStatus.LABELED)
        return {
            "ready": labeled_count >= self._retrain_threshold,
            "labeled_samples": labeled_count,
            "threshold": self._retrain_threshold,
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
                    "predicted_type": sample.predicted_type,
                    "true_type": sample.true_type,
                    "confidence": sample.confidence,
                    "alternatives": sample.alternatives,
                }
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
        return pending[:limit]

    def get_sample(self, sample_id: str) -> Optional[ActiveLearningSample]:
        """Get a sample by ID."""
        return self._samples.get(sample_id)

    def get_stats(self) -> Dict[str, int]:
        """Get statistics about samples."""
        stats = {status.value: 0 for status in SampleStatus}
        for sample in self._samples.values():
            stats[sample.status.value] += 1
        stats["total"] = len(self._samples)
        return stats


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
