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

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)


def _clean_text(value: Any) -> Optional[str]:
    """Collapse whitespace and return a non-empty string."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())
    return text or None


def _coerce_float(value: Any) -> Optional[float]:
    """Best-effort float conversion for evidence payloads."""
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _append_unique(items: List[str], value: Optional[str]) -> None:
    if value and value not in items:
        items.append(value)


def _truncate_text(value: str, limit: int = 280) -> str:
    if len(value) <= limit:
        return value
    return value[: limit - 3].rstrip() + "..."


def _build_review_evidence(score_breakdown: Dict[str, Any]) -> Dict[str, Any]:
    """Derive reviewer-facing evidence fields from score breakdown context."""
    if not isinstance(score_breakdown, dict):
        return {
            "evidence_count": 0,
            "evidence_sources": [],
            "evidence_summary": None,
            "evidence": [],
        }

    evidence: List[Dict[str, Any]] = []
    evidence_sources: List[str] = []
    summary_parts: List[str] = []

    source_contributions = score_breakdown.get("source_contributions")
    if isinstance(source_contributions, dict):
        ranked_sources: List[tuple[str, Optional[float]]] = []
        for source_name, contribution in source_contributions.items():
            clean_source = _clean_text(source_name)
            if clean_source is None:
                continue
            ranked_sources.append((clean_source, _coerce_float(contribution)))

        ranked_sources.sort(
            key=lambda item: item[1] if item[1] is not None else float("-inf"),
            reverse=True,
        )
        for source_name, contribution in ranked_sources:
            _append_unique(evidence_sources, source_name)
            entry: Dict[str, Any] = {
                "kind": "source_contribution",
                "source": source_name,
            }
            if contribution is not None:
                entry["score"] = round(contribution, 4)
            evidence.append(entry)

    hybrid_explanation = score_breakdown.get("hybrid_explanation")
    if isinstance(hybrid_explanation, dict):
        explanation_summary = _clean_text(hybrid_explanation.get("summary"))
        if explanation_summary:
            evidence.append(
                {
                    "kind": "hybrid_explanation",
                    "summary": explanation_summary,
                }
            )
            summary_parts.append(explanation_summary)

    hybrid_rejection = score_breakdown.get("hybrid_rejection")
    if isinstance(hybrid_rejection, dict):
        rejection_reason = _clean_text(hybrid_rejection.get("reason"))
        raw_source = _clean_text(hybrid_rejection.get("raw_source"))
        raw_confidence = _coerce_float(hybrid_rejection.get("raw_confidence"))
        if raw_source:
            _append_unique(evidence_sources, raw_source)
        if rejection_reason or raw_source or raw_confidence is not None:
            rejection_entry: Dict[str, Any] = {"kind": "hybrid_rejection"}
            if rejection_reason:
                rejection_entry["reason"] = rejection_reason
            if raw_source:
                rejection_entry["source"] = raw_source
            if raw_confidence is not None:
                rejection_entry["confidence"] = round(raw_confidence, 4)
            evidence.append(rejection_entry)

            rejection_summary = rejection_reason or "rejected"
            if raw_source:
                rejection_summary = f"{rejection_summary} via {raw_source}"
            if raw_confidence is not None:
                rejection_summary = f"{rejection_summary} ({raw_confidence:.3f})"
            summary_parts.append(f"Rejection: {rejection_summary}")

    decision_path = score_breakdown.get("decision_path")
    if isinstance(decision_path, list):
        steps = [_clean_text(step) for step in decision_path]
        clean_steps = [step for step in steps if step]
        if clean_steps:
            evidence.append({"kind": "decision_path", "steps": clean_steps})
            summary_parts.append(f"Path: {' -> '.join(clean_steps[:4])}")

    fusion_metadata = score_breakdown.get("fusion_metadata")
    if isinstance(fusion_metadata, dict):
        strategy = _clean_text(fusion_metadata.get("strategy"))
        agreement_score = _coerce_float(fusion_metadata.get("agreement_score"))
        num_sources = fusion_metadata.get("num_sources")
        if strategy or agreement_score is not None or num_sources is not None:
            fusion_entry: Dict[str, Any] = {"kind": "fusion_metadata"}
            if strategy:
                fusion_entry["strategy"] = strategy
            if agreement_score is not None:
                fusion_entry["agreement_score"] = round(agreement_score, 4)
            if isinstance(num_sources, int):
                fusion_entry["num_sources"] = num_sources
            evidence.append(fusion_entry)

    if evidence_sources:
        summary_parts.append(f"Sources: {', '.join(evidence_sources)}")

    evidence_summary = None
    if summary_parts:
        evidence_summary = _truncate_text(" | ".join(summary_parts))

    return {
        "evidence_count": len(evidence),
        "evidence_sources": evidence_sources,
        "evidence_summary": evidence_summary,
        "evidence": evidence,
    }


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
    evidence_count: int = 0
    evidence_sources: List[str] = Field(default_factory=list)
    evidence_summary: Optional[str] = None
    evidence: List[Dict[str, Any]] = Field(default_factory=list)
    status: SampleStatus = SampleStatus.PENDING
    true_type: Optional[str] = None
    reviewer_id: Optional[str] = None
    feedback_time: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @model_validator(mode="after")
    def _sync_review_evidence(self) -> "ActiveLearningSample":
        evidence_payload = _build_review_evidence(self.score_breakdown)
        self.evidence_count = int(evidence_payload["evidence_count"])
        self.evidence_sources = list(evidence_payload["evidence_sources"])
        self.evidence_summary = evidence_payload["evidence_summary"]
        self.evidence = list(evidence_payload["evidence"])
        return self


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
        self._retrain_threshold = int(
            os.environ.get("ACTIVE_LEARNING_RETRAIN_THRESHOLD", "10")
        )

        if self._store_type == "file":
            self._data_dir.mkdir(parents=True, exist_ok=True)
            self._load_samples()

    def _load_samples(self) -> None:
        """Load existing samples from file."""
        samples_file = self._data_dir / "samples.jsonl"
        if samples_file.exists():
            with open(samples_file, "r", encoding="utf-8") as f:
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
        with open(samples_file, "a", encoding="utf-8") as f:
            f.write(sample.model_dump_json() + "\n")

    def _update_sample_file(self) -> None:
        """Rewrite the samples file with current state."""
        if self._store_type != "file":
            return
        samples_file = self._data_dir / "samples.jsonl"
        with open(samples_file, "w", encoding="utf-8") as f:
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
        labeled_count = sum(
            1 for s in self._samples.values() if s.status == SampleStatus.LABELED
        )
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

        with open(export_file, "w", encoding="utf-8") as f:
            for sample in samples_to_export:
                export_data = {
                    "doc_id": sample.doc_id,
                    "predicted_type": sample.predicted_type,
                    "true_type": sample.true_type,
                    "confidence": sample.confidence,
                    "alternatives": sample.alternatives,
                    "score_breakdown": sample.score_breakdown,
                    "uncertainty_reason": sample.uncertainty_reason,
                    "evidence_count": sample.evidence_count,
                    "evidence_sources": sample.evidence_sources,
                    "evidence_summary": sample.evidence_summary,
                    "evidence": sample.evidence,
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
        pending = [
            s for s in self._samples.values() if s.status == SampleStatus.PENDING
        ]
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
