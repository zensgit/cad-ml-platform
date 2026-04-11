"""
Feedback Learning Pipeline.

Closes the loop: user corrections -> adaptive fusion weights -> model improvement.

The pipeline ingests corrections submitted via the feedback API, tracks per-branch
accuracy statistics, computes exponential-moving-average fusion weights, and
triggers weight updates when enough evidence has been accumulated.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.ml.learning.smart_sampler import SmartSampler

logger = logging.getLogger(__name__)

# The classifier branches whose weights we track and adapt.
BRANCH_NAMES: list[str] = [
    "filename",
    "graph2d",
    "titleblock",
    "process",
    "history_sequence",
]

# Default initial weights (mirroring hybrid_classifier defaults).
DEFAULT_WEIGHTS: dict[str, float] = {
    "filename": 0.7,
    "graph2d": 0.3,
    "titleblock": 0.0,
    "process": 0.15,
    "history_sequence": 0.2,
}


class FeedbackLearningPipeline:
    """Closes the loop: user corrections -> model improvement.

    Workflow
    -------
    1. ``ingest_correction`` stores each correction and updates per-branch stats.
    2. ``compute_adaptive_weights`` derives new fusion weights via EMA.
    3. ``get_uncertainty_samples`` delegates to :class:`SmartSampler`.
    4. ``trigger_weight_update`` writes new weights and logs history.
    """

    def __init__(
        self,
        feedback_dir: str = "data/feedback",
        min_samples: int = 20,
        alpha: float = 0.15,
        initial_weights: Optional[dict[str, float]] = None,
    ) -> None:
        """
        Parameters
        ----------
        feedback_dir:
            Directory where correction JSONL files are persisted.
        min_samples:
            Minimum corrections before adaptive weights are computed.
        alpha:
            EMA smoothing factor (0 < alpha <= 1). Higher = faster adaptation.
        initial_weights:
            Starting fusion weights per branch. Falls back to ``DEFAULT_WEIGHTS``.
        """
        self.feedback_dir = Path(feedback_dir)
        self.min_samples = max(1, min_samples)
        self.alpha = float(np.clip(alpha, 0.01, 1.0))

        self._correction_buffer: list[dict[str, Any]] = []
        self._weight_history: list[dict[str, Any]] = []

        # Per-branch accuracy tracking: {branch: {"correct": int, "total": int}}
        self._branch_stats: dict[str, dict[str, int]] = {
            branch: {"correct": 0, "total": 0} for branch in BRANCH_NAMES
        }

        # Current weights -- mutable, updated by trigger_weight_update.
        self._current_weights: dict[str, float] = dict(
            initial_weights or DEFAULT_WEIGHTS
        )

        # Smart sampler instance for uncertainty / diversity queries.
        self._sampler = SmartSampler()

        # Ensure persistence directory exists.
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "FeedbackLearningPipeline initialized",
            extra={
                "feedback_dir": str(self.feedback_dir),
                "min_samples": self.min_samples,
                "alpha": self.alpha,
                "initial_weights": self._current_weights,
            },
        )

    # ------------------------------------------------------------------
    # Correction ingestion
    # ------------------------------------------------------------------

    async def ingest_correction(
        self,
        file_id: str,
        predicted_label: str,
        corrected_label: str,
        confidence: float,
        source: str = "user",
        branch_predictions: Optional[dict[str, str]] = None,
    ) -> dict[str, Any]:
        """Ingest a single user correction and update learning state.

        Parameters
        ----------
        file_id:
            Identifier of the file whose classification is being corrected.
        predicted_label:
            The label the model originally predicted.
        corrected_label:
            The human-provided correct label.
        confidence:
            Confidence score of the original prediction (0..1).
        source:
            Origin of the correction (``"user"``, ``"reviewer"``, ``"auto"``).
        branch_predictions:
            Optional mapping ``{branch_name: predicted_label}`` so we can
            track which branches were correct.

        Returns
        -------
        dict with ``status``, ``correction_id``, ``is_correction``,
        ``total_corrections``, ``retrain_ready``.
        """
        is_correction = predicted_label != corrected_label
        correction_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)

        record: dict[str, Any] = {
            "id": correction_id,
            "file_id": file_id,
            "predicted_label": predicted_label,
            "corrected_label": corrected_label,
            "confidence": float(confidence),
            "is_correction": is_correction,
            "source": source,
            "branch_predictions": branch_predictions or {},
            "timestamp": now.isoformat(),
            "epoch": now.timestamp(),
        }

        self._correction_buffer.append(record)

        # Update per-branch accuracy stats when branch predictions are available.
        self._update_branch_stats(corrected_label, branch_predictions)

        # Persist to disk (append to JSONL).
        self._persist_correction(record)

        total = len(self._correction_buffer)
        retrain_ready = total >= self.min_samples

        logger.info(
            "Correction ingested",
            extra={
                "correction_id": correction_id,
                "is_correction": is_correction,
                "total": total,
                "retrain_ready": retrain_ready,
            },
        )

        return {
            "status": "ok",
            "correction_id": correction_id,
            "is_correction": is_correction,
            "total_corrections": total,
            "retrain_ready": retrain_ready,
        }

    # ------------------------------------------------------------------
    # Adaptive weight computation
    # ------------------------------------------------------------------

    def compute_adaptive_weights(self) -> dict[str, float]:
        """Compute updated fusion weights using exponential moving average.

        Strategy: branches that are correct more often get higher weight.

        For each branch *b*::

            accuracy_b = correct_b / total_b   (or 0.5 if no data)
            raw_b = old_weight_b * (1 - alpha) + accuracy_b * alpha

        The raw weights are then normalised so they sum to 1.0.

        Returns
        -------
        dict mapping branch name to its new weight (sums to 1.0).
        """
        total_corrections = len(self._correction_buffer)
        if total_corrections < self.min_samples:
            logger.debug(
                "Not enough corrections for adaptive weights "
                "(have %d, need %d); returning current weights.",
                total_corrections,
                self.min_samples,
            )
            return dict(self._current_weights)

        raw: dict[str, float] = {}
        for branch in BRANCH_NAMES:
            stats = self._branch_stats[branch]
            total = stats["total"]
            if total > 0:
                accuracy = stats["correct"] / total
            else:
                # No data for this branch -- neutral prior.
                accuracy = 0.5

            old_w = self._current_weights.get(branch, 0.0)
            raw[branch] = old_w * (1.0 - self.alpha) + accuracy * self.alpha

        # Normalise to sum to 1.0, guarding against degenerate all-zero case.
        total_raw = sum(raw.values())
        if total_raw <= 0:
            n = len(BRANCH_NAMES)
            return {b: 1.0 / n for b in BRANCH_NAMES}

        return {b: raw[b] / total_raw for b in BRANCH_NAMES}

    # ------------------------------------------------------------------
    # Active learning sample selection
    # ------------------------------------------------------------------

    def get_uncertainty_samples(
        self, predictions: list[dict[str, Any]], k: int = 10
    ) -> list[dict[str, Any]]:
        """Select the *k* most informative samples for human review.

        Delegates to :pyclass:`SmartSampler.combined_sampling` which blends
        uncertainty, margin, entropy, and disagreement strategies.
        """
        if not predictions:
            return []
        return self._sampler.combined_sampling(predictions, k=k)

    # ------------------------------------------------------------------
    # Status & reporting
    # ------------------------------------------------------------------

    def get_learning_status(self) -> dict[str, Any]:
        """Return a snapshot of the learning pipeline state."""
        total = len(self._correction_buffer)
        corrections_by_class: dict[str, int] = Counter(
            c["corrected_label"] for c in self._correction_buffer
        )
        accuracy_by_branch: dict[str, Optional[float]] = {}
        for branch in BRANCH_NAMES:
            stats = self._branch_stats[branch]
            if stats["total"] > 0:
                accuracy_by_branch[branch] = round(
                    stats["correct"] / stats["total"], 4
                )
            else:
                accuracy_by_branch[branch] = None

        return {
            "corrections_total": total,
            "corrections_by_class": dict(corrections_by_class),
            "weight_history": list(self._weight_history),
            "samples_until_retrain": max(0, self.min_samples - total),
            "current_weights": dict(self._current_weights),
            "accuracy_by_branch": accuracy_by_branch,
            "branch_stats": {
                b: dict(s) for b, s in self._branch_stats.items()
            },
        }

    # ------------------------------------------------------------------
    # Weight update trigger
    # ------------------------------------------------------------------

    async def trigger_weight_update(self) -> dict[str, Any]:
        """Compute and apply new fusion weights based on accumulated corrections.

        Returns
        -------
        dict with ``old_weights``, ``new_weights``, ``delta``, and ``applied``.
        """
        old_weights = dict(self._current_weights)
        new_weights = self.compute_adaptive_weights()

        # Check whether we actually have enough data to apply.
        total = len(self._correction_buffer)
        if total < self.min_samples:
            logger.info(
                "Weight update skipped: only %d corrections (need %d).",
                total,
                self.min_samples,
            )
            return {
                "old_weights": old_weights,
                "new_weights": old_weights,
                "delta": {b: 0.0 for b in BRANCH_NAMES},
                "applied": False,
                "reason": f"Only {total} corrections, need {self.min_samples}.",
            }

        # Apply the new weights.
        self._current_weights = dict(new_weights)

        delta = {
            b: round(new_weights[b] - old_weights.get(b, 0.0), 6)
            for b in BRANCH_NAMES
        }

        history_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "old_weights": old_weights,
            "new_weights": new_weights,
            "delta": delta,
            "corrections_count": total,
        }
        self._weight_history.append(history_entry)

        # Persist weight history snapshot.
        self._persist_weight_history(history_entry)

        logger.info(
            "Fusion weights updated",
            extra={"old": old_weights, "new": new_weights, "delta": delta},
        )

        return {
            "old_weights": old_weights,
            "new_weights": new_weights,
            "delta": delta,
            "applied": True,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_branch_stats(
        self,
        corrected_label: str,
        branch_predictions: Optional[dict[str, str]],
    ) -> None:
        """Update per-branch correct/total counters."""
        if not branch_predictions:
            return
        for branch in BRANCH_NAMES:
            pred = branch_predictions.get(branch)
            if pred is None:
                continue
            self._branch_stats[branch]["total"] += 1
            if pred == corrected_label:
                self._branch_stats[branch]["correct"] += 1

    def _persist_correction(self, record: dict[str, Any]) -> None:
        """Append a correction record to the JSONL file on disk."""
        try:
            path = self.feedback_dir / "corrections.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("Failed to persist correction to disk")

    def _persist_weight_history(self, entry: dict[str, Any]) -> None:
        """Append a weight-history entry to the JSONL file on disk."""
        try:
            path = self.feedback_dir / "weight_history.jsonl"
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            logger.exception("Failed to persist weight history to disk")

    def load_corrections_from_disk(self) -> int:
        """Reload persisted corrections into the in-memory buffer.

        Returns the number of records loaded.
        """
        path = self.feedback_dir / "corrections.jsonl"
        if not path.exists():
            return 0

        loaded = 0
        try:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    self._correction_buffer.append(record)
                    self._update_branch_stats(
                        record.get("corrected_label", ""),
                        record.get("branch_predictions"),
                    )
                    loaded += 1
        except (OSError, json.JSONDecodeError):
            logger.exception("Failed to load corrections from disk")

        logger.info("Loaded %d corrections from disk", loaded)
        return loaded
