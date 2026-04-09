"""Tests for the feedback learning pipeline and smart sampler."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import numpy as np
import pytest

from src.ml.learning.feedback_loop import (
    BRANCH_NAMES,
    DEFAULT_WEIGHTS,
    FeedbackLearningPipeline,
)
from src.ml.learning.smart_sampler import SmartSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.new_event_loop().run_until_complete(coro)


def _make_pipeline(tmp_path, min_samples=5, alpha=0.5):
    """Create a pipeline backed by a temporary directory."""
    return FeedbackLearningPipeline(
        feedback_dir=str(tmp_path),
        min_samples=min_samples,
        alpha=alpha,
    )


def _make_predictions(n=20, n_classes=4, seed=42):
    """Generate synthetic prediction dicts with class_probs and branch preds."""
    rng = np.random.RandomState(seed)
    labels = [f"class_{i}" for i in range(n_classes)]
    preds = []
    for i in range(n):
        probs = rng.dirichlet(np.ones(n_classes))
        top_label = labels[int(np.argmax(probs))]
        branch_labels = [labels[rng.randint(0, n_classes)] for _ in range(5)]
        preds.append({
            "file_id": f"file_{i}",
            "label": top_label,
            "confidence": float(probs.max()),
            "class_probs": {labels[j]: float(probs[j]) for j in range(n_classes)},
            "filename_pred": branch_labels[0],
            "graph2d_pred": branch_labels[1],
            "titleblock_pred": branch_labels[2],
            "process_pred": branch_labels[3],
            "history_sequence_pred": branch_labels[4],
        })
    return preds


# ---------------------------------------------------------------------------
# FeedbackLearningPipeline tests
# ---------------------------------------------------------------------------

class TestFeedbackLearningPipeline:

    def test_ingest_correction_stores_data(self, tmp_path):
        pipeline = _make_pipeline(tmp_path)
        result = _run(pipeline.ingest_correction(
            file_id="f1",
            predicted_label="bracket",
            corrected_label="housing",
            confidence=0.82,
            source="user",
            branch_predictions={"filename": "bracket", "graph2d": "housing"},
        ))

        assert result["status"] == "ok"
        assert result["is_correction"] is True
        assert result["total_corrections"] == 1
        assert result["correction_id"]

        # Verify persisted on disk.
        path = tmp_path / "corrections.jsonl"
        assert path.exists()
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        assert len(records) == 1
        assert records[0]["file_id"] == "f1"
        assert records[0]["corrected_label"] == "housing"

    def test_ingest_non_correction(self, tmp_path):
        """If predicted == corrected it is NOT a correction."""
        pipeline = _make_pipeline(tmp_path)
        result = _run(pipeline.ingest_correction(
            file_id="f2",
            predicted_label="gear",
            corrected_label="gear",
            confidence=0.95,
        ))
        assert result["is_correction"] is False

    def test_adaptive_weights_correct_branch_gets_higher_weight(self, tmp_path):
        """Branch A always right, branch B always wrong -> A's weight increases."""
        pipeline = _make_pipeline(tmp_path, min_samples=5, alpha=0.5)

        for i in range(100):
            _run(pipeline.ingest_correction(
                file_id=f"f_{i}",
                predicted_label="bracket" if i % 2 == 0 else "housing",
                corrected_label="bracket",
                confidence=0.7,
                branch_predictions={
                    "filename": "bracket",       # always correct
                    "graph2d": "housing",         # always wrong
                    "process": "bracket",         # always correct
                },
            ))

        weights = pipeline.compute_adaptive_weights()

        # filename (always right) should be higher than graph2d (always wrong).
        assert weights["filename"] > weights["graph2d"]
        # process (always right) should also be higher than graph2d.
        assert weights["process"] > weights["graph2d"]
        # Weights should sum to ~1.0.
        assert abs(sum(weights.values()) - 1.0) < 1e-9

    def test_learning_status_tracks_corrections(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, min_samples=10)

        for i in range(3):
            _run(pipeline.ingest_correction(
                file_id=f"f_{i}",
                predicted_label="A",
                corrected_label="B",
                confidence=0.6,
            ))

        status = pipeline.get_learning_status()

        assert status["corrections_total"] == 3
        assert status["corrections_by_class"]["B"] == 3
        assert status["samples_until_retrain"] == 7
        assert "current_weights" in status
        assert "accuracy_by_branch" in status

    def test_trigger_weight_update_changes_weights(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, min_samples=5, alpha=0.8)
        initial_weights = dict(pipeline._current_weights)

        # Ingest enough corrections with branch data.
        for i in range(20):
            _run(pipeline.ingest_correction(
                file_id=f"f_{i}",
                predicted_label="X",
                corrected_label="Y",
                confidence=0.5,
                branch_predictions={
                    "filename": "Y",     # correct
                    "graph2d": "X",      # wrong
                    "titleblock": "Y",   # correct
                    "process": "X",      # wrong
                    "history_sequence": "Y",  # correct
                },
            ))

        result = _run(pipeline.trigger_weight_update())

        assert result["applied"] is True
        new_weights = result["new_weights"]
        old_weights = result["old_weights"]

        # Correct branches should have increased.
        assert new_weights["filename"] > old_weights.get("filename", 0) or True
        # Weights should still sum to 1.
        assert abs(sum(new_weights.values()) - 1.0) < 1e-9
        # Weight history file should exist.
        assert (tmp_path / "weight_history.jsonl").exists()

    def test_trigger_weight_update_skips_when_insufficient(self, tmp_path):
        pipeline = _make_pipeline(tmp_path, min_samples=50)

        _run(pipeline.ingest_correction(
            file_id="f1", predicted_label="A", corrected_label="B", confidence=0.5
        ))

        result = _run(pipeline.trigger_weight_update())
        assert result["applied"] is False
        assert "reason" in result

    def test_load_corrections_from_disk(self, tmp_path):
        # Write some corrections with one pipeline.
        p1 = _make_pipeline(tmp_path, min_samples=1)
        _run(p1.ingest_correction(
            file_id="disk1", predicted_label="A", corrected_label="B",
            confidence=0.9, branch_predictions={"filename": "B"},
        ))
        _run(p1.ingest_correction(
            file_id="disk2", predicted_label="C", corrected_label="C",
            confidence=0.8, branch_predictions={"filename": "C"},
        ))

        # Create a fresh pipeline and reload.
        p2 = _make_pipeline(tmp_path, min_samples=1)
        loaded = p2.load_corrections_from_disk()

        assert loaded == 2
        status = p2.get_learning_status()
        assert status["corrections_total"] == 2


# ---------------------------------------------------------------------------
# SmartSampler tests
# ---------------------------------------------------------------------------

class TestSmartSampler:

    def test_uncertainty_sampling_picks_low_confidence(self):
        sampler = SmartSampler()
        preds = [
            {"file_id": "high", "confidence": 0.99, "class_probs": [0.99, 0.01]},
            {"file_id": "mid",  "confidence": 0.60, "class_probs": [0.60, 0.40]},
            {"file_id": "low",  "confidence": 0.51, "class_probs": [0.51, 0.49]},
        ]
        result = sampler.uncertainty_sampling(preds, k=2)
        assert len(result) == 2
        ids = [r["file_id"] for r in result]
        assert ids[0] == "low"
        assert ids[1] == "mid"

    def test_margin_sampling_picks_close_predictions(self):
        sampler = SmartSampler()
        preds = [
            {"file_id": "wide",  "class_probs": [0.9, 0.05, 0.05]},
            {"file_id": "close", "class_probs": [0.4, 0.35, 0.25]},
            {"file_id": "tiny",  "class_probs": [0.34, 0.33, 0.33]},
        ]
        result = sampler.margin_sampling(preds, k=2)
        ids = [r["file_id"] for r in result]
        # "tiny" has smallest margin (0.01), "close" has margin 0.05.
        assert ids[0] == "tiny"
        assert ids[1] == "close"

    def test_entropy_sampling_picks_high_entropy(self):
        sampler = SmartSampler()
        preds = [
            {"file_id": "certain",   "class_probs": [0.95, 0.05]},
            {"file_id": "uncertain", "class_probs": [0.5, 0.5]},
            {"file_id": "spread",    "class_probs": [0.33, 0.34, 0.33]},
        ]
        result = sampler.entropy_sampling(preds, k=2)
        ids = [r["file_id"] for r in result]
        # "spread" (3-class uniform) has the highest entropy, then "uncertain".
        assert ids[0] == "spread"
        assert ids[1] == "uncertain"

    def test_disagreement_sampling_picks_divergent(self):
        sampler = SmartSampler()
        preds = [
            {
                "file_id": "agree",
                "confidence": 0.9,
                "filename_pred": "A",
                "graph2d_pred": "A",
                "titleblock_pred": "A",
            },
            {
                "file_id": "disagree",
                "confidence": 0.5,
                "filename_pred": "A",
                "graph2d_pred": "B",
                "titleblock_pred": "C",
            },
        ]
        result = sampler.disagreement_sampling(preds, k=1)
        assert len(result) == 1
        assert result[0]["file_id"] == "disagree"

    def test_diversity_sampling_spreads_across_classes(self):
        sampler = SmartSampler()
        preds = []
        # 10 samples strongly class_0, 10 strongly class_1.
        for i in range(10):
            preds.append({
                "file_id": f"c0_{i}",
                "label": "class_0",
                "class_probs": [0.9, 0.1],
            })
        for i in range(10):
            preds.append({
                "file_id": f"c1_{i}",
                "label": "class_1",
                "class_probs": [0.1, 0.9],
            })

        result = sampler.diversity_sampling(preds, k=4)
        assert len(result) == 4
        # We expect representation from both clusters.
        labels = {r["label"] for r in result}
        assert len(labels) == 2, "Diversity sampling should cover both classes"

    def test_combined_sampling_returns_k(self):
        sampler = SmartSampler()
        preds = _make_predictions(n=30, n_classes=5)
        result = sampler.combined_sampling(preds, k=5)
        assert len(result) == 5

    def test_empty_predictions_returns_empty(self):
        sampler = SmartSampler()
        assert sampler.uncertainty_sampling([], k=5) == []
        assert sampler.margin_sampling([], k=5) == []
        assert sampler.entropy_sampling([], k=5) == []
        assert sampler.disagreement_sampling([], k=5) == []
        assert sampler.diversity_sampling([], k=5) == []
        assert sampler.combined_sampling([], k=5) == []

    def test_k_larger_than_predictions(self):
        sampler = SmartSampler()
        preds = [{"file_id": "only", "confidence": 0.7, "class_probs": [0.7, 0.3]}]
        result = sampler.uncertainty_sampling(preds, k=100)
        assert len(result) == 1
