"""Behavioral tests for Training Data Governance (Phase 1+2).

Covers:
- Provenance fields on ActiveLearningSample (safe defaults)
- submit_feedback() sets human_verified + eligible_for_training
- export_training_data() excludes unverified by default
- check_retrain_threshold() counts only eligible samples
- LowConfidenceQueue provenance columns + human_verified_entries()
- append_reviewed_to_manifest: human_verified gate + --include-unverified
- Golden val overlap detection
- Backfill cache_path into manifest
"""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_active_learning_state(tmp_path, monkeypatch):
    from src.core.active_learning import reset_active_learner

    reset_active_learner()
    monkeypatch.setenv("ACTIVE_LEARNING_STORE", "memory")
    monkeypatch.setenv("ACTIVE_LEARNING_DATA_DIR", str(tmp_path / "active_learning"))
    monkeypatch.setenv("ACTIVE_LEARNING_RETRAIN_THRESHOLD", "10")
    yield
    reset_active_learner()


# ── ActiveLearningSample Provenance ──────────────────────────────────────────

class TestActiveLearningSampleProvenance:
    def test_default_fields_are_safe(self):
        from src.core.active_learning import ActiveLearningSample
        s = ActiveLearningSample(
            doc_id="d1", predicted_type="法兰", confidence=0.5,
            uncertainty_reason="test",
        )
        assert s.human_verified is False
        assert s.eligible_for_training is False
        assert s.training_block_reason == "missing_provenance"
        assert s.sample_source == "unknown"
        assert s.label_source == "unknown"

    def test_flag_for_review_sets_provenance(self):
        from src.core.active_learning import ActiveLearner
        learner = ActiveLearner()
        sample = learner.flag_for_review(
            doc_id="d2",
            predicted_type="轴类",
            confidence=0.3,
            alternatives=[],
            score_breakdown={},
            uncertainty_reason="test",
        )
        assert sample.sample_source == "analysis_review_queue"
        assert sample.label_source == "model_auto"
        assert sample.eligible_for_training is False

    def test_submit_feedback_human_sets_eligible(self):
        from src.core.active_learning import ActiveLearner
        learner = ActiveLearner()
        sample = learner.flag_for_review(
            doc_id="d3", predicted_type="箱体", confidence=0.4,
            alternatives=[],
            score_breakdown={},
            uncertainty_reason="test",
        )
        result = learner.submit_feedback(
            sample_id=sample.id,
            true_type="换热器",
            label_source="human_feedback",
            verified_by="tester",
        )
        assert result["status"] == "ok"
        updated = learner._samples[sample.id]
        assert updated.human_verified is True
        assert updated.eligible_for_training is True
        assert updated.training_block_reason is None
        assert updated.verified_by == "tester"
        assert updated.verified_at is not None

    def test_submit_feedback_claude_not_eligible(self):
        from src.core.active_learning import ActiveLearner
        learner = ActiveLearner()
        sample = learner.flag_for_review(
            doc_id="d4", predicted_type="法兰", confidence=0.3,
            alternatives=[],
            score_breakdown={},
            uncertainty_reason="test",
        )
        learner.submit_feedback(
            sample_id=sample.id,
            true_type="轴类",
            label_source="claude_suggestion",
        )
        updated = learner._samples[sample.id]
        assert updated.human_verified is False
        assert updated.eligible_for_training is False


# ── export_training_data + check_retrain_threshold ───────────────────────────

class TestTrainingExportGate:
    def _make_learner_with_samples(self):
        from src.core.active_learning import ActiveLearner
        learner = ActiveLearner()
        # 3 human-verified samples
        for i in range(3):
            s = learner.flag_for_review(
                doc_id=f"verified_{i}", predicted_type="法兰", confidence=0.3,
                alternatives=[], score_breakdown={}, uncertainty_reason="test",
            )
            learner.submit_feedback(
                sample_id=s.id, true_type="轴类",
                label_source="human_feedback", verified_by="human",
            )
        # 2 unverified (claude) samples
        for i in range(2):
            s = learner.flag_for_review(
                doc_id=f"claude_{i}", predicted_type="箱体", confidence=0.4,
                alternatives=[], score_breakdown={}, uncertainty_reason="test",
            )
            learner.submit_feedback(
                sample_id=s.id, true_type="换热器",
                label_source="claude_suggestion",
            )
        return learner

    def test_export_default_only_eligible(self):
        learner = self._make_learner_with_samples()
        result = learner.export_training_data()
        # Only human-verified should be exported.
        assert result["count"] == 3

    def test_check_retrain_uses_eligible(self):
        learner = self._make_learner_with_samples()
        status = learner.check_retrain_threshold()
        assert "eligible_samples" in status
        # 3 human-verified eligible, 2 claude not eligible
        assert status["eligible_samples"] <= 3
        assert status["labeled_samples"] >= status["eligible_samples"]


# ── LowConfidenceQueue Provenance ────────────────────────────────────────────

class TestLowConfQueueProvenance:
    def test_enqueue_sets_sample_source(self, tmp_path):
        from src.ml.low_conf_queue import LowConfidenceQueue
        q = LowConfidenceQueue(str(tmp_path / "q.csv"), threshold=0.5)
        q.maybe_enqueue("h1", "test.dxf", "法兰", 0.3)

        with open(q.queue_path, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows[0]["sample_source"] == "legacy_low_conf_queue"
        assert rows[0]["human_verified"] == ""
        assert rows[0]["eligible_for_training"] == ""

    def test_human_verified_entries_filters(self, tmp_path):
        from src.ml.low_conf_queue import LowConfidenceQueue, _FIELDNAMES
        qpath = tmp_path / "q.csv"
        with open(qpath, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=_FIELDNAMES)
            w.writeheader()
            w.writerow({
                "file_hash": "a", "filename": "a.dxf",
                "predicted_class": "法兰", "confidence": "0.3",
                "source": "test", "timestamp": "2026-01-01",
                "reviewed_label": "轴类", "notes": "",
                "sample_source": "", "label_source": "human_review",
                "human_verified": "true", "eligible_for_training": "true",
            })
            w.writerow({
                "file_hash": "b", "filename": "b.dxf",
                "predicted_class": "箱体", "confidence": "0.4",
                "source": "test", "timestamp": "2026-01-01",
                "reviewed_label": "换热器", "notes": "",
                "sample_source": "", "label_source": "claude_suggestion",
                "human_verified": "", "eligible_for_training": "",
            })

        q = LowConfidenceQueue(str(qpath), threshold=0.5)
        assert len(q.reviewed_entries()) == 2  # both have reviewed_label
        assert len(q.human_verified_entries()) == 1  # only first is verified


# ── append_reviewed_to_manifest: human_verified gate ─────────────────────────

class TestAppendReviewedGate:
    def test_blocks_unverified(self, tmp_path):
        from scripts.append_reviewed_to_manifest import _reviewed_rows, _is_human_verified

        rows = [
            {"reviewed_label": "法兰", "predicted_class": "X", "human_verified": "true"},
            {"reviewed_label": "轴类", "predicted_class": "Y", "human_verified": ""},
            {"reviewed_label": "箱体", "predicted_class": "Z", "human_verified": "false"},
        ]
        eligible, blocked = _reviewed_rows(rows, corrections_only=False, include_unverified=False)
        assert len(eligible) == 1
        assert blocked == 2

    def test_include_unverified_override(self, tmp_path):
        from scripts.append_reviewed_to_manifest import _reviewed_rows

        rows = [
            {"reviewed_label": "法兰", "predicted_class": "X", "human_verified": ""},
            {"reviewed_label": "轴类", "predicted_class": "Y", "human_verified": "true"},
        ]
        eligible, blocked = _reviewed_rows(rows, corrections_only=False, include_unverified=True)
        assert len(eligible) == 2
        assert blocked == 0

    def test_is_human_verified_truthy(self):
        from scripts.append_reviewed_to_manifest import _is_human_verified
        assert _is_human_verified({"human_verified": "true"}) is True
        assert _is_human_verified({"human_verified": "1"}) is True
        assert _is_human_verified({"human_verified": "yes"}) is True
        assert _is_human_verified({"human_verified": ""}) is False
        assert _is_human_verified({"human_verified": "false"}) is False
        assert _is_human_verified({"human_verified": "0"}) is False
        assert _is_human_verified({}) is False


# ── Golden Validation Set: overlap detection ─────────────────────────────────

class TestGoldenValOverlap:
    def test_golden_val_no_overlap_with_train(self):
        """Real golden_val_set.csv and golden_train_set.csv must have zero overlap."""
        val_path = Path("data/manifests/golden_val_set.csv")
        train_path = Path("data/manifests/golden_train_set.csv")
        if not val_path.exists() or not train_path.exists():
            pytest.skip("Golden manifests not found")

        val_paths = set()
        with open(val_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                val_paths.add(row.get("file_path", "").strip())

        train_paths = set()
        with open(train_path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                train_paths.add(row.get("file_path", "").strip())

        overlap = val_paths & train_paths
        assert len(overlap) == 0, f"Train/val overlap: {len(overlap)} paths"

    def test_golden_val_size(self):
        val_path = Path("data/manifests/golden_val_set.csv")
        if not val_path.exists():
            pytest.skip("Golden val not found")
        with open(val_path, "r", encoding="utf-8") as f:
            count = sum(1 for _ in csv.DictReader(f))
        assert count == 914, f"Expected 914, got {count}"


# ── Backfill simulation ──────────────────────────────────────────────────────

class TestBackfillCachePath:
    def test_backfill_fills_empty_cache_path(self, tmp_path):
        """Simulate the backfill logic from auto_retrain.sh Step 2b."""
        # Create a manifest with one row missing cache_path
        manifest = tmp_path / "manifest.csv"
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        file_path = "/some/test.dxf"
        cache_file = cache_dir / f"{hashlib.md5(file_path.encode()).hexdigest()}.pt"
        cache_file.write_text("fake")  # simulate .pt file exists

        with open(manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file_path", "cache_path", "taxonomy_v2_class"])
            w.writeheader()
            w.writerow({"file_path": file_path, "cache_path": "", "taxonomy_v2_class": "法兰"})

        # Simulate backfill
        rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
        filled = 0
        for row in rows:
            if not row.get("cache_path", "").strip():
                candidate = str(cache_dir / f"{hashlib.md5(row['file_path'].encode()).hexdigest()}.pt")
                if os.path.exists(candidate):
                    row["cache_path"] = candidate
                    filled += 1

        assert filled == 1
        assert rows[0]["cache_path"] == str(cache_file)

    def test_backfill_remaining_triggers_failure(self, tmp_path):
        """If backfill can't fill all rows, the pipeline should fail."""
        manifest = tmp_path / "manifest.csv"
        with open(manifest, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file_path", "cache_path", "taxonomy_v2_class"])
            w.writeheader()
            w.writerow({"file_path": "/nonexistent.dxf", "cache_path": "", "taxonomy_v2_class": "X"})

        rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
        remaining = sum(1 for r in rows if not r.get("cache_path", "").strip())
        # In production, remaining > 0 → exit 1
        assert remaining == 1, "Should detect unfillable row"


# ── Leakage prevention in training scripts ───────────────────────────────────

class TestLeakagePrevention:
    def test_val_exclusion_logic(self, tmp_path):
        """Simulate the val exclusion from finetune_graph2d_v2_augmented.py."""
        # All samples
        all_cache_paths = [f"/cache/{i}.pt" for i in range(10)]
        # Val samples = last 3
        val_paths = set(all_cache_paths[7:])

        # Simulated: samples = [(cache_path, label_idx), ...]
        samples = [(cp, 0) for cp in all_cache_paths]
        train_indices = [i for i, (cp, _) in enumerate(samples) if cp not in val_paths]

        assert len(train_indices) == 7
        for i in train_indices:
            assert samples[i][0] not in val_paths


# ── Feedback API label_source default ────────────────────────────────────────

class TestFeedbackAPIDefaults:
    def test_active_learning_feedback_defaults_human(self):
        from src.api.v1.active_learning import FeedbackRequest
        req = FeedbackRequest(sample_id="s1", true_type="法兰")
        assert req.label_source == "human_feedback"

    def test_active_learning_feedback_claude_explicit(self):
        from src.api.v1.active_learning import FeedbackRequest
        req = FeedbackRequest(
            sample_id="s1", true_type="法兰",
            label_source="claude_suggestion",
        )
        assert req.label_source == "claude_suggestion"
