import json
import os
import shutil

import pytest

from src.core.active_learning import ActiveLearner, SampleStatus, reset_active_learner


class TestActiveLearningLoop:
    @pytest.fixture(autouse=True)
    def setup_teardown(self, tmp_path):
        # Reset singleton
        reset_active_learner()

        # Setup temp dir
        self.data_dir = tmp_path / "active_learning"
        os.environ["ACTIVE_LEARNING_DATA_DIR"] = str(self.data_dir)
        os.environ["ACTIVE_LEARNING_STORE"] = "file"
        os.environ["ACTIVE_LEARNING_RETRAIN_THRESHOLD"] = "5"

        self.learner = ActiveLearner()

        yield

        # Cleanup
        reset_active_learner()
        if os.path.exists(self.data_dir):
            shutil.rmtree(self.data_dir)

    def test_flag_for_review(self):
        sample = self.learner.flag_for_review(
            doc_id="doc1",
            predicted_type="bolt",
            confidence=0.5,
            alternatives=[{"type": "screw", "confidence": 0.4}],
            score_breakdown={},
            uncertainty_reason="low_confidence",
        )

        assert sample.status == SampleStatus.PENDING
        assert sample.doc_id == "doc1"
        assert sample.id in self.learner._samples

        # Verify persistence
        files = list(self.data_dir.glob("samples.jsonl"))
        assert len(files) == 1

    def test_feedback_submission_loop(self):
        # 1. Flag sample
        sample = self.learner.flag_for_review(
            doc_id="doc1",
            predicted_type="bolt",
            confidence=0.5,
            alternatives=[],
            score_breakdown={},
            uncertainty_reason="test",
        )

        # 2. Submit feedback
        result = self.learner.submit_feedback(
            sample_id=sample.id, true_type="screw", reviewer_id="user1"
        )

        assert result["status"] == "ok"
        assert result["is_correction"] is True
        assert self.learner._samples[sample.id].status == SampleStatus.LABELED
        assert self.learner._samples[sample.id].true_type == "screw"

    def test_retrain_threshold_trigger(self):
        # Create 5 samples and label them
        for i in range(5):
            sample = self.learner.flag_for_review(
                doc_id=f"doc{i}",
                predicted_type="bolt",
                confidence=0.5,
                alternatives=[],
                score_breakdown={},
                uncertainty_reason="test",
            )
            self.learner.submit_feedback(sample.id, "bolt")

        status = self.learner.check_retrain_threshold()
        assert status["ready"] is True
        assert status["labeled_samples"] == 5

    def test_export_training_data(self):
        # Create labeled sample
        sample = self.learner.flag_for_review(
            doc_id="doc1",
            predicted_type="bolt",
            confidence=0.5,
            alternatives=[],
            score_breakdown={},
            uncertainty_reason="test",
        )
        self.learner.submit_feedback(sample.id, "screw")

        # Export
        result = self.learner.export_training_data(format="jsonl")

        assert result["status"] == "ok"
        assert result["count"] == 1
        assert os.path.exists(result["file"])

        # Verify exported content
        with open(result["file"], "r") as f:
            data = json.loads(f.readline())
            assert data["doc_id"] == "doc1"
            assert data["true_type"] == "screw"

        # Verify status update
        assert self.learner._samples[sample.id].status == SampleStatus.EXPORTED
