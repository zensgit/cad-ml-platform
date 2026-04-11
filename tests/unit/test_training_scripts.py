"""Tests for training scripts: domain embeddings and knowledge distillation."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest import mock

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.train_domain_embeddings import build_corpus, train, evaluate


# -----------------------------------------------------------------------
# Domain embedding corpus tests
# -----------------------------------------------------------------------


class TestCorpusBuild:
    """Tests for corpus construction."""

    def test_corpus_build_returns_data(self) -> None:
        """build_corpus returns non-empty training data and eval pairs."""
        training_data, eval_pairs = build_corpus()

        assert len(training_data) > 0, "training_data should not be empty"
        assert len(eval_pairs) > 0, "eval_pairs should not be empty"

    def test_training_data_has_required_keys(self) -> None:
        """Every training record must contain 'anchor' and 'positive'."""
        training_data, _ = build_corpus()

        for record in training_data:
            assert "anchor" in record, f"Missing 'anchor' in {record}"
            assert "positive" in record, f"Missing 'positive' in {record}"

    def test_eval_pairs_have_expected_format(self) -> None:
        """Each eval pair is (text_a, text_b, expected_similarity)."""
        _, eval_pairs = build_corpus()

        for pair in eval_pairs:
            assert len(pair) == 3, f"Expected 3-tuple, got {len(pair)}"
            text_a, text_b, sim = pair
            assert isinstance(text_a, str) and text_a, "text_a must be a non-empty string"
            assert isinstance(text_b, str) and text_b, "text_b must be a non-empty string"
            assert isinstance(sim, (int, float)), "expected_similarity must be numeric"
            assert 0.0 <= sim <= 1.0, f"expected_similarity {sim} out of [0, 1]"


# -----------------------------------------------------------------------
# Trainer integration (small subset)
# -----------------------------------------------------------------------


class TestTrainerRuns:
    """Integration tests that exercise the training/evaluation path."""

    def test_trainer_runs_without_crash(self, tmp_path: Path) -> None:
        """Train on a small subset; must not crash."""
        from src.ml.embeddings.trainer import DomainEmbeddingTrainer

        training_data, _ = build_corpus()
        # Use only a tiny slice so the test is fast
        small_data = training_data[:10]

        trainer = DomainEmbeddingTrainer(output_dir=str(tmp_path / "model"))
        metrics = trainer.train(small_data, epochs=1, batch_size=4)

        assert "final_loss" in metrics
        assert "samples" in metrics
        assert metrics["samples"] == len(small_data)

    def test_evaluate_returns_metrics(self) -> None:
        """evaluate() must return spearman, mean_error, accuracy keys."""
        from src.ml.embeddings.trainer import DomainEmbeddingTrainer

        _, eval_pairs = build_corpus()
        small_eval = eval_pairs[:20]

        trainer = DomainEmbeddingTrainer()
        result = trainer.evaluate(small_eval)

        for key in ("spearman_correlation", "mean_error", "accuracy_at_threshold"):
            assert key in result, f"Missing metric key '{key}'"
            assert isinstance(result[key], float), f"'{key}' should be float"


# -----------------------------------------------------------------------
# Knowledge distillation tests
# -----------------------------------------------------------------------


class TestDistillationScript:
    """Tests for the knowledge distillation training script."""

    def test_distillation_script_importable(self) -> None:
        """The distillation module can be imported and exposes run_distillation."""
        mod = importlib.import_module("scripts.train_knowledge_distillation")
        assert hasattr(mod, "run_distillation"), "run_distillation not found"
        assert callable(mod.run_distillation)

    def test_find_teacher_model_graceful(self) -> None:
        """find_teacher_model returns None when no model files exist."""
        from scripts.train_knowledge_distillation import find_teacher_model

        # Temporarily override MODELS_DIR to a non-existent directory
        import scripts.train_knowledge_distillation as kd_mod

        original = kd_mod.MODELS_DIR
        try:
            kd_mod.MODELS_DIR = Path("/tmp/_nonexistent_models_dir_for_test")
            result = find_teacher_model()
            assert result is None, "Should return None when no models directory exists"
        finally:
            kd_mod.MODELS_DIR = original
