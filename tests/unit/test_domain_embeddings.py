"""Tests for the domain embedding fine-tuning module.

All tests are designed to pass without ``sentence-transformers`` installed
(the TF-IDF fallback is exercised instead).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.ml.embeddings.corpus_builder import ManufacturingCorpusBuilder
from src.ml.embeddings.model import DomainEmbeddingModel
from src.ml.embeddings.trainer import DomainEmbeddingTrainer


# ------------------------------------------------------------------ #
# Corpus builder tests                                                #
# ------------------------------------------------------------------ #


class TestCorpusBuilder:
    """Tests for ManufacturingCorpusBuilder."""

    def test_corpus_builder_generates_pairs(self) -> None:
        """At least 100 synonym pairs are generated."""
        builder = ManufacturingCorpusBuilder()
        builder.build_all()
        assert len(builder.synonym_pairs) >= 100, (
            f"Expected >= 100 synonym pairs, got {len(builder.synonym_pairs)}"
        )

    def test_corpus_builder_has_hard_negatives(self) -> None:
        """At least 20 hard negatives are generated."""
        builder = ManufacturingCorpusBuilder()
        builder.build_all()
        assert len(builder.hard_negatives) >= 20, (
            f"Expected >= 20 hard negatives, got {len(builder.hard_negatives)}"
        )

    def test_corpus_builder_covers_all_domains(self) -> None:
        """Pairs span materials, processes, GD&T, parts, surface, tolerances."""
        builder = ManufacturingCorpusBuilder()
        builder.build_all()
        coverage = builder.domain_coverage()
        for domain, covered in coverage.items():
            assert covered, f"Domain '{domain}' is not covered by corpus"

    def test_corpus_export_jsonl(self, tmp_path: Path) -> None:
        """Export to JSONL, reload, and verify record count."""
        builder = ManufacturingCorpusBuilder()
        builder.build_all()

        out_file = str(tmp_path / "train.jsonl")
        count = builder.export_jsonl(out_file)
        assert count > 0

        # Reload and verify
        loaded = []
        with open(out_file, encoding="utf-8") as fh:
            for line in fh:
                loaded.append(json.loads(line))
        assert len(loaded) == count

        # Every record must have anchor + positive
        for rec in loaded:
            assert "anchor" in rec
            assert "positive" in rec

    def test_build_training_data_has_negatives_when_available(self) -> None:
        """When hard negatives are present, training triplets include them."""
        builder = ManufacturingCorpusBuilder()
        builder.build_all()
        data = builder.build_training_data()
        records_with_neg = [r for r in data if "negative" in r]
        assert len(records_with_neg) == len(data), (
            "All records should have a negative when hard negatives are added"
        )


# ------------------------------------------------------------------ #
# Domain model tests                                                  #
# ------------------------------------------------------------------ #


class TestDomainEmbeddingModel:
    """Tests for DomainEmbeddingModel (using fallback in CI)."""

    def test_domain_model_encode_shape(self) -> None:
        """Encoding 3 texts yields shape (3, dim)."""
        model = DomainEmbeddingModel()
        vecs = model.encode(["法兰盘", "轴", "齿轮"])
        assert vecs.shape[0] == 3
        assert vecs.shape[1] == model.embedding_dim

    def test_domain_model_similarity_synonyms(self) -> None:
        """Synonyms should have non-trivial positive similarity."""
        model = DomainEmbeddingModel()
        sim = model.similarity("法兰盘", "法兰")
        # With the TF-IDF fallback, character overlap drives similarity.
        # "法兰盘" and "法兰" share the bigram "法兰", so expect > 0.
        assert sim > 0.0, f"Expected positive similarity, got {sim}"

    def test_domain_model_similarity_different(self) -> None:
        """Synonym pair should be more similar than unrelated pair."""
        model = DomainEmbeddingModel()
        sim_synonym = model.similarity("法兰盘", "法兰")
        sim_unrelated = model.similarity("法兰盘", "数控铣床")
        assert sim_synonym > sim_unrelated, (
            f"Synonym similarity ({sim_synonym:.4f}) should exceed "
            f"unrelated similarity ({sim_unrelated:.4f})"
        )

    def test_domain_model_search_returns_ranked(self) -> None:
        """Search results are returned in descending similarity order."""
        model = DomainEmbeddingModel()
        corpus = ["法兰", "轴承", "齿轮", "法兰盘", "螺栓"]
        results = model.search("法兰", corpus, top_k=3)
        assert len(results) == 3
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results not ranked"
        assert results[0]["rank"] == 1

    def test_model_info_has_fields(self) -> None:
        """get_model_info returns expected keys."""
        model = DomainEmbeddingModel()
        info = model.get_model_info()
        for key in ("name", "dimension", "fine_tuned"):
            assert key in info, f"Missing key '{key}' in model info"
        assert isinstance(info["dimension"], int)
        assert info["dimension"] > 0


# ------------------------------------------------------------------ #
# Trainer tests                                                       #
# ------------------------------------------------------------------ #


class TestDomainEmbeddingTrainer:
    """Tests for DomainEmbeddingTrainer (fallback mode)."""

    def test_fallback_without_sentence_transformers(self) -> None:
        """Trainer works without sentence-transformers (TF-IDF fallback)."""
        trainer = DomainEmbeddingTrainer()
        # Even if sentence-transformers IS installed the API must not crash.
        builder = ManufacturingCorpusBuilder()
        builder.build_all()
        data = builder.build_training_data()

        metrics = trainer.train(data, epochs=2)
        assert "final_loss" in metrics
        assert "samples" in metrics
        assert metrics["samples"] == len(data)

    def test_evaluate_returns_metrics(self) -> None:
        """Evaluate returns spearman, mean_error, accuracy_at_threshold."""
        trainer = DomainEmbeddingTrainer()
        test_pairs = [
            ("法兰盘", "法兰", 0.9),
            ("法兰盘", "齿轮", 0.1),
            ("轴承", "轴承座", 0.7),
        ]
        metrics = trainer.evaluate(test_pairs)
        assert "spearman_correlation" in metrics
        assert "mean_error" in metrics
        assert "accuracy_at_threshold" in metrics

    def test_save_and_load(self, tmp_path: Path) -> None:
        """Round-trip save / load works without error."""
        out = str(tmp_path / "test_model")
        trainer = DomainEmbeddingTrainer(output_dir=out)
        saved_path = trainer.save()
        assert Path(saved_path).is_dir()

        trainer2 = DomainEmbeddingTrainer()
        trainer2.load(saved_path)
        # Verify the loaded model can encode
        vecs = trainer2.model.encode(["test"], normalize_embeddings=True)
        assert vecs.shape[0] == 1

    def test_train_empty_data(self) -> None:
        """Training with empty data returns gracefully."""
        trainer = DomainEmbeddingTrainer()
        metrics = trainer.train([])
        assert metrics["samples"] == 0
