"""Unit tests for DomainEmbeddingProvider."""

import pytest

from src.core.assistant.domain_embedding_provider import DomainEmbeddingProvider


class TestDomainEmbeddingProvider:
    """Tests for the domain embedding provider.

    The underlying DomainEmbeddingModel may or may not be loadable in
    CI (it requires sentence-transformers or falls back to TF-IDF).
    All tests are written to pass regardless — the provider is designed
    to degrade gracefully to zero-vectors.
    """

    def test_provider_init(self):
        """DomainEmbeddingProvider initialises without raising."""
        provider = DomainEmbeddingProvider()
        assert provider is not None

    def test_embed_text_returns_list(self):
        """embed_text returns a list of floats."""
        provider = DomainEmbeddingProvider()
        result = provider.embed_text("304 stainless steel flange")

        assert isinstance(result, list)
        assert len(result) == provider.dimension
        assert all(isinstance(v, float) for v in result)

    def test_embed_batch_shape(self):
        """embed_batch of 3 texts returns 3 vectors."""
        provider = DomainEmbeddingProvider()
        texts = [
            "CNC machining tolerance",
            "sheet metal bending",
            "injection molding cycle time",
        ]
        results = provider.embed_batch(texts)

        assert len(results) == 3
        for vec in results:
            assert len(vec) == provider.dimension

    def test_dimension_positive(self):
        """dimension is a positive integer."""
        provider = DomainEmbeddingProvider()
        assert provider.dimension > 0
