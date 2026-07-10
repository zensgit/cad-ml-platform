"""Unit tests for DomainEmbeddingProvider.

These tests used to be false-green. The originals asserted only shape --
``isinstance(result, list)``, ``len(result) == provider.dimension``,
``all(isinstance(v, float) ...)`` -- and their docstring claimed the provider
"degrades gracefully to zero-vectors". Every one of those assertions is
satisfied by a vector of zeros, so the suite could not distinguish a working
encoder from a degraded one.

What the code actually does (verified by running it):

* ``models/embeddings/manufacturing_v2/`` ships **no encoder weights**;
* :class:`DomainEmbeddingModel` therefore lands on its "always available"
  TF-IDF char-ngram fallback and self-reports
  ``{"name": "tfidf-fallback", "fine_tuned": False, "fallback": True, ...}``;
* the provider used to **discard** that ``fallback`` flag, report
  ``available = True``, and log "DomainEmbeddingProvider loaded" -- laundering a
  disclosed fallback into an undisclosed one.

So the vectors are *not* zeros. They are real, unit-norm, and distinct -- just
TF-IDF, not the advertised fine-tuned manufacturing encoder. The tests below
pin that distinction. ``test_is_fallback_mirrors_model_self_report`` is the
regression guard: re-discarding ``fallback`` makes it fail.
"""

import math

import pytest

from src.core.assistant.domain_embedding_provider import DomainEmbeddingProvider

SAMPLE = "304 stainless steel flange"
BATCH = [
    "CNC machining tolerance",
    "sheet metal bending",
    "injection molding cycle time",
]


@pytest.fixture
def provider() -> DomainEmbeddingProvider:
    return DomainEmbeddingProvider()


def _is_all_zeros(vec) -> bool:
    return all(v == 0.0 for v in vec)


class TestPublicIntrospection:
    """``available`` / ``is_fallback`` / ``model_info`` are the caller's contract."""

    def test_construction_never_raises(self, provider):
        """Callers construct speculatively; a missing checkpoint must not raise."""
        assert provider is not None

    def test_available_is_public_and_boolean(self, provider):
        """Callers previously had to reach into the private ``_available``."""
        assert isinstance(provider.available, bool)
        assert provider.available == provider._available  # noqa: SLF001 — parity check

    def test_is_fallback_mirrors_model_self_report(self, provider):
        """THE regression guard: the provider must not discard ``fallback``.

        ``DomainEmbeddingModel.get_model_info()`` reports ``fallback``. The old
        provider read the dict and threw that key away, so a TF-IDF fallback was
        indistinguishable from a fine-tuned encoder. If anyone re-introduces
        that, ``is_fallback`` stops matching ``model_info`` and this fails.
        """
        if not provider.available:
            pytest.skip("model failed to construct; no self-report to mirror")

        assert provider.model_info, "model_info must expose the model's self-report"
        assert provider.is_fallback == bool(provider.model_info.get("fallback", False)), (
            "is_fallback must mirror model_info['fallback'] -- the provider is "
            "discarding the model's disclosed fallback status again"
        )

    def test_fallback_and_fine_tuned_are_consistent(self, provider):
        """A fallback is by definition not the fine-tuned encoder."""
        if not provider.available or not provider.model_info:
            pytest.skip("model unavailable")

        if provider.is_fallback:
            assert provider.model_info.get("fine_tuned") is False

    def test_dimension_positive(self, provider):
        assert provider.dimension > 0


class TestVectorsAreInformative:
    """The assertion the old suite lacked: vectors must carry information."""

    def test_embed_text_is_nonzero_and_unit_norm_when_available(self, provider):
        vec = provider.embed_text(SAMPLE)

        assert len(vec) == provider.dimension
        assert all(isinstance(v, float) for v in vec)

        if provider.available:
            assert not _is_all_zeros(vec), "available provider returned a zero vector"
            norm = math.sqrt(sum(v * v for v in vec))
            assert norm == pytest.approx(
                1.0, abs=1e-3
            ), f"provider encodes with normalize=True, got norm={norm}"
        else:
            # Pin the documented wrong-answer mode *explicitly* rather than by accident.
            assert _is_all_zeros(vec)

    def test_embed_batch_shape_and_content(self, provider):
        vecs = provider.embed_batch(BATCH)

        assert len(vecs) == len(BATCH)
        for vec in vecs:
            assert len(vec) == provider.dimension
            if provider.available:
                assert not _is_all_zeros(vec)
            else:
                assert _is_all_zeros(vec)

    def test_distinct_texts_give_distinct_vectors(self, provider):
        """Zero vectors would make every text identical -- the core failure mode."""
        if not provider.available:
            pytest.skip("model unavailable -> zero vectors by design")

        a = provider.embed_text("CNC machining tolerance")
        b = provider.embed_text("injection molding cycle time")
        assert a != b, "distinct inputs produced identical vectors"


class TestFalseGreenRegression:
    """Guard against reintroducing shape-only assertions."""

    def test_shape_only_assertions_cannot_detect_zero_vectors(self, provider):
        """Demonstrates *why* the old suite proved nothing.

        A vector of zeros satisfies every assertion the original tests made, so
        those assertions could not distinguish a working model from a degraded
        one. This asserts that fact directly.
        """
        zeros = [0.0] * provider.dimension

        # Verbatim the original suite's assertions:
        assert isinstance(zeros, list)
        assert len(zeros) == provider.dimension
        assert all(isinstance(v, float) for v in zeros)

        # ...and yet the vector carries no information at all.
        assert _is_all_zeros(zeros)
