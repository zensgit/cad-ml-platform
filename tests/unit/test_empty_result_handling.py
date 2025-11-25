"""Tests for empty result and query handling.

Tests that empty queries, empty vectors, and missing references are
properly rejected with appropriate error responses.
"""

from __future__ import annotations

import pytest
from src.core.similarity import compute_similarity, register_vector, has_vector


def test_compute_similarity_missing_reference():
    """Test that compute_similarity handles missing reference gracefully."""
    candidate = [1.0, 2.0, 3.0]

    result = compute_similarity("nonexistent_id", candidate)

    assert result["status"] == "reference_not_found"
    assert result["score"] == 0.0
    assert result["reference_id"] == "nonexistent_id"


def test_compute_similarity_empty_candidate_vector():
    """Test that compute_similarity handles empty candidate vector."""
    # Register a reference vector
    register_vector("ref1", [1.0, 2.0, 3.0])

    # Query with empty candidate
    result = compute_similarity("ref1", [])

    # Should return 0.0 score (zero magnitude)
    assert result["score"] == 0.0


def test_compute_similarity_zero_magnitude_vectors():
    """Test that compute_similarity handles zero-magnitude vectors."""
    # Register zero vector
    register_vector("zero_vec", [0.0, 0.0, 0.0])

    # Query with normal vector
    result = compute_similarity("zero_vec", [1.0, 2.0, 3.0])

    # Should return 0.0 (division by zero protection)
    assert result["score"] == 0.0


def test_register_vector_empty_vector():
    """Test that empty vectors can be registered (edge case)."""
    # Empty vector registration should succeed (no rejection)
    success = register_vector("empty_vec", [])

    assert success is True  # No dimension check fails for empty
    assert has_vector("empty_vec") is True


def test_compute_similarity_dimension_mismatch():
    """Test that compute_similarity handles dimension mismatches."""
    import os

    # Enable dimension checking
    original = os.getenv("ANALYSIS_VECTOR_DIM_CHECK")
    os.environ["ANALYSIS_VECTOR_DIM_CHECK"] = "1"

    try:
        # Register reference with 3 dimensions
        register_vector("ref_3d", [1.0, 2.0, 3.0])

        # Try to register vector with 4 dimensions (should be rejected)
        success = register_vector("ref_4d", [1.0, 2.0, 3.0, 4.0])

        assert success is False  # Dimension mismatch rejection

        # Query reference with mismatched candidate
        # (compute_similarity doesn't enforce dimensions, only register does)
        result = compute_similarity("ref_3d", [1.0, 2.0, 3.0, 4.0])

        # Computation should still work (no dimension enforcement in compute)
        assert "score" in result
    finally:
        # Restore original setting
        if original:
            os.environ["ANALYSIS_VECTOR_DIM_CHECK"] = original
        else:
            os.environ.pop("ANALYSIS_VECTOR_DIM_CHECK", None)


def test_compute_similarity_valid_case():
    """Test that compute_similarity works correctly for valid inputs."""
    # Register reference
    register_vector("valid_ref", [1.0, 0.0, 0.0])

    # Identical vector (cosine = 1.0)
    result = compute_similarity("valid_ref", [2.0, 0.0, 0.0])

    assert result["score"] == 1.0
    assert "status" not in result or result["status"] != "reference_not_found"


def test_compute_similarity_orthogonal_vectors():
    """Test that orthogonal vectors give score 0.0."""
    # Register reference
    register_vector("orth_ref", [1.0, 0.0, 0.0])

    # Orthogonal vector
    result = compute_similarity("orth_ref", [0.0, 1.0, 0.0])

    assert result["score"] == 0.0


def test_has_vector_empty_id():
    """Test that has_vector handles empty ID gracefully."""
    result = has_vector("")

    assert result is False


def test_compute_similarity_negative_values():
    """Test that negative cosine similarity is handled correctly."""
    # Register reference
    register_vector("neg_ref", [1.0, 1.0, 1.0])

    # Opposite direction (cosine = -1.0)
    result = compute_similarity("neg_ref", [-1.0, -1.0, -1.0])

    # Should return negative score
    assert result["score"] == pytest.approx(-1.0, abs=0.01)
