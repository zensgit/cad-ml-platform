import pytest

from src.core.similarity import register_vector, compute_similarity


def test_similarity_basic():
    register_vector("ref1", [1.0, 0.0])
    result = compute_similarity("ref1", [1.0, 0.0])
    assert result["score"] == 1.0
    assert result["method"] == "cosine"


def test_similarity_reference_missing():
    result = compute_similarity("missing", [1.0, 0.0])
    assert result["status"] == "reference_not_found"
    assert result["score"] == 0.0


def test_similarity_dimension_mismatch():
    register_vector("ref2", [1.0, 0.0, 0.0])
    result = compute_similarity("ref2", [1.0, 0.0])
    assert result["score"] == 0.0

