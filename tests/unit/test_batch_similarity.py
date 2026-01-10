"""Tests for batch similarity endpoint."""

import pytest
from fastapi.testclient import TestClient

from src.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_vector_store():
    """Reset vector store before each test."""
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE

    _VECTOR_STORE.clear()
    _VECTOR_META.clear()
    yield
    _VECTOR_STORE.clear()
    _VECTOR_META.clear()


@pytest.fixture
def sample_vectors():
    """Create sample vectors for testing."""
    from src.core.similarity import _VECTOR_META, _VECTOR_STORE

    # Create 5 test vectors with varying similarity
    vectors = {
        "vec1": [1.0, 0.0, 0.0],
        "vec2": [0.9, 0.1, 0.0],  # Similar to vec1
        "vec3": [0.0, 1.0, 0.0],  # Different from vec1
        "vec4": [0.8, 0.2, 0.0],  # Similar to vec1
        "vec5": [0.0, 0.0, 1.0],  # Different from vec1
    }

    metadata = {
        "vec1": {"material": "steel", "complexity": "high", "format": "step"},
        "vec2": {"material": "steel", "complexity": "medium", "format": "step"},
        "vec3": {"material": "aluminum", "complexity": "low", "format": "iges"},
        "vec4": {"material": "steel", "complexity": "high", "format": "step"},
        "vec5": {"material": "plastic", "complexity": "medium", "format": "stl"},
    }

    for vid, vec in vectors.items():
        _VECTOR_STORE[vid] = vec
        _VECTOR_META[vid] = metadata[vid]

    return vectors, metadata


def test_batch_similarity_success(sample_vectors):
    """Test successful batch similarity query."""
    vectors, metadata = sample_vectors

    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1", "vec3"], "top_k": 3},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 2
    assert data["successful"] == 2
    assert data["failed"] == 0
    assert len(data["items"]) == 2
    assert "batch_id" in data
    assert "duration_ms" in data

    # Check first vector results
    vec1_results = next(item for item in data["items"] if item["id"] == "vec1")
    assert vec1_results["status"] == "success"
    assert len(vec1_results["similar"]) > 0
    # vec1 should NOT include itself
    assert not any(s["id"] == "vec1" for s in vec1_results["similar"])


def test_batch_similarity_not_found(sample_vectors):
    """Test batch similarity with non-existent vectors."""
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1", "nonexistent", "vec2"], "top_k": 3},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 3
    assert data["successful"] == 2
    assert data["failed"] == 1

    # Check not_found item
    not_found = next(item for item in data["items"] if item["id"] == "nonexistent")
    assert not_found["status"] == "not_found"
    assert not_found["error"] is not None


def test_batch_similarity_with_material_filter(sample_vectors):
    """Test batch similarity with material filter."""
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1"], "top_k": 5, "material": "steel"},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200
    data = response.json()

    vec1_results = data["items"][0]
    assert vec1_results["status"] == "success"

    # All results should be steel
    for similar in vec1_results["similar"]:
        assert similar["material"] == "steel"


def test_batch_similarity_with_complexity_filter(sample_vectors):
    """Test batch similarity with complexity filter."""
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1"], "top_k": 5, "complexity": "high"},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200
    data = response.json()

    vec1_results = data["items"][0]
    assert vec1_results["status"] == "success"

    # All results should be high complexity
    for similar in vec1_results["similar"]:
        assert similar["complexity"] == "high"


def test_batch_similarity_with_min_score(sample_vectors):
    """Test batch similarity with minimum score threshold."""
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1"], "top_k": 5, "min_score": 0.8},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200
    data = response.json()

    vec1_results = data["items"][0]
    assert vec1_results["status"] == "success"

    # All results should have score >= 0.8
    for similar in vec1_results["similar"]:
        assert similar["score"] >= 0.8


def test_batch_similarity_multiple_filters(sample_vectors):
    """Test batch similarity with multiple filters combined."""
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={
            "ids": ["vec1"],
            "top_k": 5,
            "material": "steel",
            "complexity": "high",
            "format": "step",
        },
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200
    data = response.json()

    vec1_results = data["items"][0]
    assert vec1_results["status"] == "success"

    # All results should match all filters
    for similar in vec1_results["similar"]:
        assert similar["material"] == "steel"
        assert similar["complexity"] == "high"
        assert similar["format"] == "step"


def test_batch_similarity_empty_ids():
    """Test batch similarity with empty IDs list."""
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": [], "top_k": 3},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 0
    assert data["successful"] == 0
    assert data["failed"] == 0
    assert len(data["items"]) == 0


def test_batch_similarity_top_k_validation():
    """Test batch similarity with invalid top_k values."""
    # Test top_k = 0 (invalid)
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1"], "top_k": 0},
        headers={"X-API-Key": "test"},
    )
    assert response.status_code == 422  # Validation error

    # Test top_k > 50 (invalid)
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1"], "top_k": 51},
        headers={"X-API-Key": "test"},
    )
    assert response.status_code == 422


def test_batch_similarity_min_score_validation():
    """Test batch similarity with invalid min_score values."""
    # Test min_score < 0
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1"], "min_score": -0.1},
        headers={"X-API-Key": "test"},
    )
    assert response.status_code == 422

    # Test min_score > 1
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1"], "min_score": 1.1},
        headers={"X-API-Key": "test"},
    )
    assert response.status_code == 422


def test_batch_similarity_large_batch(sample_vectors):
    """Test batch similarity with large batch (metrics label: large)."""
    # Query all 5 vectors
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1", "vec2", "vec3", "vec4", "vec5"], "top_k": 2},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 5
    assert data["successful"] == 5
    assert len(data["items"]) == 5


def test_batch_similarity_response_structure(sample_vectors):
    """Test batch similarity response has correct structure."""
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1"], "top_k": 3},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200
    data = response.json()

    # Check top-level structure
    assert "total" in data
    assert "successful" in data
    assert "failed" in data
    assert "items" in data
    assert "batch_id" in data
    assert "duration_ms" in data

    # Check item structure
    item = data["items"][0]
    assert "id" in item
    assert "status" in item
    assert "similar" in item

    # Check similar vector structure
    if item["similar"]:
        similar = item["similar"][0]
        assert "id" in similar
        assert "score" in similar
        assert "material" in similar
        assert "complexity" in similar
        assert "format" in similar
        assert "dimension" in similar


def test_batch_similarity_no_api_key():
    """Test batch similarity works with default API key in test environment.

    Note: In test environment, get_api_key has a default value of "test",
    so requests without explicit API key will still succeed.
    """
    response = client.post("/api/v1/vectors/similarity/batch", json={"ids": ["vec1"], "top_k": 3})

    # In test environment, default API key is used, so request succeeds
    assert response.status_code == 200


def test_batch_similarity_mixed_success_failure(sample_vectors):
    """Test batch similarity with mix of successful and failed queries."""
    response = client.post(
        "/api/v1/vectors/similarity/batch",
        json={"ids": ["vec1", "nonexistent1", "vec2", "nonexistent2"], "top_k": 2},
        headers={"X-API-Key": "test"},
    )

    assert response.status_code == 200
    data = response.json()

    assert data["total"] == 4
    assert data["successful"] == 2
    assert data["failed"] == 2

    # Check status distribution
    statuses = [item["status"] for item in data["items"]]
    assert statuses.count("success") == 2
    assert statuses.count("not_found") == 2
