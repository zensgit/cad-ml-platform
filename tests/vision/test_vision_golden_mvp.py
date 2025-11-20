"""Vision Golden Evaluation MVP Tests - Stage A.

Tests the core evaluation logic without running the full script.
Reuses existing fixtures (sample_image_bytes) for efficiency.
"""

import sys
from pathlib import Path

import pytest

# Import evaluation functions from script
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from evaluate_vision_golden import calculate_keyword_hits, evaluate_sample

# ========== Test Fixtures ==========


@pytest.fixture
def sample_image_bytes() -> bytes:
    """Return sample image bytes (1x1 PNG) - same as other vision tests."""
    # Minimal 1x1 PNG (black pixel)
    png_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00:~\x9bU\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    return png_bytes


@pytest.fixture
def stub_expected_keywords() -> list:
    """Keywords that stub provider's response should contain."""
    return ["cylindrical", "thread", "diameter", "mechanical", "engineering"]


# ========== Keyword Matching Tests ==========


def test_calculate_keyword_hits_perfect_match():
    """Test: All keywords found in description -> 100% hit rate."""
    description = "This is a cylindrical mechanical part with thread and diameter specifications for engineering."
    keywords = ["cylindrical", "thread", "diameter", "mechanical", "engineering"]

    result = calculate_keyword_hits(description, keywords)

    assert result["total_keywords"] == 5
    assert result["hit_count"] == 5
    assert result["hit_rate"] == 1.0
    assert len(result["hits"]) == 5
    assert len(result["misses"]) == 0


def test_calculate_keyword_hits_partial_match():
    """Test: Some keywords missing -> partial hit rate."""
    description = "This is a cylindrical part with thread."
    keywords = ["cylindrical", "thread", "diameter", "mechanical", "engineering"]

    result = calculate_keyword_hits(description, keywords)

    assert result["total_keywords"] == 5
    assert result["hit_count"] == 2  # only "cylindrical" and "thread"
    assert result["hit_rate"] == 0.4  # 2/5
    assert "cylindrical" in result["hits"]
    assert "thread" in result["hits"]
    assert "diameter" in result["misses"]
    assert "mechanical" in result["misses"]
    assert "engineering" in result["misses"]


def test_calculate_keyword_hits_case_insensitive():
    """Test: Keyword matching is case-insensitive."""
    description = "This is a CYLINDRICAL part with Thread."
    keywords = ["cylindrical", "thread"]

    result = calculate_keyword_hits(description, keywords)

    assert result["hit_count"] == 2
    assert result["hit_rate"] == 1.0


def test_calculate_keyword_hits_no_keywords():
    """Test: Empty keyword list -> 0.0 hit rate (not division by zero)."""
    description = "Some text"
    keywords = []

    result = calculate_keyword_hits(description, keywords)

    assert result["total_keywords"] == 0
    assert result["hit_count"] == 0
    assert result["hit_rate"] == 0.0  # No division by zero


# ========== End-to-End Evaluation Tests ==========


@pytest.mark.asyncio
async def test_evaluate_sample_with_stub_provider(sample_image_bytes, stub_expected_keywords):
    """
    Test: evaluate_sample() with stub provider returns expected hit rate.

    Expected behavior:
    - Stub provider returns fixed description with keywords
    - All expected keywords should be found
    - Hit rate should be 1.0 (100%)
    """
    result = await evaluate_sample(
        sample_id="test_sample",
        expected_keywords=stub_expected_keywords,
        image_bytes=sample_image_bytes,
    )

    # Check success
    assert result["success"] is True
    assert "error" not in result or result["error"] is None

    # Check description was generated
    assert "description_summary" in result
    assert len(result["description_summary"]) > 0

    # Check keyword stats
    assert result["total_keywords"] == len(stub_expected_keywords)
    assert result["hit_count"] == len(stub_expected_keywords)
    assert result["hit_rate"] == 1.0

    # Verify all keywords found
    assert len(result["misses"]) == 0
    assert len(result["hits"]) == len(stub_expected_keywords)


@pytest.mark.asyncio
async def test_evaluate_sample_with_empty_image_fails():
    """
    Test: evaluate_sample() with empty image should fail gracefully.

    Expected behavior:
    - Empty image raises ValueError in stub provider
    - Result contains success=False and error message
    """
    result = await evaluate_sample(
        sample_id="empty_test", expected_keywords=["test"], image_bytes=b""  # Empty image
    )

    # Should fail gracefully
    assert result["success"] is False
    assert "error" in result
    # The actual error might be caught and wrapped, just ensure it's reported


@pytest.mark.asyncio
async def test_evaluate_sample_minimal_keywords(sample_image_bytes):
    """
    Test: evaluate_sample() with keywords not in stub response.

    Expected behavior:
    - Some keywords won't match
    - Hit rate < 1.0
    - Misses list populated
    """
    # Keywords that stub provider won't have
    keywords_not_in_stub = ["impossible", "nonexistent", "fake"]

    result = await evaluate_sample(
        sample_id="minimal_test",
        expected_keywords=keywords_not_in_stub,
        image_bytes=sample_image_bytes,
    )

    assert result["success"] is True
    assert result["hit_rate"] < 1.0  # Should not be 100%
    assert len(result["misses"]) > 0  # Should have misses


# ========== Integration with Golden Annotations ==========


def test_golden_annotation_structure():
    """
    Test: First golden annotation file exists and has correct structure.

    Expected behavior:
    - sample_001_easy.json exists
    - Contains required fields: id, expected_keywords
    """
    import json
    from pathlib import Path

    annotation_path = Path(__file__).parent / "golden" / "annotations" / "sample_001_easy.json"

    assert annotation_path.exists(), f"Golden annotation not found: {annotation_path}"

    with open(annotation_path, "r") as f:
        annotation = json.load(f)

    # Check required fields
    assert "id" in annotation
    assert "expected_keywords" in annotation

    # Check data types
    assert isinstance(annotation["id"], str)
    assert isinstance(annotation["expected_keywords"], list)
    assert len(annotation["expected_keywords"]) > 0

    # Check optional but expected fields
    assert "difficulty" in annotation
    assert annotation["difficulty"] == "easy"
