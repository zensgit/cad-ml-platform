"""Tests for drift.py to improve coverage.

Covers:
- _distribution with empty input
- psi_score with negative intermediate score
- compute_drift with empty distributions
"""

from __future__ import annotations

import pytest

from src.utils.drift import compute_drift, psi_score


class TestDistribution:
    """Tests for _distribution function (internal)."""

    def test_empty_input_returns_empty_dict(self):
        """Test _distribution with empty input returns empty dict."""
        # Import internal function
        from src.utils.drift import _distribution

        result = _distribution([])
        assert result == {}

    def test_single_item(self):
        """Test _distribution with single item."""
        from src.utils.drift import _distribution

        result = _distribution(["a"])
        assert result == {"a": 1.0}

    def test_multiple_items_same(self):
        """Test _distribution with multiple same items."""
        from src.utils.drift import _distribution

        result = _distribution(["a", "a", "a"])
        assert result == {"a": 1.0}

    def test_multiple_items_different(self):
        """Test _distribution with different items."""
        from src.utils.drift import _distribution

        result = _distribution(["a", "b", "a", "b"])
        assert result == {"a": 0.5, "b": 0.5}


class TestPsiScore:
    """Tests for psi_score function."""

    def test_identical_distributions(self):
        """Test PSI score for identical distributions is 0."""
        dist = {"a": 0.5, "b": 0.5}
        score = psi_score(dist, dist)
        assert score == pytest.approx(0.0, abs=0.001)

    def test_different_distributions(self):
        """Test PSI score for different distributions."""
        current = {"a": 0.8, "b": 0.2}
        baseline = {"a": 0.5, "b": 0.5}
        score = psi_score(current, baseline)
        assert score > 0
        assert score <= 1.0

    def test_completely_different_distributions(self):
        """Test PSI score for completely different distributions."""
        current = {"a": 1.0}
        baseline = {"b": 1.0}
        score = psi_score(current, baseline)
        # Should be clamped to 1.0 max
        assert score <= 1.0

    def test_negative_score_handled(self):
        """Test negative intermediate score is converted to absolute value."""
        # Create distributions that might produce negative intermediate
        current = {"a": 0.1, "b": 0.9}
        baseline = {"a": 0.9, "b": 0.1}
        score = psi_score(current, baseline)
        # Score should be positive (absolute value taken)
        assert score >= 0
        assert score <= 1.0

    def test_clamping_to_one(self):
        """Test score is clamped to maximum of 1.0."""
        # Very different distributions that would produce high score
        current = {"a": 1.0}
        baseline = {"z": 1.0}
        score = psi_score(current, baseline)
        assert score <= 1.0

    def test_empty_baseline_keys_in_current(self):
        """Test handling of keys missing from baseline."""
        current = {"a": 0.5, "b": 0.5}
        baseline = {"a": 1.0}
        score = psi_score(current, baseline)
        assert score >= 0
        assert score <= 1.0

    def test_empty_current_keys_in_baseline(self):
        """Test handling of keys missing from current."""
        current = {"a": 1.0}
        baseline = {"a": 0.5, "b": 0.5}
        score = psi_score(current, baseline)
        assert score >= 0
        assert score <= 1.0


class TestComputeDrift:
    """Tests for compute_drift function."""

    def test_empty_current_returns_zero(self):
        """Test compute_drift with empty current items returns 0."""
        result = compute_drift([], ["a", "b", "c"])
        assert result == 0.0

    def test_empty_baseline_returns_zero(self):
        """Test compute_drift with empty baseline items returns 0."""
        result = compute_drift(["a", "b", "c"], [])
        assert result == 0.0

    def test_both_empty_returns_zero(self):
        """Test compute_drift with both empty returns 0."""
        result = compute_drift([], [])
        assert result == 0.0

    def test_identical_items(self):
        """Test compute_drift with identical items returns ~0."""
        items = ["a", "b", "a", "b"]
        result = compute_drift(items, items)
        assert result == pytest.approx(0.0, abs=0.001)

    def test_different_items(self):
        """Test compute_drift with different items returns positive score."""
        current = ["a", "a", "a", "b"]
        baseline = ["a", "b", "b", "b"]
        result = compute_drift(current, baseline)
        assert result > 0
        assert result <= 1.0

    def test_completely_different_items(self):
        """Test compute_drift with no overlap."""
        current = ["x", "y", "z"]
        baseline = ["a", "b", "c"]
        result = compute_drift(current, baseline)
        # Should return high drift, clamped to 1.0
        assert result > 0
        assert result <= 1.0

    def test_generator_input(self):
        """Test compute_drift works with generator input."""
        def gen_current():
            yield "a"
            yield "b"

        def gen_baseline():
            yield "a"
            yield "a"

        result = compute_drift(gen_current(), gen_baseline())
        assert result >= 0
        assert result <= 1.0
