"""Tests for metrics_helpers.py to improve coverage.

Covers:
- safe_inc function with labels and without
- safe_observe function with labels and without
- safe_set function with labels and without
- Exception handling for all functions
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.utils.metrics_helpers import safe_inc, safe_observe, safe_set


class TestSafeInc:
    """Tests for safe_inc function."""

    def test_safe_inc_without_labels(self):
        """Test safe_inc without labels."""
        mock_counter = MagicMock()
        safe_inc(mock_counter)
        mock_counter.inc.assert_called_once()

    def test_safe_inc_with_labels(self):
        """Test safe_inc with labels."""
        mock_counter = MagicMock()
        safe_inc(mock_counter, provider="paddle", status="success")
        mock_counter.labels.assert_called_once_with(provider="paddle", status="success")
        mock_counter.labels().inc.assert_called_once()

    def test_safe_inc_exception_ignored(self):
        """Test safe_inc ignores exceptions."""
        mock_counter = MagicMock()
        mock_counter.inc.side_effect = Exception("Prometheus error")
        # Should not raise
        safe_inc(mock_counter)

    def test_safe_inc_labels_exception_ignored(self):
        """Test safe_inc with labels ignores exceptions."""
        mock_counter = MagicMock()
        mock_counter.labels.side_effect = Exception("Prometheus error")
        # Should not raise
        safe_inc(mock_counter, provider="test")


class TestSafeObserve:
    """Tests for safe_observe function."""

    def test_safe_observe_without_labels(self):
        """Test safe_observe without labels."""
        mock_hist = MagicMock()
        safe_observe(mock_hist, 1.5)
        mock_hist.observe.assert_called_once_with(1.5)

    def test_safe_observe_with_labels(self):
        """Test safe_observe with labels."""
        mock_hist = MagicMock()
        safe_observe(mock_hist, 2.5, stage="inference")
        mock_hist.labels.assert_called_once_with(stage="inference")
        mock_hist.labels().observe.assert_called_once_with(2.5)

    def test_safe_observe_exception_ignored(self):
        """Test safe_observe ignores exceptions."""
        mock_hist = MagicMock()
        mock_hist.observe.side_effect = Exception("Prometheus error")
        # Should not raise
        safe_observe(mock_hist, 1.0)

    def test_safe_observe_labels_exception_ignored(self):
        """Test safe_observe with labels ignores exceptions."""
        mock_hist = MagicMock()
        mock_hist.labels.side_effect = Exception("Prometheus error")
        # Should not raise
        safe_observe(mock_hist, 1.0, stage="test")


class TestSafeSet:
    """Tests for safe_set function."""

    def test_safe_set_without_labels(self):
        """Test safe_set without labels."""
        mock_gauge = MagicMock()
        safe_set(mock_gauge, 42.0)
        mock_gauge.set.assert_called_once_with(42.0)

    def test_safe_set_with_labels(self):
        """Test safe_set with labels."""
        mock_gauge = MagicMock()
        safe_set(mock_gauge, 0.5, key="test_provider")
        mock_gauge.labels.assert_called_once_with(key="test_provider")
        mock_gauge.labels().set.assert_called_once_with(0.5)

    def test_safe_set_exception_ignored(self):
        """Test safe_set ignores exceptions."""
        mock_gauge = MagicMock()
        mock_gauge.set.side_effect = Exception("Prometheus error")
        # Should not raise
        safe_set(mock_gauge, 1.0)

    def test_safe_set_labels_exception_ignored(self):
        """Test safe_set with labels ignores exceptions."""
        mock_gauge = MagicMock()
        mock_gauge.labels.side_effect = Exception("Prometheus error")
        # Should not raise
        safe_set(mock_gauge, 1.0, key="test")


class TestEdgeCases:
    """Tests for edge cases in metrics helpers."""

    def test_safe_inc_with_none_counter(self):
        """Test safe_inc handles None gracefully."""
        # Should not raise - exception caught
        safe_inc(None)

    def test_safe_observe_with_none_histogram(self):
        """Test safe_observe handles None gracefully."""
        # Should not raise - exception caught
        safe_observe(None, 1.0)

    def test_safe_set_with_none_gauge(self):
        """Test safe_set handles None gracefully."""
        # Should not raise - exception caught
        safe_set(None, 1.0)

    def test_safe_inc_multiple_labels(self):
        """Test safe_inc with multiple labels."""
        mock_counter = MagicMock()
        safe_inc(mock_counter, provider="paddle", status="error", code="timeout")
        mock_counter.labels.assert_called_once_with(
            provider="paddle", status="error", code="timeout"
        )

    def test_safe_observe_zero_value(self):
        """Test safe_observe with zero value."""
        mock_hist = MagicMock()
        safe_observe(mock_hist, 0.0)
        mock_hist.observe.assert_called_once_with(0.0)

    def test_safe_set_negative_value(self):
        """Test safe_set with negative value."""
        mock_gauge = MagicMock()
        safe_set(mock_gauge, -1.0)
        mock_gauge.set.assert_called_once_with(-1.0)
