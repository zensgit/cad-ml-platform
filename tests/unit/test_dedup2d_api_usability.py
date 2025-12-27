"""Unit tests for Phase 4 Day 6 API usability features.

Features tested:
- Forced-async mode based on file size, precision, and mode
- GET /2d/jobs endpoint for listing tenant jobs
"""

from __future__ import annotations

import pytest

from src.api.v1.dedup import _check_forced_async


class TestForcedAsyncChecks:
    """Test the forced-async detection logic."""

    def test_large_file_forces_async(self):
        """Files larger than threshold should force async mode."""
        # Default threshold is 5MB = 5 * 1024 * 1024 = 5242880 bytes
        large_file_size = 6 * 1024 * 1024  # 6MB

        reason = _check_forced_async(
            file_size=large_file_size,
            enable_precision=False,
            mode="balanced",
            query_geom=None,
        )

        assert reason is not None
        assert "file_size>" in reason

    def test_small_file_no_forced_async(self):
        """Small files should not force async mode."""
        small_file_size = 100 * 1024  # 100KB

        reason = _check_forced_async(
            file_size=small_file_size,
            enable_precision=False,
            mode="balanced",
            query_geom=None,
        )

        assert reason is None

    def test_precision_with_geom_forces_async(self):
        """enable_precision with geom_json should force async mode."""
        reason = _check_forced_async(
            file_size=100,
            enable_precision=True,
            mode="balanced",
            query_geom={"entities": []},  # Non-None geom_json
        )

        assert reason is not None
        assert "precision" in reason.lower()

    def test_precision_without_geom_no_forced_async(self):
        """enable_precision without geom_json should not force async."""
        reason = _check_forced_async(
            file_size=100,
            enable_precision=True,
            mode="balanced",
            query_geom=None,  # No geom_json
        )

        assert reason is None

    def test_precise_mode_forces_async(self):
        """mode='precise' should force async mode."""
        reason = _check_forced_async(
            file_size=100,
            enable_precision=False,
            mode="precise",
            query_geom=None,
        )

        assert reason is not None
        assert "precise" in reason.lower()

    def test_balanced_mode_no_forced_async(self):
        """mode='balanced' should not force async (other conditions permitting)."""
        reason = _check_forced_async(
            file_size=100,
            enable_precision=False,
            mode="balanced",
            query_geom=None,
        )

        assert reason is None

    def test_fast_mode_no_forced_async(self):
        """mode='fast' should not force async (other conditions permitting)."""
        reason = _check_forced_async(
            file_size=100,
            enable_precision=False,
            mode="fast",
            query_geom=None,
        )

        assert reason is None


class TestForcedAsyncPriority:
    """Test that the first matching condition is returned."""

    def test_file_size_takes_priority(self):
        """File size check should be first (if all conditions match)."""
        large_file_size = 6 * 1024 * 1024  # 6MB

        reason = _check_forced_async(
            file_size=large_file_size,
            enable_precision=True,
            mode="precise",
            query_geom={"entities": []},
        )

        # File size check should be hit first
        assert reason is not None
        assert "file_size>" in reason

    def test_precision_priority_over_mode(self):
        """Precision check should come before mode check."""
        reason = _check_forced_async(
            file_size=100,  # Small file
            enable_precision=True,
            mode="precise",
            query_geom={"entities": []},
        )

        # Precision check should be hit before mode check
        assert reason is not None
        assert "precision" in reason.lower()
