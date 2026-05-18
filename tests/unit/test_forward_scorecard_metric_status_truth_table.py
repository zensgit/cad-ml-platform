"""Truth-table coverage for forward_scorecard._metric_status.

The helper drives every per-metric component's status pick; a regression
that flips one cell of this table can silently let `release_ready` leak
into a component that should be `shadow_only` or `benchmark_ready_with_gap`.
"""

from __future__ import annotations

from typing import Optional

import pytest

from src.core.benchmark.forward_scorecard import _metric_status


# Defaults baked into `_metric_status`:
# release_threshold=0.85, gap_threshold=0.65,
# min_release_samples=30, min_gap_samples=10.


@pytest.mark.parametrize(
    "sample_size,expected",
    [
        (-1, "blocked"),
        (0, "blocked"),
        (1, "shadow_only"),
        (9, "shadow_only"),
    ],
)
def test_sample_size_below_min_gap_samples_caps_at_shadow_or_blocked(
    sample_size: int, expected: str
) -> None:
    # Even a perfect primary score cannot escape the floor when sample size
    # is below `min_gap_samples`. Zero or negative is blocked outright.
    got = _metric_status(sample_size=sample_size, primary_score=0.99)
    assert got == expected


@pytest.mark.parametrize(
    "sample_size,primary,expected",
    [
        # >= min_gap_samples, primary>=gap_threshold → benchmark_ready_with_gap
        (10, 0.65, "benchmark_ready_with_gap"),
        (10, 0.85, "benchmark_ready_with_gap"),  # below min_release_samples
        (29, 0.85, "benchmark_ready_with_gap"),
        # >= min_release_samples, primary>=release_threshold → release_ready
        (30, 0.85, "release_ready"),
        (100, 0.99, "release_ready"),
        # >= min_release_samples but primary below release_threshold
        (30, 0.84, "benchmark_ready_with_gap"),
        # >= min_gap_samples, primary < gap_threshold → shadow_only
        (50, 0.50, "shadow_only"),
        (10, 0.0, "shadow_only"),
    ],
)
def test_primary_score_thresholds_default_secondary_and_low_conf(
    sample_size: int, primary: float, expected: str
) -> None:
    got = _metric_status(sample_size=sample_size, primary_score=primary)
    assert got == expected


def test_secondary_score_below_floor_blocks_release() -> None:
    # secondary_ok requires >= max(0.5, gap_threshold - 0.1) = max(0.5, 0.55) = 0.55
    # for the default gap_threshold of 0.65.
    got = _metric_status(
        sample_size=50,
        primary_score=0.95,
        secondary_score=0.54,  # below 0.55 floor
    )
    assert got == "benchmark_ready_with_gap"  # demoted from release_ready


def test_secondary_score_at_or_above_floor_allows_release() -> None:
    got = _metric_status(
        sample_size=50,
        primary_score=0.95,
        secondary_score=0.55,  # exactly at floor
    )
    assert got == "release_ready"


def test_low_conf_rate_above_25pct_blocks_release() -> None:
    got = _metric_status(
        sample_size=50,
        primary_score=0.95,
        low_conf_rate=0.26,
    )
    assert got == "benchmark_ready_with_gap"


def test_low_conf_rate_exactly_25pct_allows_release() -> None:
    got = _metric_status(
        sample_size=50,
        primary_score=0.95,
        low_conf_rate=0.25,
    )
    assert got == "release_ready"


def test_custom_thresholds_propagate() -> None:
    # graph2d / brep callers pass different thresholds — make sure overrides
    # actually flow through.
    got = _metric_status(
        sample_size=30,
        primary_score=0.80,
        release_threshold=0.8,  # graph2d default
        gap_threshold=0.55,
    )
    assert got == "release_ready"


def test_custom_min_release_samples_propagates() -> None:
    # If a caller raises the floor to 100, the same 30 samples must demote.
    got = _metric_status(
        sample_size=30,
        primary_score=0.95,
        min_release_samples=100,
    )
    assert got == "benchmark_ready_with_gap"
