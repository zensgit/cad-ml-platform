"""Unit + facade-compat tests for the extracted health cache metric helpers."""

from __future__ import annotations

import src.api.v1.health as health_router
from src.core.health_cache_metrics import compute_cache_tuning, compute_hit_ratio


def test_helpers_reexported_from_router_module() -> None:
    # Facade: the slimmed router re-exports the moved helpers.
    assert health_router.compute_hit_ratio is compute_hit_ratio
    assert health_router.compute_cache_tuning is compute_cache_tuning


def test_compute_hit_ratio() -> None:
    assert compute_hit_ratio(None, 5) is None
    assert compute_hit_ratio(5, None) is None
    assert compute_hit_ratio(0, 0) is None
    assert compute_hit_ratio(3, 1) == 0.75


def test_cache_tuning_high_usage_with_evictions_increases_capacity() -> None:
    r = compute_cache_tuning(
        size=95, capacity=100, ttl_seconds=300, hits=60, misses=40, evictions=10
    )
    assert r["recommended_capacity"] == 150  # usage 0.95 + eviction 0.10 -> *1.5
    assert r["current_capacity"] == 100
    assert any("increase capacity" in reason for reason in r["reasons"])
    assert r["capacity_change_pct"] == 50.0
    assert r["metrics_summary"]["total_requests"] == 100


def test_cache_tuning_optimal_settings_report_no_change() -> None:
    r = compute_cache_tuning(
        size=50, capacity=100, ttl_seconds=300, hits=80, misses=20, evictions=1
    )
    assert r["recommended_capacity"] == 100
    assert r["recommended_ttl_seconds"] == 300
    assert any("optimal" in reason for reason in r["reasons"])
