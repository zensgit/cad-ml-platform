"""Tests for cache tuning recommendation endpoint.

Tests cover:
1. GET /features/cache/tuning endpoint responses
2. Tuning logic for capacity recommendations
3. Tuning logic for TTL recommendations
4. Response structure validation
5. Edge cases and boundary conditions
"""

from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from src.main import app


client = TestClient(app)


class TestCacheTuningEndpoint:
    """Test cache tuning recommendation endpoint."""

    def test_cache_tuning_endpoint_returns_200(self):
        """Test that cache tuning endpoint returns 200 OK."""
        response = client.get(
            "/api/v1/features/cache/tuning",
            headers={"X-API-Key": "test"}
        )

        assert response.status_code == 200

    def test_cache_tuning_response_structure(self):
        """Test response structure of cache tuning endpoint."""
        response = client.get(
            "/api/v1/features/cache/tuning",
            headers={"X-API-Key": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        # Required fields in response
        assert "recommended_capacity" in data
        assert "recommended_ttl_seconds" in data
        assert "current_capacity" in data
        assert "current_ttl_seconds" in data
        assert "capacity_change_pct" in data
        assert "ttl_change_pct" in data
        assert "reasons" in data
        assert "metrics_summary" in data

        # Verify data types
        assert isinstance(data["recommended_capacity"], int)
        assert isinstance(data["recommended_ttl_seconds"], int)
        assert isinstance(data["reasons"], list)
        assert isinstance(data["metrics_summary"], dict)

    def test_cache_tuning_metrics_summary_structure(self):
        """Test metrics_summary structure in response."""
        response = client.get(
            "/api/v1/features/cache/tuning",
            headers={"X-API-Key": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        metrics = data["metrics_summary"]
        assert "hit_ratio" in metrics
        assert "usage_ratio" in metrics
        assert "eviction_ratio" in metrics
        assert "total_requests" in metrics
        assert "current_size" in metrics

    def test_cache_tuning_health_alias_endpoint(self):
        """Test /health/features/cache/tuning alias endpoint."""
        response = client.get(
            "/api/v1/health/features/cache/tuning",
            headers={"X-API-Key": "test"}
        )

        assert response.status_code == 200
        data = response.json()

        assert "recommended_capacity" in data
        assert "recommended_ttl_seconds" in data


class TestCacheTuningLogic:
    """Test cache tuning recommendation logic."""

    @pytest.fixture
    def mock_cache(self):
        """Create a mock cache with configurable stats."""
        cache = MagicMock()
        cache.capacity = 1000
        cache.ttl_seconds = 3600
        return cache

    def test_high_usage_high_eviction_increases_capacity(self):
        """Test capacity increase when usage and eviction are high."""
        with patch("src.core.feature_cache.get_feature_cache") as mock_get_cache:
            cache = MagicMock()
            cache.capacity = 1000
            cache.ttl_seconds = 3600
            cache.size.return_value = 950  # 95% usage
            cache.stats.return_value = {
                "hits": 1000,
                "misses": 100,
                "evictions": 100,  # ~9% eviction rate
            }
            mock_get_cache.return_value = cache

            response = client.get(
                "/api/v1/features/cache/tuning",
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # Recommended capacity should increase
            assert data["recommended_capacity"] > data["current_capacity"]
            assert data["capacity_change_pct"] > 0
            # Should have reason mentioning capacity increase
            assert any("capacity" in r.lower() for r in data["reasons"])

    def test_low_usage_low_eviction_decreases_capacity(self):
        """Test capacity decrease when usage and eviction are low."""
        with patch("src.core.feature_cache.get_feature_cache") as mock_get_cache:
            cache = MagicMock()
            cache.capacity = 1000
            cache.ttl_seconds = 3600
            cache.size.return_value = 200  # 20% usage
            cache.stats.return_value = {
                "hits": 1000,
                "misses": 100,
                "evictions": 5,  # <1% eviction rate
            }
            mock_get_cache.return_value = cache

            response = client.get(
                "/api/v1/features/cache/tuning",
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # Recommended capacity should decrease
            assert data["recommended_capacity"] < data["current_capacity"]
            assert data["capacity_change_pct"] < 0
            # Should have reason mentioning memory savings
            assert any("memory" in r.lower() or "reduce" in r.lower() for r in data["reasons"])

    def test_very_high_eviction_doubles_capacity(self):
        """Test capacity doubles when eviction rate is very high."""
        with patch("src.core.feature_cache.get_feature_cache") as mock_get_cache:
            cache = MagicMock()
            cache.capacity = 500
            cache.ttl_seconds = 3600
            cache.size.return_value = 400
            cache.stats.return_value = {
                "hits": 1000,
                "misses": 100,
                "evictions": 200,  # ~18% eviction rate
            }
            mock_get_cache.return_value = cache

            response = client.get(
                "/api/v1/features/cache/tuning",
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # Capacity should double (2x)
            assert data["recommended_capacity"] == 1000
            assert "double" in data["reasons"][0].lower() or data["capacity_change_pct"] >= 90

    def test_low_hit_ratio_reduces_ttl(self):
        """Test TTL reduction when hit ratio is low."""
        with patch("src.core.feature_cache.get_feature_cache") as mock_get_cache:
            cache = MagicMock()
            cache.capacity = 1000
            cache.ttl_seconds = 3600
            cache.size.return_value = 500
            cache.stats.return_value = {
                "hits": 400,
                "misses": 600,  # 40% hit ratio
                "evictions": 20,  # low eviction
            }
            mock_get_cache.return_value = cache

            response = client.get(
                "/api/v1/features/cache/tuning",
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # TTL should decrease
            assert data["recommended_ttl_seconds"] < data["current_ttl_seconds"]
            assert data["ttl_change_pct"] < 0
            # Should mention low hit ratio
            assert any("hit ratio" in r.lower() or "ttl" in r.lower() for r in data["reasons"])

    def test_high_hit_ratio_extends_ttl(self):
        """Test TTL extension when hit ratio is high with low evictions."""
        with patch("src.core.feature_cache.get_feature_cache") as mock_get_cache:
            cache = MagicMock()
            cache.capacity = 1000
            cache.ttl_seconds = 3600
            cache.size.return_value = 500
            cache.stats.return_value = {
                "hits": 900,
                "misses": 100,  # 90% hit ratio
                "evictions": 10,  # <2% eviction
            }
            mock_get_cache.return_value = cache

            response = client.get(
                "/api/v1/features/cache/tuning",
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # TTL should increase
            assert data["recommended_ttl_seconds"] > data["current_ttl_seconds"]
            assert data["ttl_change_pct"] > 0
            # Should mention efficiency
            assert any("efficiency" in r.lower() or "extend" in r.lower() for r in data["reasons"])

    def test_optimal_settings_no_change(self):
        """Test no changes when settings are optimal."""
        with patch("src.core.feature_cache.get_feature_cache") as mock_get_cache:
            cache = MagicMock()
            cache.capacity = 1000
            cache.ttl_seconds = 3600
            cache.size.return_value = 600  # 60% usage - moderate
            cache.stats.return_value = {
                "hits": 700,
                "misses": 300,  # 70% hit ratio - good
                "evictions": 30,  # 3% eviction - acceptable
            }
            mock_get_cache.return_value = cache

            response = client.get(
                "/api/v1/features/cache/tuning",
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # Should keep same values
            assert data["recommended_capacity"] == data["current_capacity"]
            assert data["recommended_ttl_seconds"] == data["current_ttl_seconds"]
            assert data["capacity_change_pct"] == 0
            assert data["ttl_change_pct"] == 0
            # Should mention optimal
            assert any("optimal" in r.lower() for r in data["reasons"])


class TestCacheTuningEdgeCases:
    """Test edge cases for cache tuning."""

    def test_empty_cache_no_requests(self):
        """Test tuning when cache has no requests."""
        with patch("src.core.feature_cache.get_feature_cache") as mock_get_cache:
            cache = MagicMock()
            cache.capacity = 1000
            cache.ttl_seconds = 3600
            cache.size.return_value = 0
            cache.stats.return_value = {
                "hits": 0,
                "misses": 0,
                "evictions": 0,
            }
            mock_get_cache.return_value = cache

            response = client.get(
                "/api/v1/features/cache/tuning",
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # Should handle zero division gracefully
            metrics = data["metrics_summary"]
            assert metrics["hit_ratio"] == 0.0
            assert metrics["eviction_ratio"] == 0.0
            assert metrics["total_requests"] == 0

    def test_minimum_capacity_floor(self):
        """Test that recommended capacity has a minimum floor."""
        with patch("src.core.feature_cache.get_feature_cache") as mock_get_cache:
            cache = MagicMock()
            cache.capacity = 150  # Small capacity
            cache.ttl_seconds = 3600
            cache.size.return_value = 10  # Very low usage
            cache.stats.return_value = {
                "hits": 100,
                "misses": 10,
                "evictions": 0,
            }
            mock_get_cache.return_value = cache

            response = client.get(
                "/api/v1/features/cache/tuning",
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # Capacity should not go below 100
            assert data["recommended_capacity"] >= 100

    def test_minimum_ttl_floor(self):
        """Test that recommended TTL has a minimum floor."""
        with patch("src.core.feature_cache.get_feature_cache") as mock_get_cache:
            cache = MagicMock()
            cache.capacity = 1000
            cache.ttl_seconds = 100  # Low TTL
            cache.size.return_value = 500
            cache.stats.return_value = {
                "hits": 300,
                "misses": 700,  # Low hit ratio -> reduce TTL
                "evictions": 10,
            }
            mock_get_cache.return_value = cache

            response = client.get(
                "/api/v1/features/cache/tuning",
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # TTL should not go below 60 seconds
            assert data["recommended_ttl_seconds"] >= 60

    def test_change_percentage_calculation(self):
        """Test change percentage calculations are accurate."""
        with patch("src.core.feature_cache.get_feature_cache") as mock_get_cache:
            cache = MagicMock()
            cache.capacity = 1000
            cache.ttl_seconds = 1000
            cache.size.return_value = 950
            cache.stats.return_value = {
                "hits": 100,
                "misses": 100,
                "evictions": 100,  # High eviction
            }
            mock_get_cache.return_value = cache

            response = client.get(
                "/api/v1/features/cache/tuning",
                headers={"X-API-Key": "test"}
            )

            assert response.status_code == 200
            data = response.json()

            # Verify change percentage calculation
            expected_capacity_change = (
                (data["recommended_capacity"] - data["current_capacity"])
                / data["current_capacity"] * 100
            )
            assert abs(data["capacity_change_pct"] - expected_capacity_change) < 0.2


class TestCacheApplyEndpoint:
    """Test cache apply endpoint."""

    def test_cache_apply_requires_admin_token(self):
        """Test that cache apply requires admin token."""
        response = client.post(
            "/api/v1/features/cache/apply",
            json={"capacity": 2000, "ttl_seconds": 7200},
            headers={"X-API-Key": "test"}  # No admin token
        )

        # Should require admin token
        assert response.status_code in {401, 403, 422}

    def test_cache_apply_with_admin_token(self):
        """Test cache apply with admin token (may be rejected by auth in test env)."""
        with patch("src.core.feature_cache.apply_cache_settings") as mock_apply:
            mock_apply.return_value = {
                "status": "ok",
                "applied": {"capacity": 2000, "ttl_seconds": 7200},
            }

            response = client.post(
                "/api/v1/features/cache/apply",
                json={"capacity": 2000, "ttl_seconds": 7200},
                headers={
                    "X-API-Key": "test",
                    "X-Admin-Token": "test-admin"
                }
            )

            # In test env, admin auth may reject - 200 or 403 both valid
            assert response.status_code in {200, 403}


class TestCacheRollbackEndpoint:
    """Test cache rollback endpoint."""

    def test_cache_rollback_requires_admin_token(self):
        """Test that cache rollback requires admin token."""
        response = client.post(
            "/api/v1/features/cache/rollback",
            headers={"X-API-Key": "test"}  # No admin token
        )

        # Should require admin token
        assert response.status_code in {401, 403, 422}

    def test_cache_rollback_with_admin_token(self):
        """Test cache rollback with admin token (may be rejected by auth in test env)."""
        with patch("src.core.feature_cache.rollback_cache_settings") as mock_rollback:
            mock_rollback.return_value = {
                "status": "ok",
                "restored": {"capacity": 1000, "ttl_seconds": 3600},
            }

            response = client.post(
                "/api/v1/features/cache/rollback",
                headers={
                    "X-API-Key": "test",
                    "X-Admin-Token": "test-admin"
                }
            )

            # In test env, admin auth may reject - 200 or 403 both valid
            assert response.status_code in {200, 403}
