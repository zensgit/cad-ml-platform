"""Tests for API rate limiting middleware."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.middleware.base import BaseHTTPMiddleware


class TestRateLimitConfig:
    """Tests for RateLimitConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from src.api.middleware.rate_limiting import RateLimitConfig

        config = RateLimitConfig()

        assert config.qps == 10.0
        assert config.burst == 20
        assert config.window_seconds == 60
        assert config.max_requests == 600

    def test_custom_values(self):
        """Test custom configuration values."""
        from src.api.middleware.rate_limiting import RateLimitConfig

        config = RateLimitConfig(
            qps=5.0,
            burst=10,
            window_seconds=30,
            max_requests=150,
        )

        assert config.qps == 5.0
        assert config.burst == 10
        assert config.window_seconds == 30
        assert config.max_requests == 150


class TestRateLimitTier:
    """Tests for RateLimitTier enum."""

    def test_tier_values(self):
        """Test tier enum values."""
        from src.api.middleware.rate_limiting import RateLimitTier

        assert RateLimitTier.ANONYMOUS.value == "anonymous"
        assert RateLimitTier.FREE.value == "free"
        assert RateLimitTier.PRO.value == "pro"
        assert RateLimitTier.ENTERPRISE.value == "enterprise"
        assert RateLimitTier.INTERNAL.value == "internal"


class TestDefaultTierConfigs:
    """Tests for default tier configurations."""

    def test_anonymous_is_most_restrictive(self):
        """Test anonymous tier is most restrictive."""
        from src.api.middleware.rate_limiting import (
            DEFAULT_TIER_CONFIGS,
            RateLimitTier,
        )

        anon = DEFAULT_TIER_CONFIGS[RateLimitTier.ANONYMOUS]
        free = DEFAULT_TIER_CONFIGS[RateLimitTier.FREE]

        assert anon.qps < free.qps
        assert anon.max_requests < free.max_requests

    def test_enterprise_is_generous(self):
        """Test enterprise tier has high limits."""
        from src.api.middleware.rate_limiting import (
            DEFAULT_TIER_CONFIGS,
            RateLimitTier,
        )

        enterprise = DEFAULT_TIER_CONFIGS[RateLimitTier.ENTERPRISE]

        assert enterprise.qps >= 100.0
        assert enterprise.max_requests >= 6000

    def test_internal_has_highest_limits(self):
        """Test internal tier has highest limits."""
        from src.api.middleware.rate_limiting import (
            DEFAULT_TIER_CONFIGS,
            RateLimitTier,
        )

        internal = DEFAULT_TIER_CONFIGS[RateLimitTier.INTERNAL]
        enterprise = DEFAULT_TIER_CONFIGS[RateLimitTier.ENTERPRISE]

        assert internal.qps > enterprise.qps
        assert internal.max_requests > enterprise.max_requests


class TestSlidingWindowRateLimiter:
    """Tests for SlidingWindowRateLimiter."""

    @pytest.mark.asyncio
    async def test_local_fallback_allows_within_limit(self):
        """Test local fallback allows requests within limit."""
        from src.api.middleware.rate_limiting import (
            RateLimitConfig,
            SlidingWindowRateLimiter,
        )

        limiter = SlidingWindowRateLimiter()
        config = RateLimitConfig(max_requests=10, window_seconds=60)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            allowed, current, remaining = await limiter.check("test", config, cost=1)

        assert allowed is True
        assert current == 1
        assert remaining == 9

    @pytest.mark.asyncio
    async def test_local_fallback_blocks_over_limit(self):
        """Test local fallback blocks when over limit."""
        from src.api.middleware.rate_limiting import (
            RateLimitConfig,
            SlidingWindowRateLimiter,
        )

        limiter = SlidingWindowRateLimiter()
        config = RateLimitConfig(max_requests=3, window_seconds=60)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            with patch("src.api.middleware.rate_limiting.ocr_rate_limited_total"):
                # Use up all requests
                for _ in range(3):
                    await limiter.check("test", config, cost=1)

                # Next should be blocked
                allowed, current, remaining = await limiter.check("test", config, cost=1)

        assert allowed is False
        assert remaining == 0

    @pytest.mark.asyncio
    async def test_cost_multiplier(self):
        """Test cost multiplier consumes more quota."""
        from src.api.middleware.rate_limiting import (
            RateLimitConfig,
            SlidingWindowRateLimiter,
        )

        limiter = SlidingWindowRateLimiter()
        config = RateLimitConfig(max_requests=10, window_seconds=60)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            allowed, current, remaining = await limiter.check("test", config, cost=5)

        assert allowed is True
        assert current == 5
        assert remaining == 5

    @pytest.mark.asyncio
    async def test_different_identifiers_isolated(self):
        """Test different identifiers have separate limits."""
        from src.api.middleware.rate_limiting import (
            RateLimitConfig,
            SlidingWindowRateLimiter,
        )

        limiter = SlidingWindowRateLimiter()
        config = RateLimitConfig(max_requests=5, window_seconds=60)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            # Fill user1's quota
            for _ in range(5):
                await limiter.check("user1", config, cost=1)

            # user2 should still have quota
            allowed, _, _ = await limiter.check("user2", config, cost=1)

        assert allowed is True

    @pytest.mark.asyncio
    async def test_redis_path_success(self):
        """Test Redis path when available."""
        from src.api.middleware.rate_limiting import (
            RateLimitConfig,
            SlidingWindowRateLimiter,
        )

        limiter = SlidingWindowRateLimiter()
        config = RateLimitConfig(max_requests=10, window_seconds=60)

        mock_script = AsyncMock(return_value=[1, 1, 9])
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script

        with patch("src.api.middleware.rate_limiting.get_client", return_value=mock_client):
            allowed, current, remaining = await limiter.check("test", config, cost=1)

        assert allowed is True
        assert current == 1
        assert remaining == 9

    @pytest.mark.asyncio
    async def test_redis_path_denied(self):
        """Test Redis path when rate limited."""
        from src.api.middleware.rate_limiting import (
            RateLimitConfig,
            SlidingWindowRateLimiter,
        )

        limiter = SlidingWindowRateLimiter()
        config = RateLimitConfig(max_requests=10, window_seconds=60)

        mock_script = AsyncMock(return_value=[0, 10, 0])
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script

        with patch("src.api.middleware.rate_limiting.get_client", return_value=mock_client):
            with patch("src.api.middleware.rate_limiting.ocr_rate_limited_total"):
                allowed, current, remaining = await limiter.check("test", config, cost=1)

        assert allowed is False

    @pytest.mark.asyncio
    async def test_redis_error_fallback(self):
        """Test fallback to local on Redis error."""
        from src.api.middleware.rate_limiting import (
            RateLimitConfig,
            SlidingWindowRateLimiter,
        )

        limiter = SlidingWindowRateLimiter()
        config = RateLimitConfig(max_requests=10, window_seconds=60)

        mock_script = AsyncMock(side_effect=Exception("Redis error"))
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script

        with patch("src.api.middleware.rate_limiting.get_client", return_value=mock_client):
            # Should fall back to local and succeed
            allowed, _, _ = await limiter.check("test", config, cost=1)

        assert allowed is True


class TestRateLimitMiddleware:
    """Tests for RateLimitMiddleware."""

    def _create_app_with_middleware(self, **middleware_kwargs):
        """Create a FastAPI app with rate limiting middleware."""
        from src.api.middleware.rate_limiting import RateLimitMiddleware

        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        @app.get("/health")
        async def health_endpoint():
            return {"status": "healthy"}

        @app.post("/expensive")
        async def expensive_endpoint():
            return {"status": "processed"}

        app.add_middleware(RateLimitMiddleware, **middleware_kwargs)
        return app

    def test_excluded_paths_not_limited(self):
        """Test excluded paths bypass rate limiting."""
        app = self._create_app_with_middleware()
        client = TestClient(app)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            # Health endpoint should not be rate limited
            for _ in range(100):
                response = client.get("/health")
                assert response.status_code == 200

    def test_rate_limit_headers_present(self):
        """Test rate limit headers are added to response."""
        app = self._create_app_with_middleware()
        client = TestClient(app)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            response = client.get("/test")

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers
        assert "X-RateLimit-Reset" in response.headers
        assert "X-RateLimit-Tier" in response.headers

    def test_tier_resolution_anonymous(self):
        """Test anonymous tier for requests without auth."""
        from src.api.middleware.rate_limiting import RateLimitTier

        app = self._create_app_with_middleware()
        client = TestClient(app)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            response = client.get("/test")

        assert response.headers["X-RateLimit-Tier"] == RateLimitTier.ANONYMOUS.value

    def test_tier_resolution_with_api_key(self):
        """Test free tier for requests with API key."""
        from src.api.middleware.rate_limiting import RateLimitTier

        app = self._create_app_with_middleware()
        client = TestClient(app)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            response = client.get("/test", headers={"X-API-Key": "test-key"})

        assert response.headers["X-RateLimit-Tier"] == RateLimitTier.FREE.value

    def test_tier_resolution_enterprise(self):
        """Test enterprise tier from header."""
        from src.api.middleware.rate_limiting import RateLimitTier

        app = self._create_app_with_middleware()
        client = TestClient(app)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            response = client.get("/test", headers={"X-Tenant-Tier": "enterprise"})

        assert response.headers["X-RateLimit-Tier"] == RateLimitTier.ENTERPRISE.value

    def test_tier_resolution_internal(self):
        """Test internal tier from header."""
        from src.api.middleware.rate_limiting import RateLimitTier

        app = self._create_app_with_middleware()
        client = TestClient(app)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            response = client.get("/test", headers={"X-Internal-Service": "true"})

        assert response.headers["X-RateLimit-Tier"] == RateLimitTier.INTERNAL.value


class TestCreateRateLimitDependency:
    """Tests for create_rate_limit_dependency function."""

    @pytest.mark.asyncio
    async def test_dependency_allows_within_limit(self):
        """Test dependency allows requests within limit."""
        from src.api.middleware.rate_limiting import create_rate_limit_dependency

        dep = create_rate_limit_dependency(qps=10.0)

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "127.0.0.1"
        mock_request.url.path = "/test"

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            # Should not raise
            await dep(mock_request)

    @pytest.mark.asyncio
    async def test_dependency_blocks_over_limit(self):
        """Test dependency blocks when over limit."""
        from fastapi import HTTPException

        from src.api.middleware.rate_limiting import create_rate_limit_dependency

        # Create dependency with very low limit (1 request per window)
        dep = create_rate_limit_dependency(qps=1.0, burst=1)

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "127.0.0.1"
        mock_request.url.path = "/test-block-unique-" + str(time.time())

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            with patch("src.api.middleware.rate_limiting.ocr_rate_limited_total"):
                # First should succeed (max_requests = qps * 60 = 60)
                await dep(mock_request)

                # Exhaust the limit by making many more requests
                for _ in range(59):
                    await dep(mock_request)

                # 61st should fail
                with pytest.raises(HTTPException) as exc_info:
                    await dep(mock_request)

        assert exc_info.value.status_code == 429


class TestEndpointCosts:
    """Tests for endpoint cost multipliers."""

    def test_expensive_endpoints_have_higher_cost(self):
        """Test expensive endpoints have higher cost."""
        from src.api.middleware.rate_limiting import ENDPOINT_COSTS

        assert ENDPOINT_COSTS.get("/api/v1/ocr/extract", 1) > 1
        assert ENDPOINT_COSTS.get("/api/v1/vision/analyze", 1) > 1

    @pytest.mark.asyncio
    async def test_middleware_applies_endpoint_cost(self):
        """Test middleware applies endpoint cost."""
        from src.api.middleware.rate_limiting import RateLimitMiddleware

        # Create middleware instance to test cost calculation
        app = MagicMock()
        middleware = RateLimitMiddleware.__new__(RateLimitMiddleware)
        middleware.endpoint_costs = {"/api/v1/ocr/extract": 5}

        # POST should cost more than GET
        post_cost = middleware._get_endpoint_cost("/api/v1/test", "POST")
        get_cost = middleware._get_endpoint_cost("/api/v1/test", "GET")

        assert post_cost > get_cost


class TestConcurrentRequests:
    """Tests for concurrent request handling."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_properly_counted(self):
        """Test concurrent requests are properly counted."""
        from src.api.middleware.rate_limiting import (
            RateLimitConfig,
            SlidingWindowRateLimiter,
        )

        limiter = SlidingWindowRateLimiter()
        config = RateLimitConfig(max_requests=10, window_seconds=60)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            # Run concurrent requests
            tasks = [limiter.check("test", config, cost=1) for _ in range(5)]
            results = await asyncio.gather(*tasks)

        # All should succeed
        allowed_count = sum(1 for allowed, _, _ in results if allowed)
        assert allowed_count == 5

    @pytest.mark.asyncio
    async def test_concurrent_requests_respect_limit(self):
        """Test concurrent requests respect limit."""
        from src.api.middleware.rate_limiting import (
            RateLimitConfig,
            SlidingWindowRateLimiter,
        )

        limiter = SlidingWindowRateLimiter()
        config = RateLimitConfig(max_requests=5, window_seconds=60)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            with patch("src.api.middleware.rate_limiting.ocr_rate_limited_total"):
                # Run more concurrent requests than limit
                tasks = [limiter.check("test", config, cost=1) for _ in range(10)]
                results = await asyncio.gather(*tasks)

        # Only 5 should succeed
        allowed_count = sum(1 for allowed, _, _ in results if allowed)
        assert allowed_count == 5


class TestAdditionalCoverage:
    """Additional tests to improve coverage."""

    @pytest.mark.asyncio
    async def test_local_cleanup_old_requests(self):
        """Test local fallback cleans up old requests after 1 second."""
        from src.api.middleware.rate_limiting import (
            RateLimitConfig,
            SlidingWindowRateLimiter,
        )

        limiter = SlidingWindowRateLimiter()
        config = RateLimitConfig(max_requests=10, window_seconds=60)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            # Make first request
            await limiter.check("cleanup_test", config, cost=1)

            # Simulate time passing (>1 second for cleanup trigger)
            key = limiter._get_key("cleanup_test")
            window = limiter._local_windows.get(key)
            if window:
                window["last_cleanup"] = time.time() - 2.0  # Force cleanup on next check
                # Add an old request that should be cleaned up
                old_ts = time.time() - config.window_seconds - 10
                window["requests"].append(old_ts)

            # Make another request - this should trigger cleanup
            allowed, current, remaining = await limiter.check("cleanup_test", config, cost=1)

        assert allowed is True
        # Old request should have been cleaned up
        assert current == 2  # Only the two new requests

    def test_identifier_resolver_with_user_id(self):
        """Test default identifier resolver includes user ID."""
        from src.api.middleware.rate_limiting import RateLimitMiddleware

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "192.168.1.1"

        def mock_get(key, default=None):
            if key == "X-User-ID":
                return "user123"
            return default

        mock_request.headers.get = mock_get

        # Create a real middleware instance
        app = MagicMock()
        middleware = RateLimitMiddleware(app)

        identifier = middleware._default_identifier_resolver(mock_request)

        assert "ip:192.168.1.1" in identifier
        assert "user:user123" in identifier

    def test_identifier_resolver_with_tenant_id(self):
        """Test default identifier resolver includes tenant ID."""
        from src.api.middleware.rate_limiting import RateLimitMiddleware

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "10.0.0.1"

        def mock_get(key, default=None):
            if key == "X-Tenant-ID":
                return "tenant456"
            return default

        mock_request.headers.get = mock_get

        app = MagicMock()
        middleware = RateLimitMiddleware(app)

        identifier = middleware._default_identifier_resolver(mock_request)

        assert "ip:10.0.0.1" in identifier
        assert "tenant:tenant456" in identifier

    def test_identifier_resolver_with_user_and_tenant(self):
        """Test default identifier resolver includes both user and tenant ID."""
        from src.api.middleware.rate_limiting import RateLimitMiddleware

        mock_request = MagicMock(spec=Request)
        mock_request.client.host = "10.0.0.1"

        def mock_get(key, default=None):
            if key == "X-User-ID":
                return "user789"
            if key == "X-Tenant-ID":
                return "tenant456"
            return default

        mock_request.headers.get = mock_get

        app = MagicMock()
        middleware = RateLimitMiddleware(app)

        identifier = middleware._default_identifier_resolver(mock_request)

        assert "ip:10.0.0.1" in identifier
        assert "user:user789" in identifier
        assert "tenant:tenant456" in identifier

    def test_get_endpoint_cost_exact_match(self):
        """Test _get_endpoint_cost with exact path match."""
        from src.api.middleware.rate_limiting import RateLimitMiddleware

        middleware = RateLimitMiddleware.__new__(RateLimitMiddleware)
        middleware.endpoint_costs = {
            "/api/v1/ocr/extract": 5,
            "/api/v1/vision/analyze": 10,
        }

        # Exact match
        cost = middleware._get_endpoint_cost("/api/v1/ocr/extract", "GET")
        assert cost == 5

    def test_get_endpoint_cost_prefix_match(self):
        """Test _get_endpoint_cost with prefix path match."""
        from src.api.middleware.rate_limiting import RateLimitMiddleware

        middleware = RateLimitMiddleware.__new__(RateLimitMiddleware)
        middleware.endpoint_costs = {
            "/api/v1/ocr": 3,  # Prefix
        }

        # Prefix match - /api/v1/ocr/something starts with /api/v1/ocr
        cost = middleware._get_endpoint_cost("/api/v1/ocr/something", "GET")
        assert cost == 3

    def test_rate_limit_exceeded_returns_429(self):
        """Test rate limit exceeded returns 429 response with headers."""
        from src.api.middleware.rate_limiting import (
            RateLimitConfig,
            RateLimitMiddleware,
            RateLimitTier,
        )

        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        app.add_middleware(RateLimitMiddleware)

        client = TestClient(app)

        # Create a very low limit config
        low_limit_config = {RateLimitTier.ANONYMOUS: RateLimitConfig(max_requests=2)}

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            with patch("src.api.middleware.rate_limiting.ocr_rate_limited_total"):
                # Use custom middleware with low limit
                app2 = FastAPI()

                @app2.get("/test")
                async def test_endpoint2():
                    return {"status": "ok"}

                app2.add_middleware(
                    RateLimitMiddleware,
                    tier_configs={RateLimitTier.ANONYMOUS: RateLimitConfig(max_requests=2)},
                )

                client2 = TestClient(app2)

                # First two requests should succeed
                resp1 = client2.get("/test")
                resp2 = client2.get("/test")
                assert resp1.status_code == 200
                assert resp2.status_code == 200

                # Third request should be blocked
                resp3 = client2.get("/test")
                assert resp3.status_code == 429
                assert "Retry-After" in resp3.headers
                assert resp3.json()["detail"] == "Rate limit exceeded. Please retry later."

    def test_tier_resolution_pro(self):
        """Test pro tier from header."""
        from src.api.middleware.rate_limiting import RateLimitMiddleware, RateLimitTier

        app = FastAPI()

        @app.get("/test")
        async def test_endpoint():
            return {"status": "ok"}

        app.add_middleware(RateLimitMiddleware)
        client = TestClient(app)

        with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
            response = client.get("/test", headers={"X-Tenant-Tier": "pro"})

        assert response.headers["X-RateLimit-Tier"] == RateLimitTier.PRO.value

