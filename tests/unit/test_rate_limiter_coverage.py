"""Tests for rate_limiter.py to improve coverage.

Covers:
- RateLimiter local fallback when Redis unavailable
- RateLimiter Redis path with Lua script
- RateLimiter exception handling fallback
- Token refill logic
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.utils.rate_limiter import RateLimiter


@pytest.fixture(autouse=True)
def mock_metrics():
    """Mock metrics to avoid side effects."""
    with patch("src.utils.rate_limiter.ocr_rate_limited_total") as mock:
        mock.inc = MagicMock()
        yield mock


class TestRateLimiterLocal:
    """Tests for RateLimiter with local fallback (no Redis)."""

    @pytest.mark.asyncio
    async def test_allow_local_fallback_success(self, mock_metrics):
        """Test allow with local fallback when Redis is unavailable."""
        with patch("src.utils.rate_limiter.get_client", return_value=None):
            limiter = RateLimiter("test_key", qps=10.0, burst=10)
            result = await limiter.allow(cost=1)

        assert result is True
        mock_metrics.inc.assert_not_called()

    @pytest.mark.asyncio
    async def test_allow_local_fallback_rate_limited(self, mock_metrics):
        """Test rate limiting in local fallback mode."""
        with patch("src.utils.rate_limiter.get_client", return_value=None):
            limiter = RateLimiter("test_key", qps=1.0, burst=1)

            # First request should succeed
            result1 = await limiter.allow(cost=1)
            assert result1 is True

            # Second request should be rate limited (no tokens left)
            result2 = await limiter.allow(cost=1)
            assert result2 is False
            mock_metrics.inc.assert_called()

    @pytest.mark.asyncio
    async def test_allow_local_token_refill(self, mock_metrics):
        """Test token refill over time in local mode."""
        with patch("src.utils.rate_limiter.get_client", return_value=None):
            limiter = RateLimiter("test_key", qps=100.0, burst=2)

            # Consume all tokens
            await limiter.allow(cost=2)

            # Wait for refill
            await asyncio.sleep(0.05)

            # Should have tokens again
            result = await limiter.allow(cost=1)
            assert result is True

    @pytest.mark.asyncio
    async def test_allow_local_high_cost(self, mock_metrics):
        """Test request with cost exceeding available tokens."""
        with patch("src.utils.rate_limiter.get_client", return_value=None):
            limiter = RateLimiter("test_key", qps=1.0, burst=5)

            # Request with cost exceeding burst
            result = await limiter.allow(cost=10)
            assert result is False

    @pytest.mark.asyncio
    async def test_allow_local_concurrent_access(self, mock_metrics):
        """Test concurrent access to local limiter."""
        with patch("src.utils.rate_limiter.get_client", return_value=None):
            limiter = RateLimiter("concurrent_key", qps=100.0, burst=10)

            async def request():
                return await limiter.allow(cost=1)

            # Run multiple concurrent requests
            results = await asyncio.gather(*[request() for _ in range(10)])

            # All should succeed with burst=10
            assert sum(results) == 10


class TestRateLimiterRedis:
    """Tests for RateLimiter with Redis."""

    @pytest.mark.asyncio
    async def test_allow_redis_success(self, mock_metrics):
        """Test allow with Redis returns True when allowed."""
        mock_script = AsyncMock(return_value=1)
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script

        with patch("src.utils.rate_limiter.get_client", return_value=mock_client):
            limiter = RateLimiter("redis_key", qps=10.0, burst=10)
            result = await limiter.allow(cost=1)

        assert result is True
        mock_client.register_script.assert_called_once()

    @pytest.mark.asyncio
    async def test_allow_redis_rate_limited(self, mock_metrics):
        """Test allow with Redis returns False when rate limited."""
        mock_script = AsyncMock(return_value=0)
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script

        with patch("src.utils.rate_limiter.get_client", return_value=mock_client):
            limiter = RateLimiter("redis_limited_key", qps=10.0, burst=10)
            result = await limiter.allow(cost=1)

        assert result is False
        mock_metrics.inc.assert_called()

    @pytest.mark.asyncio
    async def test_allow_redis_script_reuse(self, mock_metrics):
        """Test that Lua script is registered only once."""
        mock_script = AsyncMock(return_value=1)
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script

        with patch("src.utils.rate_limiter.get_client", return_value=mock_client):
            limiter = RateLimiter("script_reuse_key", qps=10.0, burst=10)

            # Multiple requests
            await limiter.allow(cost=1)
            await limiter.allow(cost=1)
            await limiter.allow(cost=1)

        # Script should be registered only once
        assert mock_client.register_script.call_count == 1


class TestRateLimiterException:
    """Tests for RateLimiter exception handling."""

    @pytest.mark.asyncio
    async def test_allow_redis_exception_fallback_success(self, mock_metrics):
        """Test fallback to local limiter on Redis exception - allowed."""
        mock_script = AsyncMock(side_effect=Exception("Redis error"))
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script

        with patch("src.utils.rate_limiter.get_client", return_value=mock_client):
            limiter = RateLimiter("exception_key", qps=10.0, burst=10)
            result = await limiter.allow(cost=1)

        # Should fallback to local and succeed
        assert result is True

    @pytest.mark.asyncio
    async def test_allow_redis_exception_fallback_rate_limited(self, mock_metrics):
        """Test fallback to local limiter on Redis exception - rate limited."""
        mock_script = AsyncMock(side_effect=Exception("Redis error"))
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script

        with patch("src.utils.rate_limiter.get_client", return_value=mock_client):
            limiter = RateLimiter("exception_limited_key", qps=1.0, burst=1)

            # First call succeeds
            result1 = await limiter.allow(cost=1)
            assert result1 is True

            # Second call is rate limited in fallback
            result2 = await limiter.allow(cost=1)
            assert result2 is False
            mock_metrics.inc.assert_called()


class TestRateLimiterInit:
    """Tests for RateLimiter initialization."""

    def test_init_key_prefix(self):
        """Test key has correct prefix."""
        limiter = RateLimiter("my_key", qps=10.0, burst=10)
        assert limiter.key == "ocr:rl:my_key"

    def test_init_burst_minimum(self):
        """Test burst is at least 1."""
        limiter = RateLimiter("key", qps=10.0, burst=0)
        assert limiter.burst == 1

        limiter2 = RateLimiter("key2", qps=10.0, burst=-5)
        assert limiter2.burst == 1

    def test_init_default_values(self):
        """Test default qps and burst values."""
        limiter = RateLimiter("default_key")
        assert limiter.qps == 10.0
        assert limiter.burst == 10
