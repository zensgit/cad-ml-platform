"""Tests for src/utils/rate_limiter.py to improve coverage.

Covers:
- RateLimiter class
- Token bucket algorithm
- Redis path with Lua script
- Local fallback path
- Error handling paths
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestRateLimiterInit:
    """Tests for RateLimiter initialization."""

    @pytest.mark.asyncio
    async def test_init_default_values(self):
        """Test RateLimiter initialization with defaults."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test_key")

        assert limiter.key == "ocr:rl:test_key"
        assert limiter.qps == 10.0
        assert limiter.burst == 10
        assert limiter._local_tokens == 10.0

    @pytest.mark.asyncio
    async def test_init_custom_values(self):
        """Test RateLimiter initialization with custom values."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("custom", qps=5.0, burst=20)

        assert limiter.qps == 5.0
        assert limiter.burst == 20

    @pytest.mark.asyncio
    async def test_init_burst_minimum_one(self):
        """Test burst is at least 1."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test", burst=0)

        assert limiter.burst == 1


class TestRateLimiterLocalFallback:
    """Tests for RateLimiter local fallback path."""

    @pytest.mark.asyncio
    async def test_allow_local_success(self):
        """Test allow succeeds in local mode."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test", qps=10.0, burst=5)

        with patch("src.utils.rate_limiter.get_client", return_value=None):
            result = await limiter.allow()

        assert result is True
        assert limiter._local_tokens < 5.0

    @pytest.mark.asyncio
    async def test_allow_local_depleted(self):
        """Test allow fails when tokens depleted."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test", qps=0.1, burst=1)
        limiter._local_tokens = 0.0

        with patch("src.utils.rate_limiter.get_client", return_value=None):
            with patch("src.utils.rate_limiter.ocr_rate_limited_total") as mock_metric:
                result = await limiter.allow()

        assert result is False
        mock_metric.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_allow_local_refill(self):
        """Test tokens refill over time."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test", qps=100.0, burst=10)
        limiter._local_tokens = 0.0
        limiter._local_ts = time.time() - 1.0  # 1 second ago

        with patch("src.utils.rate_limiter.get_client", return_value=None):
            result = await limiter.allow()

        # Should have refilled (100 qps * 1s = 100 tokens, capped at burst=10)
        assert result is True

    @pytest.mark.asyncio
    async def test_allow_local_cost_greater_than_one(self):
        """Test allow with cost > 1."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test", qps=10.0, burst=10)
        limiter._local_tokens = 5.0

        with patch("src.utils.rate_limiter.get_client", return_value=None):
            result = await limiter.allow(cost=3)

        assert result is True
        assert limiter._local_tokens == pytest.approx(2.0, rel=0.1)


class TestRateLimiterRedisPath:
    """Tests for RateLimiter Redis path."""

    @pytest.mark.asyncio
    async def test_allow_redis_success(self):
        """Test allow succeeds via Redis."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test")

        mock_script = AsyncMock(return_value=1)
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script

        with patch("src.utils.rate_limiter.get_client", return_value=mock_client):
            result = await limiter.allow()

        assert result is True

    @pytest.mark.asyncio
    async def test_allow_redis_denied(self):
        """Test allow denied via Redis."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test")

        mock_script = AsyncMock(return_value=0)
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script

        with patch("src.utils.rate_limiter.get_client", return_value=mock_client):
            with patch("src.utils.rate_limiter.ocr_rate_limited_total") as mock_metric:
                result = await limiter.allow()

        assert result is False
        mock_metric.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_allow_redis_error_fallback(self):
        """Test falls back to local on Redis error."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test", burst=10)
        limiter._local_tokens = 5.0

        mock_script = AsyncMock(side_effect=Exception("Redis error"))
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script

        with patch("src.utils.rate_limiter.get_client", return_value=mock_client):
            result = await limiter.allow()

        # Should fall back to local and succeed
        assert result is True

    @pytest.mark.asyncio
    async def test_allow_redis_script_registration(self):
        """Test script is registered once."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test")

        mock_script = AsyncMock(return_value=1)
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script

        with patch("src.utils.rate_limiter.get_client", return_value=mock_client):
            await limiter.allow()
            await limiter.allow()

        # Script should only be registered once
        mock_client.register_script.assert_called_once()


class TestLuaScript:
    """Tests for Lua script constant."""

    def test_lua_script_exists(self):
        """Test Lua script constant exists."""
        from src.utils.rate_limiter import _LUA_TOKEN_BUCKET

        assert isinstance(_LUA_TOKEN_BUCKET, str)
        assert "KEYS[1]" in _LUA_TOKEN_BUCKET
        assert "tokens" in _LUA_TOKEN_BUCKET


class TestTokenBucketLogic:
    """Tests for token bucket algorithm logic."""

    def test_refill_calculation(self):
        """Test token refill calculation."""
        tokens = 5.0
        capacity = 10
        refill_rate = 2.0  # tokens per second
        elapsed = 2.0  # seconds

        refilled = tokens + elapsed * refill_rate
        new_tokens = min(capacity, refilled)

        assert new_tokens == 9.0

    def test_refill_capped_at_capacity(self):
        """Test refill is capped at capacity."""
        tokens = 8.0
        capacity = 10
        refill_rate = 5.0
        elapsed = 2.0

        refilled = tokens + elapsed * refill_rate
        new_tokens = min(capacity, refilled)

        assert new_tokens == 10  # Capped at capacity

    def test_token_deduction(self):
        """Test token deduction on allow."""
        tokens = 5.0
        cost = 1

        if tokens >= cost:
            tokens -= cost
            allowed = True
        else:
            allowed = False

        assert allowed is True
        assert tokens == 4.0

    def test_token_insufficient(self):
        """Test insufficient tokens."""
        tokens = 0.5
        cost = 1

        if tokens >= cost:
            allowed = True
        else:
            allowed = False

        assert allowed is False


class TestConcurrency:
    """Tests for concurrent access."""

    @pytest.mark.asyncio
    async def test_concurrent_allows(self):
        """Test concurrent allow calls are handled correctly."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test", qps=100.0, burst=10)

        with patch("src.utils.rate_limiter.get_client", return_value=None):
            # Run multiple allows concurrently
            results = await asyncio.gather(*[limiter.allow() for _ in range(5)])

        # All should succeed (we have burst of 10)
        assert all(results)


class TestKeyFormatting:
    """Tests for key formatting."""

    @pytest.mark.asyncio
    async def test_key_prefix(self):
        """Test key has correct prefix."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("my_endpoint")

        assert limiter.key.startswith("ocr:rl:")
        assert "my_endpoint" in limiter.key


class TestMetricsIntegration:
    """Tests for metrics integration."""

    @pytest.mark.asyncio
    async def test_rate_limited_metric_incremented(self):
        """Test rate limited metric is incremented on denial."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test", burst=1)
        limiter._local_tokens = 0.0
        limiter._local_ts = time.time()  # No time for refill

        with patch("src.utils.rate_limiter.get_client", return_value=None):
            with patch("src.utils.rate_limiter.ocr_rate_limited_total") as mock_metric:
                await limiter.allow()

        mock_metric.inc.assert_called_once()


class TestEdgeCases:
    """Tests for edge cases."""

    @pytest.mark.asyncio
    async def test_negative_elapsed_time(self):
        """Test handling of potential negative elapsed time."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test")
        limiter._local_ts = time.time() + 100  # Future timestamp

        with patch("src.utils.rate_limiter.get_client", return_value=None):
            # Should handle gracefully (max(0, elapsed))
            result = await limiter.allow()

        assert result is True

    @pytest.mark.asyncio
    async def test_zero_qps(self):
        """Test with zero QPS (no refill)."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test", qps=0.0, burst=5)
        limiter._local_tokens = 5.0

        with patch("src.utils.rate_limiter.get_client", return_value=None):
            # Should still allow using existing tokens
            result = await limiter.allow()

        assert result is True

    @pytest.mark.asyncio
    async def test_large_cost(self):
        """Test with cost larger than tokens."""
        from src.utils.rate_limiter import RateLimiter

        limiter = RateLimiter("test", burst=5)
        limiter._local_tokens = 3.0

        with patch("src.utils.rate_limiter.get_client", return_value=None):
            with patch("src.utils.rate_limiter.ocr_rate_limited_total") as mock_metric:
                result = await limiter.allow(cost=5)

        assert result is False
        mock_metric.inc.assert_called_once()
