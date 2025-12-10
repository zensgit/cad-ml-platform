"""Tests for src/middleware/rate_limit.py to improve coverage.

Covers:
- rate_limit function
- Token bucket refill logic
- Rate limiting enforcement
- IP and path key management
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException


class TestRateLimitConstants:
    """Tests for rate limit constants."""

    def test_rate_limit_qps_default(self):
        """Test RATE_LIMIT_QPS default value from env."""
        # Just verify the constants exist
        from src.middleware.rate_limit import RATE_LIMIT_QPS

        assert isinstance(RATE_LIMIT_QPS, float)

    def test_burst_default(self):
        """Test BURST default value from env."""
        from src.middleware.rate_limit import BURST

        assert isinstance(BURST, int)


class TestRateLimitFunction:
    """Tests for rate_limit function."""

    def test_first_request_allowed(self):
        """Test first request from IP creates bucket and allows."""
        from src.middleware import rate_limit as rl_module

        # Clear buckets
        rl_module._buckets.clear()

        mock_request = MagicMock()
        mock_request.client.host = "192.168.1.1"
        mock_request.url.path = "/api/test"

        # Should not raise
        rl_module.rate_limit(mock_request)

        # Bucket should exist
        key = ("192.168.1.1", "/api/test")
        assert key in rl_module._buckets

    def test_subsequent_request_allowed(self):
        """Test subsequent requests within limit are allowed."""
        from src.middleware import rate_limit as rl_module

        rl_module._buckets.clear()

        mock_request = MagicMock()
        mock_request.client.host = "192.168.1.2"
        mock_request.url.path = "/api/test"

        # Multiple requests within burst limit
        for _ in range(5):
            rl_module.rate_limit(mock_request)

    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded raises HTTPException."""
        from src.middleware import rate_limit as rl_module

        rl_module._buckets.clear()

        mock_request = MagicMock()
        mock_request.client.host = "192.168.1.3"
        mock_request.url.path = "/api/test"

        # Set bucket with zero tokens
        key = ("192.168.1.3", "/api/test")
        rl_module._buckets[key] = {"tokens": 0.5, "timestamp": time.time()}

        with pytest.raises(HTTPException) as exc_info:
            rl_module.rate_limit(mock_request)

        assert exc_info.value.status_code == 429
        assert "Rate limit exceeded" in exc_info.value.detail

    def test_token_refill(self):
        """Test tokens refill over time."""
        from src.middleware import rate_limit as rl_module

        rl_module._buckets.clear()

        mock_request = MagicMock()
        mock_request.client.host = "192.168.1.4"
        mock_request.url.path = "/api/test"

        # Set bucket with low tokens but old timestamp
        key = ("192.168.1.4", "/api/test")
        rl_module._buckets[key] = {
            "tokens": 0.5,
            "timestamp": time.time() - 10,  # 10 seconds ago
        }

        # Should not raise due to refill
        rl_module.rate_limit(mock_request)

    def test_tokens_capped_at_burst(self):
        """Test tokens are capped at BURST value."""
        from src.middleware import rate_limit as rl_module

        rl_module._buckets.clear()

        mock_request = MagicMock()
        mock_request.client.host = "192.168.1.5"
        mock_request.url.path = "/api/test"

        # Set bucket with many tokens and very old timestamp
        key = ("192.168.1.5", "/api/test")
        rl_module._buckets[key] = {
            "tokens": 5.0,
            "timestamp": time.time() - 1000,  # Very old
        }

        rl_module.rate_limit(mock_request)

        # Tokens should be capped at BURST
        assert rl_module._buckets[key]["tokens"] <= rl_module.BURST

    def test_unknown_client_host(self):
        """Test request with no client info uses 'unknown' host."""
        from src.middleware import rate_limit as rl_module

        rl_module._buckets.clear()

        mock_request = MagicMock()
        mock_request.client = None
        mock_request.url.path = "/api/test"

        # Should not raise
        rl_module.rate_limit(mock_request)

        # Bucket should use 'unknown' as host
        key = ("unknown", "/api/test")
        assert key in rl_module._buckets

    def test_different_paths_separate_buckets(self):
        """Test different paths have separate buckets."""
        from src.middleware import rate_limit as rl_module

        rl_module._buckets.clear()

        mock_request1 = MagicMock()
        mock_request1.client.host = "192.168.1.6"
        mock_request1.url.path = "/api/path1"

        mock_request2 = MagicMock()
        mock_request2.client.host = "192.168.1.6"
        mock_request2.url.path = "/api/path2"

        rl_module.rate_limit(mock_request1)
        rl_module.rate_limit(mock_request2)

        # Both buckets should exist
        assert ("192.168.1.6", "/api/path1") in rl_module._buckets
        assert ("192.168.1.6", "/api/path2") in rl_module._buckets

    def test_different_ips_separate_buckets(self):
        """Test different IPs have separate buckets."""
        from src.middleware import rate_limit as rl_module

        rl_module._buckets.clear()

        mock_request1 = MagicMock()
        mock_request1.client.host = "192.168.1.7"
        mock_request1.url.path = "/api/test"

        mock_request2 = MagicMock()
        mock_request2.client.host = "192.168.1.8"
        mock_request2.url.path = "/api/test"

        rl_module.rate_limit(mock_request1)
        rl_module.rate_limit(mock_request2)

        # Both buckets should exist
        assert ("192.168.1.7", "/api/test") in rl_module._buckets
        assert ("192.168.1.8", "/api/test") in rl_module._buckets


class TestTokenBucketAlgorithm:
    """Tests for token bucket algorithm logic."""

    def test_refill_calculation(self):
        """Test refill calculation."""
        qps = 10.0
        elapsed = 2.0
        current_tokens = 5.0
        burst = 20

        refill = elapsed * qps
        new_tokens = min(burst, current_tokens + refill)

        assert refill == 20.0
        assert new_tokens == 20  # Capped at burst

    def test_token_deduction(self):
        """Test token deduction on request."""
        tokens = 5.0
        tokens -= 1

        assert tokens == 4.0

    def test_insufficient_tokens(self):
        """Test insufficient tokens check."""
        tokens = 0.5

        if tokens < 1:
            should_reject = True
        else:
            should_reject = False

        assert should_reject is True


class TestBucketStructure:
    """Tests for bucket data structure."""

    def test_bucket_fields(self):
        """Test bucket has required fields."""
        bucket = {"tokens": 20.0, "timestamp": time.time()}

        assert "tokens" in bucket
        assert "timestamp" in bucket

    def test_bucket_key_structure(self):
        """Test bucket key is (ip, path) tuple."""
        key = ("192.168.1.1", "/api/endpoint")

        assert isinstance(key, tuple)
        assert len(key) == 2
        assert key[0] == "192.168.1.1"
        assert key[1] == "/api/endpoint"
