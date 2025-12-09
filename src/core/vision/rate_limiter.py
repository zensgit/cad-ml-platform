"""Vision provider rate limiting.

Provides:
- Token bucket rate limiter
- Per-provider rate limits
- Async-safe operations
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

from .base import VisionDescription, VisionProviderError

logger = logging.getLogger(__name__)


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_minute: float = 60.0
    requests_per_hour: float = 1000.0
    burst_size: int = 10  # Max burst requests
    retry_after_seconds: float = 1.0  # Wait time when limited


@dataclass
class RateLimitStats:
    """Rate limit statistics."""

    total_requests: int = 0
    allowed_requests: int = 0
    rejected_requests: int = 0
    current_tokens: float = 0.0
    last_refill: float = 0.0

    @property
    def rejection_rate(self) -> float:
        """Calculate rejection rate."""
        if self.total_requests == 0:
            return 0.0
        return self.rejected_requests / self.total_requests


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.

    Allows bursting while maintaining average rate.
    """

    def __init__(
        self,
        rate: float,  # Tokens per second
        capacity: int,  # Maximum tokens (burst size)
    ):
        """
        Initialize rate limiter.

        Args:
            rate: Token refill rate per second
            capacity: Maximum token capacity
        """
        self.rate = rate
        self.capacity = capacity
        self._tokens = float(capacity)
        self._last_refill = time.time()
        self._lock = asyncio.Lock()
        self._stats = RateLimitStats(current_tokens=capacity)

    async def acquire(self, tokens: int = 1, wait: bool = True) -> bool:
        """
        Acquire tokens from the bucket.

        Args:
            tokens: Number of tokens to acquire
            wait: If True, wait for tokens; if False, return immediately

        Returns:
            True if tokens acquired, False if rejected
        """
        async with self._lock:
            self._refill()
            self._stats.total_requests += 1

            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats.allowed_requests += 1
                self._stats.current_tokens = self._tokens
                return True

            if not wait:
                self._stats.rejected_requests += 1
                return False

        # Wait for tokens
        wait_time = (tokens - self._tokens) / self.rate
        logger.debug(f"Rate limited, waiting {wait_time:.2f}s for tokens")
        await asyncio.sleep(wait_time)

        async with self._lock:
            self._refill()
            if self._tokens >= tokens:
                self._tokens -= tokens
                self._stats.allowed_requests += 1
                self._stats.current_tokens = self._tokens
                return True

            self._stats.rejected_requests += 1
            return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self._last_refill
        self._tokens = min(
            self.capacity,
            self._tokens + elapsed * self.rate,
        )
        self._last_refill = now
        self._stats.last_refill = now

    @property
    def available_tokens(self) -> float:
        """Get current available tokens (approximate)."""
        return self._tokens

    @property
    def stats(self) -> RateLimitStats:
        """Get rate limit statistics."""
        self._stats.current_tokens = self._tokens
        return self._stats

    def reset(self) -> None:
        """Reset rate limiter to full capacity."""
        self._tokens = float(self.capacity)
        self._last_refill = time.time()
        self._stats = RateLimitStats(current_tokens=self.capacity)


class RateLimitError(VisionProviderError):
    """Raised when rate limit is exceeded."""

    def __init__(self, provider: str, retry_after: float):
        self.retry_after = retry_after
        super().__init__(provider, f"Rate limit exceeded, retry after {retry_after:.1f}s")


class RateLimitedVisionProvider:
    """
    Wrapper that adds rate limiting to any VisionProvider.

    Enforces requests per minute limits with token bucket algorithm.
    """

    def __init__(
        self,
        provider,  # VisionProvider
        config: Optional[RateLimitConfig] = None,
    ):
        """
        Initialize rate-limited provider.

        Args:
            provider: The underlying vision provider
            config: Rate limit configuration
        """
        self._provider = provider
        self._config = config or RateLimitConfig()

        # Convert RPM to tokens per second
        rate = self._config.requests_per_minute / 60.0
        self._limiter = TokenBucketRateLimiter(
            rate=rate,
            capacity=self._config.burst_size,
        )

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        wait_for_rate_limit: bool = True,
    ) -> VisionDescription:
        """
        Analyze image with rate limiting.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description
            wait_for_rate_limit: If True, wait when limited; if False, raise error

        Returns:
            VisionDescription with analysis results

        Raises:
            RateLimitError: When rate limit exceeded and wait_for_rate_limit=False
        """
        acquired = await self._limiter.acquire(
            tokens=1,
            wait=wait_for_rate_limit,
        )

        if not acquired:
            raise RateLimitError(
                self._provider.provider_name,
                self._config.retry_after_seconds,
            )

        return await self._provider.analyze_image(image_data, include_description)

    @property
    def provider_name(self) -> str:
        """Return wrapped provider name."""
        return self._provider.provider_name

    @property
    def rate_limit_stats(self) -> RateLimitStats:
        """Get rate limit statistics."""
        return self._limiter.stats

    @property
    def available_requests(self) -> float:
        """Get approximate available requests."""
        return self._limiter.available_tokens


# Default rate limits per provider
DEFAULT_RATE_LIMITS: Dict[str, RateLimitConfig] = {
    "openai": RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        burst_size=10,
    ),
    "anthropic": RateLimitConfig(
        requests_per_minute=60,
        requests_per_hour=1000,
        burst_size=10,
    ),
    "deepseek": RateLimitConfig(
        requests_per_minute=120,
        requests_per_hour=2000,
        burst_size=20,
    ),
    "deepseek_stub": RateLimitConfig(
        requests_per_minute=1000,  # High limit for testing
        requests_per_hour=10000,
        burst_size=100,
    ),
}


def create_rate_limited_provider(
    provider,
    requests_per_minute: Optional[float] = None,
    burst_size: Optional[int] = None,
) -> RateLimitedVisionProvider:
    """
    Factory to create a rate-limited provider wrapper.

    Uses default limits based on provider name if not specified.

    Args:
        provider: The underlying vision provider
        requests_per_minute: Override RPM limit
        burst_size: Override burst size

    Returns:
        RateLimitedVisionProvider wrapping the original

    Example:
        >>> provider = create_vision_provider("openai")
        >>> limited = create_rate_limited_provider(provider, requests_per_minute=30)
        >>> result = await limited.analyze_image(image_bytes)
    """
    provider_name = provider.provider_name

    # Get default config or create new one
    if provider_name in DEFAULT_RATE_LIMITS:
        config = DEFAULT_RATE_LIMITS[provider_name]
    else:
        config = RateLimitConfig()

    # Apply overrides
    if requests_per_minute is not None:
        config = RateLimitConfig(
            requests_per_minute=requests_per_minute,
            requests_per_hour=config.requests_per_hour,
            burst_size=burst_size or config.burst_size,
        )
    elif burst_size is not None:
        config = RateLimitConfig(
            requests_per_minute=config.requests_per_minute,
            requests_per_hour=config.requests_per_hour,
            burst_size=burst_size,
        )

    return RateLimitedVisionProvider(provider=provider, config=config)
