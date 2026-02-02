"""Rate Limiter Module.

Provides rate limiting:
- Token bucket algorithm
- Sliding window algorithm
- Fixed window algorithm
- Distributed limiting support
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class RateLimitAlgorithm(Enum):
    """Rate limiting algorithms."""
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"
    FIXED_WINDOW = "fixed_window"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_second: float = 10.0
    burst_size: int = 10
    window_size_seconds: float = 1.0
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET


@dataclass
class RateLimitResult:
    """Result of rate limit check."""

    allowed: bool
    remaining: int = 0
    reset_after_seconds: float = 0.0
    retry_after_seconds: float = 0.0
    limit: int = 0
    current_usage: int = 0

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(time.time() + self.reset_after_seconds)),
        }
        if not self.allowed:
            headers["Retry-After"] = str(int(self.retry_after_seconds) + 1)
        return headers


class RateLimiter(ABC):
    """Abstract rate limiter."""

    @abstractmethod
    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed."""
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        pass


class TokenBucketLimiter(RateLimiter):
    """Token bucket rate limiter."""

    def __init__(
        self,
        rate: float = 10.0,  # tokens per second
        capacity: int = 10,  # max tokens
    ):
        self._rate = rate
        self._capacity = capacity
        self._buckets: Dict[str, Tuple[float, float]] = {}  # key -> (tokens, last_update)
        self._lock = asyncio.Lock()

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed using token bucket."""
        async with self._lock:
            now = time.time()

            if key not in self._buckets:
                self._buckets[key] = (self._capacity, now)

            tokens, last_update = self._buckets[key]

            # Add tokens based on time elapsed
            elapsed = now - last_update
            tokens = min(self._capacity, tokens + elapsed * self._rate)

            # Check if request can be allowed
            if tokens >= cost:
                tokens -= cost
                self._buckets[key] = (tokens, now)
                return RateLimitResult(
                    allowed=True,
                    remaining=int(tokens),
                    reset_after_seconds=(self._capacity - tokens) / self._rate,
                    limit=self._capacity,
                    current_usage=self._capacity - int(tokens),
                )
            else:
                # Calculate wait time
                wait_time = (cost - tokens) / self._rate
                self._buckets[key] = (tokens, now)
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_after_seconds=self._capacity / self._rate,
                    retry_after_seconds=wait_time,
                    limit=self._capacity,
                    current_usage=self._capacity,
                )

    async def reset(self, key: str) -> None:
        """Reset bucket for a key."""
        async with self._lock:
            self._buckets[key] = (self._capacity, time.time())


class SlidingWindowLimiter(RateLimiter):
    """Sliding window rate limiter."""

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
    ):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._requests: Dict[str, List[float]] = {}  # key -> list of timestamps
        self._lock = asyncio.Lock()

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed using sliding window."""
        async with self._lock:
            now = time.time()
            window_start = now - self._window_seconds

            if key not in self._requests:
                self._requests[key] = []

            # Remove old requests outside window
            self._requests[key] = [
                ts for ts in self._requests[key]
                if ts > window_start
            ]

            current_count = len(self._requests[key])

            if current_count + cost <= self._max_requests:
                # Add current requests
                for _ in range(cost):
                    self._requests[key].append(now)

                return RateLimitResult(
                    allowed=True,
                    remaining=self._max_requests - current_count - cost,
                    reset_after_seconds=self._window_seconds,
                    limit=self._max_requests,
                    current_usage=current_count + cost,
                )
            else:
                # Calculate when oldest request will expire
                if self._requests[key]:
                    oldest = min(self._requests[key])
                    retry_after = oldest + self._window_seconds - now
                else:
                    retry_after = 0

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_after_seconds=self._window_seconds,
                    retry_after_seconds=max(0, retry_after),
                    limit=self._max_requests,
                    current_usage=current_count,
                )

    async def reset(self, key: str) -> None:
        """Reset window for a key."""
        async with self._lock:
            self._requests[key] = []


class FixedWindowLimiter(RateLimiter):
    """Fixed window rate limiter."""

    def __init__(
        self,
        max_requests: int = 100,
        window_seconds: float = 60.0,
    ):
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._windows: Dict[str, Tuple[int, int]] = {}  # key -> (count, window_start)
        self._lock = asyncio.Lock()

    def _get_window_start(self, now: float) -> int:
        """Get the start time of the current window."""
        return int(now / self._window_seconds) * int(self._window_seconds)

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed using fixed window."""
        async with self._lock:
            now = time.time()
            window_start = self._get_window_start(now)

            if key not in self._windows or self._windows[key][1] != window_start:
                # New window
                self._windows[key] = (0, window_start)

            count, _ = self._windows[key]

            if count + cost <= self._max_requests:
                self._windows[key] = (count + cost, window_start)
                return RateLimitResult(
                    allowed=True,
                    remaining=self._max_requests - count - cost,
                    reset_after_seconds=window_start + self._window_seconds - now,
                    limit=self._max_requests,
                    current_usage=count + cost,
                )
            else:
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_after_seconds=window_start + self._window_seconds - now,
                    retry_after_seconds=window_start + self._window_seconds - now,
                    limit=self._max_requests,
                    current_usage=count,
                )

    async def reset(self, key: str) -> None:
        """Reset window for a key."""
        async with self._lock:
            if key in self._windows:
                _, window_start = self._windows[key]
                self._windows[key] = (0, window_start)


class LeakyBucketLimiter(RateLimiter):
    """Leaky bucket rate limiter (smooths traffic)."""

    def __init__(
        self,
        rate: float = 10.0,  # requests per second
        capacity: int = 10,  # bucket size
    ):
        self._rate = rate
        self._capacity = capacity
        self._buckets: Dict[str, Tuple[int, float]] = {}  # key -> (level, last_leak)
        self._lock = asyncio.Lock()

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed using leaky bucket."""
        async with self._lock:
            now = time.time()

            if key not in self._buckets:
                self._buckets[key] = (0, now)

            level, last_leak = self._buckets[key]

            # Leak water based on time elapsed
            elapsed = now - last_leak
            leaked = int(elapsed * self._rate)
            level = max(0, level - leaked)

            # Try to add request to bucket
            if level + cost <= self._capacity:
                level += cost
                self._buckets[key] = (level, now)
                return RateLimitResult(
                    allowed=True,
                    remaining=self._capacity - level,
                    reset_after_seconds=level / self._rate,
                    limit=self._capacity,
                    current_usage=level,
                )
            else:
                # Bucket full, calculate wait time
                wait_time = (level + cost - self._capacity) / self._rate
                self._buckets[key] = (level, now)
                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    reset_after_seconds=level / self._rate,
                    retry_after_seconds=wait_time,
                    limit=self._capacity,
                    current_usage=level,
                )

    async def reset(self, key: str) -> None:
        """Reset bucket for a key."""
        async with self._lock:
            self._buckets[key] = (0, time.time())


class CompositeRateLimiter(RateLimiter):
    """Combines multiple rate limiters."""

    def __init__(self, limiters: List[Tuple[str, RateLimiter]]):
        self._limiters = limiters

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check all limiters, fail if any fails."""
        results = []

        for name, limiter in self._limiters:
            result = await limiter.check(key, cost)
            results.append((name, result))

            if not result.allowed:
                # Return first failure
                return result

        # All passed, return most restrictive
        if results:
            min_remaining = min(r.remaining for _, r in results)
            max_reset = max(r.reset_after_seconds for _, r in results)
            return RateLimitResult(
                allowed=True,
                remaining=min_remaining,
                reset_after_seconds=max_reset,
                limit=min(r.limit for _, r in results),
                current_usage=max(r.current_usage for _, r in results),
            )

        return RateLimitResult(allowed=True)

    async def reset(self, key: str) -> None:
        """Reset all limiters for a key."""
        for _, limiter in self._limiters:
            await limiter.reset(key)


class RateLimiterMiddleware:
    """Middleware for rate limiting."""

    def __init__(
        self,
        limiter: RateLimiter,
        key_func: Optional[Callable[[Any], str]] = None,
        on_limited: Optional[Callable[[str, RateLimitResult], Any]] = None,
    ):
        self._limiter = limiter
        self._key_func = key_func or (lambda req: "default")
        self._on_limited = on_limited

    async def __call__(self, request: Any, next_handler: Callable) -> Any:
        """Process request through rate limiter."""
        key = self._key_func(request)
        result = await self._limiter.check(key)

        if not result.allowed:
            if self._on_limited:
                return await self._on_limited(key, result)
            raise RateLimitExceeded(result)

        return await next_handler(request)


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded."""

    def __init__(self, result: RateLimitResult):
        self.result = result
        super().__init__(
            f"Rate limit exceeded. Retry after {result.retry_after_seconds:.1f}s"
        )


def create_rate_limiter(
    algorithm: RateLimitAlgorithm = RateLimitAlgorithm.TOKEN_BUCKET,
    **kwargs,
) -> RateLimiter:
    """Factory function to create rate limiters."""
    if algorithm == RateLimitAlgorithm.TOKEN_BUCKET:
        return TokenBucketLimiter(
            rate=kwargs.get("rate", 10.0),
            capacity=kwargs.get("capacity", 10),
        )
    elif algorithm == RateLimitAlgorithm.SLIDING_WINDOW:
        return SlidingWindowLimiter(
            max_requests=kwargs.get("max_requests", 100),
            window_seconds=kwargs.get("window_seconds", 60.0),
        )
    elif algorithm == RateLimitAlgorithm.FIXED_WINDOW:
        return FixedWindowLimiter(
            max_requests=kwargs.get("max_requests", 100),
            window_seconds=kwargs.get("window_seconds", 60.0),
        )
    elif algorithm == RateLimitAlgorithm.LEAKY_BUCKET:
        return LeakyBucketLimiter(
            rate=kwargs.get("rate", 10.0),
            capacity=kwargs.get("capacity", 10),
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


__all__ = [
    "RateLimitAlgorithm",
    "RateLimitConfig",
    "RateLimitResult",
    "RateLimiter",
    "TokenBucketLimiter",
    "SlidingWindowLimiter",
    "FixedWindowLimiter",
    "LeakyBucketLimiter",
    "CompositeRateLimiter",
    "RateLimiterMiddleware",
    "RateLimitExceeded",
    "create_rate_limiter",
]
