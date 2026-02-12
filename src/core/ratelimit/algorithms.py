"""Rate Limiting Algorithms.

Provides various rate limiting algorithms:
- Token Bucket
- Leaky Bucket
- Sliding Window Log
- Sliding Window Counter
- Fixed Window Counter
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""
    allowed: bool
    remaining: int
    limit: int
    reset_at: float  # Unix timestamp
    retry_after: Optional[float] = None  # Seconds to wait

    @property
    def headers(self) -> Dict[str, str]:
        """Generate standard rate limit headers."""
        headers = {
            "X-RateLimit-Limit": str(self.limit),
            "X-RateLimit-Remaining": str(max(0, self.remaining)),
            "X-RateLimit-Reset": str(int(self.reset_at)),
        }
        if self.retry_after is not None:
            headers["Retry-After"] = str(int(self.retry_after))
        return headers


class RateLimiter(ABC):
    """Abstract base class for rate limiters."""

    @abstractmethod
    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        """Try to acquire permission for a request.

        Args:
            key: Identifier for the rate limit (e.g., user_id, ip_address)
            cost: Cost of this request (default 1)

        Returns:
            RateLimitResult indicating if request is allowed.
        """
        pass

    @abstractmethod
    async def get_status(self, key: str) -> RateLimitResult:
        """Get current rate limit status without consuming.

        Args:
            key: Identifier for the rate limit

        Returns:
            Current RateLimitResult.
        """
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        pass


class TokenBucketLimiter(RateLimiter):
    """Token Bucket algorithm.

    Tokens are added at a constant rate. Each request consumes tokens.
    Allows bursting up to bucket capacity.
    """

    def __init__(
        self,
        capacity: int,
        refill_rate: float,
        refill_interval: float = 1.0,
    ):
        """Initialize token bucket.

        Args:
            capacity: Maximum tokens in bucket
            refill_rate: Tokens added per refill interval
            refill_interval: Seconds between refills
        """
        self.capacity = capacity
        self.refill_rate = refill_rate
        self.refill_interval = refill_interval

        # {key: (tokens, last_refill_time)}
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _refill(self, key: str, now: float) -> float:
        """Calculate current tokens after refill."""
        if key not in self._buckets:
            return float(self.capacity)

        tokens, last_refill = self._buckets[key]
        elapsed = now - last_refill
        intervals = elapsed / self.refill_interval
        new_tokens = min(self.capacity, tokens + intervals * self.refill_rate)
        return new_tokens

    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            tokens = self._refill(key, now)

            if tokens >= cost:
                # Consume tokens
                self._buckets[key] = (tokens - cost, now)
                return RateLimitResult(
                    allowed=True,
                    remaining=int(tokens - cost),
                    limit=self.capacity,
                    reset_at=now + self.refill_interval,
                )
            else:
                # Calculate wait time
                needed = cost - tokens
                wait_intervals = needed / self.refill_rate
                retry_after = wait_intervals * self.refill_interval

                return RateLimitResult(
                    allowed=False,
                    remaining=int(tokens),
                    limit=self.capacity,
                    reset_at=now + retry_after,
                    retry_after=retry_after,
                )

    async def get_status(self, key: str) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            tokens = self._refill(key, now)

            return RateLimitResult(
                allowed=tokens >= 1,
                remaining=int(tokens),
                limit=self.capacity,
                reset_at=now + self.refill_interval,
            )

    async def reset(self, key: str) -> None:
        async with self._get_lock():
            if key in self._buckets:
                del self._buckets[key]


class LeakyBucketLimiter(RateLimiter):
    """Leaky Bucket algorithm.

    Requests are processed at a constant rate.
    Excess requests queue up to bucket capacity.
    """

    def __init__(
        self,
        capacity: int,
        leak_rate: float,
    ):
        """Initialize leaky bucket.

        Args:
            capacity: Maximum queue size
            leak_rate: Requests processed per second
        """
        self.capacity = capacity
        self.leak_rate = leak_rate

        # {key: (queue_size, last_leak_time)}
        self._buckets: Dict[str, Tuple[float, float]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _leak(self, key: str, now: float) -> float:
        """Calculate current queue size after leaking."""
        if key not in self._buckets:
            return 0.0

        queue_size, last_leak = self._buckets[key]
        elapsed = now - last_leak
        leaked = elapsed * self.leak_rate
        return max(0, queue_size - leaked)

    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            queue_size = self._leak(key, now)

            if queue_size + cost <= self.capacity:
                # Add to queue
                self._buckets[key] = (queue_size + cost, now)
                remaining = int(self.capacity - queue_size - cost)

                return RateLimitResult(
                    allowed=True,
                    remaining=remaining,
                    limit=self.capacity,
                    reset_at=now + (queue_size + cost) / self.leak_rate,
                )
            else:
                # Queue full
                wait_time = (queue_size + cost - self.capacity) / self.leak_rate

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=self.capacity,
                    reset_at=now + wait_time,
                    retry_after=wait_time,
                )

    async def get_status(self, key: str) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            queue_size = self._leak(key, now)
            remaining = int(self.capacity - queue_size)

            return RateLimitResult(
                allowed=remaining >= 1,
                remaining=remaining,
                limit=self.capacity,
                reset_at=now + queue_size / self.leak_rate if queue_size > 0 else now,
            )

    async def reset(self, key: str) -> None:
        async with self._get_lock():
            if key in self._buckets:
                del self._buckets[key]


class SlidingWindowLogLimiter(RateLimiter):
    """Sliding Window Log algorithm.

    Tracks timestamps of all requests in the window.
    Most accurate but uses more memory.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
    ):
        """Initialize sliding window log.

        Args:
            limit: Maximum requests per window
            window_seconds: Window duration in seconds
        """
        self.limit = limit
        self.window_seconds = window_seconds

        # {key: [timestamps]}
        self._logs: Dict[str, List[float]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _cleanup(self, key: str, now: float) -> List[float]:
        """Remove expired timestamps."""
        if key not in self._logs:
            return []

        cutoff = now - self.window_seconds
        self._logs[key] = [t for t in self._logs[key] if t > cutoff]
        return self._logs[key]

    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            timestamps = self._cleanup(key, now)
            current_count = len(timestamps)

            if current_count + cost <= self.limit:
                # Add timestamps
                if key not in self._logs:
                    self._logs[key] = []
                for _ in range(cost):
                    self._logs[key].append(now)

                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit - current_count - cost,
                    limit=self.limit,
                    reset_at=now + self.window_seconds,
                )
            else:
                # Find when oldest request will expire
                if timestamps:
                    oldest = min(timestamps)
                    retry_after = oldest + self.window_seconds - now
                else:
                    retry_after = 0

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=self.limit,
                    reset_at=now + retry_after,
                    retry_after=max(0, retry_after),
                )

    async def get_status(self, key: str) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            timestamps = self._cleanup(key, now)
            current_count = len(timestamps)
            remaining = self.limit - current_count

            return RateLimitResult(
                allowed=remaining >= 1,
                remaining=remaining,
                limit=self.limit,
                reset_at=now + self.window_seconds,
            )

    async def reset(self, key: str) -> None:
        async with self._get_lock():
            if key in self._logs:
                del self._logs[key]


class SlidingWindowCounterLimiter(RateLimiter):
    """Sliding Window Counter algorithm.

    Combines fixed windows with weighted counts.
    More memory efficient than log-based approach.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
    ):
        """Initialize sliding window counter.

        Args:
            limit: Maximum requests per window
            window_seconds: Window duration in seconds
        """
        self.limit = limit
        self.window_seconds = window_seconds

        # {key: {window_start: count}}
        self._counters: Dict[str, Dict[int, int]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_window_key(self, timestamp: float) -> int:
        """Get window key for timestamp."""
        return int(timestamp // self.window_seconds)

    def _get_weighted_count(self, key: str, now: float) -> float:
        """Calculate weighted count across sliding window."""
        if key not in self._counters:
            return 0

        current_window = self._get_window_key(now)
        previous_window = current_window - 1

        # Position within current window (0 to 1)
        window_position = (now % self.window_seconds) / self.window_seconds

        counters = self._counters[key]
        current_count = counters.get(current_window, 0)
        previous_count = counters.get(previous_window, 0)

        # Weighted average
        return current_count + previous_count * (1 - window_position)

    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            weighted_count = self._get_weighted_count(key, now)

            if weighted_count + cost <= self.limit:
                # Increment current window
                current_window = self._get_window_key(now)
                if key not in self._counters:
                    self._counters[key] = {}

                if current_window not in self._counters[key]:
                    self._counters[key][current_window] = 0
                self._counters[key][current_window] += cost

                # Cleanup old windows
                self._cleanup_old_windows(key, current_window)

                return RateLimitResult(
                    allowed=True,
                    remaining=int(self.limit - weighted_count - cost),
                    limit=self.limit,
                    reset_at=now + self.window_seconds,
                )
            else:
                # Estimate retry time
                excess = weighted_count + cost - self.limit
                retry_after = excess / self.limit * self.window_seconds

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=self.limit,
                    reset_at=now + self.window_seconds,
                    retry_after=retry_after,
                )

    def _cleanup_old_windows(self, key: str, current_window: int) -> None:
        """Remove windows older than previous."""
        if key not in self._counters:
            return

        old_windows = [w for w in self._counters[key] if w < current_window - 1]
        for w in old_windows:
            del self._counters[key][w]

    async def get_status(self, key: str) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            weighted_count = self._get_weighted_count(key, now)
            remaining = int(self.limit - weighted_count)

            return RateLimitResult(
                allowed=remaining >= 1,
                remaining=remaining,
                limit=self.limit,
                reset_at=now + self.window_seconds,
            )

    async def reset(self, key: str) -> None:
        async with self._get_lock():
            if key in self._counters:
                del self._counters[key]


class FixedWindowCounterLimiter(RateLimiter):
    """Fixed Window Counter algorithm.

    Simple counter that resets at fixed intervals.
    May allow 2x burst at window boundaries.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
    ):
        """Initialize fixed window counter.

        Args:
            limit: Maximum requests per window
            window_seconds: Window duration in seconds
        """
        self.limit = limit
        self.window_seconds = window_seconds

        # {key: (count, window_start)}
        self._counters: Dict[str, Tuple[int, int]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_window_key(self, timestamp: float) -> int:
        """Get window key for timestamp."""
        return int(timestamp // self.window_seconds)

    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            current_window = self._get_window_key(now)

            # Get or create counter for current window
            if key in self._counters:
                count, window = self._counters[key]
                if window != current_window:
                    # New window, reset counter
                    count = 0
            else:
                count = 0

            if count + cost <= self.limit:
                # Increment counter
                self._counters[key] = (count + cost, current_window)

                # Calculate reset time (end of current window)
                window_end = (current_window + 1) * self.window_seconds

                return RateLimitResult(
                    allowed=True,
                    remaining=self.limit - count - cost,
                    limit=self.limit,
                    reset_at=window_end,
                )
            else:
                # Window limit exceeded
                window_end = (current_window + 1) * self.window_seconds
                retry_after = window_end - now

                return RateLimitResult(
                    allowed=False,
                    remaining=0,
                    limit=self.limit,
                    reset_at=window_end,
                    retry_after=retry_after,
                )

    async def get_status(self, key: str) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            current_window = self._get_window_key(now)
            window_end = (current_window + 1) * self.window_seconds

            if key in self._counters:
                count, window = self._counters[key]
                if window != current_window:
                    count = 0
            else:
                count = 0

            remaining = self.limit - count

            return RateLimitResult(
                allowed=remaining >= 1,
                remaining=remaining,
                limit=self.limit,
                reset_at=window_end,
            )

    async def reset(self, key: str) -> None:
        async with self._get_lock():
            if key in self._counters:
                del self._counters[key]
