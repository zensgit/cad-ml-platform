"""Rate Limiting Implementation.

Provides multiple rate limiting strategies:
- Fixed Window
- Sliding Window
- Token Bucket
- Leaky Bucket
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Dict, Optional, TypeVar, Union

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class RateLimitStrategy(str, Enum):
    """Rate limiting strategies."""

    FIXED_WINDOW = "fixed_window"
    SLIDING_WINDOW = "sliding_window"
    TOKEN_BUCKET = "token_bucket"
    LEAKY_BUCKET = "leaky_bucket"


@dataclass
class RateLimitConfig:
    """Rate limit configuration."""

    requests_per_second: float = 10.0
    requests_per_minute: float = 100.0
    requests_per_hour: float = 1000.0
    burst_size: int = 20
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW

    # Per-endpoint overrides
    endpoint_limits: Dict[str, Dict[str, float]] = field(default_factory=dict)

    # Per-tier limits (multipliers)
    tier_multipliers: Dict[str, float] = field(default_factory=lambda: {
        "free": 0.5,
        "standard": 1.0,
        "professional": 2.0,
        "enterprise": 10.0,
    })

    def get_limit_for_tier(self, tier: str, period: str = "minute") -> float:
        """Get rate limit for a specific tier and period."""
        multiplier = self.tier_multipliers.get(tier, 1.0)
        if period == "second":
            return self.requests_per_second * multiplier
        elif period == "minute":
            return self.requests_per_minute * multiplier
        elif period == "hour":
            return self.requests_per_hour * multiplier
        return self.requests_per_minute * multiplier


@dataclass
class RateLimitResult:
    """Result of a rate limit check."""

    allowed: bool
    remaining: int
    reset_at: float
    retry_after: Optional[float] = None
    limit: int = 0

    def to_headers(self) -> Dict[str, str]:
        """Convert to HTTP headers."""
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
    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check if request is allowed.

        Args:
            key: Unique identifier (user_id, api_key, ip)
            cost: Cost of this request (default 1)

        Returns:
            RateLimitResult with allowed status and metadata
        """
        pass

    @abstractmethod
    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        pass


class FixedWindowLimiter(RateLimiter):
    """Fixed window rate limiter.

    Simple but can have burst issues at window boundaries.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._windows: Dict[str, Dict[str, Any]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            window_start = int(now / 60) * 60  # 1-minute windows
            window_key = f"{key}:{window_start}"

            if window_key not in self._windows:
                self._windows[window_key] = {"count": 0, "start": window_start}
                # Cleanup old windows
                self._cleanup_old_windows(now)

            window = self._windows[window_key]
            limit = int(self.config.requests_per_minute)

            if window["count"] + cost > limit:
                return RateLimitResult(
                    allowed=False,
                    remaining=limit - window["count"],
                    reset_at=window_start + 60,
                    retry_after=window_start + 60 - now,
                    limit=limit,
                )

            window["count"] += cost
            return RateLimitResult(
                allowed=True,
                remaining=limit - window["count"],
                reset_at=window_start + 60,
                limit=limit,
            )

    async def reset(self, key: str) -> None:
        async with self._get_lock():
            keys_to_remove = [k for k in self._windows if k.startswith(f"{key}:")]
            for k in keys_to_remove:
                del self._windows[k]

    def _cleanup_old_windows(self, now: float) -> None:
        """Remove windows older than 2 minutes."""
        cutoff = now - 120
        keys_to_remove = [
            k for k, v in self._windows.items()
            if v["start"] < cutoff
        ]
        for k in keys_to_remove:
            del self._windows[k]


class SlidingWindowLimiter(RateLimiter):
    """Sliding window rate limiter.

    More accurate than fixed window, smooths out boundary issues.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._requests: Dict[str, list] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            window_size = 60.0  # 1 minute
            limit = int(self.config.requests_per_minute)

            if key not in self._requests:
                self._requests[key] = []

            # Remove old requests outside the window
            cutoff = now - window_size
            self._requests[key] = [
                ts for ts in self._requests[key] if ts > cutoff
            ]

            current_count = len(self._requests[key])

            if current_count + cost > limit:
                # Find when the oldest request will expire
                if self._requests[key]:
                    oldest = min(self._requests[key])
                    retry_after = oldest + window_size - now
                else:
                    retry_after = 0

                return RateLimitResult(
                    allowed=False,
                    remaining=limit - current_count,
                    reset_at=now + window_size,
                    retry_after=max(0, retry_after),
                    limit=limit,
                )

            # Add current request(s)
            for _ in range(cost):
                self._requests[key].append(now)

            return RateLimitResult(
                allowed=True,
                remaining=limit - current_count - cost,
                reset_at=now + window_size,
                limit=limit,
            )

    async def reset(self, key: str) -> None:
        async with self._get_lock():
            self._requests.pop(key, None)


class TokenBucketLimiter(RateLimiter):
    """Token bucket rate limiter.

    Allows bursts while maintaining average rate.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._buckets: Dict[str, Dict[str, float]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            bucket_size = self.config.burst_size
            refill_rate = self.config.requests_per_second

            if key not in self._buckets:
                self._buckets[key] = {
                    "tokens": float(bucket_size),
                    "last_update": now,
                }

            bucket = self._buckets[key]

            # Refill tokens based on elapsed time
            elapsed = now - bucket["last_update"]
            bucket["tokens"] = min(
                bucket_size,
                bucket["tokens"] + elapsed * refill_rate
            )
            bucket["last_update"] = now

            if bucket["tokens"] >= cost:
                bucket["tokens"] -= cost
                return RateLimitResult(
                    allowed=True,
                    remaining=int(bucket["tokens"]),
                    reset_at=now + (bucket_size - bucket["tokens"]) / refill_rate,
                    limit=bucket_size,
                )

            # Calculate when enough tokens will be available
            tokens_needed = cost - bucket["tokens"]
            retry_after = tokens_needed / refill_rate

            return RateLimitResult(
                allowed=False,
                remaining=int(bucket["tokens"]),
                reset_at=now + retry_after,
                retry_after=retry_after,
                limit=bucket_size,
            )

    async def reset(self, key: str) -> None:
        async with self._get_lock():
            self._buckets.pop(key, None)


class LeakyBucketLimiter(RateLimiter):
    """Leaky bucket rate limiter.

    Processes requests at a constant rate, queuing excess.
    """

    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._buckets: Dict[str, Dict[str, Any]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        async with self._get_lock():
            now = time.time()
            bucket_size = self.config.burst_size
            leak_rate = self.config.requests_per_second

            if key not in self._buckets:
                self._buckets[key] = {
                    "water_level": 0.0,
                    "last_update": now,
                }

            bucket = self._buckets[key]

            # Leak water based on elapsed time
            elapsed = now - bucket["last_update"]
            bucket["water_level"] = max(
                0.0,
                bucket["water_level"] - elapsed * leak_rate
            )
            bucket["last_update"] = now

            if bucket["water_level"] + cost <= bucket_size:
                bucket["water_level"] += cost
                return RateLimitResult(
                    allowed=True,
                    remaining=int(bucket_size - bucket["water_level"]),
                    reset_at=now + bucket["water_level"] / leak_rate,
                    limit=bucket_size,
                )

            # Calculate when space will be available
            overflow = bucket["water_level"] + cost - bucket_size
            retry_after = overflow / leak_rate

            return RateLimitResult(
                allowed=False,
                remaining=0,
                reset_at=now + retry_after,
                retry_after=retry_after,
                limit=bucket_size,
            )

    async def reset(self, key: str) -> None:
        async with self._get_lock():
            self._buckets.pop(key, None)


class CompositeRateLimiter:
    """Combines multiple rate limiters for layered protection."""

    def __init__(self, limiters: Dict[str, RateLimiter]):
        """Initialize with named limiters.

        Args:
            limiters: Dict of name -> limiter (e.g., {"per_second": ..., "per_minute": ...})
        """
        self.limiters = limiters

    async def check(self, key: str, cost: int = 1) -> RateLimitResult:
        """Check all limiters, return first rejection or last success."""
        results = []
        for name, limiter in self.limiters.items():
            result = await limiter.check(key, cost)
            results.append((name, result))
            if not result.allowed:
                logger.debug(f"Rate limit hit on {name} for key {key}")
                return result

        # All passed, return the most restrictive remaining
        if results:
            min_result = min(results, key=lambda x: x[1].remaining)
            return min_result[1]

        return RateLimitResult(allowed=True, remaining=999, reset_at=time.time() + 60)

    async def reset(self, key: str) -> None:
        """Reset all limiters for a key."""
        for limiter in self.limiters.values():
            await limiter.reset(key)


def create_rate_limiter(config: RateLimitConfig) -> RateLimiter:
    """Factory function to create appropriate rate limiter."""
    if config.strategy == RateLimitStrategy.FIXED_WINDOW:
        return FixedWindowLimiter(config)
    elif config.strategy == RateLimitStrategy.SLIDING_WINDOW:
        return SlidingWindowLimiter(config)
    elif config.strategy == RateLimitStrategy.TOKEN_BUCKET:
        return TokenBucketLimiter(config)
    elif config.strategy == RateLimitStrategy.LEAKY_BUCKET:
        return LeakyBucketLimiter(config)
    else:
        return SlidingWindowLimiter(config)


# Global rate limiter instance
_rate_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = create_rate_limiter(RateLimitConfig())
    return _rate_limiter


def set_rate_limiter(limiter: RateLimiter) -> None:
    """Set global rate limiter."""
    global _rate_limiter
    _rate_limiter = limiter


def rate_limit(
    key_func: Optional[Callable[..., str]] = None,
    cost: int = 1,
    limiter: Optional[RateLimiter] = None,
) -> Callable[[F], F]:
    """Decorator for rate limiting functions.

    Args:
        key_func: Function to extract rate limit key from args
        cost: Cost of this operation
        limiter: Optional specific limiter to use

    Example:
        @rate_limit(key_func=lambda request: request.client.host)
        async def my_endpoint(request: Request):
            ...
    """
    def decorator(func: F) -> F:
        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            # Extract key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = "default"

            # Get limiter
            rl = limiter or get_rate_limiter()

            # Check rate limit
            result = await rl.check(key, cost)

            if not result.allowed:
                from fastapi import HTTPException
                from fastapi.responses import JSONResponse

                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "retry_after": result.retry_after,
                        "limit": result.limit,
                    },
                    headers=result.to_headers(),
                )

            return await func(*args, **kwargs)

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            # For sync functions, run async check in event loop
            import asyncio

            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = "default"

            rl = limiter or get_rate_limiter()

            # Run async check
            loop = asyncio.new_event_loop()
            try:
                result = loop.run_until_complete(rl.check(key, cost))
            finally:
                loop.close()

            if not result.allowed:
                from fastapi import HTTPException
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Rate limit exceeded",
                        "retry_after": result.retry_after,
                    },
                    headers=result.to_headers(),
                )

            return func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore

    return decorator


class RateLimitMiddleware:
    """FastAPI middleware for global rate limiting."""

    def __init__(
        self,
        app: Any,
        config: Optional[RateLimitConfig] = None,
        key_func: Optional[Callable[[Any], str]] = None,
    ):
        self.app = app
        self.limiter = create_rate_limiter(config or RateLimitConfig())
        self.key_func = key_func or self._default_key_func

    def _default_key_func(self, scope: dict) -> str:
        """Extract rate limit key from request scope."""
        # Try to get client IP
        client = scope.get("client")
        if client:
            return f"ip:{client[0]}"

        # Try to get from headers
        headers = dict(scope.get("headers", []))
        if b"x-forwarded-for" in headers:
            return f"ip:{headers[b'x-forwarded-for'].decode().split(',')[0].strip()}"

        if b"x-api-key" in headers:
            return f"key:{headers[b'x-api-key'].decode()}"

        return "anonymous"

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        key = self.key_func(scope)
        result = await self.limiter.check(key)

        if not result.allowed:
            # Return 429 response
            response_body = b'{"error": "Rate limit exceeded"}'
            headers = [
                (b"content-type", b"application/json"),
                (b"retry-after", str(int(result.retry_after or 60)).encode()),
            ]
            for k, v in result.to_headers().items():
                headers.append((k.lower().encode(), v.encode()))

            await send({
                "type": "http.response.start",
                "status": 429,
                "headers": headers,
            })
            await send({
                "type": "http.response.body",
                "body": response_body,
            })
            return

        # Add rate limit headers to response
        original_send = send

        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                headers = list(message.get("headers", []))
                for k, v in result.to_headers().items():
                    headers.append((k.lower().encode(), v.encode()))
                message["headers"] = headers
            await original_send(message)

        await self.app(scope, receive, send_with_headers)
