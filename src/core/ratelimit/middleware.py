"""Rate Limiting Middleware and Decorators.

Provides integration utilities:
- HTTP middleware
- Function decorators
- Multi-tier rate limiting
- Key extraction strategies
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar, Union

from src.core.ratelimit.algorithms import RateLimiter, RateLimitResult

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class KeyStrategy(Enum):
    """Strategy for extracting rate limit keys."""
    IP_ADDRESS = "ip"
    USER_ID = "user"
    API_KEY = "api_key"
    ENDPOINT = "endpoint"
    COMPOSITE = "composite"
    CUSTOM = "custom"


class KeyExtractor(ABC):
    """Base class for rate limit key extraction."""

    @abstractmethod
    def extract(self, context: Dict[str, Any]) -> str:
        """Extract rate limit key from request context."""
        pass


class IPKeyExtractor(KeyExtractor):
    """Extract key from client IP address."""

    def __init__(
        self,
        headers: List[str] = None,
        default_key: str = "unknown",
    ):
        self.headers = headers or [
            "X-Forwarded-For",
            "X-Real-IP",
            "CF-Connecting-IP",
        ]
        self.default_key = default_key

    def extract(self, context: Dict[str, Any]) -> str:
        headers = context.get("headers", {})

        # Check forwarding headers
        for header in self.headers:
            value = headers.get(header)
            if value:
                # X-Forwarded-For may contain multiple IPs
                return value.split(",")[0].strip()

        # Fall back to remote address
        return context.get("remote_addr", self.default_key)


class UserKeyExtractor(KeyExtractor):
    """Extract key from authenticated user."""

    def __init__(
        self,
        user_field: str = "user_id",
        fallback_to_ip: bool = True,
    ):
        self.user_field = user_field
        self.fallback_to_ip = fallback_to_ip
        self._ip_extractor = IPKeyExtractor()

    def extract(self, context: Dict[str, Any]) -> str:
        user_id = context.get(self.user_field)
        if user_id:
            return f"user:{user_id}"

        if self.fallback_to_ip:
            return f"ip:{self._ip_extractor.extract(context)}"

        return "anonymous"


class APIKeyExtractor(KeyExtractor):
    """Extract key from API key header."""

    def __init__(
        self,
        header_name: str = "X-API-Key",
        query_param: str = "api_key",
    ):
        self.header_name = header_name
        self.query_param = query_param

    def extract(self, context: Dict[str, Any]) -> str:
        headers = context.get("headers", {})

        # Check header
        api_key = headers.get(self.header_name)
        if api_key:
            # Hash the key for privacy
            return f"api:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

        # Check query param
        query_params = context.get("query_params", {})
        api_key = query_params.get(self.query_param)
        if api_key:
            return f"api:{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"

        return "no_key"


class EndpointKeyExtractor(KeyExtractor):
    """Extract key from endpoint path."""

    def __init__(self, include_method: bool = True):
        self.include_method = include_method

    def extract(self, context: Dict[str, Any]) -> str:
        method = context.get("method", "GET")
        path = context.get("path", "/")

        if self.include_method:
            return f"{method}:{path}"
        return path


class CompositeKeyExtractor(KeyExtractor):
    """Combine multiple extractors."""

    def __init__(self, extractors: List[KeyExtractor], separator: str = ":"):
        self.extractors = extractors
        self.separator = separator

    def extract(self, context: Dict[str, Any]) -> str:
        parts = [e.extract(context) for e in self.extractors]
        return self.separator.join(parts)


@dataclass
class RateLimitTier:
    """A tier of rate limiting."""
    name: str
    limiter: RateLimiter
    key_extractor: KeyExtractor
    priority: int = 0  # Higher = checked first
    skip_on_pass: bool = False  # Skip lower tiers if this passes


class MultiTierRateLimiter:
    """Multi-tier rate limiting.

    Supports different limits for different contexts:
    - Global limits
    - Per-user limits
    - Per-endpoint limits
    """

    def __init__(self):
        self._tiers: List[RateLimitTier] = []

    def add_tier(
        self,
        name: str,
        limiter: RateLimiter,
        key_extractor: KeyExtractor,
        priority: int = 0,
        skip_on_pass: bool = False,
    ) -> None:
        """Add a rate limiting tier."""
        tier = RateLimitTier(
            name=name,
            limiter=limiter,
            key_extractor=key_extractor,
            priority=priority,
            skip_on_pass=skip_on_pass,
        )
        self._tiers.append(tier)
        self._tiers.sort(key=lambda t: -t.priority)  # Higher priority first

    async def acquire(
        self,
        context: Dict[str, Any],
        cost: int = 1,
    ) -> Tuple[bool, Dict[str, RateLimitResult]]:
        """Check all rate limit tiers.

        Returns:
            Tuple of (allowed, {tier_name: result})
        """
        results: Dict[str, RateLimitResult] = {}
        all_allowed = True

        for tier in self._tiers:
            key = tier.key_extractor.extract(context)
            result = await tier.limiter.acquire(key, cost)
            results[tier.name] = result

            if not result.allowed:
                all_allowed = False
                # Continue checking other tiers for complete status

            if result.allowed and tier.skip_on_pass:
                # Skip lower priority tiers
                break

        return all_allowed, results

    async def get_status(
        self,
        context: Dict[str, Any],
    ) -> Dict[str, RateLimitResult]:
        """Get status of all tiers without consuming."""
        results: Dict[str, RateLimitResult] = {}

        for tier in self._tiers:
            key = tier.key_extractor.extract(context)
            results[tier.name] = await tier.limiter.get_status(key)

        return results


def rate_limit(
    limiter: RateLimiter,
    key_func: Optional[Callable[..., str]] = None,
    cost_func: Optional[Callable[..., int]] = None,
    on_limited: Optional[Callable[..., Any]] = None,
) -> Callable[[F], F]:
    """Decorator for rate limiting async functions.

    Args:
        limiter: Rate limiter to use
        key_func: Function to extract key from arguments
        cost_func: Function to calculate cost from arguments
        on_limited: Handler when rate limited (raise exception if None)
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            # Extract key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = func.__name__

            # Calculate cost
            if cost_func:
                cost = cost_func(*args, **kwargs)
            else:
                cost = 1

            # Check rate limit
            result = await limiter.acquire(key, cost)

            if not result.allowed:
                if on_limited:
                    return on_limited(result, *args, **kwargs)
                else:
                    raise RateLimitExceeded(result)

            return await func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


class RateLimitExceeded(Exception):
    """Exception raised when rate limit is exceeded."""

    def __init__(self, result: RateLimitResult):
        self.result = result
        super().__init__(
            f"Rate limit exceeded. Retry after {result.retry_after:.1f}s"
        )


@dataclass
class RateLimitMiddlewareConfig:
    """Configuration for rate limit middleware."""
    limiter: RateLimiter
    key_extractor: KeyExtractor
    skip_paths: List[str] = field(default_factory=list)
    skip_methods: List[str] = field(default_factory=list)
    include_headers: bool = True
    custom_response: Optional[Callable[[RateLimitResult], Any]] = None


class RateLimitMiddleware:
    """HTTP middleware for rate limiting."""

    def __init__(self, config: RateLimitMiddlewareConfig):
        self.config = config

    async def __call__(
        self,
        request: Dict[str, Any],
        call_next: Callable,
    ) -> Dict[str, Any]:
        """Process request through rate limiter."""
        # Check skip conditions
        path = request.get("path", "")
        method = request.get("method", "")

        if path in self.config.skip_paths:
            return await call_next(request)

        if method in self.config.skip_methods:
            return await call_next(request)

        # Extract key and check limit
        key = self.config.key_extractor.extract(request)
        result = await self.config.limiter.acquire(key)

        if not result.allowed:
            if self.config.custom_response:
                response = self.config.custom_response(result)
            else:
                response = {
                    "status_code": 429,
                    "body": {
                        "error": "Too Many Requests",
                        "retry_after": result.retry_after,
                    },
                    "headers": {},
                }

            if self.config.include_headers:
                response["headers"].update(result.headers)

            return response

        # Process request
        response = await call_next(request)

        # Add rate limit headers to response
        if self.config.include_headers:
            if "headers" not in response:
                response["headers"] = {}
            response["headers"].update(result.headers)

        return response


class AdaptiveRateLimiter:
    """Rate limiter that adapts based on system load."""

    def __init__(
        self,
        base_limiter: RateLimiter,
        load_threshold: float = 0.8,
        reduction_factor: float = 0.5,
        load_provider: Optional[Callable[[], float]] = None,
    ):
        """Initialize adaptive rate limiter.

        Args:
            base_limiter: Base rate limiter
            load_threshold: System load threshold (0-1) to trigger reduction
            reduction_factor: Factor to reduce limits by when overloaded
            load_provider: Function that returns current system load (0-1)
        """
        self.base_limiter = base_limiter
        self.load_threshold = load_threshold
        self.reduction_factor = reduction_factor
        self.load_provider = load_provider or self._default_load_provider

    def _default_load_provider(self) -> float:
        """Default load provider using asyncio pending tasks."""
        try:
            loop = asyncio.get_event_loop()
            # Rough estimate based on pending tasks
            pending = len([t for t in asyncio.all_tasks(loop) if not t.done()])
            # Normalize to 0-1 range (assuming 100 tasks = full load)
            return min(1.0, pending / 100)
        except Exception:
            return 0.5

    async def acquire(self, key: str, cost: int = 1) -> RateLimitResult:
        """Acquire with adaptive cost adjustment."""
        current_load = self.load_provider()

        if current_load > self.load_threshold:
            # Increase effective cost when system is overloaded
            adjusted_cost = int(cost / self.reduction_factor)
            logger.debug(
                f"System load {current_load:.2f} > {self.load_threshold}, "
                f"adjusted cost from {cost} to {adjusted_cost}"
            )
            cost = adjusted_cost

        return await self.base_limiter.acquire(key, cost)

    async def get_status(self, key: str) -> RateLimitResult:
        return await self.base_limiter.get_status(key)

    async def reset(self, key: str) -> None:
        await self.base_limiter.reset(key)


@dataclass
class QuotaConfig:
    """Configuration for quota-based rate limiting."""
    daily_limit: int
    monthly_limit: Optional[int] = None
    burst_limit: Optional[int] = None
    burst_window_seconds: float = 60.0


class QuotaManager:
    """Manage usage quotas with multiple time windows."""

    def __init__(
        self,
        daily_limiter: RateLimiter,
        monthly_limiter: Optional[RateLimiter] = None,
        burst_limiter: Optional[RateLimiter] = None,
    ):
        self.daily = daily_limiter
        self.monthly = monthly_limiter
        self.burst = burst_limiter

    async def check_quota(
        self,
        key: str,
        cost: int = 1,
    ) -> Tuple[bool, Dict[str, RateLimitResult]]:
        """Check all quota limits."""
        results: Dict[str, RateLimitResult] = {}

        # Check burst limit first (most restrictive time window)
        if self.burst:
            result = await self.burst.acquire(key, cost)
            results["burst"] = result
            if not result.allowed:
                return False, results

        # Check daily limit
        result = await self.daily.acquire(key, cost)
        results["daily"] = result
        if not result.allowed:
            return False, results

        # Check monthly limit
        if self.monthly:
            result = await self.monthly.acquire(key, cost)
            results["monthly"] = result
            if not result.allowed:
                return False, results

        return True, results

    async def get_quota_status(self, key: str) -> Dict[str, RateLimitResult]:
        """Get status of all quota limits."""
        results: Dict[str, RateLimitResult] = {}

        if self.burst:
            results["burst"] = await self.burst.get_status(key)

        results["daily"] = await self.daily.get_status(key)

        if self.monthly:
            results["monthly"] = await self.monthly.get_status(key)

        return results
