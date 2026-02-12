"""Enterprise API Rate Limiting Middleware.

Features:
- Multi-tier rate limiting (per-IP, per-user, per-tenant, global)
- Sliding window algorithm with Redis backend
- Graceful degradation to local limiting
- Configurable limits per endpoint/method
- Rate limit headers in responses
- Metrics integration
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Optional

from fastapi import HTTPException, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.cache import get_client
from src.utils.metrics import ocr_rate_limited_total

logger = logging.getLogger(__name__)


class RateLimitTier(Enum):
    """Rate limit tiers for different client types."""

    ANONYMOUS = "anonymous"
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"
    INTERNAL = "internal"


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""

    # Requests per second
    qps: float = 10.0
    # Burst capacity
    burst: int = 20
    # Window size in seconds (for sliding window)
    window_seconds: int = 60
    # Max requests per window
    max_requests: int = 600
    # Cost multiplier for specific endpoints
    endpoint_costs: Dict[str, int] = field(default_factory=dict)


# Default configs per tier
DEFAULT_TIER_CONFIGS: Dict[RateLimitTier, RateLimitConfig] = {
    RateLimitTier.ANONYMOUS: RateLimitConfig(qps=1.0, burst=5, max_requests=60),
    RateLimitTier.FREE: RateLimitConfig(qps=5.0, burst=10, max_requests=300),
    RateLimitTier.PRO: RateLimitConfig(qps=20.0, burst=50, max_requests=1200),
    RateLimitTier.ENTERPRISE: RateLimitConfig(qps=100.0, burst=200, max_requests=6000),
    RateLimitTier.INTERNAL: RateLimitConfig(qps=1000.0, burst=1000, max_requests=60000),
}

# Endpoint-specific cost multipliers
ENDPOINT_COSTS: Dict[str, int] = {
    "/api/v1/ocr/extract": 5,
    "/api/v1/vision/analyze": 10,
    "/api/v1/assistant/ask": 3,
    "/api/v1/model/predict": 2,
}

# Lua script for Redis sliding window rate limiting
_LUA_SLIDING_WINDOW = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local max_requests = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])

-- Remove old entries outside the window
local window_start = now - window
redis.call('ZREMRANGEBYSCORE', key, '-inf', window_start)

-- Count current requests in window
local current = redis.call('ZCARD', key)

-- Check if we can allow this request
if current + cost > max_requests then
    return {0, current, max_requests - current}
end

-- Add the request(s) to the sorted set
for i = 1, cost do
    redis.call('ZADD', key, now, now .. ':' .. i .. ':' .. math.random())
end

-- Set expiration
redis.call('EXPIRE', key, window)

return {1, current + cost, max_requests - current - cost}
"""


class SlidingWindowRateLimiter:
    """Sliding window rate limiter with Redis backend and local fallback."""

    def __init__(self, prefix: str = "api:rl"):
        self.prefix = prefix
        self._script: Optional[Any] = None
        self._local_windows: Dict[str, Dict[str, Any]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        """Get or create asyncio lock (lazy initialization for Python 3.9 compatibility)."""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def _get_key(self, identifier: str) -> str:
        return f"{self.prefix}:{identifier}"

    async def check(
        self,
        identifier: str,
        config: RateLimitConfig,
        cost: int = 1,
    ) -> tuple[bool, int, int]:
        """Check if request is allowed.

        Returns:
            Tuple of (allowed, current_count, remaining)
        """
        client = get_client()
        now = time.time()
        key = self._get_key(identifier)

        if client is not None:
            try:
                return await self._check_redis(
                    client, key, now, config, cost
                )
            except Exception as e:
                logger.warning(f"Redis rate limit check failed, falling back to local: {e}")

        # Local fallback
        return await self._check_local(key, now, config, cost)

    async def _check_redis(
        self,
        client: Any,
        key: str,
        now: float,
        config: RateLimitConfig,
        cost: int,
    ) -> tuple[bool, int, int]:
        """Check rate limit using Redis."""
        if self._script is None:
            self._script = client.register_script(_LUA_SLIDING_WINDOW)

        result = await self._script(
            keys=[key],
            args=[now, config.window_seconds, config.max_requests, cost],
        )

        allowed = bool(result[0])
        current = int(result[1])
        remaining = int(result[2])

        if not allowed:
            ocr_rate_limited_total.inc()

        return allowed, current, remaining

    async def _check_local(
        self,
        key: str,
        now: float,
        config: RateLimitConfig,
        cost: int,
    ) -> tuple[bool, int, int]:
        """Check rate limit using local sliding window."""
        async with self._get_lock():
            window = self._local_windows.get(key)
            window_start = now - config.window_seconds

            if window is None:
                window = {"requests": [], "last_cleanup": now}
                self._local_windows[key] = window

            # Clean up old requests
            if now - window["last_cleanup"] > 1.0:
                window["requests"] = [
                    ts for ts in window["requests"] if ts > window_start
                ]
                window["last_cleanup"] = now

            current = len(window["requests"])
            remaining = config.max_requests - current

            if current + cost > config.max_requests:
                ocr_rate_limited_total.inc()
                return False, current, max(0, remaining)

            # Add requests
            for _ in range(cost):
                window["requests"].append(now)

            return True, current + cost, remaining - cost


class RateLimitMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for rate limiting."""

    def __init__(
        self,
        app: Any,
        tier_resolver: Optional[Callable[[Request], RateLimitTier]] = None,
        identifier_resolver: Optional[Callable[[Request], str]] = None,
        tier_configs: Optional[Dict[RateLimitTier, RateLimitConfig]] = None,
        endpoint_costs: Optional[Dict[str, int]] = None,
        exclude_paths: Optional[list[str]] = None,
    ):
        super().__init__(app)
        self.limiter = SlidingWindowRateLimiter()
        self.tier_resolver = tier_resolver or self._default_tier_resolver
        self.identifier_resolver = identifier_resolver or self._default_identifier_resolver
        self.tier_configs = tier_configs or DEFAULT_TIER_CONFIGS
        self.endpoint_costs = endpoint_costs or ENDPOINT_COSTS
        self.exclude_paths = exclude_paths or [
            "/health",
            "/healthz",
            "/ready",
            "/metrics",
            "/docs",
            "/redoc",
            "/openapi.json",
        ]

    def _default_tier_resolver(self, request: Request) -> RateLimitTier:
        """Default tier resolution based on headers/auth."""
        # Check for internal service header
        if request.headers.get("X-Internal-Service"):
            return RateLimitTier.INTERNAL

        # Check for tenant tier from auth
        tenant_tier = request.headers.get("X-Tenant-Tier", "").lower()
        tier_map = {
            "enterprise": RateLimitTier.ENTERPRISE,
            "pro": RateLimitTier.PRO,
            "free": RateLimitTier.FREE,
        }
        if tenant_tier in tier_map:
            return tier_map[tenant_tier]

        # Check for API key (at least free tier)
        if request.headers.get("X-API-Key"):
            return RateLimitTier.FREE

        return RateLimitTier.ANONYMOUS

    def _default_identifier_resolver(self, request: Request) -> str:
        """Default identifier resolution (IP + user/tenant)."""
        parts = []

        # IP address
        client_ip = request.client.host if request.client else "unknown"
        parts.append(f"ip:{client_ip}")

        # User ID if present
        user_id = request.headers.get("X-User-ID")
        if user_id:
            parts.append(f"user:{user_id}")

        # Tenant ID if present
        tenant_id = request.headers.get("X-Tenant-ID")
        if tenant_id:
            parts.append(f"tenant:{tenant_id}")

        return ":".join(parts)

    def _get_endpoint_cost(self, path: str, method: str) -> int:
        """Get cost multiplier for endpoint."""
        # Exact match first
        if path in self.endpoint_costs:
            return self.endpoint_costs[path]

        # Prefix match
        for endpoint, cost in self.endpoint_costs.items():
            if path.startswith(endpoint):
                return cost

        # POST/PUT typically cost more
        if method in ("POST", "PUT", "PATCH"):
            return 2

        return 1

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        path = request.url.path

        # Skip excluded paths
        for exclude in self.exclude_paths:
            if path.startswith(exclude):
                return await call_next(request)

        # Resolve tier and identifier
        tier = self.tier_resolver(request)
        identifier = self.identifier_resolver(request)
        config = self.tier_configs.get(tier, DEFAULT_TIER_CONFIGS[RateLimitTier.ANONYMOUS])

        # Get endpoint cost
        cost = self._get_endpoint_cost(path, request.method)

        # Check rate limit
        allowed, current, remaining = await self.limiter.check(
            identifier, config, cost
        )

        # Add rate limit headers
        headers = {
            "X-RateLimit-Limit": str(config.max_requests),
            "X-RateLimit-Remaining": str(max(0, remaining)),
            "X-RateLimit-Reset": str(int(time.time()) + config.window_seconds),
            "X-RateLimit-Tier": tier.value,
        }

        if not allowed:
            logger.warning(
                f"Rate limit exceeded for {identifier} on {path} "
                f"(tier={tier.value}, current={current})"
            )
            response = Response(
                content='{"detail": "Rate limit exceeded. Please retry later."}',
                status_code=429,
                media_type="application/json",
            )
            for key, value in headers.items():
                response.headers[key] = value
            response.headers["Retry-After"] = str(config.window_seconds)
            return response

        # Process request
        response = await call_next(request)

        # Add headers to response
        for key, value in headers.items():
            response.headers[key] = value

        return response


def create_rate_limit_dependency(
    tier: Optional[RateLimitTier] = None,
    qps: Optional[float] = None,
    burst: Optional[int] = None,
    cost: int = 1,
) -> Callable:
    """Create a rate limit dependency for specific endpoints.

    Usage:
        @router.post("/expensive-operation")
        async def expensive_op(
            _: None = Depends(create_rate_limit_dependency(qps=1.0, cost=10))
        ):
            ...
    """
    limiter = SlidingWindowRateLimiter(prefix="api:dep:rl")

    config = RateLimitConfig(
        qps=qps or 10.0,
        burst=burst or 20,
        max_requests=int((qps or 10.0) * 60),
    )

    async def rate_limit_check(request: Request) -> None:
        # Build identifier
        client_ip = request.client.host if request.client else "unknown"
        identifier = f"{client_ip}:{request.url.path}"

        allowed, current, remaining = await limiter.check(identifier, config, cost)

        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={
                    "X-RateLimit-Limit": str(config.max_requests),
                    "X-RateLimit-Remaining": str(remaining),
                    "Retry-After": str(config.window_seconds),
                },
            )

    return rate_limit_check
