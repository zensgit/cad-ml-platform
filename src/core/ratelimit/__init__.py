"""Rate Limiting Module.

Provides comprehensive rate limiting:
- Multiple algorithms (Token Bucket, Leaky Bucket, Sliding Window, Fixed Window)
- Distributed rate limiting with Redis
- Middleware and decorators
- Multi-tier and adaptive limiting
"""

from src.core.ratelimit.algorithms import (
    RateLimitResult,
    RateLimiter,
    TokenBucketLimiter,
    LeakyBucketLimiter,
    SlidingWindowLogLimiter,
    SlidingWindowCounterLimiter,
    FixedWindowCounterLimiter,
)
from src.core.ratelimit.distributed import (
    InMemoryRedis,
    DistributedTokenBucketLimiter,
    DistributedSlidingWindowLimiter,
    RateLimitConfig,
    create_distributed_limiter,
)
from src.core.ratelimit.middleware import (
    KeyStrategy,
    KeyExtractor,
    IPKeyExtractor,
    UserKeyExtractor,
    APIKeyExtractor,
    EndpointKeyExtractor,
    CompositeKeyExtractor,
    RateLimitTier,
    MultiTierRateLimiter,
    rate_limit,
    RateLimitExceeded,
    RateLimitMiddlewareConfig,
    RateLimitMiddleware,
    AdaptiveRateLimiter,
    QuotaConfig,
    QuotaManager,
)

__all__ = [
    # Algorithms
    "RateLimitResult",
    "RateLimiter",
    "TokenBucketLimiter",
    "LeakyBucketLimiter",
    "SlidingWindowLogLimiter",
    "SlidingWindowCounterLimiter",
    "FixedWindowCounterLimiter",
    # Distributed
    "InMemoryRedis",
    "DistributedTokenBucketLimiter",
    "DistributedSlidingWindowLimiter",
    "RateLimitConfig",
    "create_distributed_limiter",
    # Middleware
    "KeyStrategy",
    "KeyExtractor",
    "IPKeyExtractor",
    "UserKeyExtractor",
    "APIKeyExtractor",
    "EndpointKeyExtractor",
    "CompositeKeyExtractor",
    "RateLimitTier",
    "MultiTierRateLimiter",
    "rate_limit",
    "RateLimitExceeded",
    "RateLimitMiddlewareConfig",
    "RateLimitMiddleware",
    "AdaptiveRateLimiter",
    "QuotaConfig",
    "QuotaManager",
]
