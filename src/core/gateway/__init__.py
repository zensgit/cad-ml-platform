"""API Gateway Integration for CAD ML Platform.

Provides:
- Rate limiting with multiple strategies
- Request/response transformation
- API key management
- Circuit breaker patterns
- Request validation
"""

from src.core.gateway.rate_limiter import (
    RateLimiter,
    RateLimitConfig,
    RateLimitStrategy,
    SlidingWindowLimiter,
    TokenBucketLimiter,
    rate_limit,
)
from src.core.gateway.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    circuit_breaker,
)
from src.core.gateway.api_key import (
    APIKey,
    APIKeyManager,
    APIKeyScope,
    validate_api_key,
)

__all__ = [
    # Rate Limiting
    "RateLimiter",
    "RateLimitConfig",
    "RateLimitStrategy",
    "SlidingWindowLimiter",
    "TokenBucketLimiter",
    "rate_limit",
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "circuit_breaker",
    # API Key
    "APIKey",
    "APIKeyManager",
    "APIKeyScope",
    "validate_api_key",
]
