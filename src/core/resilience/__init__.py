"""
Resilience Layer - 统一的弹性和容错抽象层
提供 Circuit Breaker、Rate Limiter、Retry Policy 和 Bulkhead 模式
"""

from .bulkhead import Bulkhead, BulkheadError, ThreadPoolBulkhead
from .circuit_breaker import CircuitBreaker, CircuitBreakerError, CircuitState
from .metrics import ResilienceMetrics
from .rate_limiter import RateLimiter, RateLimitError, TokenBucket
from .resilience_manager import ResilienceManager
from .retry_policy import ExponentialBackoff, RetryError, RetryPolicy

__version__ = "1.0.0"

__all__ = [
    # Circuit Breaker
    "CircuitBreaker",
    "CircuitBreakerError",
    "CircuitState",
    # Rate Limiter
    "RateLimiter",
    "RateLimitError",
    "TokenBucket",
    # Retry Policy
    "RetryPolicy",
    "RetryError",
    "ExponentialBackoff",
    # Bulkhead
    "Bulkhead",
    "BulkheadError",
    "ThreadPoolBulkhead",
    # Manager
    "ResilienceManager",
    # Metrics
    "ResilienceMetrics",
]

# 默认配置
DEFAULT_CONFIG = {
    "circuit_breaker": {
        "failure_threshold": 5,
        "recovery_timeout": 60,
        "expected_exception": Exception,
        "half_open_max_calls": 3,
    },
    "rate_limiter": {
        "rate": 100,  # requests per second
        "burst": 150,
        "algorithm": "token_bucket",
    },
    "retry_policy": {
        "max_attempts": 3,
        "base_delay": 1.0,
        "max_delay": 30.0,
        "exponential_base": 2,
        "jitter": True,
    },
    "bulkhead": {
        "max_concurrent_calls": 10,
        "max_wait_duration": 0,  # 0 means no waiting
        "type": "threadpool",
    },
}
