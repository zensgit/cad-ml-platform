"""Enhanced Resilience Module.

Provides advanced resilience patterns:
- Enhanced circuit breaker
- Bulkhead isolation
- Timeout policies
- Fallback strategies
"""

from src.core.resilience_enhanced.circuit_breaker import (
    CircuitState,
    CircuitBreakerConfig,
    CircuitBreakerMetrics,
    FailureCounter,
    CountBasedFailureCounter,
    TimeBasedFailureCounter,
    CircuitBreakerError,
    CircuitBreaker,
    CircuitBreakerRegistry,
    get_circuit_breaker_registry,
)
from src.core.resilience_enhanced.bulkhead import (
    BulkheadError,
    BulkheadConfig,
    BulkheadMetrics,
    Bulkhead,
    SemaphoreBulkhead,
    ThreadPoolBulkhead,
    AdaptiveBulkhead,
    BulkheadRegistry,
)
from src.core.resilience_enhanced.timeout import (
    TimeoutError,
    TimeoutConfig,
    TimeoutMetrics,
    TimeoutPolicy,
    SimpleTimeout,
    AdaptiveTimeout,
    HedgedRequest,
)
from src.core.resilience_enhanced.fallback import (
    FallbackMetrics,
    FallbackStrategy,
    StaticFallback,
    FunctionFallback,
    CachedFallback,
    FallbackChain,
    GracefulDegradation,
    FallbackDecorator,
    with_fallback,
)

__all__ = [
    # Circuit Breaker
    "CircuitState",
    "CircuitBreakerConfig",
    "CircuitBreakerMetrics",
    "FailureCounter",
    "CountBasedFailureCounter",
    "TimeBasedFailureCounter",
    "CircuitBreakerError",
    "CircuitBreaker",
    "CircuitBreakerRegistry",
    "get_circuit_breaker_registry",
    # Bulkhead
    "BulkheadError",
    "BulkheadConfig",
    "BulkheadMetrics",
    "Bulkhead",
    "SemaphoreBulkhead",
    "ThreadPoolBulkhead",
    "AdaptiveBulkhead",
    "BulkheadRegistry",
    # Timeout
    "TimeoutError",
    "TimeoutConfig",
    "TimeoutMetrics",
    "TimeoutPolicy",
    "SimpleTimeout",
    "AdaptiveTimeout",
    "HedgedRequest",
    # Fallback
    "FallbackMetrics",
    "FallbackStrategy",
    "StaticFallback",
    "FunctionFallback",
    "CachedFallback",
    "FallbackChain",
    "GracefulDegradation",
    "FallbackDecorator",
    "with_fallback",
]
