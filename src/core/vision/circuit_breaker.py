"""Circuit-breaker vision decorators (+ back-compat re-exports).

Phase 0 slice A2a moved the generic breaker core to
`src.core.resilience.advanced_circuit_breaker` so that src/core/dedupcad_vision.py
no longer depends on src/core/vision/. The generic names are re-exported here
purely for backward compatibility with existing importers (vision/__init__.py,
tests/unit/test_vision_phase7.py).

Only the two symbols below genuinely belong to vision: they decorate a
VisionProvider and therefore import vision's .base. They are part of the decorator
zoo pruned in Phase 0 slice 2.
"""

from .base import VisionDescription, VisionProvider
from src.core.resilience.advanced_circuit_breaker import (  # noqa: F401  (re-export)
    CallRecord,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerError,
    CircuitBreakerStats,
    CircuitState,
    FailureRecord,
    FailureType,
    RecoveryStrategy,
    SlidingWindow,
    get_circuit_breaker,
)

__all__ = [
    "CallRecord",
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitBreakerError",
    "CircuitBreakerStats",
    "CircuitBreakerVisionProvider",
    "CircuitState",
    "FailureRecord",
    "FailureType",
    "RecoveryStrategy",
    "SlidingWindow",
    "create_circuit_breaker_provider",
    "get_circuit_breaker",
]


class CircuitBreakerVisionProvider(VisionProvider):
    """Vision provider with circuit breaker protection."""

    def __init__(
        self,
        provider: VisionProvider,
        circuit_breaker: CircuitBreaker,
        fallback: Optional[Callable[[], VisionDescription]] = None,
    ) -> None:
        """Initialize circuit breaker provider.

        Args:
            provider: Underlying vision provider
            circuit_breaker: Circuit breaker
            fallback: Optional fallback function
        """
        self._provider = provider
        self._circuit_breaker = circuit_breaker
        self._fallback = fallback

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"circuit_breaker_{self._provider.provider_name}"

    @property
    def circuit_breaker(self) -> CircuitBreaker:
        """Return circuit breaker."""
        return self._circuit_breaker

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with circuit breaker protection.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        if not self._circuit_breaker.can_execute():
            self._circuit_breaker.record_rejection()

            if self._fallback:
                return self._fallback()

            raise CircuitBreakerError(
                self._circuit_breaker.name,
                self._circuit_breaker.state,
            )

        start_time = time.time()

        try:
            result = await self._provider.analyze_image(image_data, include_description)
            duration_ms = (time.time() - start_time) * 1000
            self._circuit_breaker.record_success(duration_ms)
            return result

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self._circuit_breaker.record_failure(
                duration_ms,
                FailureType.EXCEPTION,
                str(e),
            )
            raise


# Global circuit breaker registry
_circuit_breakers: Dict[str, CircuitBreaker] = {}


def create_circuit_breaker_provider(
    provider: VisionProvider,
    name: Optional[str] = None,
    config: Optional[CircuitBreakerConfig] = None,
    fallback: Optional[Callable[[], VisionDescription]] = None,
) -> CircuitBreakerVisionProvider:
    """Create a circuit breaker provider.

    Args:
        provider: Underlying vision provider
        name: Circuit breaker name
        config: Circuit breaker configuration
        fallback: Optional fallback function

    Returns:
        CircuitBreakerVisionProvider instance
    """
    cb_name = name or f"cb_{provider.provider_name}"
    circuit_breaker = get_circuit_breaker(cb_name, config)

    return CircuitBreakerVisionProvider(
        provider=provider,
        circuit_breaker=circuit_breaker,
        fallback=fallback,
    )
