"""Graceful degradation strategies for Vision Provider system.

This module provides graceful degradation including:
- Fallback chains
- Degraded mode responses
- Quality-based degradation
- Resource-aware degradation
- Recovery strategies
"""

import asyncio
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, TypeVar, Union

from .base import VisionDescription, VisionProvider


class DegradationLevel(Enum):
    """Level of degradation."""

    NORMAL = "normal"  # Full functionality
    REDUCED = "reduced"  # Some features disabled
    MINIMAL = "minimal"  # Basic functionality only
    OFFLINE = "offline"  # Cached/static responses only


class DegradationReason(Enum):
    """Reason for degradation."""

    NONE = "none"
    HIGH_LATENCY = "high_latency"
    HIGH_ERROR_RATE = "high_error_rate"
    RESOURCE_PRESSURE = "resource_pressure"
    DEPENDENCY_FAILURE = "dependency_failure"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"
    MANUAL = "manual"


class RecoveryStrategy(Enum):
    """Strategy for recovery from degradation."""

    IMMEDIATE = "immediate"  # Recover immediately when conditions improve
    GRADUAL = "gradual"  # Gradually increase service level
    MANUAL = "manual"  # Require manual intervention
    TIMED = "timed"  # Recover after fixed time


@dataclass
class DegradationThresholds:
    """Thresholds for triggering degradation."""

    max_latency_ms: float = 5000.0
    max_error_rate: float = 0.3
    max_cpu_percent: float = 80.0
    max_memory_percent: float = 80.0
    min_success_rate: float = 0.7
    consecutive_failures: int = 5


@dataclass
class DegradationState:
    """Current degradation state."""

    level: DegradationLevel = DegradationLevel.NORMAL
    reason: DegradationReason = DegradationReason.NONE
    started_at: Optional[datetime] = None
    updated_at: datetime = field(default_factory=datetime.now)
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level.value,
            "reason": self.reason.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "updated_at": self.updated_at.isoformat(),
            "message": self.message,
            "metadata": dict(self.metadata),
        }


@dataclass
class DegradationMetrics:
    """Metrics for degradation decisions."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    consecutive_failures: int = 0
    last_success_at: Optional[datetime] = None
    last_failure_at: Optional[datetime] = None

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return 1.0 - self.error_rate

    @property
    def average_latency_ms(self) -> float:
        """Calculate average latency."""
        if self.successful_requests == 0:
            return 0.0
        return self.total_latency_ms / self.successful_requests

    def record_success(self, latency_ms: float) -> None:
        """Record successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
        self.consecutive_failures = 0
        self.last_success_at = datetime.now()

    def record_failure(self) -> None:
        """Record failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.consecutive_failures += 1
        self.last_failure_at = datetime.now()

    def reset(self) -> None:
        """Reset metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
        self.consecutive_failures = 0


@dataclass
class FallbackResponse:
    """Fallback response for degraded mode."""

    summary: str
    details: List[str] = field(default_factory=list)
    confidence: float = 0.0
    is_fallback: bool = True
    fallback_reason: str = ""

    def to_vision_description(self) -> VisionDescription:
        """Convert to VisionDescription."""
        return VisionDescription(
            summary=self.summary,
            details=self.details,
            confidence=self.confidence,
        )


class FallbackProvider(ABC):
    """Abstract base class for fallback providers."""

    @abstractmethod
    def get_fallback(
        self,
        image_data: bytes,
        degradation_level: DegradationLevel,
    ) -> FallbackResponse:
        """Get fallback response.

        Args:
            image_data: Image data
            degradation_level: Current degradation level

        Returns:
            Fallback response
        """
        pass


class StaticFallbackProvider(FallbackProvider):
    """Fallback provider with static responses."""

    def __init__(
        self,
        default_summary: str = "Image analysis temporarily unavailable",
        default_details: Optional[List[str]] = None,
    ) -> None:
        """Initialize provider.

        Args:
            default_summary: Default summary
            default_details: Default details
        """
        self._default_summary = default_summary
        self._default_details = (
            default_details if default_details is not None else ["Please try again later"]
        )
        self._responses: Dict[DegradationLevel, FallbackResponse] = {}

    def set_response(
        self,
        level: DegradationLevel,
        response: FallbackResponse,
    ) -> None:
        """Set response for degradation level.

        Args:
            level: Degradation level
            response: Fallback response
        """
        self._responses[level] = response

    def get_fallback(
        self,
        image_data: bytes,
        degradation_level: DegradationLevel,
    ) -> FallbackResponse:
        """Get fallback response."""
        if degradation_level in self._responses:
            return self._responses[degradation_level]

        return FallbackResponse(
            summary=self._default_summary,
            details=self._default_details,
            fallback_reason=f"Service degraded to {degradation_level.value}",
        )


class CachedFallbackProvider(FallbackProvider):
    """Fallback provider using cached responses."""

    def __init__(self, max_cache_size: int = 1000) -> None:
        """Initialize provider.

        Args:
            max_cache_size: Maximum cache size
        """
        self._cache: Dict[str, FallbackResponse] = {}
        self._max_size = max_cache_size
        self._lock = threading.Lock()

    def cache_response(
        self,
        image_hash: str,
        response: FallbackResponse,
    ) -> None:
        """Cache a response.

        Args:
            image_hash: Image hash
            response: Response to cache
        """
        with self._lock:
            if len(self._cache) >= self._max_size:
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            self._cache[image_hash] = response

    def get_fallback(
        self,
        image_data: bytes,
        degradation_level: DegradationLevel,
    ) -> FallbackResponse:
        """Get fallback from cache."""
        import hashlib

        image_hash = hashlib.md5(image_data).hexdigest()

        with self._lock:
            if image_hash in self._cache:
                cached = self._cache[image_hash]
                return FallbackResponse(
                    summary=cached.summary,
                    details=cached.details,
                    confidence=cached.confidence * 0.8,  # Reduce confidence for cached
                    is_fallback=True,
                    fallback_reason="Cached response",
                )

        return FallbackResponse(
            summary="Image analysis unavailable",
            details=["No cached response available"],
            fallback_reason=f"Service degraded to {degradation_level.value}",
        )


class DegradationPolicy:
    """Policy for degradation decisions."""

    def __init__(
        self,
        thresholds: Optional[DegradationThresholds] = None,
    ) -> None:
        """Initialize policy.

        Args:
            thresholds: Degradation thresholds
        """
        self._thresholds = thresholds or DegradationThresholds()

    @property
    def thresholds(self) -> DegradationThresholds:
        """Return thresholds."""
        return self._thresholds

    def evaluate(
        self,
        metrics: DegradationMetrics,
        current_state: DegradationState,
    ) -> DegradationState:
        """Evaluate metrics and determine degradation state.

        Args:
            metrics: Current metrics
            current_state: Current state

        Returns:
            New degradation state
        """
        # Check consecutive failures
        if metrics.consecutive_failures >= self._thresholds.consecutive_failures:
            return DegradationState(
                level=DegradationLevel.OFFLINE,
                reason=DegradationReason.HIGH_ERROR_RATE,
                started_at=current_state.started_at or datetime.now(),
                message=f"Too many consecutive failures: {metrics.consecutive_failures}",
            )

        # Check error rate
        if metrics.error_rate > self._thresholds.max_error_rate:
            return DegradationState(
                level=DegradationLevel.MINIMAL,
                reason=DegradationReason.HIGH_ERROR_RATE,
                started_at=current_state.started_at or datetime.now(),
                message=f"High error rate: {metrics.error_rate:.2%}",
            )

        # Check latency
        if metrics.average_latency_ms > self._thresholds.max_latency_ms:
            return DegradationState(
                level=DegradationLevel.REDUCED,
                reason=DegradationReason.HIGH_LATENCY,
                started_at=current_state.started_at or datetime.now(),
                message=f"High latency: {metrics.average_latency_ms:.0f}ms",
            )

        # Check success rate
        if metrics.success_rate < self._thresholds.min_success_rate:
            return DegradationState(
                level=DegradationLevel.REDUCED,
                reason=DegradationReason.HIGH_ERROR_RATE,
                started_at=current_state.started_at or datetime.now(),
                message=f"Low success rate: {metrics.success_rate:.2%}",
            )

        # Normal operation
        return DegradationState(
            level=DegradationLevel.NORMAL,
            reason=DegradationReason.NONE,
        )


class RecoveryManager:
    """Manager for recovery from degradation."""

    def __init__(
        self,
        strategy: RecoveryStrategy = RecoveryStrategy.GRADUAL,
        recovery_window_seconds: float = 60.0,
        min_success_for_recovery: int = 5,
    ) -> None:
        """Initialize manager.

        Args:
            strategy: Recovery strategy
            recovery_window_seconds: Time window for recovery
            min_success_for_recovery: Minimum successes to recover
        """
        self._strategy = strategy
        self._recovery_window = recovery_window_seconds
        self._min_success = min_success_for_recovery
        self._recovery_attempts = 0
        self._recovery_successes = 0
        self._last_recovery_attempt: Optional[datetime] = None

    @property
    def strategy(self) -> RecoveryStrategy:
        """Return recovery strategy."""
        return self._strategy

    def can_attempt_recovery(self, current_state: DegradationState) -> bool:
        """Check if recovery attempt is allowed.

        Args:
            current_state: Current state

        Returns:
            True if recovery can be attempted
        """
        if current_state.level == DegradationLevel.NORMAL:
            return False

        if self._strategy == RecoveryStrategy.MANUAL:
            return False

        if self._strategy == RecoveryStrategy.IMMEDIATE:
            return True

        if self._strategy == RecoveryStrategy.TIMED:
            if current_state.started_at:
                elapsed = (datetime.now() - current_state.started_at).total_seconds()
                return elapsed >= self._recovery_window
            return True

        if self._strategy == RecoveryStrategy.GRADUAL:
            if self._last_recovery_attempt:
                elapsed = (datetime.now() - self._last_recovery_attempt).total_seconds()
                return elapsed >= 10.0  # Minimum 10 seconds between attempts
            return True

        return False

    def record_recovery_attempt(self, success: bool) -> None:
        """Record recovery attempt.

        Args:
            success: Whether attempt succeeded
        """
        self._recovery_attempts += 1
        self._last_recovery_attempt = datetime.now()

        if success:
            self._recovery_successes += 1
        else:
            self._recovery_successes = 0

    def should_recover(self) -> bool:
        """Check if system should recover to normal.

        Returns:
            True if should recover
        """
        if self._strategy == RecoveryStrategy.IMMEDIATE:
            return self._recovery_successes >= 1

        if self._strategy == RecoveryStrategy.GRADUAL:
            return self._recovery_successes >= self._min_success

        return False

    def reset(self) -> None:
        """Reset recovery state."""
        self._recovery_attempts = 0
        self._recovery_successes = 0
        self._last_recovery_attempt = None


@dataclass
class DegradationConfig:
    """Configuration for degradation manager."""

    enabled: bool = True
    thresholds: DegradationThresholds = field(default_factory=DegradationThresholds)
    recovery_strategy: RecoveryStrategy = RecoveryStrategy.GRADUAL
    recovery_window_seconds: float = 60.0
    min_success_for_recovery: int = 5
    metrics_window_seconds: float = 300.0  # 5 minutes
    auto_recovery: bool = True


class DegradationManager:
    """Manager for graceful degradation."""

    def __init__(
        self,
        config: Optional[DegradationConfig] = None,
        fallback_provider: Optional[FallbackProvider] = None,
    ) -> None:
        """Initialize manager.

        Args:
            config: Degradation configuration
            fallback_provider: Fallback provider
        """
        self._config = config or DegradationConfig()
        self._fallback = fallback_provider or StaticFallbackProvider()
        self._policy = DegradationPolicy(self._config.thresholds)
        self._recovery = RecoveryManager(
            strategy=self._config.recovery_strategy,
            recovery_window_seconds=self._config.recovery_window_seconds,
            min_success_for_recovery=self._config.min_success_for_recovery,
        )
        self._state = DegradationState()
        self._metrics = DegradationMetrics()
        self._lock = threading.Lock()
        self._listeners: List[Callable[[DegradationState], None]] = []

    @property
    def config(self) -> DegradationConfig:
        """Return configuration."""
        return self._config

    @property
    def state(self) -> DegradationState:
        """Return current state."""
        return self._state

    @property
    def metrics(self) -> DegradationMetrics:
        """Return current metrics."""
        return self._metrics

    def add_listener(
        self,
        listener: Callable[[DegradationState], None],
    ) -> None:
        """Add state change listener.

        Args:
            listener: Listener function
        """
        self._listeners.append(listener)

    def record_success(self, latency_ms: float) -> None:
        """Record successful request.

        Args:
            latency_ms: Request latency
        """
        with self._lock:
            self._metrics.record_success(latency_ms)

            if self._config.auto_recovery:
                if self._state.level != DegradationLevel.NORMAL:
                    self._recovery.record_recovery_attempt(True)

                    if self._recovery.should_recover():
                        self._update_state(
                            DegradationState(
                                level=DegradationLevel.NORMAL,
                                reason=DegradationReason.NONE,
                            )
                        )
                        self._recovery.reset()

    def record_failure(self) -> None:
        """Record failed request."""
        with self._lock:
            self._metrics.record_failure()

            if self._config.enabled:
                new_state = self._policy.evaluate(self._metrics, self._state)
                if new_state.level != self._state.level:
                    self._update_state(new_state)

            if self._state.level != DegradationLevel.NORMAL:
                self._recovery.record_recovery_attempt(False)

    def force_degradation(
        self,
        level: DegradationLevel,
        reason: str = "",
    ) -> None:
        """Force degradation to specified level.

        Args:
            level: Degradation level
            reason: Reason for degradation
        """
        with self._lock:
            self._update_state(
                DegradationState(
                    level=level,
                    reason=DegradationReason.MANUAL,
                    started_at=datetime.now(),
                    message=reason,
                )
            )

    def force_recovery(self) -> None:
        """Force recovery to normal state."""
        with self._lock:
            self._update_state(
                DegradationState(
                    level=DegradationLevel.NORMAL,
                    reason=DegradationReason.NONE,
                )
            )
            self._recovery.reset()
            self._metrics.reset()

    def get_fallback(self, image_data: bytes) -> FallbackResponse:
        """Get fallback response.

        Args:
            image_data: Image data

        Returns:
            Fallback response
        """
        return self._fallback.get_fallback(image_data, self._state.level)

    def should_use_fallback(self) -> bool:
        """Check if fallback should be used.

        Returns:
            True if should use fallback
        """
        return self._state.level in [
            DegradationLevel.MINIMAL,
            DegradationLevel.OFFLINE,
        ]

    def can_attempt_request(self) -> bool:
        """Check if request can be attempted.

        Returns:
            True if request allowed
        """
        if not self._config.enabled:
            return True

        if self._state.level == DegradationLevel.OFFLINE:
            return self._recovery.can_attempt_recovery(self._state)

        return True

    def _update_state(self, new_state: DegradationState) -> None:
        """Update degradation state.

        Args:
            new_state: New state
        """
        old_level = self._state.level
        self._state = new_state

        if old_level != new_state.level:
            for listener in self._listeners:
                try:
                    listener(new_state)
                except Exception:
                    pass


class GracefulDegradationVisionProvider(VisionProvider):
    """Vision provider with graceful degradation."""

    def __init__(
        self,
        provider: VisionProvider,
        degradation_manager: DegradationManager,
    ) -> None:
        """Initialize provider.

        Args:
            provider: Underlying vision provider
            degradation_manager: Degradation manager
        """
        self._provider = provider
        self._degradation = degradation_manager

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return f"graceful_{self._provider.provider_name}"

    @property
    def degradation_manager(self) -> DegradationManager:
        """Return degradation manager."""
        return self._degradation

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image with graceful degradation.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        # Check if we should use fallback
        if self._degradation.should_use_fallback():
            if not self._degradation.can_attempt_request():
                fallback = self._degradation.get_fallback(image_data)
                return fallback.to_vision_description()

        # Attempt request
        start_time = time.time()

        try:
            result = await self._provider.analyze_image(image_data, include_description)

            latency_ms = (time.time() - start_time) * 1000
            self._degradation.record_success(latency_ms)

            return result

        except Exception as e:
            self._degradation.record_failure()

            # Return fallback if in degraded mode
            if self._degradation.should_use_fallback():
                fallback = self._degradation.get_fallback(image_data)
                return fallback.to_vision_description()

            raise


class FallbackChain:
    """Chain of fallback providers."""

    def __init__(self) -> None:
        """Initialize chain."""
        self._providers: List[VisionProvider] = []
        self._fallback: Optional[FallbackProvider] = None

    def add_provider(self, provider: VisionProvider) -> "FallbackChain":
        """Add provider to chain.

        Args:
            provider: Vision provider

        Returns:
            Self for chaining
        """
        self._providers.append(provider)
        return self

    def set_fallback(self, fallback: FallbackProvider) -> "FallbackChain":
        """Set final fallback provider.

        Args:
            fallback: Fallback provider

        Returns:
            Self for chaining
        """
        self._fallback = fallback
        return self

    async def analyze(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image using fallback chain.

        Args:
            image_data: Image data
            include_description: Whether to include description

        Returns:
            Vision analysis result
        """
        last_error: Optional[Exception] = None

        for provider in self._providers:
            try:
                return await provider.analyze_image(image_data, include_description)
            except Exception as e:
                last_error = e
                continue

        # Use fallback if all providers failed
        if self._fallback:
            response = self._fallback.get_fallback(
                image_data,
                DegradationLevel.OFFLINE,
            )
            return response.to_vision_description()

        if last_error:
            raise last_error

        raise RuntimeError("No providers in fallback chain")


class FallbackChainVisionProvider(VisionProvider):
    """Vision provider using fallback chain."""

    def __init__(self, chain: FallbackChain) -> None:
        """Initialize provider.

        Args:
            chain: Fallback chain
        """
        self._chain = chain

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "fallback_chain"

    @property
    def chain(self) -> FallbackChain:
        """Return fallback chain."""
        return self._chain

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image using chain.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        return await self._chain.analyze(image_data, include_description)


def create_graceful_provider(
    provider: VisionProvider,
    config: Optional[DegradationConfig] = None,
    fallback_provider: Optional[FallbackProvider] = None,
) -> GracefulDegradationVisionProvider:
    """Create a graceful degradation vision provider.

    Args:
        provider: Underlying vision provider
        config: Degradation configuration
        fallback_provider: Optional fallback provider

    Returns:
        GracefulDegradationVisionProvider instance
    """
    manager = DegradationManager(
        config=config,
        fallback_provider=fallback_provider,
    )

    return GracefulDegradationVisionProvider(
        provider=provider,
        degradation_manager=manager,
    )


def create_fallback_chain(
    providers: List[VisionProvider],
    fallback_message: str = "Image analysis unavailable",
) -> FallbackChainVisionProvider:
    """Create a fallback chain vision provider.

    Args:
        providers: List of providers in priority order
        fallback_message: Final fallback message

    Returns:
        FallbackChainVisionProvider instance
    """
    chain = FallbackChain()

    for provider in providers:
        chain.add_provider(provider)

    chain.set_fallback(
        StaticFallbackProvider(
            default_summary=fallback_message,
        )
    )

    return FallbackChainVisionProvider(chain)
