"""Provider quotas and throttling for Vision Provider system.

This module provides quota management and request throttling capabilities
including:
- Per-provider request quotas (daily, hourly, per-minute)
- Token/cost quotas
- Adaptive throttling based on provider response times
- Burst limiting and smoothing
- Quota reset scheduling
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .base import VisionDescription, VisionProvider


class QuotaPeriod(Enum):
    """Time periods for quota tracking."""

    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"
    PER_WEEK = "per_week"
    PER_MONTH = "per_month"


class ThrottleStrategy(Enum):
    """Throttling strategies."""

    FIXED_DELAY = "fixed_delay"  # Fixed delay between requests
    ADAPTIVE = "adaptive"  # Adjust based on response times
    TOKEN_BUCKET = "token_bucket"  # Token bucket algorithm
    LEAKY_BUCKET = "leaky_bucket"  # Leaky bucket algorithm
    SLIDING_WINDOW = "sliding_window"  # Sliding window rate limit


class QuotaAction(Enum):
    """Actions to take when quota is exceeded."""

    BLOCK = "block"  # Block the request
    QUEUE = "queue"  # Queue for later execution
    DEGRADE = "degrade"  # Degrade to lower quality
    FALLBACK = "fallback"  # Use fallback provider
    WARN = "warn"  # Allow but warn


@dataclass
class QuotaLimit:
    """Configuration for a quota limit."""

    period: QuotaPeriod
    max_requests: int
    max_tokens: Optional[int] = None
    max_cost: Optional[float] = None
    action_on_exceed: QuotaAction = QuotaAction.BLOCK
    warning_threshold: float = 0.8  # Warn at 80% usage


@dataclass
class QuotaUsage:
    """Current usage for a quota period."""

    period: QuotaPeriod
    requests_used: int = 0
    tokens_used: int = 0
    cost_used: float = 0.0
    period_start: datetime = field(default_factory=datetime.now)
    last_request: Optional[datetime] = None

    @property
    def period_duration(self) -> timedelta:
        """Get the duration for this period."""
        durations = {
            QuotaPeriod.PER_MINUTE: timedelta(minutes=1),
            QuotaPeriod.PER_HOUR: timedelta(hours=1),
            QuotaPeriod.PER_DAY: timedelta(days=1),
            QuotaPeriod.PER_WEEK: timedelta(weeks=1),
            QuotaPeriod.PER_MONTH: timedelta(days=30),
        }
        return durations[self.period]

    @property
    def period_end(self) -> datetime:
        """Get when this period ends."""
        return self.period_start + self.period_duration

    @property
    def is_expired(self) -> bool:
        """Check if this period has expired."""
        return datetime.now() >= self.period_end

    def reset(self) -> None:
        """Reset usage for new period."""
        self.requests_used = 0
        self.tokens_used = 0
        self.cost_used = 0.0
        self.period_start = datetime.now()


@dataclass
class ThrottleConfig:
    """Configuration for request throttling."""

    strategy: ThrottleStrategy = ThrottleStrategy.TOKEN_BUCKET
    requests_per_second: float = 10.0
    burst_size: int = 20
    min_delay_ms: float = 10.0
    max_delay_ms: float = 5000.0
    adaptive_target_latency_ms: float = 500.0
    adaptive_adjustment_factor: float = 0.1


@dataclass
class ThrottleState:
    """Current state for throttling."""

    tokens: float = 0.0
    last_refill: float = field(default_factory=time.time)
    current_delay_ms: float = 100.0
    recent_latencies: List[float] = field(default_factory=list)
    window_requests: List[float] = field(default_factory=list)


@dataclass
class QuotaCheckResult:
    """Result of a quota check."""

    allowed: bool
    action: QuotaAction
    reason: Optional[str] = None
    wait_time_seconds: Optional[float] = None
    usage_percentage: float = 0.0
    warnings: List[str] = field(default_factory=list)


class QuotaManager:
    """Manages quotas for providers."""

    def __init__(self) -> None:
        """Initialize quota manager."""
        self._limits: Dict[str, List[QuotaLimit]] = {}
        self._usage: Dict[str, Dict[QuotaPeriod, QuotaUsage]] = {}
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    def set_limits(self, provider_id: str, limits: List[QuotaLimit]) -> None:
        """Set quota limits for a provider."""
        self._limits[provider_id] = limits
        self._usage[provider_id] = {}

        for limit in limits:
            self._usage[provider_id][limit.period] = QuotaUsage(period=limit.period)

    def get_limits(self, provider_id: str) -> List[QuotaLimit]:
        """Get quota limits for a provider."""
        return self._limits.get(provider_id, [])

    async def check_quota(
        self,
        provider_id: str,
        estimated_tokens: int = 0,
        estimated_cost: float = 0.0,
    ) -> QuotaCheckResult:
        """Check if a request is within quota."""
        async with self._get_lock():
            limits = self._limits.get(provider_id, [])
            usage_dict = self._usage.get(provider_id, {})

            if not limits:
                return QuotaCheckResult(
                    allowed=True,
                    action=QuotaAction.WARN,
                    usage_percentage=0.0,
                )

            warnings: List[str] = []
            max_usage_pct = 0.0

            for limit in limits:
                usage = usage_dict.get(limit.period)
                if not usage:
                    continue

                # Reset if period expired
                if usage.is_expired:
                    usage.reset()

                # Calculate usage percentages
                request_pct = (usage.requests_used + 1) / limit.max_requests
                token_pct = 0.0
                cost_pct = 0.0

                if limit.max_tokens and estimated_tokens:
                    token_pct = (usage.tokens_used + estimated_tokens) / limit.max_tokens

                if limit.max_cost and estimated_cost:
                    cost_pct = (usage.cost_used + estimated_cost) / limit.max_cost

                usage_pct = max(request_pct, token_pct, cost_pct)
                max_usage_pct = max(max_usage_pct, usage_pct)

                # Check for warnings
                if usage_pct >= limit.warning_threshold:
                    warnings.append(f"{limit.period.value}: {usage_pct:.1%} of quota used")

                # Check if exceeded
                if usage_pct >= 1.0:
                    wait_time = None
                    if limit.action_on_exceed == QuotaAction.QUEUE:
                        wait_time = (usage.period_end - datetime.now()).total_seconds()
                        wait_time = max(0, wait_time)

                    return QuotaCheckResult(
                        allowed=False,
                        action=limit.action_on_exceed,
                        reason=f"Quota exceeded for {limit.period.value}",
                        wait_time_seconds=wait_time,
                        usage_percentage=usage_pct,
                        warnings=warnings,
                    )

            return QuotaCheckResult(
                allowed=True,
                action=QuotaAction.WARN if warnings else QuotaAction.BLOCK,
                usage_percentage=max_usage_pct,
                warnings=warnings,
            )

    async def record_usage(
        self,
        provider_id: str,
        tokens_used: int = 0,
        cost_used: float = 0.0,
    ) -> None:
        """Record usage after a request."""
        async with self._get_lock():
            usage_dict = self._usage.get(provider_id, {})

            for usage in usage_dict.values():
                if usage.is_expired:
                    usage.reset()

                usage.requests_used += 1
                usage.tokens_used += tokens_used
                usage.cost_used += cost_used
                usage.last_request = datetime.now()

    def get_usage(self, provider_id: str) -> Dict[QuotaPeriod, QuotaUsage]:
        """Get current usage for a provider."""
        return self._usage.get(provider_id, {})

    def reset_usage(self, provider_id: str, period: Optional[QuotaPeriod] = None) -> None:
        """Reset usage for a provider."""
        usage_dict = self._usage.get(provider_id, {})

        if period:
            if period in usage_dict:
                usage_dict[period].reset()
        else:
            for usage in usage_dict.values():
                usage.reset()

    def get_remaining(self, provider_id: str) -> Dict[QuotaPeriod, Dict[str, Union[int, float]]]:
        """Get remaining quota for each period."""
        limits = self._limits.get(provider_id, [])
        usage_dict = self._usage.get(provider_id, {})
        remaining: Dict[QuotaPeriod, Dict[str, Union[int, float]]] = {}

        for limit in limits:
            usage = usage_dict.get(limit.period)
            if not usage:
                continue

            period_remaining: Dict[str, Union[int, float]] = {
                "requests": max(0, limit.max_requests - usage.requests_used),
                "seconds_until_reset": max(0, (usage.period_end - datetime.now()).total_seconds()),
            }

            if limit.max_tokens:
                period_remaining["tokens"] = max(0, limit.max_tokens - usage.tokens_used)

            if limit.max_cost:
                period_remaining["cost"] = max(0, limit.max_cost - usage.cost_used)

            remaining[limit.period] = period_remaining

        return remaining


class Throttler:
    """Implements request throttling."""

    def __init__(self, config: Optional[ThrottleConfig] = None) -> None:
        """Initialize throttler."""
        self._config = config or ThrottleConfig()
        self._state = ThrottleState(tokens=float(self._config.burst_size))
        self._lock: Optional[asyncio.Lock] = None

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @property
    def config(self) -> ThrottleConfig:
        """Get throttle configuration."""
        return self._config

    async def acquire(self) -> float:
        """Acquire permission to make a request.

        Returns the delay in seconds before the request should be made.
        """
        async with self._get_lock():
            if self._config.strategy == ThrottleStrategy.FIXED_DELAY:
                return self._fixed_delay()
            elif self._config.strategy == ThrottleStrategy.ADAPTIVE:
                return self._adaptive_delay()
            elif self._config.strategy == ThrottleStrategy.TOKEN_BUCKET:
                return await self._token_bucket_delay()
            elif self._config.strategy == ThrottleStrategy.LEAKY_BUCKET:
                return self._leaky_bucket_delay()
            elif self._config.strategy == ThrottleStrategy.SLIDING_WINDOW:
                return self._sliding_window_delay()
            else:
                return 0.0

    def _fixed_delay(self) -> float:
        """Calculate fixed delay."""
        return self._config.min_delay_ms / 1000.0

    def _adaptive_delay(self) -> float:
        """Calculate adaptive delay based on recent latencies."""
        if not self._state.recent_latencies:
            return self._config.min_delay_ms / 1000.0

        avg_latency = sum(self._state.recent_latencies) / len(self._state.recent_latencies)
        target = self._config.adaptive_target_latency_ms

        if avg_latency > target:
            # Increase delay
            adjustment = 1 + self._config.adaptive_adjustment_factor
            self._state.current_delay_ms = min(
                self._state.current_delay_ms * adjustment,
                self._config.max_delay_ms,
            )
        else:
            # Decrease delay
            adjustment = 1 - self._config.adaptive_adjustment_factor
            self._state.current_delay_ms = max(
                self._state.current_delay_ms * adjustment,
                self._config.min_delay_ms,
            )

        return self._state.current_delay_ms / 1000.0

    async def _token_bucket_delay(self) -> float:
        """Calculate delay using token bucket algorithm."""
        now = time.time()
        elapsed = now - self._state.last_refill

        # Refill tokens
        refill_amount = elapsed * self._config.requests_per_second
        self._state.tokens = min(
            self._state.tokens + refill_amount,
            float(self._config.burst_size),
        )
        self._state.last_refill = now

        if self._state.tokens >= 1.0:
            self._state.tokens -= 1.0
            return 0.0
        else:
            # Calculate wait time for next token
            wait_time = (1.0 - self._state.tokens) / self._config.requests_per_second
            return wait_time

    def _leaky_bucket_delay(self) -> float:
        """Calculate delay using leaky bucket algorithm."""
        # Leaky bucket: constant outflow rate
        return 1.0 / self._config.requests_per_second

    def _sliding_window_delay(self) -> float:
        """Calculate delay using sliding window."""
        now = time.time()
        window_start = now - 1.0  # 1 second window

        # Remove old requests
        self._state.window_requests = [t for t in self._state.window_requests if t > window_start]

        if len(self._state.window_requests) < self._config.requests_per_second:
            self._state.window_requests.append(now)
            return 0.0
        else:
            # Wait until oldest request exits window
            oldest = min(self._state.window_requests)
            wait_time = oldest + 1.0 - now
            return max(0.0, wait_time)

    def record_latency(self, latency_ms: float) -> None:
        """Record a request latency for adaptive throttling."""
        self._state.recent_latencies.append(latency_ms)
        # Keep only recent samples
        if len(self._state.recent_latencies) > 100:
            self._state.recent_latencies = self._state.recent_latencies[-50:]

    def reset(self) -> None:
        """Reset throttler state."""
        self._state = ThrottleState(tokens=float(self._config.burst_size))


@dataclass
class ProviderQuotaConfig:
    """Complete quota configuration for a provider."""

    provider_id: str
    limits: List[QuotaLimit] = field(default_factory=list)
    throttle_config: Optional[ThrottleConfig] = None
    fallback_provider_id: Optional[str] = None
    degrade_options: Dict[str, Any] = field(default_factory=dict)


class QuotaVisionProvider:
    """Vision provider wrapper with quota management and throttling."""

    def __init__(
        self,
        provider: VisionProvider,
        quota_manager: QuotaManager,
        throttler: Optional[Throttler] = None,
        provider_id: Optional[str] = None,
        on_quota_exceeded: Optional[Callable[[QuotaCheckResult], None]] = None,
        fallback_provider: Optional[VisionProvider] = None,
    ) -> None:
        """Initialize quota-managed provider."""
        self._provider = provider
        self._quota_manager = quota_manager
        self._throttler = throttler or Throttler()
        self._provider_id = provider_id or "default"
        self._on_quota_exceeded = on_quota_exceeded
        self._fallback_provider = fallback_provider

    @property
    def provider(self) -> VisionProvider:
        """Get underlying provider."""
        return self._provider

    @property
    def quota_manager(self) -> QuotaManager:
        """Get quota manager."""
        return self._quota_manager

    @property
    def throttler(self) -> Throttler:
        """Get throttler."""
        return self._throttler

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
        estimated_tokens: int = 1000,
        estimated_cost: float = 0.01,
    ) -> Tuple[Optional[VisionDescription], QuotaCheckResult]:
        """Analyze image with quota checks.

        Returns:
            Tuple of (result, quota_check_result)
        """
        # Check quota
        quota_result = await self._quota_manager.check_quota(
            self._provider_id,
            estimated_tokens=estimated_tokens,
            estimated_cost=estimated_cost,
        )

        if not quota_result.allowed:
            if self._on_quota_exceeded:
                self._on_quota_exceeded(quota_result)

            # Handle based on action
            if quota_result.action == QuotaAction.FALLBACK:
                if self._fallback_provider:
                    result = await self._fallback_provider.analyze_image(
                        image_data, include_description
                    )
                    return result, quota_result

            if quota_result.action == QuotaAction.QUEUE:
                # Wait and retry
                if quota_result.wait_time_seconds:
                    await asyncio.sleep(quota_result.wait_time_seconds)
                    return await self.analyze_image(
                        image_data,
                        include_description,
                        estimated_tokens,
                        estimated_cost,
                    )

            if quota_result.action == QuotaAction.BLOCK:
                return None, quota_result

        # Apply throttling
        delay = await self._throttler.acquire()
        if delay > 0:
            await asyncio.sleep(delay)

        # Make request
        start_time = time.time()
        result = await self._provider.analyze_image(image_data, include_description)
        latency_ms = (time.time() - start_time) * 1000

        # Record usage and latency
        await self._quota_manager.record_usage(
            self._provider_id,
            tokens_used=estimated_tokens,
            cost_used=estimated_cost,
        )
        self._throttler.record_latency(latency_ms)

        return result, quota_result

    def get_remaining_quota(self) -> Dict[QuotaPeriod, Dict[str, Union[int, float]]]:
        """Get remaining quota."""
        return self._quota_manager.get_remaining(self._provider_id)

    def get_usage(self) -> Dict[QuotaPeriod, QuotaUsage]:
        """Get current usage."""
        return self._quota_manager.get_usage(self._provider_id)


# Preset quota configurations
QUOTA_PRESETS: Dict[str, List[QuotaLimit]] = {
    "free_tier": [
        QuotaLimit(
            period=QuotaPeriod.PER_DAY,
            max_requests=100,
            max_tokens=100000,
            action_on_exceed=QuotaAction.BLOCK,
        ),
        QuotaLimit(
            period=QuotaPeriod.PER_MINUTE,
            max_requests=10,
            action_on_exceed=QuotaAction.QUEUE,
        ),
    ],
    "standard": [
        QuotaLimit(
            period=QuotaPeriod.PER_DAY,
            max_requests=1000,
            max_tokens=1000000,
            max_cost=10.0,
            action_on_exceed=QuotaAction.WARN,
        ),
        QuotaLimit(
            period=QuotaPeriod.PER_MINUTE,
            max_requests=60,
            action_on_exceed=QuotaAction.QUEUE,
        ),
    ],
    "premium": [
        QuotaLimit(
            period=QuotaPeriod.PER_DAY,
            max_requests=10000,
            max_tokens=10000000,
            max_cost=100.0,
            action_on_exceed=QuotaAction.WARN,
        ),
        QuotaLimit(
            period=QuotaPeriod.PER_MINUTE,
            max_requests=300,
            action_on_exceed=QuotaAction.QUEUE,
        ),
    ],
    "enterprise": [
        QuotaLimit(
            period=QuotaPeriod.PER_DAY,
            max_requests=100000,
            max_tokens=100000000,
            max_cost=1000.0,
            action_on_exceed=QuotaAction.WARN,
            warning_threshold=0.9,
        ),
        QuotaLimit(
            period=QuotaPeriod.PER_MINUTE,
            max_requests=1000,
            action_on_exceed=QuotaAction.QUEUE,
        ),
    ],
    "burst_limited": [
        QuotaLimit(
            period=QuotaPeriod.PER_MINUTE,
            max_requests=30,
            action_on_exceed=QuotaAction.BLOCK,
        ),
        QuotaLimit(
            period=QuotaPeriod.PER_HOUR,
            max_requests=500,
            action_on_exceed=QuotaAction.QUEUE,
        ),
    ],
    "cost_capped": [
        QuotaLimit(
            period=QuotaPeriod.PER_DAY,
            max_requests=10000,
            max_cost=50.0,
            action_on_exceed=QuotaAction.BLOCK,
        ),
        QuotaLimit(
            period=QuotaPeriod.PER_MONTH,
            max_requests=100000,
            max_cost=500.0,
            action_on_exceed=QuotaAction.BLOCK,
        ),
    ],
}

# Preset throttle configurations
THROTTLE_PRESETS: Dict[str, ThrottleConfig] = {
    "relaxed": ThrottleConfig(
        strategy=ThrottleStrategy.TOKEN_BUCKET,
        requests_per_second=20.0,
        burst_size=50,
    ),
    "moderate": ThrottleConfig(
        strategy=ThrottleStrategy.TOKEN_BUCKET,
        requests_per_second=10.0,
        burst_size=20,
    ),
    "strict": ThrottleConfig(
        strategy=ThrottleStrategy.FIXED_DELAY,
        requests_per_second=5.0,
        burst_size=5,
        min_delay_ms=200.0,
    ),
    "adaptive": ThrottleConfig(
        strategy=ThrottleStrategy.ADAPTIVE,
        requests_per_second=10.0,
        adaptive_target_latency_ms=500.0,
        adaptive_adjustment_factor=0.1,
    ),
    "sliding_window": ThrottleConfig(
        strategy=ThrottleStrategy.SLIDING_WINDOW,
        requests_per_second=15.0,
        burst_size=30,
    ),
}


def get_quota_preset(name: str) -> List[QuotaLimit]:
    """Get a preset quota configuration by name."""
    return QUOTA_PRESETS.get(name, QUOTA_PRESETS["standard"])


def get_throttle_preset(name: str) -> ThrottleConfig:
    """Get a preset throttle configuration by name."""
    return THROTTLE_PRESETS.get(name, THROTTLE_PRESETS["moderate"])


def create_quota_provider(
    provider: VisionProvider,
    provider_id: str,
    quota_preset: str = "standard",
    throttle_preset: str = "moderate",
    fallback_provider: Optional[VisionProvider] = None,
    on_quota_exceeded: Optional[Callable[[QuotaCheckResult], None]] = None,
) -> QuotaVisionProvider:
    """Create a quota-managed vision provider.

    Args:
        provider: The underlying vision provider
        provider_id: Unique identifier for this provider
        quota_preset: Name of quota preset to use
        throttle_preset: Name of throttle preset to use
        fallback_provider: Optional fallback provider when quota exceeded
        on_quota_exceeded: Callback when quota is exceeded

    Returns:
        QuotaVisionProvider instance
    """
    quota_manager = QuotaManager()
    quota_manager.set_limits(provider_id, get_quota_preset(quota_preset))

    throttler = Throttler(get_throttle_preset(throttle_preset))

    return QuotaVisionProvider(
        provider=provider,
        quota_manager=quota_manager,
        throttler=throttler,
        provider_id=provider_id,
        on_quota_exceeded=on_quota_exceeded,
        fallback_provider=fallback_provider,
    )


__all__ = [
    # Enums
    "QuotaPeriod",
    "ThrottleStrategy",
    "QuotaAction",
    # Data classes
    "QuotaLimit",
    "QuotaUsage",
    "ThrottleConfig",
    "ThrottleState",
    "QuotaCheckResult",
    "ProviderQuotaConfig",
    # Core classes
    "QuotaManager",
    "Throttler",
    "QuotaVisionProvider",
    # Presets
    "QUOTA_PRESETS",
    "THROTTLE_PRESETS",
    # Factory functions
    "get_quota_preset",
    "get_throttle_preset",
    "create_quota_provider",
]
