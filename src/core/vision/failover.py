"""Multi-region failover support for vision analysis.

Provides:
- Geographic failover
- Provider prioritization
- Health-based routing
- Automatic recovery
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import VisionDescription, VisionProvider, VisionProviderError

logger = logging.getLogger(__name__)


class ProviderHealth(Enum):
    """Health status of a provider."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class FailoverStrategy(Enum):
    """Strategy for failover selection."""

    PRIORITY = "priority"  # Use priority order
    ROUND_ROBIN = "round_robin"  # Rotate through healthy providers
    LATENCY = "latency"  # Choose lowest latency
    WEIGHTED = "weighted"  # Use weighted random selection


@dataclass
class ProviderEndpoint:
    """A provider endpoint with health tracking."""

    provider: VisionProvider
    priority: int = 0
    weight: float = 1.0
    region: str = "default"
    health: ProviderHealth = ProviderHealth.UNKNOWN
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    consecutive_failures: int = 0
    avg_latency_ms: float = 0.0
    _latency_samples: List[float] = field(default_factory=list)

    def record_success(self, latency_ms: float) -> None:
        """Record a successful request."""
        self.last_success = datetime.now()
        self.consecutive_failures = 0
        self.health = ProviderHealth.HEALTHY

        # Update latency average (keep last 10 samples)
        self._latency_samples.append(latency_ms)
        if len(self._latency_samples) > 10:
            self._latency_samples.pop(0)
        self.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def record_failure(self) -> None:
        """Record a failed request."""
        self.last_failure = datetime.now()
        self.consecutive_failures += 1

        if self.consecutive_failures >= 5:
            self.health = ProviderHealth.UNHEALTHY
        elif self.consecutive_failures >= 2:
            self.health = ProviderHealth.DEGRADED

    def is_available(self) -> bool:
        """Check if provider is available for requests."""
        return self.health != ProviderHealth.UNHEALTHY


@dataclass
class FailoverConfig:
    """Configuration for failover behavior."""

    max_retries: int = 3
    retry_delay: float = 1.0
    health_check_interval: float = 60.0  # Seconds
    recovery_threshold: int = 2  # Successes needed to mark healthy
    failover_timeout: float = 30.0  # Max time for failover attempts
    strategy: FailoverStrategy = FailoverStrategy.PRIORITY


@dataclass
class FailoverResult:
    """Result of a failover operation."""

    success: bool
    result: Optional[VisionDescription]
    provider_used: str
    attempts: List[tuple[str, Optional[str]]]  # (provider, error)
    total_time_ms: float


class FailoverManager:
    """
    Manages failover across multiple providers.

    Features:
    - Multiple failover strategies
    - Health-based routing
    - Automatic recovery
    - Regional prioritization
    """

    def __init__(
        self,
        endpoints: List[ProviderEndpoint],
        config: Optional[FailoverConfig] = None,
    ):
        """
        Initialize failover manager.

        Args:
            endpoints: List of provider endpoints
            config: Failover configuration
        """
        self._endpoints = endpoints
        self._config = config or FailoverConfig()
        self._round_robin_index = 0
        self._lock = asyncio.Lock()

    def add_endpoint(self, endpoint: ProviderEndpoint) -> None:
        """Add a new endpoint."""
        self._endpoints.append(endpoint)

    def remove_endpoint(self, provider_name: str) -> bool:
        """Remove an endpoint by provider name."""
        for i, ep in enumerate(self._endpoints):
            if ep.provider.provider_name == provider_name:
                self._endpoints.pop(i)
                return True
        return False

    def get_endpoints(self) -> List[ProviderEndpoint]:
        """Get all endpoints."""
        return list(self._endpoints)

    def get_healthy_endpoints(self) -> List[ProviderEndpoint]:
        """Get all healthy endpoints."""
        return [ep for ep in self._endpoints if ep.is_available()]

    async def select_endpoint(self) -> Optional[ProviderEndpoint]:
        """
        Select an endpoint based on the configured strategy.

        Returns:
            Selected ProviderEndpoint or None if none available
        """
        available = self.get_healthy_endpoints()
        if not available:
            # Try to use degraded endpoints
            available = [
                ep for ep in self._endpoints
                if ep.health != ProviderHealth.UNHEALTHY
            ]
            if not available:
                return None

        if self._config.strategy == FailoverStrategy.PRIORITY:
            # Sort by priority (lower is better)
            available.sort(key=lambda ep: ep.priority)
            return available[0]

        elif self._config.strategy == FailoverStrategy.ROUND_ROBIN:
            async with self._lock:
                self._round_robin_index = (
                    self._round_robin_index % len(available)
                )
                selected = available[self._round_robin_index]
                self._round_robin_index += 1
                return selected

        elif self._config.strategy == FailoverStrategy.LATENCY:
            # Sort by average latency
            available.sort(key=lambda ep: ep.avg_latency_ms or float("inf"))
            return available[0]

        elif self._config.strategy == FailoverStrategy.WEIGHTED:
            # Weighted random selection
            import random
            total_weight = sum(ep.weight for ep in available)
            if total_weight == 0:
                return available[0]
            r = random.uniform(0, total_weight)
            cumulative = 0.0
            for ep in available:
                cumulative += ep.weight
                if r <= cumulative:
                    return ep
            return available[-1]

        return available[0] if available else None

    async def analyze_with_failover(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> FailoverResult:
        """
        Analyze image with automatic failover.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description

        Returns:
            FailoverResult with result and attempt history
        """
        start_time = time.time()
        attempts: List[tuple[str, Optional[str]]] = []
        tried_providers: set[str] = set()

        for attempt in range(self._config.max_retries):
            # Select endpoint
            endpoint = await self.select_endpoint()
            if endpoint is None:
                break

            provider_name = endpoint.provider.provider_name

            # Skip if already tried
            if provider_name in tried_providers:
                # Try to find another endpoint
                available = [
                    ep for ep in self.get_healthy_endpoints()
                    if ep.provider.provider_name not in tried_providers
                ]
                if available:
                    endpoint = available[0]
                    provider_name = endpoint.provider.provider_name
                else:
                    break

            tried_providers.add(provider_name)
            request_start = time.time()

            try:
                result = await endpoint.provider.analyze_image(
                    image_data, include_description
                )
                latency_ms = (time.time() - request_start) * 1000
                endpoint.record_success(latency_ms)

                attempts.append((provider_name, None))
                total_time_ms = (time.time() - start_time) * 1000

                logger.info(
                    f"Failover success: {provider_name} on attempt {attempt + 1}"
                )

                return FailoverResult(
                    success=True,
                    result=result,
                    provider_used=provider_name,
                    attempts=attempts,
                    total_time_ms=total_time_ms,
                )

            except Exception as e:
                endpoint.record_failure()
                attempts.append((provider_name, str(e)))
                logger.warning(
                    f"Failover attempt {attempt + 1} failed for {provider_name}: {e}"
                )

                # Wait before retry
                if attempt < self._config.max_retries - 1:
                    await asyncio.sleep(self._config.retry_delay)

        # All attempts failed
        total_time_ms = (time.time() - start_time) * 1000
        return FailoverResult(
            success=False,
            result=None,
            provider_used="",
            attempts=attempts,
            total_time_ms=total_time_ms,
        )

    async def health_check(self, endpoint: ProviderEndpoint) -> ProviderHealth:
        """
        Perform health check on an endpoint.

        Args:
            endpoint: Endpoint to check

        Returns:
            Updated health status
        """
        # Simple health check - try a minimal request
        test_image = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100

        try:
            start_time = time.time()
            await asyncio.wait_for(
                endpoint.provider.analyze_image(test_image, False),
                timeout=10.0,
            )
            latency_ms = (time.time() - start_time) * 1000
            endpoint.record_success(latency_ms)
            endpoint.last_check = datetime.now()
            return ProviderHealth.HEALTHY

        except asyncio.TimeoutError:
            endpoint.record_failure()
            endpoint.last_check = datetime.now()
            return ProviderHealth.UNHEALTHY

        except Exception:
            # For health checks, any error indicates degraded/unhealthy
            endpoint.record_failure()
            endpoint.last_check = datetime.now()
            return endpoint.health

    async def check_all_endpoints(self) -> Dict[str, ProviderHealth]:
        """Check health of all endpoints."""
        results: Dict[str, ProviderHealth] = {}
        tasks = [
            self.health_check(ep) for ep in self._endpoints
        ]
        healths = await asyncio.gather(*tasks, return_exceptions=True)

        for ep, health in zip(self._endpoints, healths):
            if isinstance(health, Exception):
                results[ep.provider.provider_name] = ProviderHealth.UNHEALTHY
            else:
                results[ep.provider.provider_name] = health

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get current status of all endpoints."""
        return {
            "endpoints": [
                {
                    "provider": ep.provider.provider_name,
                    "region": ep.region,
                    "priority": ep.priority,
                    "health": ep.health.value,
                    "consecutive_failures": ep.consecutive_failures,
                    "avg_latency_ms": ep.avg_latency_ms,
                    "last_success": (
                        ep.last_success.isoformat() if ep.last_success else None
                    ),
                    "last_failure": (
                        ep.last_failure.isoformat() if ep.last_failure else None
                    ),
                }
                for ep in self._endpoints
            ],
            "strategy": self._config.strategy.value,
            "healthy_count": len(self.get_healthy_endpoints()),
            "total_count": len(self._endpoints),
        }


class FailoverVisionProvider:
    """
    Wrapper that adds failover capability to vision providers.

    Automatically fails over to backup providers on failure.
    """

    def __init__(
        self,
        failover_manager: FailoverManager,
    ):
        """
        Initialize failover provider.

        Args:
            failover_manager: FailoverManager instance
        """
        self._manager = failover_manager

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """
        Analyze image with automatic failover.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description

        Returns:
            VisionDescription with analysis results

        Raises:
            VisionProviderError: If all providers fail
        """
        result = await self._manager.analyze_with_failover(
            image_data, include_description
        )

        if result.success and result.result:
            return result.result

        # All providers failed
        error_msg = "; ".join(
            f"{p}: {e}" for p, e in result.attempts if e
        )
        raise VisionProviderError(
            "failover",
            f"All providers failed: {error_msg}",
        )

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "failover"

    @property
    def failover_manager(self) -> FailoverManager:
        """Get the failover manager."""
        return self._manager


def create_failover_provider(
    providers: List[VisionProvider],
    strategy: FailoverStrategy = FailoverStrategy.PRIORITY,
    max_retries: int = 3,
) -> FailoverVisionProvider:
    """
    Factory to create a failover provider from a list of providers.

    Args:
        providers: List of vision providers in priority order
        strategy: Failover strategy to use
        max_retries: Maximum retry attempts

    Returns:
        FailoverVisionProvider wrapping all providers

    Example:
        >>> providers = [
        ...     create_vision_provider("openai"),
        ...     create_vision_provider("anthropic"),
        ...     create_vision_provider("deepseek"),
        ... ]
        >>> failover = create_failover_provider(providers)
        >>> result = await failover.analyze_image(image_bytes)
    """
    endpoints = [
        ProviderEndpoint(
            provider=p,
            priority=i,
            health=ProviderHealth.UNKNOWN,
        )
        for i, p in enumerate(providers)
    ]

    config = FailoverConfig(
        strategy=strategy,
        max_retries=max_retries,
    )

    manager = FailoverManager(endpoints=endpoints, config=config)
    return FailoverVisionProvider(manager)
