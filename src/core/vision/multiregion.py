"""Multi-region routing module for Vision Provider system.

This module provides geographic routing and failover capabilities including:
- Region-based provider selection
- Latency-based routing
- Geographic failover
- Regional health monitoring
- Cross-region replication
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from .base import VisionDescription, VisionProvider


class Region(Enum):
    """Geographic regions."""

    US_EAST = "us-east"
    US_WEST = "us-west"
    EU_WEST = "eu-west"
    EU_CENTRAL = "eu-central"
    ASIA_EAST = "asia-east"
    ASIA_SOUTH = "asia-south"
    AUSTRALIA = "australia"
    SOUTH_AMERICA = "south-america"
    AFRICA = "africa"
    MIDDLE_EAST = "middle-east"


class RoutingStrategy(Enum):
    """Routing strategy options."""

    NEAREST = "nearest"  # Route to nearest region
    LATENCY = "latency"  # Route based on measured latency
    ROUND_ROBIN = "round_robin"  # Rotate through regions
    WEIGHTED = "weighted"  # Weight-based distribution
    FAILOVER = "failover"  # Primary with failover
    GEOFENCED = "geofenced"  # Restrict to specific regions


class RegionStatus(Enum):
    """Health status of a region."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"


@dataclass
class RegionConfig:
    """Configuration for a region."""

    region: Region
    provider: VisionProvider
    weight: float = 1.0
    priority: int = 0  # Lower is higher priority
    is_primary: bool = False
    enabled: bool = True
    maintenance_window: Optional[Tuple[datetime, datetime]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegionHealth:
    """Health status of a region."""

    region: Region
    status: RegionStatus = RegionStatus.HEALTHY
    latency_ms: float = 0.0
    success_rate: float = 1.0
    last_check: datetime = field(default_factory=datetime.now)
    last_success: Optional[datetime] = None
    consecutive_failures: int = 0
    total_requests: int = 0
    total_successes: int = 0
    total_failures: int = 0

    def update_success(self, latency_ms: float) -> None:
        """Update after successful request."""
        self.total_requests += 1
        self.total_successes += 1
        self.consecutive_failures = 0
        self.last_success = datetime.now()
        self.last_check = datetime.now()

        # Update moving average latency
        if self.total_requests == 1:
            self.latency_ms = latency_ms
        else:
            self.latency_ms = self.latency_ms * 0.9 + latency_ms * 0.1

        self._update_status()

    def update_failure(self) -> None:
        """Update after failed request."""
        self.total_requests += 1
        self.total_failures += 1
        self.consecutive_failures += 1
        self.last_check = datetime.now()

        self._update_status()

    def _update_status(self) -> None:
        """Update health status based on metrics."""
        if self.total_requests > 0:
            self.success_rate = self.total_successes / self.total_requests

        if self.consecutive_failures >= 5:
            self.status = RegionStatus.UNHEALTHY
        elif self.consecutive_failures >= 2 or self.success_rate < 0.9:
            self.status = RegionStatus.DEGRADED
        else:
            self.status = RegionStatus.HEALTHY


@dataclass
class RoutingDecision:
    """Record of a routing decision."""

    request_id: str
    selected_region: Region
    strategy_used: RoutingStrategy
    fallback_regions: List[Region]
    decision_time_ms: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RegionLatency:
    """Latency information for a region."""

    region: Region
    latency_ms: float
    samples: int = 1
    last_measured: datetime = field(default_factory=datetime.now)


class LatencyMeasurer:
    """Measures and tracks region latencies."""

    def __init__(self, measurement_interval: timedelta = timedelta(minutes=5)) -> None:
        """Initialize the latency measurer."""
        self._latencies: Dict[Region, RegionLatency] = {}
        self._measurement_interval = measurement_interval

    def record_latency(self, region: Region, latency_ms: float) -> None:
        """Record a latency measurement."""
        if region not in self._latencies:
            self._latencies[region] = RegionLatency(region=region, latency_ms=latency_ms)
        else:
            existing = self._latencies[region]
            # Exponential moving average
            existing.latency_ms = existing.latency_ms * 0.8 + latency_ms * 0.2
            existing.samples += 1
            existing.last_measured = datetime.now()

    def get_latency(self, region: Region) -> Optional[float]:
        """Get the current latency estimate for a region."""
        if region in self._latencies:
            return self._latencies[region].latency_ms
        return None

    def get_sorted_regions(self) -> List[Tuple[Region, float]]:
        """Get regions sorted by latency (lowest first)."""
        return sorted(
            [(r, l.latency_ms) for r, l in self._latencies.items()],
            key=lambda x: x[1],
        )

    def is_measurement_stale(self, region: Region) -> bool:
        """Check if measurement is stale."""
        if region not in self._latencies:
            return True

        elapsed = datetime.now() - self._latencies[region].last_measured
        return elapsed > self._measurement_interval


class RegionRouter:
    """Routes requests to appropriate regions."""

    def __init__(
        self,
        strategy: RoutingStrategy = RoutingStrategy.LATENCY,
        allowed_regions: Optional[List[Region]] = None,
        blocked_regions: Optional[List[Region]] = None,
    ) -> None:
        """Initialize the router.

        Args:
            strategy: Routing strategy to use
            allowed_regions: Regions that can be used (geofencing)
            blocked_regions: Regions that cannot be used
        """
        self._strategy = strategy
        self._allowed_regions = set(allowed_regions) if allowed_regions else None
        self._blocked_regions = set(blocked_regions) if blocked_regions else set()
        self._regions: Dict[Region, RegionConfig] = {}
        self._health: Dict[Region, RegionHealth] = {}
        self._latency_measurer = LatencyMeasurer()
        self._round_robin_index = 0
        self._decisions: List[RoutingDecision] = []

    def add_region(self, config: RegionConfig) -> None:
        """Add a region configuration."""
        self._regions[config.region] = config
        self._health[config.region] = RegionHealth(region=config.region)

    def remove_region(self, region: Region) -> None:
        """Remove a region."""
        self._regions.pop(region, None)
        self._health.pop(region, None)

    def set_strategy(self, strategy: RoutingStrategy) -> None:
        """Set the routing strategy."""
        self._strategy = strategy

    def get_available_regions(self) -> List[Region]:
        """Get list of available regions."""
        available = []

        for region, config in self._regions.items():
            if not config.enabled:
                continue

            if region in self._blocked_regions:
                continue

            if self._allowed_regions and region not in self._allowed_regions:
                continue

            # Check maintenance window
            if config.maintenance_window:
                start, end = config.maintenance_window
                now = datetime.now()
                if start <= now <= end:
                    continue

            # Check health
            health = self._health.get(region)
            if health and health.status == RegionStatus.OFFLINE:
                continue

            available.append(region)

        return available

    def select_region(
        self, request_id: str, preferred_region: Optional[Region] = None
    ) -> Tuple[Optional[Region], List[Region]]:
        """Select a region for a request.

        Args:
            request_id: Unique request identifier
            preferred_region: Optional preferred region

        Returns:
            Tuple of (selected_region, fallback_regions)
        """
        import time

        start_time = time.time()
        available = self.get_available_regions()

        if not available:
            return None, []

        # Use preferred region if available and healthy
        if preferred_region and preferred_region in available:
            health = self._health.get(preferred_region)
            if health and health.status == RegionStatus.HEALTHY:
                fallbacks = [r for r in available if r != preferred_region]
                self._record_decision(
                    request_id,
                    preferred_region,
                    fallbacks,
                    time.time() - start_time,
                    "Preferred region requested",
                )
                return preferred_region, fallbacks

        # Apply routing strategy
        if self._strategy == RoutingStrategy.NEAREST:
            selected, fallbacks = self._route_nearest(available)
        elif self._strategy == RoutingStrategy.LATENCY:
            selected, fallbacks = self._route_by_latency(available)
        elif self._strategy == RoutingStrategy.ROUND_ROBIN:
            selected, fallbacks = self._route_round_robin(available)
        elif self._strategy == RoutingStrategy.WEIGHTED:
            selected, fallbacks = self._route_weighted(available)
        elif self._strategy == RoutingStrategy.FAILOVER:
            selected, fallbacks = self._route_failover(available)
        else:
            selected = available[0]
            fallbacks = available[1:]

        self._record_decision(
            request_id,
            selected,
            fallbacks,
            time.time() - start_time,
            f"Strategy: {self._strategy.value}",
        )

        return selected, fallbacks

    def _route_nearest(self, available: List[Region]) -> Tuple[Region, List[Region]]:
        """Route to nearest region (by priority for now)."""
        # Sort by priority (lower is better)
        sorted_regions = sorted(
            available,
            key=lambda r: self._regions[r].priority,
        )
        return sorted_regions[0], sorted_regions[1:]

    def _route_by_latency(self, available: List[Region]) -> Tuple[Region, List[Region]]:
        """Route based on measured latency."""
        # Get regions with latency data
        regions_with_latency = []
        regions_without_latency = []

        for region in available:
            latency = self._latency_measurer.get_latency(region)
            if latency is not None:
                regions_with_latency.append((region, latency))
            else:
                regions_without_latency.append(region)

        # Sort by latency
        regions_with_latency.sort(key=lambda x: x[1])

        # Combine: lowest latency first, then regions without data
        sorted_regions = [r for r, _ in regions_with_latency] + regions_without_latency

        if not sorted_regions:
            sorted_regions = available

        return sorted_regions[0], sorted_regions[1:]

    def _route_round_robin(self, available: List[Region]) -> Tuple[Region, List[Region]]:
        """Route using round-robin."""
        self._round_robin_index = (self._round_robin_index + 1) % len(available)
        selected = available[self._round_robin_index]
        fallbacks = [r for r in available if r != selected]
        return selected, fallbacks

    def _route_weighted(self, available: List[Region]) -> Tuple[Region, List[Region]]:
        """Route based on weights."""
        import random

        weights = [self._regions[r].weight for r in available]
        selected = random.choices(available, weights=weights, k=1)[0]
        fallbacks = [r for r in available if r != selected]
        return selected, fallbacks

    def _route_failover(self, available: List[Region]) -> Tuple[Region, List[Region]]:
        """Route with primary/failover pattern."""
        # Find primary region
        primary = None
        for region in available:
            if self._regions[region].is_primary:
                health = self._health.get(region)
                if health and health.status == RegionStatus.HEALTHY:
                    primary = region
                    break

        if primary:
            fallbacks = [r for r in available if r != primary]
            return primary, fallbacks

        # Fall back to priority-based selection
        return self._route_nearest(available)

    def _record_decision(
        self,
        request_id: str,
        selected: Region,
        fallbacks: List[Region],
        decision_time: float,
        reason: str,
    ) -> None:
        """Record a routing decision."""
        decision = RoutingDecision(
            request_id=request_id,
            selected_region=selected,
            strategy_used=self._strategy,
            fallback_regions=fallbacks,
            decision_time_ms=decision_time * 1000,
            reason=reason,
        )
        self._decisions.append(decision)

        # Keep only recent decisions
        if len(self._decisions) > 1000:
            self._decisions = self._decisions[-1000:]

    def record_result(self, region: Region, success: bool, latency_ms: float) -> None:
        """Record a request result."""
        health = self._health.get(region)
        if health:
            if success:
                health.update_success(latency_ms)
                self._latency_measurer.record_latency(region, latency_ms)
            else:
                health.update_failure()

    def get_health(self, region: Region) -> Optional[RegionHealth]:
        """Get health status for a region."""
        return self._health.get(region)

    def get_all_health(self) -> Dict[Region, RegionHealth]:
        """Get health status for all regions."""
        return dict(self._health)

    def get_recent_decisions(self, limit: int = 100) -> List[RoutingDecision]:
        """Get recent routing decisions."""
        return self._decisions[-limit:]


class MultiRegionVisionProvider(VisionProvider):
    """Vision provider with multi-region support."""

    def __init__(
        self,
        router: RegionRouter,
        max_retries: int = 2,
        retry_on_regions: bool = True,
    ) -> None:
        """Initialize the multi-region provider.

        Args:
            router: The region router
            max_retries: Maximum retry attempts
            retry_on_regions: Whether to retry on different regions
        """
        self._router = router
        self._max_retries = max_retries
        self._retry_on_regions = retry_on_regions

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "multiregion"

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """Analyze image using multi-region routing.

        Args:
            image_data: Raw image bytes
            include_description: Whether to include description

        Returns:
            Vision analysis description
        """
        import uuid

        request_id = str(uuid.uuid4())
        selected, fallbacks = self._router.select_region(request_id)

        if not selected:
            raise RuntimeError("No available regions")

        regions_to_try = [selected] + (fallbacks if self._retry_on_regions else [])
        regions_to_try = regions_to_try[: self._max_retries + 1]

        last_error = None

        for region in regions_to_try:
            config = self._router._regions.get(region)
            if not config:
                continue

            start_time = time.time()

            try:
                result = await config.provider.analyze_image(image_data, include_description)
                latency_ms = (time.time() - start_time) * 1000

                self._router.record_result(region, True, latency_ms)
                return result

            except Exception as e:
                latency_ms = (time.time() - start_time) * 1000
                self._router.record_result(region, False, latency_ms)
                last_error = e

        raise last_error or RuntimeError("All regions failed")

    def get_router(self) -> RegionRouter:
        """Get the region router."""
        return self._router


def create_multiregion_provider(
    regions: List[Tuple[Region, VisionProvider, float]],
    strategy: RoutingStrategy = RoutingStrategy.LATENCY,
    primary_region: Optional[Region] = None,
) -> MultiRegionVisionProvider:
    """Create a multi-region provider.

    Args:
        regions: List of (region, provider, weight) tuples
        strategy: Routing strategy
        primary_region: Primary region for failover

    Returns:
        Multi-region provider
    """
    router = RegionRouter(strategy=strategy)

    for i, (region, provider, weight) in enumerate(regions):
        config = RegionConfig(
            region=region,
            provider=provider,
            weight=weight,
            priority=i,
            is_primary=(region == primary_region),
        )
        router.add_region(config)

    return MultiRegionVisionProvider(router=router)
