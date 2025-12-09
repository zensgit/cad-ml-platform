"""Load balancing for vision providers.

Provides:
- Multiple load balancing algorithms
- Provider weight management
- Connection pooling awareness
- Traffic distribution metrics
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .base import VisionDescription, VisionProvider, VisionProviderError

logger = logging.getLogger(__name__)


class LoadBalancingAlgorithm(Enum):
    """Load balancing algorithms."""

    ROUND_ROBIN = "round_robin"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_LEAST_CONNECTIONS = "weighted_least_connections"
    RANDOM = "random"
    WEIGHTED_RANDOM = "weighted_random"
    LEAST_RESPONSE_TIME = "least_response_time"
    ADAPTIVE = "adaptive"  # Combines multiple factors


@dataclass
class ProviderNode:
    """A provider node in the load balancer."""

    provider: VisionProvider
    weight: float = 1.0
    max_connections: int = 100
    enabled: bool = True

    # Runtime state
    current_connections: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_response_time_ms: float = 0.0
    last_response_time_ms: float = 0.0
    last_failure: Optional[datetime] = None
    _latency_samples: List[float] = field(default_factory=list)

    @property
    def avg_response_time_ms(self) -> float:
        """Calculate average response time."""
        if self.successful_requests == 0:
            return float("inf")
        return self.total_response_time_ms / self.successful_requests

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    @property
    def effective_weight(self) -> float:
        """Calculate effective weight based on performance."""
        # Reduce weight for poor performance
        if self.success_rate < 0.5:
            return self.weight * 0.1
        elif self.success_rate < 0.8:
            return self.weight * 0.5
        return self.weight

    @property
    def connection_utilization(self) -> float:
        """Calculate connection utilization."""
        if self.max_connections == 0:
            return 1.0
        return self.current_connections / self.max_connections

    def record_request_start(self) -> None:
        """Record start of a request."""
        self.current_connections += 1
        self.total_requests += 1

    def record_request_end(self, success: bool, response_time_ms: float) -> None:
        """Record end of a request."""
        self.current_connections = max(0, self.current_connections - 1)
        self.last_response_time_ms = response_time_ms

        if success:
            self.successful_requests += 1
            self.total_response_time_ms += response_time_ms

            # Keep recent latency samples
            self._latency_samples.append(response_time_ms)
            if len(self._latency_samples) > 100:
                self._latency_samples.pop(0)
        else:
            self.failed_requests += 1
            self.last_failure = datetime.now()

    def is_available(self) -> bool:
        """Check if node is available for requests."""
        if not self.enabled:
            return False
        if self.current_connections >= self.max_connections:
            return False
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provider": self.provider.provider_name,
            "weight": self.weight,
            "effective_weight": self.effective_weight,
            "enabled": self.enabled,
            "current_connections": self.current_connections,
            "max_connections": self.max_connections,
            "connection_utilization": self.connection_utilization,
            "total_requests": self.total_requests,
            "success_rate": self.success_rate,
            "avg_response_time_ms": self.avg_response_time_ms,
            "last_response_time_ms": self.last_response_time_ms,
        }


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer."""

    algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN
    health_check_interval: float = 30.0
    retry_on_failure: bool = True
    max_retries: int = 2
    sticky_sessions: bool = False  # Not typically needed for vision API
    connection_timeout_ms: float = 5000.0


@dataclass
class LoadBalancerStats:
    """Statistics for load balancer."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    requests_per_provider: Dict[str, int]
    success_rate_per_provider: Dict[str, float]
    current_algorithm: LoadBalancingAlgorithm

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "success_rate": (
                self.successful_requests / self.total_requests
                if self.total_requests > 0 else 0
            ),
            "avg_response_time_ms": self.avg_response_time_ms,
            "requests_per_provider": self.requests_per_provider,
            "success_rate_per_provider": self.success_rate_per_provider,
            "algorithm": self.current_algorithm.value,
        }


class LoadBalancer:
    """
    Load balancer for vision providers.

    Features:
    - Multiple algorithms
    - Weighted distribution
    - Connection awareness
    - Adaptive balancing
    """

    def __init__(
        self,
        nodes: List[ProviderNode],
        config: Optional[LoadBalancerConfig] = None,
    ):
        """
        Initialize load balancer.

        Args:
            nodes: List of provider nodes
            config: Load balancer configuration
        """
        self._nodes = nodes
        self._config = config or LoadBalancerConfig()
        self._round_robin_index = 0
        self._lock = asyncio.Lock()

    def add_node(self, node: ProviderNode) -> None:
        """Add a provider node."""
        self._nodes.append(node)

    def remove_node(self, provider_name: str) -> bool:
        """Remove a provider node by name."""
        for i, node in enumerate(self._nodes):
            if node.provider.provider_name == provider_name:
                self._nodes.pop(i)
                return True
        return False

    def enable_node(self, provider_name: str) -> bool:
        """Enable a provider node."""
        for node in self._nodes:
            if node.provider.provider_name == provider_name:
                node.enabled = True
                return True
        return False

    def disable_node(self, provider_name: str) -> bool:
        """Disable a provider node."""
        for node in self._nodes:
            if node.provider.provider_name == provider_name:
                node.enabled = False
                return True
        return False

    def get_available_nodes(self) -> List[ProviderNode]:
        """Get all available nodes."""
        return [n for n in self._nodes if n.is_available()]

    async def select_node(self) -> Optional[ProviderNode]:
        """
        Select a node based on the configured algorithm.

        Returns:
            Selected ProviderNode or None if none available
        """
        available = self.get_available_nodes()
        if not available:
            return None

        algorithm = self._config.algorithm

        if algorithm == LoadBalancingAlgorithm.ROUND_ROBIN:
            return await self._round_robin(available)

        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN:
            return await self._weighted_round_robin(available)

        elif algorithm == LoadBalancingAlgorithm.LEAST_CONNECTIONS:
            return self._least_connections(available)

        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_LEAST_CONNECTIONS:
            return self._weighted_least_connections(available)

        elif algorithm == LoadBalancingAlgorithm.RANDOM:
            return random.choice(available)

        elif algorithm == LoadBalancingAlgorithm.WEIGHTED_RANDOM:
            return self._weighted_random(available)

        elif algorithm == LoadBalancingAlgorithm.LEAST_RESPONSE_TIME:
            return self._least_response_time(available)

        elif algorithm == LoadBalancingAlgorithm.ADAPTIVE:
            return self._adaptive(available)

        return available[0] if available else None

    async def _round_robin(self, nodes: List[ProviderNode]) -> ProviderNode:
        """Simple round-robin selection."""
        async with self._lock:
            self._round_robin_index = self._round_robin_index % len(nodes)
            selected = nodes[self._round_robin_index]
            self._round_robin_index += 1
            return selected

    async def _weighted_round_robin(
        self, nodes: List[ProviderNode]
    ) -> ProviderNode:
        """Weighted round-robin selection."""
        # Build weighted list
        weighted_nodes: List[ProviderNode] = []
        for node in nodes:
            # Add node multiple times based on weight
            count = max(1, int(node.effective_weight * 10))
            weighted_nodes.extend([node] * count)

        if not weighted_nodes:
            return nodes[0]

        async with self._lock:
            self._round_robin_index = self._round_robin_index % len(weighted_nodes)
            selected = weighted_nodes[self._round_robin_index]
            self._round_robin_index += 1
            return selected

    def _least_connections(self, nodes: List[ProviderNode]) -> ProviderNode:
        """Select node with least connections."""
        return min(nodes, key=lambda n: n.current_connections)

    def _weighted_least_connections(
        self, nodes: List[ProviderNode]
    ) -> ProviderNode:
        """Select node with least weighted connections."""
        # Score = connections / weight (lower is better)
        return min(
            nodes,
            key=lambda n: (
                n.current_connections / n.effective_weight
                if n.effective_weight > 0 else float("inf")
            ),
        )

    def _weighted_random(self, nodes: List[ProviderNode]) -> ProviderNode:
        """Weighted random selection."""
        total_weight = sum(n.effective_weight for n in nodes)
        if total_weight == 0:
            return random.choice(nodes)

        r = random.uniform(0, total_weight)
        cumulative = 0.0
        for node in nodes:
            cumulative += node.effective_weight
            if r <= cumulative:
                return node
        return nodes[-1]

    def _least_response_time(self, nodes: List[ProviderNode]) -> ProviderNode:
        """Select node with lowest average response time."""
        return min(nodes, key=lambda n: n.avg_response_time_ms)

    def _adaptive(self, nodes: List[ProviderNode]) -> ProviderNode:
        """
        Adaptive selection combining multiple factors.

        Score = (response_time * 0.4) + (connection_util * 0.3) + (1/success_rate * 0.3)
        Lower score is better.
        """
        def score(node: ProviderNode) -> float:
            # Normalize response time (assume 10s is max)
            rt_score = min(node.avg_response_time_ms / 10000, 1.0)

            # Connection utilization already 0-1
            conn_score = node.connection_utilization

            # Inverse success rate (1 = 0%, 0 = 100%)
            sr_score = 1.0 - node.success_rate

            return (rt_score * 0.4) + (conn_score * 0.3) + (sr_score * 0.3)

        return min(nodes, key=score)

    def get_stats(self) -> LoadBalancerStats:
        """Get load balancer statistics."""
        total_requests = sum(n.total_requests for n in self._nodes)
        successful_requests = sum(n.successful_requests for n in self._nodes)
        failed_requests = sum(n.failed_requests for n in self._nodes)

        # Calculate average response time
        total_time = sum(n.total_response_time_ms for n in self._nodes)
        avg_time = total_time / successful_requests if successful_requests > 0 else 0

        return LoadBalancerStats(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_time,
            requests_per_provider={
                n.provider.provider_name: n.total_requests for n in self._nodes
            },
            success_rate_per_provider={
                n.provider.provider_name: n.success_rate for n in self._nodes
            },
            current_algorithm=self._config.algorithm,
        )

    def get_node_status(self) -> List[Dict[str, Any]]:
        """Get status of all nodes."""
        return [n.to_dict() for n in self._nodes]


class LoadBalancedVisionProvider:
    """
    Wrapper that adds load balancing to vision providers.

    Distributes requests across multiple providers.
    """

    def __init__(
        self,
        load_balancer: LoadBalancer,
        retry_on_failure: bool = True,
        max_retries: int = 2,
    ):
        """
        Initialize load-balanced provider.

        Args:
            load_balancer: LoadBalancer instance
            retry_on_failure: Retry with different node on failure
            max_retries: Maximum retry attempts
        """
        self._balancer = load_balancer
        self._retry_on_failure = retry_on_failure
        self._max_retries = max_retries

    async def analyze_image(
        self,
        image_data: bytes,
        include_description: bool = True,
    ) -> VisionDescription:
        """
        Analyze image using load-balanced providers.

        Args:
            image_data: Raw image bytes
            include_description: Whether to generate description

        Returns:
            VisionDescription with analysis results

        Raises:
            VisionProviderError: If all providers fail
        """
        tried_providers: set = set()
        last_error: Optional[str] = None

        for attempt in range(self._max_retries + 1):
            # Select node
            node = await self._balancer.select_node()
            if node is None:
                break

            provider_name = node.provider.provider_name

            # Skip if already tried
            if provider_name in tried_providers:
                # Try to find another node
                available = [
                    n for n in self._balancer.get_available_nodes()
                    if n.provider.provider_name not in tried_providers
                ]
                if available:
                    node = available[0]
                    provider_name = node.provider.provider_name
                else:
                    break

            tried_providers.add(provider_name)
            node.record_request_start()
            start_time = time.time()

            try:
                result = await node.provider.analyze_image(
                    image_data, include_description
                )
                response_time_ms = (time.time() - start_time) * 1000
                node.record_request_end(True, response_time_ms)

                logger.debug(
                    f"Load balancer: {provider_name} responded in "
                    f"{response_time_ms:.0f}ms"
                )
                return result

            except Exception as e:
                response_time_ms = (time.time() - start_time) * 1000
                node.record_request_end(False, response_time_ms)
                last_error = str(e)

                logger.warning(
                    f"Load balancer: {provider_name} failed: {e}"
                )

                if not self._retry_on_failure:
                    break

        # All attempts failed
        raise VisionProviderError(
            "load_balancer",
            f"All providers failed. Last error: {last_error}",
        )

    @property
    def provider_name(self) -> str:
        """Return provider name."""
        return "load_balancer"

    @property
    def load_balancer(self) -> LoadBalancer:
        """Get the load balancer."""
        return self._balancer

    def get_stats(self) -> LoadBalancerStats:
        """Get load balancer statistics."""
        return self._balancer.get_stats()


def create_load_balanced_provider(
    providers: List[VisionProvider],
    weights: Optional[List[float]] = None,
    algorithm: LoadBalancingAlgorithm = LoadBalancingAlgorithm.WEIGHTED_ROUND_ROBIN,
    max_connections_per_provider: int = 100,
) -> LoadBalancedVisionProvider:
    """
    Factory to create a load-balanced provider.

    Args:
        providers: List of vision providers
        weights: Optional weights for each provider (default: equal)
        algorithm: Load balancing algorithm to use
        max_connections_per_provider: Max concurrent connections per provider

    Returns:
        LoadBalancedVisionProvider wrapping all providers

    Example:
        >>> providers = [
        ...     create_vision_provider("openai"),
        ...     create_vision_provider("anthropic"),
        ...     create_vision_provider("deepseek"),
        ... ]
        >>> balanced = create_load_balanced_provider(
        ...     providers,
        ...     weights=[2.0, 1.5, 1.0],  # OpenAI gets most traffic
        ...     algorithm=LoadBalancingAlgorithm.WEIGHTED_LEAST_CONNECTIONS,
        ... )
        >>> result = await balanced.analyze_image(image_bytes)
    """
    if weights is None:
        weights = [1.0] * len(providers)

    if len(weights) != len(providers):
        raise ValueError("Number of weights must match number of providers")

    nodes = [
        ProviderNode(
            provider=p,
            weight=w,
            max_connections=max_connections_per_provider,
        )
        for p, w in zip(providers, weights)
    ]

    config = LoadBalancerConfig(algorithm=algorithm)
    balancer = LoadBalancer(nodes=nodes, config=config)

    return LoadBalancedVisionProvider(load_balancer=balancer)
