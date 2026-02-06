"""Load Balancing.

Provides load balancing strategies:
- Round robin
- Weighted round robin
- Least connections
- Random
- Consistent hashing
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
import time
import threading
from abc import ABC, abstractmethod
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Set, TypeVar

from src.core.service_mesh.discovery import ServiceInstance, ServiceStatus

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class LoadBalancerStats:
    """Statistics for a load balancer."""
    total_requests: int
    requests_by_instance: Dict[str, int]
    active_connections: Dict[str, int]
    average_latency: Dict[str, float]


class LoadBalancer(ABC):
    """Abstract base class for load balancers."""

    @abstractmethod
    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ServiceInstance]:
        """Select an instance to route request to."""
        pass

    @abstractmethod
    def record_success(
        self,
        instance: ServiceInstance,
        latency: float,
    ) -> None:
        """Record a successful request."""
        pass

    @abstractmethod
    def record_failure(
        self,
        instance: ServiceInstance,
        error: Exception,
    ) -> None:
        """Record a failed request."""
        pass

    @abstractmethod
    def get_stats(self) -> LoadBalancerStats:
        """Get load balancer statistics."""
        pass


class RoundRobinBalancer(LoadBalancer):
    """Round-robin load balancing."""

    def __init__(self):
        self._index = 0
        self._requests: Dict[str, int] = {}
        self._lock = threading.Lock()

    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ServiceInstance]:
        if not instances:
            return None

        healthy = [i for i in instances if i.is_available()]
        if not healthy:
            return None

        instance = healthy[self._index % len(healthy)]
        self._index += 1

        self._requests[instance.instance_id] = (
            self._requests.get(instance.instance_id, 0) + 1
        )

        return instance

    def record_success(self, instance: ServiceInstance, latency: float) -> None:
        pass

    def record_failure(self, instance: ServiceInstance, error: Exception) -> None:
        pass

    def get_stats(self) -> LoadBalancerStats:
        return LoadBalancerStats(
            total_requests=sum(self._requests.values()),
            requests_by_instance=self._requests.copy(),
            active_connections={},
            average_latency={},
        )


class WeightedRoundRobinBalancer(LoadBalancer):
    """Weighted round-robin load balancing."""

    def __init__(self):
        self._current_weight = 0
        self._index = 0
        self._requests: Dict[str, int] = {}

    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ServiceInstance]:
        if not instances:
            return None

        healthy = [i for i in instances if i.is_available()]
        if not healthy:
            return None

        # Calculate weights
        total_weight = sum(i.weight for i in healthy)
        if total_weight == 0:
            return healthy[0]

        # Weighted selection
        max_weight = max(i.weight for i in healthy)
        gcd_weight = self._gcd_weights([i.weight for i in healthy])

        while True:
            self._index = (self._index + 1) % len(healthy)
            if self._index == 0:
                self._current_weight -= gcd_weight
                if self._current_weight <= 0:
                    self._current_weight = max_weight

            instance = healthy[self._index]
            if instance.weight >= self._current_weight:
                self._requests[instance.instance_id] = (
                    self._requests.get(instance.instance_id, 0) + 1
                )
                return instance

    def _gcd_weights(self, weights: List[int]) -> int:
        """Calculate GCD of all weights."""
        from math import gcd
        from functools import reduce
        return reduce(gcd, weights)

    def record_success(self, instance: ServiceInstance, latency: float) -> None:
        pass

    def record_failure(self, instance: ServiceInstance, error: Exception) -> None:
        pass

    def get_stats(self) -> LoadBalancerStats:
        return LoadBalancerStats(
            total_requests=sum(self._requests.values()),
            requests_by_instance=self._requests.copy(),
            active_connections={},
            average_latency={},
        )


class LeastConnectionsBalancer(LoadBalancer):
    """Least connections load balancing."""

    def __init__(self):
        self._connections: Dict[str, int] = {}
        self._requests: Dict[str, int] = {}
        self._latencies: Dict[str, List[float]] = {}

    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ServiceInstance]:
        if not instances:
            return None

        healthy = [i for i in instances if i.is_available()]
        if not healthy:
            return None

        # Select instance with fewest connections
        min_conns = float('inf')
        selected = None

        for instance in healthy:
            conns = self._connections.get(instance.instance_id, 0)
            # Consider weight
            weighted_conns = conns / instance.weight if instance.weight > 0 else conns
            if weighted_conns < min_conns:
                min_conns = weighted_conns
                selected = instance

        if selected:
            self._connections[selected.instance_id] = (
                self._connections.get(selected.instance_id, 0) + 1
            )
            self._requests[selected.instance_id] = (
                self._requests.get(selected.instance_id, 0) + 1
            )

        return selected

    def record_success(self, instance: ServiceInstance, latency: float) -> None:
        self._connections[instance.instance_id] = max(
            0, self._connections.get(instance.instance_id, 1) - 1
        )

        if instance.instance_id not in self._latencies:
            self._latencies[instance.instance_id] = []
        self._latencies[instance.instance_id].append(latency)
        # Keep only last 100 latencies
        if len(self._latencies[instance.instance_id]) > 100:
            self._latencies[instance.instance_id] = self._latencies[instance.instance_id][-100:]

    def record_failure(self, instance: ServiceInstance, error: Exception) -> None:
        self._connections[instance.instance_id] = max(
            0, self._connections.get(instance.instance_id, 1) - 1
        )

    def get_stats(self) -> LoadBalancerStats:
        avg_latency = {}
        for instance_id, latencies in self._latencies.items():
            if latencies:
                avg_latency[instance_id] = sum(latencies) / len(latencies)

        return LoadBalancerStats(
            total_requests=sum(self._requests.values()),
            requests_by_instance=self._requests.copy(),
            active_connections=self._connections.copy(),
            average_latency=avg_latency,
        )


class RandomBalancer(LoadBalancer):
    """Random load balancing."""

    def __init__(self, weighted: bool = False):
        self.weighted = weighted
        self._requests: Dict[str, int] = {}

    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ServiceInstance]:
        if not instances:
            return None

        healthy = [i for i in instances if i.is_available()]
        if not healthy:
            return None

        if self.weighted:
            weights = [i.weight for i in healthy]
            instance = random.choices(healthy, weights=weights)[0]
        else:
            instance = random.choice(healthy)

        self._requests[instance.instance_id] = (
            self._requests.get(instance.instance_id, 0) + 1
        )

        return instance

    def record_success(self, instance: ServiceInstance, latency: float) -> None:
        pass

    def record_failure(self, instance: ServiceInstance, error: Exception) -> None:
        pass

    def get_stats(self) -> LoadBalancerStats:
        return LoadBalancerStats(
            total_requests=sum(self._requests.values()),
            requests_by_instance=self._requests.copy(),
            active_connections={},
            average_latency={},
        )


class ConsistentHashBalancer(LoadBalancer):
    """Consistent hashing load balancing.

    Useful for sticky sessions and cache affinity.
    """

    def __init__(self, replicas: int = 100):
        self.replicas = replicas
        self._ring: List[int] = []
        self._ring_map: Dict[int, str] = {}
        self._instance_map: Dict[str, ServiceInstance] = {}
        self._requests: Dict[str, int] = {}

    def _hash(self, key: str) -> int:
        return int(hashlib.md5(key.encode()).hexdigest(), 16)  # nosec B324 - consistent hashing

    def _rebuild_ring(self, instances: List[ServiceInstance]) -> None:
        """Rebuild the hash ring."""
        self._ring = []
        self._ring_map = {}
        self._instance_map = {}

        for instance in instances:
            if instance.is_available():
                self._instance_map[instance.instance_id] = instance
                for i in range(self.replicas):
                    key = f"{instance.instance_id}:{i}"
                    hash_val = self._hash(key)
                    self._ring.append(hash_val)
                    self._ring_map[hash_val] = instance.instance_id

        self._ring.sort()

    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ServiceInstance]:
        if not instances:
            return None

        # Rebuild ring if instances changed
        self._rebuild_ring(instances)

        if not self._ring:
            return None

        # Get hash key from context or generate one
        hash_key = "default"
        if context:
            # Use user_id, session_id, or request_id for affinity
            hash_key = context.get("user_id") or context.get("session_id") or context.get("request_id", "default")

        hash_val = self._hash(str(hash_key))

        # Find the first instance in the ring
        idx = bisect_left(self._ring, hash_val)
        if idx >= len(self._ring):
            idx = 0

        instance_id = self._ring_map[self._ring[idx]]
        instance = self._instance_map.get(instance_id)

        if instance:
            self._requests[instance.instance_id] = (
                self._requests.get(instance.instance_id, 0) + 1
            )

        return instance

    def record_success(self, instance: ServiceInstance, latency: float) -> None:
        pass

    def record_failure(self, instance: ServiceInstance, error: Exception) -> None:
        pass

    def get_stats(self) -> LoadBalancerStats:
        return LoadBalancerStats(
            total_requests=sum(self._requests.values()),
            requests_by_instance=self._requests.copy(),
            active_connections={},
            average_latency={},
        )


class AdaptiveBalancer(LoadBalancer):
    """Adaptive load balancing based on response times and error rates."""

    def __init__(
        self,
        decay_factor: float = 0.9,
        penalty_factor: float = 2.0,
    ):
        self.decay_factor = decay_factor
        self.penalty_factor = penalty_factor
        self._scores: Dict[str, float] = {}
        self._requests: Dict[str, int] = {}
        self._latencies: Dict[str, List[float]] = {}

    def select(
        self,
        instances: List[ServiceInstance],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[ServiceInstance]:
        if not instances:
            return None

        healthy = [i for i in instances if i.is_available()]
        if not healthy:
            return None

        # Initialize scores for new instances
        for instance in healthy:
            if instance.instance_id not in self._scores:
                self._scores[instance.instance_id] = 1.0

        # Select instance with best score (higher is better)
        best_score = -1
        selected = None

        for instance in healthy:
            score = self._scores.get(instance.instance_id, 1.0)
            # Consider weight
            weighted_score = score * instance.weight
            if weighted_score > best_score:
                best_score = weighted_score
                selected = instance

        if selected:
            self._requests[selected.instance_id] = (
                self._requests.get(selected.instance_id, 0) + 1
            )

        return selected

    def record_success(self, instance: ServiceInstance, latency: float) -> None:
        # Update score based on latency
        instance_id = instance.instance_id
        current_score = self._scores.get(instance_id, 1.0)

        # Lower latency = higher score
        latency_score = 1.0 / (1.0 + latency)

        # Exponential moving average
        new_score = self.decay_factor * current_score + (1 - self.decay_factor) * latency_score
        self._scores[instance_id] = new_score

        # Track latency
        if instance_id not in self._latencies:
            self._latencies[instance_id] = []
        self._latencies[instance_id].append(latency)
        if len(self._latencies[instance_id]) > 100:
            self._latencies[instance_id] = self._latencies[instance_id][-100:]

    def record_failure(self, instance: ServiceInstance, error: Exception) -> None:
        # Penalize failed instances
        instance_id = instance.instance_id
        current_score = self._scores.get(instance_id, 1.0)
        self._scores[instance_id] = current_score / self.penalty_factor

    def get_stats(self) -> LoadBalancerStats:
        avg_latency = {}
        for instance_id, latencies in self._latencies.items():
            if latencies:
                avg_latency[instance_id] = sum(latencies) / len(latencies)

        return LoadBalancerStats(
            total_requests=sum(self._requests.values()),
            requests_by_instance=self._requests.copy(),
            active_connections={},
            average_latency=avg_latency,
        )


class LoadBalancerFactory:
    """Factory for creating load balancers."""

    @staticmethod
    def create(strategy: str, **kwargs) -> LoadBalancer:
        """Create a load balancer by strategy name."""
        strategies = {
            "round_robin": RoundRobinBalancer,
            "weighted_round_robin": WeightedRoundRobinBalancer,
            "least_connections": LeastConnectionsBalancer,
            "random": lambda: RandomBalancer(weighted=False),
            "weighted_random": lambda: RandomBalancer(weighted=True),
            "consistent_hash": lambda: ConsistentHashBalancer(**kwargs),
            "adaptive": lambda: AdaptiveBalancer(**kwargs),
        }

        if strategy not in strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        factory = strategies[strategy]
        if callable(factory):
            return factory()
        return factory
