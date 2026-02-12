"""API Gateway Load Balancing.

Provides load balancing strategies:
- Round robin
- Weighted
- Least connections
- Random
"""

from __future__ import annotations

import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BackendState(Enum):
    """State of a backend."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"


@dataclass
class Backend:
    """A backend service instance."""
    id: str
    host: str
    port: int
    weight: int = 1
    state: BackendState = BackendState.HEALTHY
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Runtime state
    active_connections: int = 0
    total_requests: int = 0
    failed_requests: int = 0
    last_health_check: Optional[datetime] = None
    last_failure: Optional[datetime] = None

    @property
    def address(self) -> str:
        return f"{self.host}:{self.port}"

    @property
    def is_available(self) -> bool:
        return self.state == BackendState.HEALTHY

    @property
    def error_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.failed_requests / self.total_requests


@dataclass
class BackendPool:
    """Pool of backend instances."""
    name: str
    backends: List[Backend] = field(default_factory=list)

    def add(self, backend: Backend) -> None:
        self.backends.append(backend)

    def remove(self, backend_id: str) -> bool:
        initial = len(self.backends)
        self.backends = [b for b in self.backends if b.id != backend_id]
        return len(self.backends) < initial

    def get_healthy(self) -> List[Backend]:
        return [b for b in self.backends if b.is_available]

    def get_by_id(self, backend_id: str) -> Optional[Backend]:
        for b in self.backends:
            if b.id == backend_id:
                return b
        return None


class LoadBalancer(ABC):
    """Abstract load balancer."""

    @abstractmethod
    def select(self, pool: BackendPool) -> Optional[Backend]:
        """Select a backend from the pool."""
        pass

    def on_request_start(self, backend: Backend) -> None:
        """Called when request starts."""
        backend.active_connections += 1
        backend.total_requests += 1

    def on_request_end(self, backend: Backend, success: bool) -> None:
        """Called when request ends."""
        backend.active_connections = max(0, backend.active_connections - 1)
        if not success:
            backend.failed_requests += 1
            backend.last_failure = datetime.utcnow()


class RoundRobinBalancer(LoadBalancer):
    """Round robin load balancing."""

    def __init__(self):
        self._counters: Dict[str, int] = {}

    def select(self, pool: BackendPool) -> Optional[Backend]:
        healthy = pool.get_healthy()
        if not healthy:
            return None

        counter = self._counters.get(pool.name, 0)
        backend = healthy[counter % len(healthy)]
        self._counters[pool.name] = counter + 1

        return backend


class WeightedRoundRobinBalancer(LoadBalancer):
    """Weighted round robin load balancing."""

    def __init__(self):
        self._current_weights: Dict[str, Dict[str, int]] = {}

    def select(self, pool: BackendPool) -> Optional[Backend]:
        healthy = pool.get_healthy()
        if not healthy:
            return None

        # Initialize weights
        if pool.name not in self._current_weights:
            self._current_weights[pool.name] = {
                b.id: 0 for b in healthy
            }

        weights = self._current_weights[pool.name]

        # Update weights
        total_weight = sum(b.weight for b in healthy)
        for backend in healthy:
            if backend.id not in weights:
                weights[backend.id] = 0
            weights[backend.id] += backend.weight

        # Select backend with highest current weight
        selected = max(healthy, key=lambda b: weights.get(b.id, 0))

        # Decrease selected backend's weight
        weights[selected.id] -= total_weight

        return selected


class LeastConnectionsBalancer(LoadBalancer):
    """Least connections load balancing."""

    def select(self, pool: BackendPool) -> Optional[Backend]:
        healthy = pool.get_healthy()
        if not healthy:
            return None

        # Select backend with fewest active connections
        return min(healthy, key=lambda b: b.active_connections)


class WeightedLeastConnectionsBalancer(LoadBalancer):
    """Weighted least connections load balancing."""

    def select(self, pool: BackendPool) -> Optional[Backend]:
        healthy = pool.get_healthy()
        if not healthy:
            return None

        # Calculate weighted score (lower is better)
        def score(backend: Backend) -> float:
            if backend.weight == 0:
                return float('inf')
            return backend.active_connections / backend.weight

        return min(healthy, key=score)


class RandomBalancer(LoadBalancer):
    """Random load balancing."""

    def select(self, pool: BackendPool) -> Optional[Backend]:
        healthy = pool.get_healthy()
        if not healthy:
            return None

        return random.choice(healthy)


class WeightedRandomBalancer(LoadBalancer):
    """Weighted random load balancing."""

    def select(self, pool: BackendPool) -> Optional[Backend]:
        healthy = pool.get_healthy()
        if not healthy:
            return None

        total_weight = sum(b.weight for b in healthy)
        if total_weight == 0:
            return random.choice(healthy)

        r = random.uniform(0, total_weight)
        cumulative = 0

        for backend in healthy:
            cumulative += backend.weight
            if r <= cumulative:
                return backend

        return healthy[-1]


class IPHashBalancer(LoadBalancer):
    """IP hash load balancing for session affinity."""

    def __init__(self):
        self._client_ip: Optional[str] = None

    def set_client_ip(self, ip: str) -> None:
        self._client_ip = ip

    def select(self, pool: BackendPool) -> Optional[Backend]:
        healthy = pool.get_healthy()
        if not healthy:
            return None

        if self._client_ip:
            hash_value = hash(self._client_ip)
            return healthy[hash_value % len(healthy)]

        return random.choice(healthy)


class AdaptiveBalancer(LoadBalancer):
    """Adaptive load balancing based on response times and error rates."""

    def __init__(
        self,
        error_rate_threshold: float = 0.1,
        response_time_window: int = 100,
    ):
        self._error_threshold = error_rate_threshold
        self._window_size = response_time_window
        self._response_times: Dict[str, List[float]] = {}

    def record_response_time(self, backend: Backend, time_ms: float) -> None:
        """Record response time for backend."""
        if backend.id not in self._response_times:
            self._response_times[backend.id] = []

        times = self._response_times[backend.id]
        times.append(time_ms)

        # Keep window size
        if len(times) > self._window_size:
            self._response_times[backend.id] = times[-self._window_size:]

    def select(self, pool: BackendPool) -> Optional[Backend]:
        healthy = pool.get_healthy()
        if not healthy:
            return None

        # Filter out backends with high error rates
        candidates = [
            b for b in healthy
            if b.error_rate < self._error_threshold
        ]

        if not candidates:
            # Fall back to all healthy if all have high error rates
            candidates = healthy

        # Score based on average response time
        def score(backend: Backend) -> float:
            times = self._response_times.get(backend.id, [])
            if not times:
                return 0  # Unknown backends get priority

            avg_time = sum(times) / len(times)
            # Also factor in weight and active connections
            connection_factor = 1 + (backend.active_connections * 0.1)
            weight_factor = 1 / max(backend.weight, 1)

            return avg_time * connection_factor * weight_factor

        return min(candidates, key=score)


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer."""
    algorithm: str = "round_robin"
    health_check_interval: float = 10.0
    unhealthy_threshold: int = 3
    healthy_threshold: int = 2


def create_load_balancer(algorithm: str) -> LoadBalancer:
    """Create load balancer from algorithm name."""
    balancers = {
        "round_robin": RoundRobinBalancer,
        "weighted_round_robin": WeightedRoundRobinBalancer,
        "least_connections": LeastConnectionsBalancer,
        "weighted_least_connections": WeightedLeastConnectionsBalancer,
        "random": RandomBalancer,
        "weighted_random": WeightedRandomBalancer,
        "ip_hash": IPHashBalancer,
        "adaptive": AdaptiveBalancer,
    }

    balancer_class = balancers.get(algorithm, RoundRobinBalancer)
    return balancer_class()
