"""
Request router for model serving.

Provides:
- Request routing strategies
- Load balancing
- Model selection
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Callable

from src.ml.serving.request import InferenceRequest

logger = logging.getLogger(__name__)


class RoutingStrategy(str, Enum):
    """Request routing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    LATENCY_BASED = "latency_based"
    RANDOM = "random"


@dataclass
class RouteConfig:
    """Configuration for request routing."""
    strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN
    weights: Optional[Dict[str, float]] = None  # For weighted routing
    latency_window: int = 100  # Number of requests for latency calculation
    sticky_sessions: bool = False


@dataclass
class WorkerState:
    """State tracking for a worker."""
    worker_id: str
    model_name: str
    weight: float = 1.0
    active_requests: int = 0
    total_requests: int = 0
    total_latency: float = 0.0
    recent_latencies: List[float] = field(default_factory=list)
    is_healthy: bool = True
    last_used: float = 0.0

    @property
    def avg_latency(self) -> float:
        """Calculate average latency."""
        if not self.recent_latencies:
            return 0.0
        return sum(self.recent_latencies) / len(self.recent_latencies)

    def record_request(self, latency: float, max_window: int = 100) -> None:
        """Record request completion."""
        self.total_requests += 1
        self.total_latency += latency
        self.recent_latencies.append(latency)
        if len(self.recent_latencies) > max_window:
            self.recent_latencies.pop(0)
        self.last_used = time.time()


class RequestRouter:
    """
    Router for distributing requests to workers.

    Supports:
    - Multiple routing strategies
    - Load balancing
    - Health-aware routing
    """

    def __init__(self, config: Optional[RouteConfig] = None):
        """
        Initialize request router.

        Args:
            config: Routing configuration
        """
        self._config = config or RouteConfig()
        self._workers: Dict[str, WorkerState] = {}
        self._round_robin_index: Dict[str, int] = {}  # model_name -> index

    def register_worker(
        self,
        worker_id: str,
        model_name: str,
        weight: float = 1.0,
    ) -> None:
        """
        Register a worker for routing.

        Args:
            worker_id: Unique worker identifier
            model_name: Model served by worker
            weight: Weight for weighted routing
        """
        # Override weight from config if provided
        if self._config.weights and model_name in self._config.weights:
            weight = self._config.weights[model_name]

        self._workers[worker_id] = WorkerState(
            worker_id=worker_id,
            model_name=model_name,
            weight=weight,
        )

        if model_name not in self._round_robin_index:
            self._round_robin_index[model_name] = 0

        logger.debug(f"Registered worker {worker_id} for model {model_name}")

    def unregister_worker(self, worker_id: str) -> None:
        """
        Unregister a worker.

        Args:
            worker_id: Worker identifier
        """
        if worker_id in self._workers:
            del self._workers[worker_id]
            logger.debug(f"Unregistered worker {worker_id}")

    def route(self, request: InferenceRequest) -> Optional[str]:
        """
        Route a request to a worker.

        Args:
            request: Inference request

        Returns:
            Worker ID or None if no worker available
        """
        model_name = request.model_name

        # Get healthy workers for this model
        candidates = [
            w for w in self._workers.values()
            if w.model_name == model_name and w.is_healthy
        ]

        if not candidates:
            logger.warning(f"No healthy workers available for model {model_name}")
            return None

        # Apply routing strategy
        if self._config.strategy == RoutingStrategy.ROUND_ROBIN:
            worker = self._route_round_robin(model_name, candidates)
        elif self._config.strategy == RoutingStrategy.LEAST_CONNECTIONS:
            worker = self._route_least_connections(candidates)
        elif self._config.strategy == RoutingStrategy.WEIGHTED:
            worker = self._route_weighted(candidates)
        elif self._config.strategy == RoutingStrategy.LATENCY_BASED:
            worker = self._route_latency_based(candidates)
        elif self._config.strategy == RoutingStrategy.RANDOM:
            worker = self._route_random(candidates)
        else:
            worker = candidates[0]

        if worker:
            worker.active_requests += 1
            return worker.worker_id

        return None

    def release(self, worker_id: str, latency: float) -> None:
        """
        Release a worker after request completion.

        Args:
            worker_id: Worker identifier
            latency: Request latency
        """
        if worker_id in self._workers:
            worker = self._workers[worker_id]
            worker.active_requests = max(0, worker.active_requests - 1)
            worker.record_request(latency, self._config.latency_window)

    def mark_unhealthy(self, worker_id: str) -> None:
        """Mark worker as unhealthy."""
        if worker_id in self._workers:
            self._workers[worker_id].is_healthy = False
            logger.warning(f"Worker {worker_id} marked unhealthy")

    def mark_healthy(self, worker_id: str) -> None:
        """Mark worker as healthy."""
        if worker_id in self._workers:
            self._workers[worker_id].is_healthy = True
            logger.info(f"Worker {worker_id} marked healthy")

    def _route_round_robin(
        self,
        model_name: str,
        candidates: List[WorkerState],
    ) -> WorkerState:
        """Round-robin routing."""
        index = self._round_robin_index.get(model_name, 0)
        worker = candidates[index % len(candidates)]
        self._round_robin_index[model_name] = index + 1
        return worker

    def _route_least_connections(
        self,
        candidates: List[WorkerState],
    ) -> WorkerState:
        """Route to worker with fewest active connections."""
        return min(candidates, key=lambda w: w.active_requests)

    def _route_weighted(
        self,
        candidates: List[WorkerState],
    ) -> WorkerState:
        """Weighted random routing."""
        total_weight = sum(w.weight for w in candidates)
        r = random.random() * total_weight

        cumulative = 0.0
        for worker in candidates:
            cumulative += worker.weight
            if r <= cumulative:
                return worker

        return candidates[-1]

    def _route_latency_based(
        self,
        candidates: List[WorkerState],
    ) -> WorkerState:
        """Route to worker with lowest average latency."""
        # For workers with no latency data, use a default
        def latency_score(w: WorkerState) -> float:
            if not w.recent_latencies:
                return float("inf")
            return w.avg_latency

        return min(candidates, key=latency_score)

    def _route_random(
        self,
        candidates: List[WorkerState],
    ) -> WorkerState:
        """Random routing."""
        return random.choice(candidates)

    def get_workers_for_model(self, model_name: str) -> List[str]:
        """Get all worker IDs for a model."""
        return [
            w.worker_id for w in self._workers.values()
            if w.model_name == model_name
        ]

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        stats_by_model: Dict[str, Dict[str, Any]] = {}

        for worker in self._workers.values():
            model = worker.model_name
            if model not in stats_by_model:
                stats_by_model[model] = {
                    "workers": 0,
                    "healthy_workers": 0,
                    "total_requests": 0,
                    "active_requests": 0,
                    "avg_latency_ms": 0.0,
                }

            stats_by_model[model]["workers"] += 1
            if worker.is_healthy:
                stats_by_model[model]["healthy_workers"] += 1
            stats_by_model[model]["total_requests"] += worker.total_requests
            stats_by_model[model]["active_requests"] += worker.active_requests

        # Calculate average latencies
        for model in stats_by_model:
            workers = [w for w in self._workers.values() if w.model_name == model]
            latencies = [w.avg_latency for w in workers if w.recent_latencies]
            if latencies:
                stats_by_model[model]["avg_latency_ms"] = round(
                    sum(latencies) / len(latencies) * 1000, 2
                )

        return {
            "strategy": self._config.strategy.value,
            "total_workers": len(self._workers),
            "models": stats_by_model,
        }
