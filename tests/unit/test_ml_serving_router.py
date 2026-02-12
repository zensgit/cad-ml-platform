"""Tests for src/ml/serving/router.py to improve coverage.

Covers:
- RequestRouter routing strategies
- Worker registration and lifecycle
- Load balancing algorithms
- Health-aware routing
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest

from src.ml.serving.router import (
    RequestRouter,
    RouteConfig,
    RoutingStrategy,
    WorkerState,
)


class TestWorkerState:
    """Tests for WorkerState dataclass."""

    def test_default_values(self):
        """Test default WorkerState values."""
        worker = WorkerState(worker_id="w1", model_name="test_model")

        assert worker.worker_id == "w1"
        assert worker.model_name == "test_model"
        assert worker.weight == 1.0
        assert worker.active_requests == 0
        assert worker.total_requests == 0
        assert worker.total_latency == 0.0
        assert worker.recent_latencies == []
        assert worker.is_healthy is True
        assert worker.last_used == 0.0

    def test_avg_latency_empty(self):
        """Test avg_latency with no data."""
        worker = WorkerState(worker_id="w1", model_name="test")

        assert worker.avg_latency == 0.0

    def test_avg_latency_with_data(self):
        """Test avg_latency calculation."""
        worker = WorkerState(worker_id="w1", model_name="test")
        worker.recent_latencies = [0.1, 0.2, 0.3]

        assert worker.avg_latency == pytest.approx(0.2, rel=0.01)

    def test_record_request(self):
        """Test record_request updates state."""
        worker = WorkerState(worker_id="w1", model_name="test")

        worker.record_request(0.5, max_window=10)

        assert worker.total_requests == 1
        assert worker.total_latency == 0.5
        assert worker.recent_latencies == [0.5]
        assert worker.last_used > 0

    def test_record_request_window_limit(self):
        """Test record_request respects max_window."""
        worker = WorkerState(worker_id="w1", model_name="test")

        # Record more than max_window requests
        for i in range(15):
            worker.record_request(0.1 * i, max_window=10)

        assert len(worker.recent_latencies) == 10
        assert worker.total_requests == 15


class TestRouteConfig:
    """Tests for RouteConfig dataclass."""

    def test_default_values(self):
        """Test default RouteConfig values."""
        config = RouteConfig()

        assert config.strategy == RoutingStrategy.ROUND_ROBIN
        assert config.weights is None
        assert config.latency_window == 100
        assert config.sticky_sessions is False

    def test_custom_values(self):
        """Test custom RouteConfig values."""
        config = RouteConfig(
            strategy=RoutingStrategy.WEIGHTED,
            weights={"model_a": 0.7, "model_b": 0.3},
            latency_window=50,
            sticky_sessions=True,
        )

        assert config.strategy == RoutingStrategy.WEIGHTED
        assert config.weights == {"model_a": 0.7, "model_b": 0.3}
        assert config.latency_window == 50
        assert config.sticky_sessions is True


class TestRoutingStrategy:
    """Tests for RoutingStrategy enum."""

    def test_strategy_values(self):
        """Test strategy enum values."""
        assert RoutingStrategy.ROUND_ROBIN.value == "round_robin"
        assert RoutingStrategy.LEAST_CONNECTIONS.value == "least_connections"
        assert RoutingStrategy.WEIGHTED.value == "weighted"
        assert RoutingStrategy.LATENCY_BASED.value == "latency_based"
        assert RoutingStrategy.RANDOM.value == "random"


class TestRequestRouter:
    """Tests for RequestRouter class."""

    def _create_request(self, model_name: str = "test_model"):
        """Create a mock inference request."""
        request = MagicMock()
        request.model_name = model_name
        return request

    def test_init_default_config(self):
        """Test router initialization with default config."""
        router = RequestRouter()

        assert router._config.strategy == RoutingStrategy.ROUND_ROBIN

    def test_init_custom_config(self):
        """Test router initialization with custom config."""
        config = RouteConfig(strategy=RoutingStrategy.RANDOM)
        router = RequestRouter(config=config)

        assert router._config.strategy == RoutingStrategy.RANDOM

    def test_register_worker(self):
        """Test worker registration."""
        router = RequestRouter()

        router.register_worker("w1", "model_a", weight=2.0)

        assert "w1" in router._workers
        assert router._workers["w1"].model_name == "model_a"
        assert router._workers["w1"].weight == 2.0

    def test_register_worker_weight_from_config(self):
        """Test worker registration with weight from config."""
        config = RouteConfig(weights={"model_a": 5.0})
        router = RequestRouter(config=config)

        router.register_worker("w1", "model_a", weight=1.0)

        # Config weight should override
        assert router._workers["w1"].weight == 5.0

    def test_unregister_worker(self):
        """Test worker unregistration."""
        router = RequestRouter()
        router.register_worker("w1", "model_a")

        router.unregister_worker("w1")

        assert "w1" not in router._workers

    def test_unregister_nonexistent_worker(self):
        """Test unregistering a worker that doesn't exist."""
        router = RequestRouter()

        # Should not raise
        router.unregister_worker("nonexistent")

    def test_route_no_workers(self):
        """Test routing with no workers available."""
        router = RequestRouter()
        request = self._create_request("model_a")

        result = router.route(request)

        assert result is None

    def test_route_no_healthy_workers(self):
        """Test routing with no healthy workers."""
        router = RequestRouter()
        router.register_worker("w1", "model_a")
        router.mark_unhealthy("w1")
        request = self._create_request("model_a")

        result = router.route(request)

        assert result is None

    def test_route_round_robin(self):
        """Test round-robin routing strategy."""
        router = RequestRouter(RouteConfig(strategy=RoutingStrategy.ROUND_ROBIN))
        router.register_worker("w1", "model_a")
        router.register_worker("w2", "model_a")

        results = []
        for _ in range(4):
            request = self._create_request("model_a")
            results.append(router.route(request))

        # Should cycle through workers
        assert results == ["w1", "w2", "w1", "w2"]

    def test_route_least_connections(self):
        """Test least-connections routing strategy."""
        config = RouteConfig(strategy=RoutingStrategy.LEAST_CONNECTIONS)
        router = RequestRouter(config=config)
        router.register_worker("w1", "model_a")
        router.register_worker("w2", "model_a")

        # Simulate w1 having more connections
        router._workers["w1"].active_requests = 5
        router._workers["w2"].active_requests = 1

        request = self._create_request("model_a")
        result = router.route(request)

        # Should route to w2 (fewer connections)
        assert result == "w2"

    def test_route_weighted(self):
        """Test weighted routing strategy."""
        config = RouteConfig(strategy=RoutingStrategy.WEIGHTED)
        router = RequestRouter(config=config)
        router.register_worker("w1", "model_a")
        router._workers["w1"].weight = 1.0

        request = self._create_request("model_a")
        result = router.route(request)

        assert result == "w1"

    def test_route_weighted_multiple_workers(self):
        """Test weighted routing with multiple workers."""
        config = RouteConfig(strategy=RoutingStrategy.WEIGHTED)
        router = RequestRouter(config=config)
        router.register_worker("w1", "model_a")
        router.register_worker("w2", "model_a")
        router._workers["w1"].weight = 100.0  # High weight
        router._workers["w2"].weight = 0.001  # Very low weight

        # With such extreme weights, w1 should be chosen most of the time
        results = []
        for _ in range(10):
            request = self._create_request("model_a")
            result = router.route(request)
            results.append(result)
            # Release to allow re-routing
            router.release(result, 0.01)

        # w1 should be chosen predominantly
        assert results.count("w1") >= 8

    def test_route_latency_based(self):
        """Test latency-based routing strategy."""
        config = RouteConfig(strategy=RoutingStrategy.LATENCY_BASED)
        router = RequestRouter(config=config)
        router.register_worker("w1", "model_a")
        router.register_worker("w2", "model_a")

        # w1 has higher latency
        router._workers["w1"].recent_latencies = [0.5, 0.6, 0.7]
        router._workers["w2"].recent_latencies = [0.1, 0.1, 0.1]

        request = self._create_request("model_a")
        result = router.route(request)

        # Should route to w2 (lower latency)
        assert result == "w2"

    def test_route_latency_based_no_latency_data(self):
        """Test latency-based routing with no latency data."""
        config = RouteConfig(strategy=RoutingStrategy.LATENCY_BASED)
        router = RequestRouter(config=config)
        router.register_worker("w1", "model_a")
        router.register_worker("w2", "model_a")

        # w2 has latency data, w1 doesn't
        router._workers["w2"].recent_latencies = [0.1]

        request = self._create_request("model_a")
        result = router.route(request)

        # Should route to w2 (has data)
        assert result == "w2"

    def test_route_random(self):
        """Test random routing strategy."""
        config = RouteConfig(strategy=RoutingStrategy.RANDOM)
        router = RequestRouter(config=config)
        router.register_worker("w1", "model_a")
        router.register_worker("w2", "model_a")

        request = self._create_request("model_a")
        result = router.route(request)

        assert result in ["w1", "w2"]

    def test_route_unknown_strategy_fallback(self):
        """Test routing with unknown strategy falls back to first worker."""
        router = RequestRouter()
        router.register_worker("w1", "model_a")
        router.register_worker("w2", "model_a")

        # Manually set an invalid strategy to test fallback
        router._config.strategy = "invalid"

        request = self._create_request("model_a")
        result = router.route(request)

        # Should use first candidate
        assert result in ["w1", "w2"]

    def test_route_increments_active_requests(self):
        """Test routing increments active request count."""
        router = RequestRouter()
        router.register_worker("w1", "model_a")

        request = self._create_request("model_a")
        router.route(request)

        assert router._workers["w1"].active_requests == 1

    def test_release(self):
        """Test releasing a worker after request."""
        router = RequestRouter()
        router.register_worker("w1", "model_a")

        # Route a request
        request = self._create_request("model_a")
        router.route(request)
        assert router._workers["w1"].active_requests == 1

        # Release
        router.release("w1", latency=0.5)

        assert router._workers["w1"].active_requests == 0
        assert router._workers["w1"].total_requests == 1

    def test_release_nonexistent_worker(self):
        """Test releasing a nonexistent worker."""
        router = RequestRouter()

        # Should not raise
        router.release("nonexistent", 0.1)

    def test_mark_unhealthy(self):
        """Test marking a worker unhealthy."""
        router = RequestRouter()
        router.register_worker("w1", "model_a")

        router.mark_unhealthy("w1")

        assert router._workers["w1"].is_healthy is False

    def test_mark_unhealthy_nonexistent(self):
        """Test marking nonexistent worker unhealthy."""
        router = RequestRouter()

        # Should not raise
        router.mark_unhealthy("nonexistent")

    def test_mark_healthy(self):
        """Test marking a worker healthy."""
        router = RequestRouter()
        router.register_worker("w1", "model_a")
        router.mark_unhealthy("w1")

        router.mark_healthy("w1")

        assert router._workers["w1"].is_healthy is True

    def test_mark_healthy_nonexistent(self):
        """Test marking nonexistent worker healthy."""
        router = RequestRouter()

        # Should not raise
        router.mark_healthy("nonexistent")

    def test_get_workers_for_model(self):
        """Test getting workers for a model."""
        router = RequestRouter()
        router.register_worker("w1", "model_a")
        router.register_worker("w2", "model_a")
        router.register_worker("w3", "model_b")

        workers = router.get_workers_for_model("model_a")

        assert set(workers) == {"w1", "w2"}

    def test_get_workers_for_model_empty(self):
        """Test getting workers for a model with no workers."""
        router = RequestRouter()

        workers = router.get_workers_for_model("nonexistent")

        assert workers == []

    def test_get_stats_empty(self):
        """Test getting stats with no workers."""
        router = RequestRouter()

        stats = router.get_stats()

        assert stats["strategy"] == "round_robin"
        assert stats["total_workers"] == 0
        assert stats["models"] == {}

    def test_get_stats_with_workers(self):
        """Test getting stats with workers."""
        router = RequestRouter()
        router.register_worker("w1", "model_a")
        router.register_worker("w2", "model_a")
        router.register_worker("w3", "model_b")

        # Simulate some activity
        router._workers["w1"].total_requests = 10
        router._workers["w1"].active_requests = 2
        router._workers["w1"].recent_latencies = [0.1, 0.2]
        router._workers["w2"].total_requests = 5
        router._workers["w3"].total_requests = 3

        stats = router.get_stats()

        assert stats["total_workers"] == 3
        assert "model_a" in stats["models"]
        assert "model_b" in stats["models"]
        assert stats["models"]["model_a"]["workers"] == 2
        assert stats["models"]["model_a"]["total_requests"] == 15
        assert stats["models"]["model_a"]["active_requests"] == 2

    def test_get_stats_with_unhealthy_workers(self):
        """Test getting stats with unhealthy workers."""
        router = RequestRouter()
        router.register_worker("w1", "model_a")
        router.register_worker("w2", "model_a")
        router.mark_unhealthy("w2")

        stats = router.get_stats()

        assert stats["models"]["model_a"]["workers"] == 2
        assert stats["models"]["model_a"]["healthy_workers"] == 1

    def test_get_stats_latency_calculation(self):
        """Test stats include latency calculation."""
        router = RequestRouter()
        router.register_worker("w1", "model_a")
        router._workers["w1"].recent_latencies = [0.1, 0.2, 0.3]

        stats = router.get_stats()

        # avg_latency = 0.2, in ms = 200
        assert stats["models"]["model_a"]["avg_latency_ms"] == pytest.approx(200, rel=0.1)
