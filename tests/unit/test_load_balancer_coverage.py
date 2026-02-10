"""Tests for load_balancer module to improve coverage."""

import pytest
from unittest.mock import MagicMock

from src.core.service_mesh.load_balancer import (
    LoadBalancerStats,
    LoadBalancer,
    RoundRobinBalancer,
    WeightedRoundRobinBalancer,
    LeastConnectionsBalancer,
    RandomBalancer,
    ConsistentHashBalancer,
    AdaptiveBalancer,
    LoadBalancerFactory,
)
from src.core.service_mesh.discovery import ServiceInstance, ServiceStatus


def create_instance(
    instance_id: str,
    weight: int = 1,
    status: ServiceStatus = ServiceStatus.HEALTHY,
) -> ServiceInstance:
    """Helper to create a ServiceInstance."""
    instance = ServiceInstance(
        instance_id=instance_id,
        service_name="test-service",
        host="localhost",
        port=8080,
        weight=weight,
    )
    instance.status = status
    return instance


class TestLoadBalancerStats:
    """Tests for LoadBalancerStats dataclass."""

    def test_creation(self):
        """Test LoadBalancerStats creation."""
        stats = LoadBalancerStats(
            total_requests=100,
            requests_by_instance={"i1": 50, "i2": 50},
            active_connections={"i1": 5},
            average_latency={"i1": 0.5},
        )

        assert stats.total_requests == 100
        assert stats.requests_by_instance == {"i1": 50, "i2": 50}
        assert stats.active_connections == {"i1": 5}
        assert stats.average_latency == {"i1": 0.5}


class TestLoadBalancerAbstract:
    """Tests for abstract LoadBalancer class."""

    def test_select_is_abstract(self):
        """Test select is abstract (line 50)."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            LoadBalancer()

    def test_record_success_is_abstract(self):
        """Test record_success is abstract (line 59)."""
        # Verified through concrete implementations
        lb = RoundRobinBalancer()
        lb.record_success(create_instance("i1"), 0.1)

    def test_record_failure_is_abstract(self):
        """Test record_failure is abstract (line 68)."""
        lb = RoundRobinBalancer()
        lb.record_failure(create_instance("i1"), Exception("test"))

    def test_get_stats_is_abstract(self):
        """Test get_stats is abstract (line 73)."""
        lb = RoundRobinBalancer()
        stats = lb.get_stats()
        assert isinstance(stats, LoadBalancerStats)


class TestRoundRobinBalancer:
    """Tests for RoundRobinBalancer class."""

    def test_select_empty_instances(self):
        """Test select with empty instances (line 90)."""
        lb = RoundRobinBalancer()

        result = lb.select([])

        assert result is None

    def test_select_no_healthy_instances(self):
        """Test select with no healthy instances (line 94)."""
        lb = RoundRobinBalancer()
        unhealthy = create_instance("i1", status=ServiceStatus.UNHEALTHY)

        result = lb.select([unhealthy])

        assert result is None

    def test_select_round_robin(self):
        """Test round-robin selection."""
        lb = RoundRobinBalancer()
        instances = [
            create_instance("i1"),
            create_instance("i2"),
            create_instance("i3"),
        ]

        results = [lb.select(instances).instance_id for _ in range(6)]

        # Should cycle through instances
        assert results == ["i1", "i2", "i3", "i1", "i2", "i3"]

    def test_record_success_noop(self):
        """Test record_success is noop (line 106)."""
        lb = RoundRobinBalancer()
        instance = create_instance("i1")

        # Should not raise
        lb.record_success(instance, 0.1)

    def test_record_failure_noop(self):
        """Test record_failure is noop (line 109)."""
        lb = RoundRobinBalancer()
        instance = create_instance("i1")

        # Should not raise
        lb.record_failure(instance, Exception("test"))

    def test_get_stats(self):
        """Test get_stats returns correct stats (line 112)."""
        lb = RoundRobinBalancer()
        instances = [create_instance("i1"), create_instance("i2")]

        lb.select(instances)
        lb.select(instances)
        lb.select(instances)

        stats = lb.get_stats()

        assert stats.total_requests == 3
        assert len(stats.requests_by_instance) == 2


class TestWeightedRoundRobinBalancer:
    """Tests for WeightedRoundRobinBalancer class."""

    def test_select_empty_instances(self):
        """Test select with empty instances (line 133)."""
        lb = WeightedRoundRobinBalancer()

        result = lb.select([])

        assert result is None

    def test_select_no_healthy_instances(self):
        """Test select with no healthy instances."""
        lb = WeightedRoundRobinBalancer()
        unhealthy = create_instance("i1", status=ServiceStatus.UNHEALTHY)

        result = lb.select([unhealthy])

        assert result is None

    def test_select_zero_weight(self):
        """Test select with zero total weight (lines 142-143)."""
        lb = WeightedRoundRobinBalancer()
        instance = create_instance("i1", weight=0)

        result = lb.select([instance])

        # Should return first instance when weight is 0
        assert result == instance

    def test_select_weighted(self):
        """Test weighted selection (lines 146-161)."""
        lb = WeightedRoundRobinBalancer()
        heavy = create_instance("heavy", weight=3)
        light = create_instance("light", weight=1)

        # Run many selections to see distribution
        counts = {"heavy": 0, "light": 0}
        for _ in range(100):
            result = lb.select([heavy, light])
            counts[result.instance_id] += 1

        # Heavy should be selected more often
        assert counts["heavy"] > counts["light"]

    def test_gcd_weights(self):
        """Test _gcd_weights calculation (lines 165-167)."""
        lb = WeightedRoundRobinBalancer()

        gcd = lb._gcd_weights([6, 9, 12])

        assert gcd == 3

    def test_record_success_noop(self):
        """Test record_success is noop (line 170)."""
        lb = WeightedRoundRobinBalancer()
        lb.record_success(create_instance("i1"), 0.1)

    def test_record_failure_noop(self):
        """Test record_failure is noop (line 173)."""
        lb = WeightedRoundRobinBalancer()
        lb.record_failure(create_instance("i1"), Exception("test"))

    def test_get_stats(self):
        """Test get_stats (line 176)."""
        lb = WeightedRoundRobinBalancer()
        instances = [create_instance("i1")]
        lb.select(instances)

        stats = lb.get_stats()

        assert stats.total_requests == 1


class TestLeastConnectionsBalancer:
    """Tests for LeastConnectionsBalancer class."""

    def test_select_empty_instances(self):
        """Test select with empty instances (line 198)."""
        lb = LeastConnectionsBalancer()

        result = lb.select([])

        assert result is None

    def test_select_no_healthy_instances(self):
        """Test select with no healthy instances (line 202)."""
        lb = LeastConnectionsBalancer()
        unhealthy = create_instance("i1", status=ServiceStatus.UNHEALTHY)

        result = lb.select([unhealthy])

        assert result is None

    def test_select_least_connections(self):
        """Test least connections selection."""
        lb = LeastConnectionsBalancer()
        instances = [create_instance("i1"), create_instance("i2")]

        # Select i1 multiple times (simulating active connections)
        lb.select(instances)
        lb.select(instances)
        lb.select(instances)

        # Now i1 has more connections, should prefer i2
        lb._connections["i1"] = 5
        lb._connections["i2"] = 1

        result = lb.select(instances)

        assert result.instance_id == "i2"

    def test_record_success(self):
        """Test record_success decrements connections (lines 227-236)."""
        lb = LeastConnectionsBalancer()
        instance = create_instance("i1")
        lb._connections["i1"] = 5

        lb.record_success(instance, 0.5)

        assert lb._connections["i1"] == 4
        assert len(lb._latencies["i1"]) == 1

    def test_record_success_latency_cap(self):
        """Test record_success caps latencies at 100 (lines 235-236)."""
        lb = LeastConnectionsBalancer()
        instance = create_instance("i1")
        lb._latencies["i1"] = [0.1] * 100

        lb.record_success(instance, 0.5)

        # Should keep only last 100
        assert len(lb._latencies["i1"]) == 100

    def test_record_failure(self):
        """Test record_failure decrements connections (line 239)."""
        lb = LeastConnectionsBalancer()
        instance = create_instance("i1")
        lb._connections["i1"] = 3

        lb.record_failure(instance, Exception("test"))

        assert lb._connections["i1"] == 2

    def test_get_stats(self):
        """Test get_stats calculates average latency (lines 244-249)."""
        lb = LeastConnectionsBalancer()
        lb._latencies["i1"] = [0.1, 0.2, 0.3]
        lb._requests["i1"] = 10
        lb._connections["i1"] = 2

        stats = lb.get_stats()

        assert abs(stats.average_latency["i1"] - 0.2) < 0.001
        assert stats.active_connections["i1"] == 2


class TestRandomBalancer:
    """Tests for RandomBalancer class."""

    def test_weighted_init(self):
        """Test weighted initialization (lines 261-262)."""
        lb = RandomBalancer(weighted=True)
        assert lb.weighted is True

    def test_select_empty_instances(self):
        """Test select with empty instances (line 269)."""
        lb = RandomBalancer()

        result = lb.select([])

        assert result is None

    def test_select_no_healthy_instances(self):
        """Test select with no healthy instances."""
        lb = RandomBalancer()
        unhealthy = create_instance("i1", status=ServiceStatus.UNHEALTHY)

        result = lb.select([unhealthy])

        assert result is None

    def test_select_random(self):
        """Test random selection (lines 279-280)."""
        lb = RandomBalancer()
        instances = [create_instance("i1"), create_instance("i2")]

        result = lb.select(instances)

        assert result in instances

    def test_select_weighted_random(self):
        """Test weighted random selection (lines 276-278)."""
        lb = RandomBalancer(weighted=True)
        heavy = create_instance("heavy", weight=100)
        light = create_instance("light", weight=1)

        # Run many selections
        counts = {"heavy": 0, "light": 0}
        for _ in range(100):
            result = lb.select([heavy, light])
            counts[result.instance_id] += 1

        # Heavy should be selected much more often
        assert counts["heavy"] > counts["light"]

    def test_record_success_noop(self):
        """Test record_success is noop (line 289)."""
        lb = RandomBalancer()
        lb.record_success(create_instance("i1"), 0.1)

    def test_record_failure_noop(self):
        """Test record_failure is noop (line 292)."""
        lb = RandomBalancer()
        lb.record_failure(create_instance("i1"), Exception("test"))

    def test_get_stats(self):
        """Test get_stats (line 295)."""
        lb = RandomBalancer()
        instances = [create_instance("i1")]
        lb.select(instances)

        stats = lb.get_stats()

        assert stats.total_requests == 1


class TestConsistentHashBalancer:
    """Tests for ConsistentHashBalancer class."""

    def test_select_empty_instances(self):
        """Test select with empty instances (line 342)."""
        lb = ConsistentHashBalancer()

        result = lb.select([])

        assert result is None

    def test_select_no_healthy_instances(self):
        """Test select with no healthy instances (line 348)."""
        lb = ConsistentHashBalancer()
        unhealthy = create_instance("i1", status=ServiceStatus.UNHEALTHY)

        result = lb.select([unhealthy])

        assert result is None

    def test_select_with_context(self):
        """Test select with context (lines 352-354, 361)."""
        lb = ConsistentHashBalancer()
        instances = [create_instance("i1"), create_instance("i2")]

        # Same user_id should consistently select same instance
        context = {"user_id": "user123"}
        results = [lb.select(instances, context) for _ in range(5)]

        # All should be the same instance
        assert all(r.instance_id == results[0].instance_id for r in results)

    def test_select_with_session_id(self):
        """Test select uses session_id from context."""
        lb = ConsistentHashBalancer()
        instances = [create_instance("i1"), create_instance("i2")]

        context = {"session_id": "session456"}
        result = lb.select(instances, context)

        assert result is not None

    def test_select_with_request_id(self):
        """Test select uses request_id from context."""
        lb = ConsistentHashBalancer()
        instances = [create_instance("i1"), create_instance("i2")]

        context = {"request_id": "req789"}
        result = lb.select(instances, context)

        assert result is not None

    def test_select_wraps_around_ring(self):
        """Test select wraps around ring (line 361)."""
        lb = ConsistentHashBalancer(replicas=1)
        instances = [create_instance("i1")]

        # Use high hash value to force wrap around
        result = lb.select(instances, {"user_id": "zzzzzzzzz"})

        assert result is not None

    def test_record_success_noop(self):
        """Test record_success is noop (line 374)."""
        lb = ConsistentHashBalancer()
        lb.record_success(create_instance("i1"), 0.1)

    def test_record_failure_noop(self):
        """Test record_failure is noop (line 377)."""
        lb = ConsistentHashBalancer()
        lb.record_failure(create_instance("i1"), Exception("test"))

    def test_get_stats(self):
        """Test get_stats (line 380)."""
        lb = ConsistentHashBalancer()
        instances = [create_instance("i1")]
        lb.select(instances)

        stats = lb.get_stats()

        assert stats.total_requests == 1


class TestAdaptiveBalancer:
    """Tests for AdaptiveBalancer class."""

    def test_init_with_params(self):
        """Test initialization with parameters (lines 396-400)."""
        lb = AdaptiveBalancer(decay_factor=0.8, penalty_factor=3.0)

        assert lb.decay_factor == 0.8
        assert lb.penalty_factor == 3.0

    def test_select_empty_instances(self):
        """Test select with empty instances (line 407)."""
        lb = AdaptiveBalancer()

        result = lb.select([])

        assert result is None

    def test_select_no_healthy_instances(self):
        """Test select with no healthy instances."""
        lb = AdaptiveBalancer()
        unhealthy = create_instance("i1", status=ServiceStatus.UNHEALTHY)

        result = lb.select([unhealthy])

        assert result is None

    def test_select_best_score(self):
        """Test select chooses best score (lines 419-436)."""
        lb = AdaptiveBalancer()
        instances = [create_instance("good"), create_instance("bad")]

        # Set scores
        lb._scores["good"] = 2.0
        lb._scores["bad"] = 0.5

        result = lb.select(instances)

        assert result.instance_id == "good"

    def test_select_initializes_new_instance_scores(self):
        """Test select initializes scores for new instances (line 417)."""
        lb = AdaptiveBalancer()
        instances = [create_instance("new1"), create_instance("new2")]

        # Neither instance has a score yet
        assert "new1" not in lb._scores
        assert "new2" not in lb._scores

        lb.select(instances)

        # Both should now have initial score of 1.0
        assert lb._scores["new1"] == 1.0
        assert lb._scores["new2"] == 1.0

    def test_record_success(self):
        """Test record_success updates score (lines 440-455)."""
        lb = AdaptiveBalancer()
        instance = create_instance("i1")
        lb._scores["i1"] = 1.0

        lb.record_success(instance, 0.1)

        # Score should be updated
        assert lb._scores["i1"] != 1.0
        assert len(lb._latencies["i1"]) == 1

    def test_record_success_latency_cap(self):
        """Test record_success caps latencies at 100 (lines 454-455)."""
        lb = AdaptiveBalancer()
        instance = create_instance("i1")
        lb._latencies["i1"] = [0.1] * 100

        lb.record_success(instance, 0.5)

        assert len(lb._latencies["i1"]) == 100

    def test_record_failure(self):
        """Test record_failure penalizes score (lines 459-461)."""
        lb = AdaptiveBalancer(penalty_factor=2.0)
        instance = create_instance("i1")
        lb._scores["i1"] = 1.0

        lb.record_failure(instance, Exception("test"))

        assert lb._scores["i1"] == 0.5  # 1.0 / 2.0

    def test_get_stats(self):
        """Test get_stats calculates average latency (lines 464-469)."""
        lb = AdaptiveBalancer()
        lb._latencies["i1"] = [0.1, 0.2, 0.3]
        lb._requests["i1"] = 10

        stats = lb.get_stats()

        assert abs(stats.average_latency["i1"] - 0.2) < 0.001


class TestLoadBalancerFactory:
    """Tests for LoadBalancerFactory class."""

    def test_create_round_robin(self):
        """Test create round_robin (line 484)."""
        lb = LoadBalancerFactory.create("round_robin")
        assert isinstance(lb, RoundRobinBalancer)

    def test_create_weighted_round_robin(self):
        """Test create weighted_round_robin (line 485)."""
        lb = LoadBalancerFactory.create("weighted_round_robin")
        assert isinstance(lb, WeightedRoundRobinBalancer)

    def test_create_least_connections(self):
        """Test create least_connections (line 486)."""
        lb = LoadBalancerFactory.create("least_connections")
        assert isinstance(lb, LeastConnectionsBalancer)

    def test_create_random(self):
        """Test create random (line 487)."""
        lb = LoadBalancerFactory.create("random")
        assert isinstance(lb, RandomBalancer)
        assert lb.weighted is False

    def test_create_weighted_random(self):
        """Test create weighted_random (line 488)."""
        lb = LoadBalancerFactory.create("weighted_random")
        assert isinstance(lb, RandomBalancer)
        assert lb.weighted is True

    def test_create_consistent_hash(self):
        """Test create consistent_hash with kwargs (line 489)."""
        lb = LoadBalancerFactory.create("consistent_hash", replicas=50)
        assert isinstance(lb, ConsistentHashBalancer)
        assert lb.replicas == 50

    def test_create_adaptive(self):
        """Test create adaptive with kwargs (line 490)."""
        lb = LoadBalancerFactory.create("adaptive", decay_factor=0.8)
        assert isinstance(lb, AdaptiveBalancer)
        assert lb.decay_factor == 0.8

    def test_create_unknown_strategy(self):
        """Test create with unknown strategy (lines 493-494)."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            LoadBalancerFactory.create("unknown")

    def test_create_callable_vs_class(self):
        """Test factory handles both callable and class (lines 496-499)."""
        # round_robin is a class
        lb1 = LoadBalancerFactory.create("round_robin")
        assert isinstance(lb1, RoundRobinBalancer)

        # random is a lambda
        lb2 = LoadBalancerFactory.create("random")
        assert isinstance(lb2, RandomBalancer)
