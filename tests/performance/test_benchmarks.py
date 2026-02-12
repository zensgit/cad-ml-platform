"""
Performance Benchmark Tests for CAD Assistant Enterprise Features.

Measures performance of:
- Streaming response throughput
- Multi-model selection latency
- Multi-tenant quota checking
- RBAC permission resolution
- Cache operations
"""

import pytest
import time
import asyncio
import statistics
from typing import List, Dict, Any
from unittest.mock import MagicMock

# Module imports
from src.core.assistant.streaming import StreamingResponse, StreamEventType
from src.core.assistant.multi_model import (
    ModelSelector, ModelConfig, ModelProvider, LoadBalancingStrategy
)
from src.core.assistant.multi_tenant import (
    TenantManager, TenantTier, TenantQuota, Tenant, TenantContext
)
from src.core.assistant.rbac import RBACManager, Permission
from src.core.assistant.caching import LRUCache


class BenchmarkResult:
    """Container for benchmark results."""

    def __init__(self, name: str, times: List[float]):
        self.name = name
        self.times = times

    @property
    def mean(self) -> float:
        return statistics.mean(self.times)

    @property
    def median(self) -> float:
        return statistics.median(self.times)

    @property
    def std_dev(self) -> float:
        return statistics.stdev(self.times) if len(self.times) > 1 else 0

    @property
    def min(self) -> float:
        return min(self.times)

    @property
    def max(self) -> float:
        return max(self.times)

    @property
    def ops_per_sec(self) -> float:
        return 1.0 / self.mean if self.mean > 0 else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": len(self.times),
            "mean_ms": self.mean * 1000,
            "median_ms": self.median * 1000,
            "std_dev_ms": self.std_dev * 1000,
            "min_ms": self.min * 1000,
            "max_ms": self.max * 1000,
            "ops_per_sec": self.ops_per_sec,
        }


def benchmark(iterations: int = 1000):
    """Decorator to benchmark a function."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)
            return BenchmarkResult(func.__name__, times)
        return wrapper
    return decorator


class TestStreamingPerformance:
    """Performance tests for streaming responses."""

    @pytest.mark.asyncio
    async def test_streaming_throughput(self):
        """Benchmark streaming throughput."""
        streamer = StreamingResponse(chunk_size=100, delay_ms=0)
        text = "A" * 10000  # 10KB text

        iterations = 100
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            chunks = []
            async for event in streamer.stream_text(text):
                if event.event_type == StreamEventType.CHUNK:
                    chunks.append(event.data["text"])
            end = time.perf_counter()
            times.append(end - start)

        result = BenchmarkResult("streaming_throughput", times)
        print(f"\nStreaming Throughput: {result.mean * 1000:.3f}ms avg, "
              f"{result.ops_per_sec:.1f} ops/sec")

        # Performance assertion: should process 10KB in < 50ms
        assert result.mean < 0.05

    @pytest.mark.asyncio
    async def test_chunk_generation_speed(self):
        """Benchmark individual chunk generation."""
        streamer = StreamingResponse(chunk_size=50, delay_ms=0)
        text = "A" * 500

        iterations = 500
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            count = 0
            async for event in streamer.stream_text(text):
                if event.event_type == StreamEventType.CHUNK:
                    count += 1
            end = time.perf_counter()
            times.append(end - start)

        result = BenchmarkResult("chunk_generation", times)
        print(f"\nChunk Generation: {result.mean * 1000:.3f}ms avg, "
              f"{result.ops_per_sec:.1f} ops/sec")

        # Should generate chunks very fast
        assert result.mean < 0.01


class TestModelSelectorPerformance:
    """Performance tests for model selection."""

    @pytest.fixture
    def selector_with_models(self):
        """Create selector with multiple models."""
        selector = ModelSelector(strategy=LoadBalancingStrategy.PRIORITY)
        for i, provider in enumerate(ModelProvider):
            selector.register_model(ModelConfig(
                provider=provider,
                model_name=f"model-{provider.value}",
                priority=i + 1,
                weight=1.0 / (i + 1),
            ))
        return selector

    def test_model_selection_speed(self, selector_with_models):
        """Benchmark model selection speed."""
        iterations = 10000
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            selector_with_models.select_model()
            end = time.perf_counter()
            times.append(end - start)

        result = BenchmarkResult("model_selection", times)
        print(f"\nModel Selection: {result.mean * 1000000:.3f}µs avg, "
              f"{result.ops_per_sec:.0f} ops/sec")

        # Should select in < 100µs
        assert result.mean < 0.0001

    def test_fallback_list_generation(self, selector_with_models):
        """Benchmark fallback list generation."""
        iterations = 10000
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            selector_with_models.select_with_fallback()
            end = time.perf_counter()
            times.append(end - start)

        result = BenchmarkResult("fallback_generation", times)
        print(f"\nFallback Generation: {result.mean * 1000000:.3f}µs avg, "
              f"{result.ops_per_sec:.0f} ops/sec")

        assert result.mean < 0.0001

    def test_health_update_speed(self, selector_with_models):
        """Benchmark health status update."""
        iterations = 10000
        times = []

        from src.core.assistant.multi_model import ModelStatus

        for i in range(iterations):
            provider = list(ModelProvider)[i % len(ModelProvider)]
            start = time.perf_counter()
            selector_with_models.update_health(
                provider,
                ModelStatus.AVAILABLE,
                latency_ms=100 + i % 1000,
            )
            end = time.perf_counter()
            times.append(end - start)

        result = BenchmarkResult("health_update", times)
        print(f"\nHealth Update: {result.mean * 1000000:.3f}µs avg, "
              f"{result.ops_per_sec:.0f} ops/sec")

        assert result.mean < 0.0001


class TestTenantPerformance:
    """Performance tests for multi-tenant operations."""

    def test_quota_check_speed(self):
        """Benchmark quota checking speed."""
        tenant = Tenant(
            id="perf-tenant",
            name="Performance Test",
            tier=TenantTier.PROFESSIONAL,
            quota=TenantQuota.for_tier(TenantTier.PROFESSIONAL),
        )

        iterations = 100000
        times = []
        resources = ["conversations", "messages", "knowledge", "api_calls"]

        for i in range(iterations):
            resource = resources[i % len(resources)]
            start = time.perf_counter()
            tenant.check_quota(resource)
            end = time.perf_counter()
            times.append(end - start)

        result = BenchmarkResult("quota_check", times)
        print(f"\nQuota Check: {result.mean * 1000000:.3f}µs avg, "
              f"{result.ops_per_sec:.0f} ops/sec")

        # Should check quota in < 10µs
        assert result.mean < 0.00001

    def test_tenant_context_speed(self):
        """Benchmark tenant context operations."""
        tenant = Tenant(
            id="ctx-tenant",
            name="Context Test",
            tier=TenantTier.BASIC,
            quota=TenantQuota.for_tier(TenantTier.BASIC),
        )

        iterations = 10000
        times = []

        for _ in range(iterations):
            start = time.perf_counter()
            with TenantContext(tenant) as ctx:
                ctx.check_quota("messages")
            end = time.perf_counter()
            times.append(end - start)

        result = BenchmarkResult("tenant_context", times)
        print(f"\nTenant Context: {result.mean * 1000000:.3f}µs avg, "
              f"{result.ops_per_sec:.0f} ops/sec")

        assert result.mean < 0.0001

    def test_tenant_manager_lookup(self):
        """Benchmark tenant lookup speed."""
        manager = TenantManager()

        # Create 100 tenants
        tenant_ids = []
        for i in range(100):
            tid = manager.create_tenant(f"Tenant {i}", TenantTier.BASIC)
            tenant_ids.append(tid)

        iterations = 50000
        times = []

        for i in range(iterations):
            tid = tenant_ids[i % len(tenant_ids)]
            start = time.perf_counter()
            manager.get_tenant(tid)
            end = time.perf_counter()
            times.append(end - start)

        result = BenchmarkResult("tenant_lookup", times)
        print(f"\nTenant Lookup: {result.mean * 1000000:.3f}µs avg, "
              f"{result.ops_per_sec:.0f} ops/sec")

        assert result.mean < 0.00001


class TestRBACPerformance:
    """Performance tests for RBAC operations."""

    @pytest.fixture
    def rbac_with_users(self):
        """Create RBAC with many users and roles."""
        rbac = RBACManager()
        rbac.create_default_roles()

        # Create 100 users with various roles
        roles = ["guest", "user", "engineer", "manager", "admin"]
        for i in range(100):
            user_id = f"user-{i}"
            rbac.create_user(user_id, f"user{i}")
            rbac.assign_role(user_id, roles[i % len(roles)])

        return rbac

    def test_permission_check_speed(self, rbac_with_users):
        """Benchmark permission checking speed."""
        iterations = 50000
        times = []
        permissions = list(Permission)

        for i in range(iterations):
            user_id = f"user-{i % 100}"
            perm = permissions[i % len(permissions)]
            start = time.perf_counter()
            rbac_with_users.check_permission(user_id, perm)
            end = time.perf_counter()
            times.append(end - start)

        result = BenchmarkResult("permission_check", times)
        print(f"\nPermission Check: {result.mean * 1000000:.3f}µs avg, "
              f"{result.ops_per_sec:.0f} ops/sec")

        # Should check permission in < 50µs
        assert result.mean < 0.00005

    def test_effective_permissions_speed(self, rbac_with_users):
        """Benchmark getting all effective permissions."""
        iterations = 10000
        times = []

        for i in range(iterations):
            user_id = f"user-{i % 100}"
            start = time.perf_counter()
            rbac_with_users.get_user_permissions(user_id)
            end = time.perf_counter()
            times.append(end - start)

        result = BenchmarkResult("effective_permissions", times)
        print(f"\nEffective Permissions: {result.mean * 1000000:.3f}µs avg, "
              f"{result.ops_per_sec:.0f} ops/sec")

        assert result.mean < 0.0001

    def test_role_inheritance_resolution(self, rbac_with_users):
        """Benchmark role inheritance resolution."""
        iterations = 10000
        times = []
        roles = ["admin", "manager", "engineer", "user", "guest"]

        for i in range(iterations):
            role = roles[i % len(roles)]
            start = time.perf_counter()
            rbac_with_users.get_effective_permissions(role)
            end = time.perf_counter()
            times.append(end - start)

        result = BenchmarkResult("role_inheritance", times)
        print(f"\nRole Inheritance: {result.mean * 1000000:.3f}µs avg, "
              f"{result.ops_per_sec:.0f} ops/sec")

        assert result.mean < 0.0001


class TestCachePerformance:
    """Performance tests for caching operations."""

    def test_cache_get_set_speed(self):
        """Benchmark cache get/set operations."""
        cache = LRUCache(max_size=10000)

        # Pre-populate cache
        for i in range(5000):
            cache.set(f"key-{i}", f"value-{i}")

        iterations = 100000
        times = []

        for i in range(iterations):
            key = f"key-{i % 10000}"
            start = time.perf_counter()
            if i % 2 == 0:
                cache.get(key)
            else:
                cache.set(key, f"value-{i}")
            end = time.perf_counter()
            times.append(end - start)

        result = BenchmarkResult("cache_operations", times)
        print(f"\nCache Operations: {result.mean * 1000000:.3f}µs avg, "
              f"{result.ops_per_sec:.0f} ops/sec")

        # Should be very fast
        assert result.mean < 0.00001

    def test_cache_hit_rate_under_load(self):
        """Test cache hit rate with realistic access patterns."""
        cache = LRUCache(max_size=1000)

        # Simulate zipf-like access pattern (some keys accessed much more often)
        import random
        random.seed(42)

        hits = 0
        misses = 0

        for i in range(10000):
            # 80% of accesses to 20% of keys (zipf-like)
            if random.random() < 0.8:
                key = f"hot-{i % 200}"
            else:
                key = f"cold-{random.randint(0, 5000)}"

            result = cache.get(key)
            if result is not None:
                hits += 1
            else:
                misses += 1
                cache.set(key, f"value-{key}")

        hit_rate = hits / (hits + misses)
        print(f"\nCache Hit Rate: {hit_rate:.2%} ({hits} hits, {misses} misses)")

        # With zipf pattern and LRU, expect reasonable hit rate
        assert hit_rate > 0.3


class TestCombinedPerformance:
    """Combined performance tests simulating real workloads."""

    @pytest.mark.asyncio
    async def test_full_request_simulation(self):
        """Simulate full request processing pipeline."""
        # Setup
        tenant_mgr = TenantManager()
        tenant_id = tenant_mgr.create_tenant("Perf Corp", TenantTier.PROFESSIONAL)
        tenant = tenant_mgr.get_tenant(tenant_id)

        rbac = RBACManager()
        rbac.create_default_roles()
        rbac.create_user("perf-user", "perfuser", tenant_id=tenant_id)
        rbac.assign_role("perf-user", "engineer")

        streamer = StreamingResponse(chunk_size=100, delay_ms=0)

        iterations = 100
        times = []

        for _ in range(iterations):
            start = time.perf_counter()

            # 1. Check tenant context and quota
            with TenantContext(tenant) as ctx:
                ctx.check_quota("messages")

                # 2. Check permissions
                rbac.check_permission("perf-user", Permission.CONVERSATION_CREATE)

                # 3. Stream response
                response_text = "Sample response " * 50
                async for event in streamer.stream_text(response_text):
                    pass

                # 4. Use quota
                ctx.use_quota("messages")

            end = time.perf_counter()
            times.append(end - start)

        result = BenchmarkResult("full_request", times)
        print(f"\nFull Request Pipeline: {result.mean * 1000:.3f}ms avg, "
              f"{result.ops_per_sec:.1f} ops/sec")

        # Full pipeline should complete in < 10ms
        assert result.mean < 0.01


def run_all_benchmarks():
    """Run all benchmarks and generate summary report."""
    print("=" * 60)
    print("CAD Assistant Performance Benchmark Report")
    print("=" * 60)

    # Note: This is a helper function, tests are run via pytest
    print("\nRun with: pytest tests/performance/test_benchmarks.py -v -s")
