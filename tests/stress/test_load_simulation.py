"""
Stress Tests for CAD Assistant Platform.

Simulates production-level load to verify:
- Concurrent request handling
- Memory stability under load
- Throughput and latency under pressure
- Resource cleanup and leak detection
"""

import asyncio
import gc
import os
import statistics
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Tuple
from unittest.mock import MagicMock

import pytest

# Core imports
from src.core.assistant.multi_tenant import (
    TenantManager, TenantTier, Tenant, TenantContext
)
from src.core.assistant.rbac import RBACManager, Permission, AccessContext
from src.core.assistant.multi_model import (
    ModelProvider, ModelConfig, ModelSelector,
    MultiModelAssistant, LoadBalancingStrategy
)
from src.core.assistant.streaming import StreamingResponse, StreamEventType
from src.core.assistant.caching import LRUCache

IS_CI_ENV = (
    os.getenv("CI", "").strip().lower() in {"1", "true", "yes", "on"}
    or os.getenv("GITHUB_ACTIONS", "").strip().lower() in {"1", "true", "yes", "on"}
)


def _throughput_threshold(env_name: str, *, default_local: float, default_ci: float) -> float:
    """Resolve throughput threshold with optional env override."""
    raw = os.getenv(env_name)
    if raw:
        try:
            value = float(raw)
            if value > 0:
                return value
        except ValueError:
            pass
    return default_ci if IS_CI_ENV else default_local


class LoadTestResult:
    """Container for load test results."""

    def __init__(self, name: str):
        self.name = name
        self.latencies: List[float] = []
        self.errors: List[str] = []
        self.start_time: float = 0
        self.end_time: float = 0

    def add_latency(self, latency: float):
        self.latencies.append(latency)

    def add_error(self, error: str):
        self.errors.append(error)

    @property
    def total_requests(self) -> int:
        return len(self.latencies) + len(self.errors)

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0
        return len(self.latencies) / self.total_requests

    @property
    def avg_latency(self) -> float:
        if not self.latencies:
            return 0
        return statistics.mean(self.latencies)

    @property
    def p50_latency(self) -> float:
        if not self.latencies:
            return 0
        return statistics.median(self.latencies)

    @property
    def p95_latency(self) -> float:
        if len(self.latencies) < 2:
            return self.avg_latency
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[idx]

    @property
    def p99_latency(self) -> float:
        if len(self.latencies) < 2:
            return self.avg_latency
        sorted_latencies = sorted(self.latencies)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[idx]

    @property
    def throughput(self) -> float:
        duration = self.end_time - self.start_time
        if duration == 0:
            return 0
        return len(self.latencies) / duration

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_requests": self.total_requests,
            "success_count": len(self.latencies),
            "error_count": len(self.errors),
            "success_rate": f"{self.success_rate:.2%}",
            "avg_latency_ms": f"{self.avg_latency * 1000:.2f}",
            "p50_latency_ms": f"{self.p50_latency * 1000:.2f}",
            "p95_latency_ms": f"{self.p95_latency * 1000:.2f}",
            "p99_latency_ms": f"{self.p99_latency * 1000:.2f}",
            "throughput_rps": f"{self.throughput:.1f}",
        }


class TestConcurrentLoad:
    """Concurrent load tests."""

    @pytest.fixture
    def stress_setup(self):
        """Set up for stress testing."""
        tenant_mgr = TenantManager()
        tenant_id = tenant_mgr.create_tenant("Stress Test Corp", TenantTier.ENTERPRISE)

        rbac = RBACManager()
        rbac.create_default_roles()

        # Create multiple users
        for i in range(100):
            rbac.create_user(f"user_{i}", f"user_{i}", tenant_id=tenant_id)
            rbac.assign_role(f"user_{i}", "engineer")

        return {
            "tenant_mgr": tenant_mgr,
            "rbac": rbac,
            "tenant_id": tenant_id,
        }

    def test_concurrent_tenant_lookups(self, stress_setup):
        """Test concurrent tenant lookup performance."""
        tenant_mgr = stress_setup["tenant_mgr"]
        tenant_id = stress_setup["tenant_id"]

        result = LoadTestResult("concurrent_tenant_lookups")
        num_requests = 10000
        num_threads = 10

        def lookup_tenant():
            start = time.perf_counter()
            try:
                tenant = tenant_mgr.get_tenant(tenant_id)
                assert tenant is not None
                result.add_latency(time.perf_counter() - start)
            except Exception as e:
                result.add_error(str(e))

        result.start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(lookup_tenant) for _ in range(num_requests)]
            for f in futures:
                f.result()
        result.end_time = time.perf_counter()

        summary = result.summary()
        print(f"\n{summary}")

        # Assertions
        assert result.success_rate > 0.99  # 99% success rate
        assert result.avg_latency < 0.001  # < 1ms average
        assert result.throughput > 1000  # > 1000 RPS

    def test_concurrent_permission_checks(self, stress_setup):
        """Test concurrent permission check performance."""
        rbac = stress_setup["rbac"]

        result = LoadTestResult("concurrent_permission_checks")
        num_requests = 10000
        num_threads = 10
        permissions = list(Permission)

        def check_permission():
            start = time.perf_counter()
            try:
                user_id = f"user_{hash(time.time()) % 100}"
                perm = permissions[hash(time.time()) % len(permissions)]
                rbac.check_permission(user_id, perm)
                result.add_latency(time.perf_counter() - start)
            except Exception as e:
                result.add_error(str(e))

        result.start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(check_permission) for _ in range(num_requests)]
            for f in futures:
                f.result()
        result.end_time = time.perf_counter()

        summary = result.summary()
        print(f"\n{summary}")

        assert result.success_rate > 0.99
        assert result.avg_latency < 0.001  # < 1ms
        assert result.throughput > 500

    def test_concurrent_quota_operations(self, stress_setup):
        """Test concurrent quota check and use operations."""
        tenant_mgr = stress_setup["tenant_mgr"]
        tenant_id = stress_setup["tenant_id"]
        tenant = tenant_mgr.get_tenant(tenant_id)

        result = LoadTestResult("concurrent_quota_operations")
        num_requests = 5000
        num_threads = 10
        resources = ["conversations", "messages", "knowledge", "api_calls"]

        def quota_operation():
            start = time.perf_counter()
            try:
                resource = resources[hash(time.time()) % len(resources)]
                with TenantContext(tenant) as ctx:
                    ctx.check_quota(resource)
                result.add_latency(time.perf_counter() - start)
            except Exception as e:
                result.add_error(str(e))

        result.start_time = time.perf_counter()
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(quota_operation) for _ in range(num_requests)]
            for f in futures:
                f.result()
        result.end_time = time.perf_counter()

        summary = result.summary()
        print(f"\n{summary}")

        assert result.success_rate > 0.99
        assert result.avg_latency < 0.005  # < 5ms


class TestMemoryStability:
    """Memory stability and leak detection tests."""

    def test_cache_memory_stability(self):
        """Test cache doesn't leak memory under heavy use."""
        tracemalloc.start()

        cache = LRUCache(max_size=1000)

        # Warm up
        for i in range(1000):
            cache.set(f"key_{i}", f"value_{i}" * 100)

        snapshot1 = tracemalloc.take_snapshot()

        # Heavy usage
        for iteration in range(10):
            for i in range(10000):
                key = f"key_{i % 2000}"
                if i % 2 == 0:
                    cache.set(key, f"value_{i}" * 100)
                else:
                    cache.get(key)

            # Force garbage collection
            gc.collect()

        snapshot2 = tracemalloc.take_snapshot()

        # Compare memory
        top_stats = snapshot2.compare_to(snapshot1, 'lineno')

        # Check for significant memory growth
        total_diff = sum(stat.size_diff for stat in top_stats[:10])
        print(f"\nMemory diff after heavy cache use: {total_diff / 1024:.2f} KB")

        tracemalloc.stop()

        # Memory growth should be bounded
        assert total_diff < 10 * 1024 * 1024  # < 10MB growth

    def test_tenant_manager_memory_stability(self):
        """Test tenant manager doesn't leak memory."""
        tracemalloc.start()

        manager = TenantManager()

        # Create and delete tenants repeatedly
        for iteration in range(100):
            tenant_ids = []
            for i in range(10):
                tid = manager.create_tenant(f"Tenant_{iteration}_{i}", TenantTier.BASIC)
                tenant_ids.append(tid)

            for tid in tenant_ids:
                manager.delete_tenant(tid)

            gc.collect()

        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')

        total_memory = sum(stat.size for stat in top_stats[:10])
        print(f"\nTotal memory after tenant churn: {total_memory / 1024:.2f} KB")

        tracemalloc.stop()

        # Should not have excessive memory
        assert total_memory < 5 * 1024 * 1024  # < 5MB


class TestThroughputLimits:
    """Tests for throughput limits and capacity."""

    @pytest.mark.asyncio
    async def test_streaming_throughput(self):
        """Test streaming response throughput."""
        streamer = StreamingResponse(chunk_size=100, delay_ms=0)
        text = "A" * 10000  # 10KB

        result = LoadTestResult("streaming_throughput")
        num_iterations = 100

        result.start_time = time.perf_counter()
        for _ in range(num_iterations):
            start = time.perf_counter()
            chunks = []
            async for event in streamer.stream_text(text):
                if event.event_type == StreamEventType.CHUNK:
                    chunks.append(event.data["text"])
            result.add_latency(time.perf_counter() - start)
        result.end_time = time.perf_counter()

        summary = result.summary()
        print(f"\n{summary}")

        # Each 10KB should stream in < 100ms
        assert result.avg_latency < 0.1
        assert result.throughput > 10  # > 10 streams/sec

    def test_model_selector_throughput(self):
        """Test model selector throughput under load."""
        selector = ModelSelector(strategy=LoadBalancingStrategy.PRIORITY)

        # Register multiple models
        for i, provider in enumerate(ModelProvider):
            selector.register_model(ModelConfig(
                provider=provider,
                model_name=f"model-{provider.value}",
                priority=i + 1,
            ))

        result = LoadTestResult("model_selector_throughput")
        num_selections = 100000

        result.start_time = time.perf_counter()
        for _ in range(num_selections):
            start = time.perf_counter()
            selector.select_model()
            result.add_latency(time.perf_counter() - start)
        result.end_time = time.perf_counter()

        summary = result.summary()
        print(f"\n{summary}")

        min_rps = _throughput_threshold(
            "STRESS_MODEL_SELECTOR_MIN_RPS",
            default_local=100000.0,
            default_ci=50000.0,
        )
        assert result.throughput > min_rps

    def test_cache_throughput(self):
        """Test cache read/write throughput."""
        cache = LRUCache(max_size=10000)

        # Pre-populate
        for i in range(5000):
            cache.set(f"key_{i}", f"value_{i}")

        result = LoadTestResult("cache_throughput")
        num_operations = 100000

        result.start_time = time.perf_counter()
        for i in range(num_operations):
            start = time.perf_counter()
            key = f"key_{i % 10000}"
            if i % 2 == 0:
                cache.get(key)
            else:
                cache.set(key, f"value_{i}")
            result.add_latency(time.perf_counter() - start)
        result.end_time = time.perf_counter()

        summary = result.summary()
        print(f"\n{summary}")

        min_rps = _throughput_threshold(
            "STRESS_CACHE_MIN_RPS",
            default_local=500000.0,
            default_ci=250000.0,
        )
        assert result.throughput > min_rps


class TestSustainedLoad:
    """Tests for sustained load over time."""

    def test_sustained_permission_checks(self):
        """Test sustained permission check load over time."""
        rbac = RBACManager()
        rbac.create_default_roles()

        # Create users
        for i in range(50):
            rbac.create_user(f"user_{i}", f"user_{i}")
            rbac.assign_role(f"user_{i}", ["guest", "user", "engineer", "manager", "admin"][i % 5])

        result = LoadTestResult("sustained_permission_checks")
        duration_seconds = 5  # 5 second sustained load
        permissions = list(Permission)

        result.start_time = time.perf_counter()
        end_time = result.start_time + duration_seconds

        while time.perf_counter() < end_time:
            start = time.perf_counter()
            user_id = f"user_{hash(time.time()) % 50}"
            perm = permissions[hash(time.time()) % len(permissions)]
            rbac.check_permission(user_id, perm)
            result.add_latency(time.perf_counter() - start)

        result.end_time = time.perf_counter()

        summary = result.summary()
        print(f"\n{summary}")

        # Should maintain > 50K checks/sec for 5 seconds
        assert result.throughput > 50000
        # Latency should stay low
        assert result.p99_latency < 0.001  # < 1ms p99


class TestResourceCleanup:
    """Tests for proper resource cleanup."""

    def test_tenant_context_cleanup(self):
        """Test tenant context properly cleans up."""
        manager = TenantManager()
        tenant_id = manager.create_tenant("Cleanup Test", TenantTier.BASIC)
        tenant = manager.get_tenant(tenant_id)

        initial_refs = len(gc.get_referrers(tenant))

        # Create and exit many contexts
        for _ in range(1000):
            with TenantContext(tenant) as ctx:
                ctx.check_quota("messages")

        gc.collect()
        final_refs = len(gc.get_referrers(tenant))

        # Reference count should not grow significantly
        print(f"\nInitial refs: {initial_refs}, Final refs: {final_refs}")
        assert final_refs <= initial_refs + 10  # Allow small variance

    def test_access_context_cleanup(self):
        """Test access context properly cleans up."""
        rbac = RBACManager()
        rbac.create_default_roles()
        rbac.create_user("cleanup_user", "cleanup_user")
        rbac.assign_role("cleanup_user", "engineer")

        initial_gc_count = len(gc.get_objects())

        # Create and exit many contexts
        for _ in range(1000):
            with AccessContext(rbac, "cleanup_user") as ctx:
                ctx.can(Permission.CONVERSATION_CREATE)

        gc.collect()
        final_gc_count = len(gc.get_objects())

        growth = final_gc_count - initial_gc_count
        print(f"\nObject growth after 1000 contexts: {growth}")

        # Object count should not grow excessively
        assert growth < 1000  # Less than 1 object per context
