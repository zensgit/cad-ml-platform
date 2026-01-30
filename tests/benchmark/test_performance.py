"""Performance benchmarking suite.

Features:
- API endpoint benchmarks
- Model inference benchmarks
- Database query benchmarks
- Memory usage tracking
- Latency percentiles
"""

from __future__ import annotations

import asyncio
import gc
import statistics
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import pytest


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""

    name: str
    iterations: int
    total_time_s: float
    mean_time_ms: float
    median_time_ms: float
    std_dev_ms: float
    min_time_ms: float
    max_time_ms: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    throughput_rps: float
    memory_peak_mb: float = 0.0
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "total_time_s": round(self.total_time_s, 3),
            "mean_time_ms": round(self.mean_time_ms, 3),
            "median_time_ms": round(self.median_time_ms, 3),
            "std_dev_ms": round(self.std_dev_ms, 3),
            "min_time_ms": round(self.min_time_ms, 3),
            "max_time_ms": round(self.max_time_ms, 3),
            "p50_ms": round(self.p50_ms, 3),
            "p90_ms": round(self.p90_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "throughput_rps": round(self.throughput_rps, 2),
            "memory_peak_mb": round(self.memory_peak_mb, 2),
            "errors": self.errors,
        }


def percentile(data: List[float], p: float) -> float:
    """Calculate percentile of a list."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_data) else f
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


class Benchmark:
    """Benchmark runner."""

    def __init__(
        self,
        name: str,
        iterations: int = 100,
        warmup: int = 10,
        track_memory: bool = True,
    ):
        self.name = name
        self.iterations = iterations
        self.warmup = warmup
        self.track_memory = track_memory
        self._timings: List[float] = []
        self._errors = 0

    def run_sync(self, func: Callable, *args: Any, **kwargs: Any) -> BenchmarkResult:
        """Run a synchronous benchmark."""
        # Warmup
        for _ in range(self.warmup):
            try:
                func(*args, **kwargs)
            except Exception:
                pass

        # Force GC before benchmark
        gc.collect()

        # Start memory tracking
        if self.track_memory:
            tracemalloc.start()

        self._timings = []
        self._errors = 0
        total_start = time.perf_counter()

        for _ in range(self.iterations):
            start = time.perf_counter()
            try:
                func(*args, **kwargs)
            except Exception:
                self._errors += 1
            end = time.perf_counter()
            self._timings.append((end - start) * 1000)  # Convert to ms

        total_time = time.perf_counter() - total_start

        # Get memory stats
        memory_peak = 0.0
        if self.track_memory:
            _, peak = tracemalloc.get_traced_memory()
            memory_peak = peak / (1024 * 1024)  # Convert to MB
            tracemalloc.stop()

        return self._create_result(total_time, memory_peak)

    async def run_async(
        self, func: Callable, *args: Any, **kwargs: Any
    ) -> BenchmarkResult:
        """Run an asynchronous benchmark."""
        # Warmup
        for _ in range(self.warmup):
            try:
                await func(*args, **kwargs)
            except Exception:
                pass

        # Force GC before benchmark
        gc.collect()

        # Start memory tracking
        if self.track_memory:
            tracemalloc.start()

        self._timings = []
        self._errors = 0
        total_start = time.perf_counter()

        for _ in range(self.iterations):
            start = time.perf_counter()
            try:
                await func(*args, **kwargs)
            except Exception:
                self._errors += 1
            end = time.perf_counter()
            self._timings.append((end - start) * 1000)

        total_time = time.perf_counter() - total_start

        # Get memory stats
        memory_peak = 0.0
        if self.track_memory:
            _, peak = tracemalloc.get_traced_memory()
            memory_peak = peak / (1024 * 1024)
            tracemalloc.stop()

        return self._create_result(total_time, memory_peak)

    def _create_result(self, total_time: float, memory_peak: float) -> BenchmarkResult:
        """Create benchmark result from timings."""
        if not self._timings:
            return BenchmarkResult(
                name=self.name,
                iterations=0,
                total_time_s=0,
                mean_time_ms=0,
                median_time_ms=0,
                std_dev_ms=0,
                min_time_ms=0,
                max_time_ms=0,
                p50_ms=0,
                p90_ms=0,
                p95_ms=0,
                p99_ms=0,
                throughput_rps=0,
                memory_peak_mb=memory_peak,
                errors=self._errors,
            )

        return BenchmarkResult(
            name=self.name,
            iterations=self.iterations,
            total_time_s=total_time,
            mean_time_ms=statistics.mean(self._timings),
            median_time_ms=statistics.median(self._timings),
            std_dev_ms=statistics.stdev(self._timings) if len(self._timings) > 1 else 0,
            min_time_ms=min(self._timings),
            max_time_ms=max(self._timings),
            p50_ms=percentile(self._timings, 50),
            p90_ms=percentile(self._timings, 90),
            p95_ms=percentile(self._timings, 95),
            p99_ms=percentile(self._timings, 99),
            throughput_rps=self.iterations / total_time if total_time > 0 else 0,
            memory_peak_mb=memory_peak,
            errors=self._errors,
        )


# ============================================
# Benchmark Tests
# ============================================


class TestAPIBenchmarks:
    """API endpoint benchmarks."""

    @pytest.mark.benchmark
    def test_health_endpoint_benchmark(self):
        """Benchmark health check endpoint."""
        from fastapi.testclient import TestClient

        from src.main import app

        client = TestClient(app)

        def call_health():
            client.get("/health")

        bench = Benchmark("health_endpoint", iterations=1000, warmup=50)
        result = bench.run_sync(call_health)

        assert result.mean_time_ms < 10, f"Health check too slow: {result.mean_time_ms}ms"
        assert result.throughput_rps > 100, f"Throughput too low: {result.throughput_rps} RPS"

        print(f"\n{result.name}: {result.throughput_rps:.0f} RPS, p99={result.p99_ms:.2f}ms")


class TestModelBenchmarks:
    """Model inference benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_rate_limiter_benchmark(self):
        """Benchmark rate limiter performance."""
        from unittest.mock import patch

        from src.api.middleware.rate_limiting import RateLimitConfig, SlidingWindowRateLimiter

        limiter = SlidingWindowRateLimiter()
        config = RateLimitConfig(max_requests=100000, window_seconds=60)

        async def check_rate():
            with patch("src.api.middleware.rate_limiting.get_client", return_value=None):
                await limiter.check(f"user-{id(check_rate)}", config)

        bench = Benchmark("rate_limiter", iterations=10000, warmup=100)
        result = await bench.run_async(check_rate)

        assert result.throughput_rps > 10000, f"Rate limiter too slow: {result.throughput_rps} RPS"

        print(f"\n{result.name}: {result.throughput_rps:.0f} RPS, p99={result.p99_ms:.3f}ms")

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_audit_logger_benchmark(self):
        """Benchmark audit logger performance."""
        from src.core.audit.service import (
            AuditAction,
            AuditActor,
            AuditLogger,
            MemoryAuditStorage,
        )

        storage = MemoryAuditStorage(max_events=100000)
        logger = AuditLogger(storage=storage, batch_size=1000)
        actor = AuditActor(id="bench-user")

        async def log_event():
            await logger.log(AuditAction.API_CALL, actor=actor, outcome="success")

        bench = Benchmark("audit_logger", iterations=10000, warmup=100)
        result = await bench.run_async(log_event)

        assert result.throughput_rps > 50000, f"Audit logger too slow: {result.throughput_rps} RPS"

        print(f"\n{result.name}: {result.throughput_rps:.0f} RPS, p99={result.p99_ms:.3f}ms")


class TestCacheBenchmarks:
    """Cache operation benchmarks."""

    @pytest.mark.benchmark
    def test_lru_cache_benchmark(self):
        """Benchmark LRU cache operations."""
        from functools import lru_cache

        @lru_cache(maxsize=1000)
        def cached_func(x: int) -> int:
            return x * 2

        def run_cache_ops():
            for i in range(100):
                cached_func(i % 50)  # 50% hit rate

        bench = Benchmark("lru_cache", iterations=10000, warmup=100)
        result = bench.run_sync(run_cache_ops)

        print(f"\n{result.name}: {result.throughput_rps:.0f} RPS, p99={result.p99_ms:.3f}ms")


class TestWebSocketBenchmarks:
    """WebSocket benchmarks."""

    @pytest.mark.benchmark
    @pytest.mark.asyncio
    async def test_websocket_manager_benchmark(self):
        """Benchmark WebSocket message routing."""
        from unittest.mock import AsyncMock

        from src.core.websocket.manager import WebSocketManager

        manager = WebSocketManager()

        # Setup connections
        for i in range(100):
            ws = AsyncMock()
            await manager.connect(ws, user_id=f"user-{i}")

        async def send_message():
            await manager.send_to_user("user-50", {"test": "message"})

        bench = Benchmark("websocket_routing", iterations=10000, warmup=100)
        result = await bench.run_async(send_message)

        assert result.throughput_rps > 10000

        print(f"\n{result.name}: {result.throughput_rps:.0f} RPS, p99={result.p99_ms:.3f}ms")


def run_all_benchmarks():
    """Run all benchmarks and print summary."""
    print("\n" + "=" * 60)
    print("Performance Benchmark Suite")
    print("=" * 60)

    # Run with pytest
    pytest.main([__file__, "-v", "-m", "benchmark", "--tb=short"])


if __name__ == "__main__":
    run_all_benchmarks()
