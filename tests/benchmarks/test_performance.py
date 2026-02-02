"""
Performance Benchmark Tests for CAD-ML Platform.

Tests cover:
- API endpoint response times
- ML pipeline throughput
- Memory usage patterns
- Concurrent request handling
"""

import asyncio
import io
import json
import os
import time
from typing import Dict, List

import pytest
import httpx


# Skip all benchmarks by default, run with: pytest tests/benchmarks -v --benchmark
pytestmark = pytest.mark.benchmark


@pytest.fixture(scope="module")
async def async_client():
    """Create async test client."""
    from src.main import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


@pytest.fixture
def sample_dxf_payload() -> bytes:
    """Create minimal DXF payload for testing."""
    return b"0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF\n"


@pytest.fixture
def api_headers() -> Dict[str, str]:
    """Standard API headers."""
    return {"x-api-key": os.getenv("API_KEY", "test")}


class TestAnalyzeEndpointPerformance:
    """Performance tests for /api/v1/analyze endpoint."""

    @pytest.mark.anyio
    async def test_analyze_single_request_latency(
        self, async_client, sample_dxf_payload, api_headers
    ):
        """Measure single request latency."""
        options = {"extract_features": True, "classify_parts": True}

        # Warm up to avoid cold-start penalties in latency check
        await async_client.post(
            "/api/v1/analyze/",
            files={"file": ("warmup.dxf", io.BytesIO(sample_dxf_payload), "application/dxf")},
            data={"options": json.dumps({"extract_features": False, "classify_parts": False})},
            headers=api_headers,
        )

        start = time.perf_counter()
        resp = await async_client.post(
            "/api/v1/analyze/",
            files={"file": ("test.dxf", io.BytesIO(sample_dxf_payload), "application/dxf")},
            data={"options": json.dumps(options)},
            headers=api_headers,
        )
        elapsed = time.perf_counter() - start

        assert resp.status_code == 200
        # Single request should complete within 2 seconds
        assert elapsed < 2.0, f"Request took {elapsed:.2f}s, expected < 2s"

    @pytest.mark.anyio
    async def test_analyze_throughput(self, async_client, sample_dxf_payload, api_headers):
        """Measure throughput with sequential requests."""
        options = {"extract_features": True, "classify_parts": False}
        num_requests = 10

        start = time.perf_counter()
        for _ in range(num_requests):
            resp = await async_client.post(
                "/api/v1/analyze/",
                files={"file": ("test.dxf", io.BytesIO(sample_dxf_payload), "application/dxf")},
                data={"options": json.dumps(options)},
                headers=api_headers,
            )
            assert resp.status_code == 200
        elapsed = time.perf_counter() - start

        throughput = num_requests / elapsed
        # Should handle at least 2 requests per second
        assert throughput >= 2.0, f"Throughput {throughput:.2f} req/s, expected >= 2"

    @pytest.mark.anyio
    async def test_analyze_concurrent_requests(self, async_client, sample_dxf_payload, api_headers):
        """Measure concurrent request handling."""
        options = {"extract_features": True, "classify_parts": False}
        num_concurrent = 5

        async def make_request():
            return await async_client.post(
                "/api/v1/analyze/",
                files={"file": ("test.dxf", io.BytesIO(sample_dxf_payload), "application/dxf")},
                data={"options": json.dumps(options)},
                headers=api_headers,
            )

        start = time.perf_counter()
        responses = await asyncio.gather(*[make_request() for _ in range(num_concurrent)])
        elapsed = time.perf_counter() - start

        # All requests should succeed
        for resp in responses:
            assert resp.status_code == 200

        # Concurrent requests should complete faster than sequential
        # (allowing some overhead for thread coordination)
        assert elapsed < num_concurrent * 1.0, f"Concurrent took {elapsed:.2f}s"


class TestHealthEndpointPerformance:
    """Performance tests for health check endpoint."""

    @pytest.mark.anyio
    async def test_health_check_latency(self, async_client):
        """Health check should be extremely fast."""
        start = time.perf_counter()
        resp = await async_client.get("/health")
        elapsed = time.perf_counter() - start

        assert resp.status_code == 200
        # Health check should complete in < 100ms
        assert elapsed < 0.1, f"Health check took {elapsed*1000:.1f}ms, expected < 100ms"

    @pytest.mark.anyio
    async def test_health_check_burst(self, async_client):
        """Health check should handle burst traffic."""
        num_requests = 100

        start = time.perf_counter()
        for _ in range(num_requests):
            resp = await async_client.get("/health")
            assert resp.status_code == 200
        elapsed = time.perf_counter() - start

        # Should handle 100+ health checks per second
        throughput = num_requests / elapsed
        assert throughput >= 100, f"Health throughput {throughput:.1f}/s, expected >= 100"


class TestMLPipelinePerformance:
    """Performance tests for ML pipeline components."""

    def test_fusion_creation_time(self):
        """Measure fusion system initialization time."""
        from src.ml.hybrid import MultiSourceFusion, FusionStrategy

        start = time.perf_counter()
        fusion = MultiSourceFusion(default_strategy=FusionStrategy.WEIGHTED_AVERAGE)
        elapsed = time.perf_counter() - start

        assert fusion is not None
        # Fusion creation should be instant
        assert elapsed < 0.1, f"Fusion creation took {elapsed*1000:.1f}ms"

    def test_pipeline_creation_time(self):
        """Measure pipeline creation time."""
        from src.ml.pipeline import Pipeline, PipelineConfig

        start = time.perf_counter()
        config = PipelineConfig(name="benchmark_pipeline")
        pipeline = Pipeline(config=config)
        elapsed = time.perf_counter() - start

        assert pipeline is not None
        # Pipeline creation should be instant
        assert elapsed < 0.1, f"Pipeline creation took {elapsed*1000:.1f}ms"

    def test_geometry_extractor_init(self):
        """Measure geometry extractor initialization."""
        from src.core.cad.geometry import GeometryExtractor

        start = time.perf_counter()
        extractor = GeometryExtractor()
        elapsed = time.perf_counter() - start

        assert extractor is not None
        # Extractor init should be fast
        assert elapsed < 0.5, f"Extractor init took {elapsed*1000:.1f}ms"


class TestMonitoringPerformance:
    """Performance tests for monitoring components."""

    def test_metrics_collector_creation(self):
        """Measure metrics collector initialization."""
        from src.ml.monitoring import MetricsCollector

        start = time.perf_counter()
        collector = MetricsCollector()
        elapsed = time.perf_counter() - start

        assert collector is not None
        # Collector creation should be instant
        assert elapsed < 0.1, f"Collector creation took {elapsed*1000:.1f}ms"

    def test_drift_monitor_creation(self):
        """Measure drift monitor initialization."""
        from src.ml.monitoring.drift import DriftMonitor

        start = time.perf_counter()
        monitor = DriftMonitor(window_size=100)
        elapsed = time.perf_counter() - start

        assert monitor is not None
        # Monitor creation should be fast
        assert elapsed < 0.1, f"Monitor creation took {elapsed*1000:.1f}ms"


class TestMemoryUsage:
    """Memory usage tests."""

    @pytest.mark.anyio
    async def test_analyze_memory_stability(
        self, async_client, sample_dxf_payload, api_headers
    ):
        """Check memory doesn't grow unboundedly during requests."""
        import gc
        import tracemalloc

        options = {"extract_features": True, "classify_parts": True}

        # Warm up
        for _ in range(3):
            await async_client.post(
                "/api/v1/analyze/",
                files={"file": ("test.dxf", io.BytesIO(sample_dxf_payload), "application/dxf")},
                data={"options": json.dumps(options)},
                headers=api_headers,
            )

        gc.collect()
        tracemalloc.start()
        baseline_snapshot = tracemalloc.take_snapshot()

        # Run more requests
        for _ in range(10):
            await async_client.post(
                "/api/v1/analyze/",
                files={"file": ("test.dxf", io.BytesIO(sample_dxf_payload), "application/dxf")},
                data={"options": json.dumps(options)},
                headers=api_headers,
            )

        gc.collect()
        final_snapshot = tracemalloc.take_snapshot()
        stats = final_snapshot.compare_to(baseline_snapshot, "filename")
        total_diff = sum(stat.size_diff for stat in stats)
        tracemalloc.stop()

        # Allow small allocation growth (10MB) to reduce flakiness
        assert total_diff < 10 * 1024 * 1024, f"Memory grew by {total_diff / (1024 * 1024):.2f} MB"


@pytest.fixture
def benchmark_results() -> Dict[str, List[float]]:
    """Collect benchmark results."""
    return {}


def test_print_benchmark_summary(benchmark_results):
    """Print benchmark summary (runs last)."""
    if benchmark_results:
        print("\n=== Benchmark Summary ===")
        for name, times in benchmark_results.items():
            avg = sum(times) / len(times)
            print(f"{name}: avg={avg*1000:.2f}ms")
