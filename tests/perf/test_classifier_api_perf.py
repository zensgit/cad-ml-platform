"""
Performance benchmark for V16 Classifier API.

Measures:
- Single file classification latency
- Batch classification throughput
- Cache hit performance
- Parallel processing efficiency
"""

import os
import statistics
import time
from pathlib import Path

import pytest

# Test data directory
DATA_DIR = Path(__file__).parent.parent.parent / "data" / "training_v7"


def get_sample_files(category: str, count: int = 5) -> list:
    """Get sample DXF files from training data."""
    cat_dir = DATA_DIR / category
    if not cat_dir.exists():
        return []
    files = list(cat_dir.glob("*.dxf"))[:count]
    return files


@pytest.fixture
def classifier_client(monkeypatch):
    """Create test client with mocked model loading."""
    from fastapi.testclient import TestClient
    from src.inference import classifier_api

    # Use real classifier for performance testing
    # but skip if models not available
    try:
        classifier_api.classifier.load()
    except Exception as e:
        pytest.skip(f"Models not available: {e}")

    return TestClient(classifier_api.app)


class TestSingleFileLatency:
    """Benchmark single file classification latency."""

    @pytest.mark.perf
    def test_single_file_latency_cold(self, classifier_client):
        """Test latency for cold (non-cached) classification."""
        files = get_sample_files("壳体类", 1)
        if not files:
            pytest.skip("No sample files available")

        # Clear cache first
        classifier_client.post(
            "/cache/clear",
            headers={"X-Admin-Token": "test"}
        )

        latencies = []
        for _ in range(5):
            with open(files[0], "rb") as f:
                content = f.read()

            # Use different "filename" to avoid cache
            start = time.perf_counter()
            response = classifier_client.post(
                "/classify",
                files={"file": (f"test_{time.time()}.dxf", content, "application/dxf")},
            )
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)  # ms

            assert response.status_code == 200

            # Clear cache between runs
            classifier_client.post(
                "/cache/clear",
                headers={"X-Admin-Token": "test"}
            )

        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nCold classification latency:")
        print(f"  Average: {avg:.1f}ms")
        print(f"  P95: {p95:.1f}ms")
        print(f"  Min: {min(latencies):.1f}ms")
        print(f"  Max: {max(latencies):.1f}ms")

        # Target: < 2000ms for cold classification (includes DXF parsing + model inference)
        assert avg < 2000, f"Average latency {avg:.1f}ms exceeds 2000ms target"

    @pytest.mark.perf
    def test_single_file_latency_warm(self, classifier_client):
        """Test latency for warm (cached) classification."""
        files = get_sample_files("轴类", 1)
        if not files:
            pytest.skip("No sample files available")

        with open(files[0], "rb") as f:
            content = f.read()

        # First call to populate cache
        classifier_client.post(
            "/classify",
            files={"file": ("warmup.dxf", content, "application/dxf")},
        )

        # Measure cached performance
        latencies = []
        for _ in range(10):
            start = time.perf_counter()
            response = classifier_client.post(
                "/classify",
                files={"file": ("warmup.dxf", content, "application/dxf")},
            )
            elapsed = time.perf_counter() - start
            latencies.append(elapsed * 1000)  # ms

            assert response.status_code == 200

        avg = statistics.mean(latencies)
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]

        print(f"\nWarm (cached) classification latency:")
        print(f"  Average: {avg:.1f}ms")
        print(f"  P95: {p95:.1f}ms")
        print(f"  Min: {min(latencies):.1f}ms")
        print(f"  Max: {max(latencies):.1f}ms")

        # Target: < 10ms for cached classification
        assert avg < 10, f"Average cached latency {avg:.1f}ms exceeds 10ms target"


class TestBatchThroughput:
    """Benchmark batch classification throughput."""

    @pytest.mark.perf
    def test_batch_throughput(self, classifier_client):
        """Test batch classification throughput."""
        # Collect files from different categories
        all_files = []
        for cat in ["壳体类", "轴类", "传动件", "连接件"]:
            all_files.extend(get_sample_files(cat, 3))

        if len(all_files) < 5:
            pytest.skip("Not enough sample files available")

        # Clear cache
        classifier_client.post(
            "/cache/clear",
            headers={"X-Admin-Token": "test"}
        )

        # Prepare files
        files_data = []
        for f in all_files[:10]:
            with open(f, "rb") as fp:
                files_data.append(
                    ("files", (f.name, fp.read(), "application/dxf"))
                )

        # Measure batch processing
        start = time.perf_counter()
        response = classifier_client.post(
            "/classify/batch",
            files=files_data,
        )
        elapsed = time.perf_counter() - start

        assert response.status_code == 200
        data = response.json()

        throughput = data["total"] / elapsed
        latency_per_file = (elapsed * 1000) / data["total"]

        print(f"\nBatch classification performance:")
        print(f"  Files processed: {data['total']}")
        print(f"  Successful: {data['success']}")
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.1f} files/sec")
        print(f"  Avg latency per file: {latency_per_file:.1f}ms")

        # Target: > 2 files/sec throughput
        assert throughput > 2, f"Throughput {throughput:.1f} files/sec below 2 target"


class TestCacheEfficiency:
    """Benchmark cache efficiency."""

    @pytest.mark.perf
    def test_cache_hit_rate(self, classifier_client):
        """Test cache hit rate with repeated requests."""
        files = get_sample_files("壳体类", 3)
        if len(files) < 3:
            pytest.skip("Not enough sample files")

        # Clear cache
        classifier_client.post(
            "/cache/clear",
            headers={"X-Admin-Token": "test"}
        )

        # First pass - all misses
        for f in files:
            with open(f, "rb") as fp:
                classifier_client.post(
                    "/classify",
                    files={"file": (f.name, fp.read(), "application/dxf")},
                )

        # Check stats after first pass
        stats1 = classifier_client.get(
            "/cache/stats",
            headers={"X-Admin-Token": "test"}
        ).json()

        # Second pass - all hits
        for f in files:
            with open(f, "rb") as fp:
                classifier_client.post(
                    "/classify",
                    files={"file": (f.name, fp.read(), "application/dxf")},
                )

        # Check stats after second pass
        stats2 = classifier_client.get(
            "/cache/stats",
            headers={"X-Admin-Token": "test"}
        ).json()

        print(f"\nCache efficiency:")
        print(f"  After first pass: {stats1}")
        print(f"  After second pass: {stats2}")
        print(f"  Hit rate: {stats2['hit_rate']}")

        # Second pass should have 100% hits for repeated files
        expected_hits = len(files)
        actual_hits = stats2["hits"] - stats1["hits"]
        assert actual_hits == expected_hits, f"Expected {expected_hits} cache hits, got {actual_hits}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "perf"])
