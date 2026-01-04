import os
import statistics
import subprocess
import sys
import time

import numpy as np
import pytest

from src.core.similarity import FaissVectorStore, InMemoryVectorStore


def _faiss_import_ok() -> tuple[bool, str | None]:
    result = subprocess.run(
        [sys.executable, "-c", "import faiss"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        if not message:
            message = f"exit code {result.returncode}"
        return False, message
    return True, None


@pytest.mark.asyncio
async def test_vector_search_latency():
    """
    Performance test for vector search.
    Goal: < 10ms for 10k vectors.
    """
    # Setup
    dim = 64  # Metric learning dim
    count = 10000

    # Create synthetic data
    vectors = np.random.rand(count, dim).astype(np.float32)
    ids = [f"vec_{i}" for i in range(count)]

    # Test both backends if available
    stores = [InMemoryVectorStore()]
    if os.getenv("RUN_FAISS_PERF_TESTS", "0") == "1":
        ok, reason = _faiss_import_ok()
        if not ok:
            pytest.skip(f"Faiss perf tests requested but faiss import failed: {reason}")
        try:
            faiss_store = FaissVectorStore()
            if faiss_store._available:
                stores.append(faiss_store)
            else:
                pytest.skip("Faiss perf tests requested but faiss is unavailable")
        except Exception as e:
            pytest.skip(f"FaissVectorStore init failed: {e}")

    for store in stores:
        backend_name = store.__class__.__name__
        print(f"\nTesting {backend_name} with {count} vectors...")

        # Add vectors
        start_add = time.perf_counter()
        for i in range(count):
            store.add(ids[i], vectors[i].tolist())
        add_time = (time.perf_counter() - start_add) * 1000
        print(f"  Add time: {add_time:.2f} ms")

        # Query
        query_vec = np.random.rand(dim).astype(np.float32).tolist()

        latencies = []
        iterations = 100

        for _ in range(iterations):
            start_q = time.perf_counter()
            store.query(query_vec, top_k=10)
            end_q = time.perf_counter()
            latencies.append((end_q - start_q) * 1000)

        avg_latency = statistics.mean(latencies)
        p99_latency = statistics.quantiles(latencies, n=100)[98]

        print(f"  Query Latency (N={iterations}):")
        print(f"    Average: {avg_latency:.4f} ms")
        print(f"    P99:     {p99_latency:.4f} ms")

        # Thresholds
        if "Faiss" in backend_name:
            assert avg_latency < 10, f"Faiss average latency {avg_latency}ms > 10ms"
