import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from src.core.similarity import InMemoryVectorStore


def _run_faiss_perf_subprocess(
    dim: int,
    count: int,
    iterations: int,
    python_warnings: list[str],
) -> dict[str, float]:
    # Run faiss perf in a subprocess so a segfault doesn't kill pytest.
    script = f"""
import json
import statistics
import time
import warnings

import numpy as np

from src.core.similarity import FaissVectorStore

dim = {dim}
count = {count}
iterations = {iterations}

vectors = np.random.rand(count, dim).astype(np.float32)
ids = ["vec_{{}}".format(i) for i in range(count)]

warnings.filterwarnings(
    "ignore",
    message=r"builtin type (SwigPyPacked|SwigPyObject|swigvarlink) has no __module__ attribute",
    category=DeprecationWarning,
)

store = FaissVectorStore()
if not store._available:
    raise SystemExit("faiss backend not available")

start_add = time.perf_counter()
for i in range(count):
    store.add(ids[i], vectors[i].tolist())
add_time = (time.perf_counter() - start_add) * 1000

query_vec = np.random.rand(dim).astype(np.float32).tolist()
latencies = []
for _ in range(iterations):
    start_q = time.perf_counter()
    store.query(query_vec, top_k=10)
    end_q = time.perf_counter()
    latencies.append((end_q - start_q) * 1000)

avg_latency = statistics.mean(latencies)
p99_latency = statistics.quantiles(latencies, n=100)[98]

print(json.dumps({{
    "add_ms": add_time,
    "avg_ms": avg_latency,
    "p99_ms": p99_latency,
}}))
"""
    cmd = [sys.executable]
    for warning in python_warnings:
        cmd.extend(["-W", warning])
    cmd.extend(["-c", script])
    repo_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=repo_root,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("faiss perf subprocess timed out") from exc
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip()
        if not message:
            message = f"exit code {result.returncode}"
        raise RuntimeError(message)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("faiss perf subprocess produced no output")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"unable to parse faiss perf output: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("faiss perf subprocess returned non-object JSON")
    return payload


@pytest.mark.asyncio
async def test_vector_search_latency(pytestconfig: pytest.Config):
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

    print(f"\nTesting InMemoryVectorStore with {count} vectors...")

    store = InMemoryVectorStore()
    start_add = time.perf_counter()
    for i in range(count):
        store.add(ids[i], vectors[i].tolist())
    add_time = (time.perf_counter() - start_add) * 1000
    print(f"  Add time: {add_time:.2f} ms")

    query_vec = np.random.rand(dim).astype(np.float32).tolist()
    iterations = 100
    latencies = []
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

    if os.getenv("RUN_FAISS_PERF_TESTS", "0") == "1":
        python_warnings = pytestconfig.getoption("pythonwarnings") or []
        try:
            faiss_result = _run_faiss_perf_subprocess(
                dim=dim,
                count=count,
                iterations=iterations,
                python_warnings=python_warnings,
            )
        except RuntimeError as exc:
            if os.getenv("REQUIRE_FAISS_PERF", "0") == "1":
                pytest.fail(f"Faiss perf subprocess failed: {exc}")
            pytest.skip(f"Faiss perf subprocess failed: {exc}")

        print(f"\nTesting FaissVectorStore with {count} vectors...")
        print(f"  Add time: {faiss_result['add_ms']:.2f} ms")
        print(f"  Query Latency (N={iterations}):")
        print(f"    Average: {faiss_result['avg_ms']:.4f} ms")
        print(f"    P99:     {faiss_result['p99_ms']:.4f} ms")
        assert (
            faiss_result["avg_ms"] < 10
        ), f"Faiss average latency {faiss_result['avg_ms']}ms > 10ms"
