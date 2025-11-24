#!/usr/bin/env python3
"""Performance baseline capture script.

Captures performance metrics for key operations to establish baseline.
Run at Day 0 and Day 6 to compare before/after performance.
"""

import json
import time
import statistics
from pathlib import Path
from typing import List, Dict, Any
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def time_operation(func, *args, runs: int = 10, **kwargs) -> Dict[str, float]:
    """Time an operation multiple times and return statistics."""
    times = []
    for _ in range(runs):
        start = time.time()
        try:
            func(*args, **kwargs)
            duration = time.time() - start
            times.append(duration)
        except Exception as e:
            print(f"Warning: {func.__name__} failed: {e}")
            continue

    if not times:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "runs": 0}

    sorted_times = sorted(times)
    return {
        "p50": statistics.median(sorted_times),
        "p95": sorted_times[int(len(sorted_times) * 0.95)] if len(sorted_times) > 1 else sorted_times[0],
        "p99": sorted_times[int(len(sorted_times) * 0.99)] if len(sorted_times) > 1 else sorted_times[0],
        "mean": statistics.mean(sorted_times),
        "runs": len(times),
    }


def benchmark_feature_extraction_v3():
    """Benchmark v3 feature extraction."""
    # Mock implementation - replace with actual
    time.sleep(0.001)  # Simulate work


def benchmark_feature_extraction_v4():
    """Benchmark v4 feature extraction."""
    # Mock implementation - replace with actual
    time.sleep(0.0012)  # Simulate slightly slower work


def benchmark_batch_similarity(batch_size: int):
    """Benchmark batch similarity query."""
    # Mock implementation - replace with actual
    time.sleep(0.01 * (batch_size / 10))  # Simulate batch work


def benchmark_model_loading():
    """Benchmark model cold load."""
    # Mock implementation - replace with actual
    time.sleep(0.05)  # Simulate model loading


def main():
    """Run all benchmarks and save baseline."""
    print("🏁 Starting performance baseline capture...")
    print()

    baseline = {
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {},
    }

    # 1. Feature extraction v3
    print("📊 Benchmarking feature extraction v3...")
    baseline["benchmarks"]["feature_extraction_v3"] = time_operation(
        benchmark_feature_extraction_v3, runs=50
    )
    print(f"   p95: {baseline['benchmarks']['feature_extraction_v3']['p95']*1000:.2f}ms")

    # 2. Feature extraction v4 (if implemented)
    print("📊 Benchmarking feature extraction v4...")
    baseline["benchmarks"]["feature_extraction_v4"] = time_operation(
        benchmark_feature_extraction_v4, runs=50
    )
    print(f"   p95: {baseline['benchmarks']['feature_extraction_v4']['p95']*1000:.2f}ms")

    # Calculate v4 overhead
    v3_p95 = baseline["benchmarks"]["feature_extraction_v3"]["p95"]
    v4_p95 = baseline["benchmarks"]["feature_extraction_v4"]["p95"]
    if v3_p95 > 0:
        overhead = ((v4_p95 - v3_p95) / v3_p95) * 100
        baseline["benchmarks"]["v4_overhead_pct"] = overhead
        print(f"   v4 overhead: {overhead:+.1f}%")

    # 3. Batch similarity (various sizes)
    for size in [5, 20, 50]:
        print(f"📊 Benchmarking batch similarity (size={size})...")
        key = f"batch_similarity_{size}ids"
        baseline["benchmarks"][key] = time_operation(
            benchmark_batch_similarity, size, runs=20
        )
        print(f"   p95: {baseline['benchmarks'][key]['p95']*1000:.2f}ms")

    # 4. Model loading
    print("📊 Benchmarking model cold load...")
    baseline["benchmarks"]["model_cold_load"] = time_operation(
        benchmark_model_loading, runs=10
    )
    print(f"   p95: {baseline['benchmarks']['model_cold_load']['p95']*1000:.2f}ms")

    # Save baseline
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / "performance_baseline_day0.json"
    with open(output_file, "w") as f:
        json.dump(baseline, f, indent=2)

    print()
    print(f"✅ Baseline saved to {output_file}")
    print()

    # Display summary table
    print("Performance Summary:")
    print("=" * 60)
    print(f"{'Operation':<40} {'p50':>8} {'p95':>8}")
    print("=" * 60)
    for name, stats in baseline["benchmarks"].items():
        if isinstance(stats, dict) and "p50" in stats:
            p50_ms = stats["p50"] * 1000
            p95_ms = stats["p95"] * 1000
            print(f"{name:<40} {p50_ms:>7.2f}ms {p95_ms:>7.2f}ms")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    exit(main())
