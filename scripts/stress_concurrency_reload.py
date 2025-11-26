"""
Concurrent model reload stress test.

Validates _MODEL_LOCK effectiveness and _MODEL_LOAD_SEQ monotonicity
by triggering multiple concurrent reload requests.

Usage:
  python scripts/stress_concurrency_reload.py [--threads N] [--iterations N] [--url URL]

Environment:
  STRESS_API_URL: Base API URL (default: http://localhost:8000)
  STRESS_API_KEY: API key for authentication
  STRESS_ADMIN_TOKEN: Admin token for reload operations
  STRESS_THREADS: Number of concurrent threads (default: 10)
  STRESS_ITERATIONS: Number of reload iterations per thread (default: 10)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Support running with or without requests
try:
    import requests
except ImportError:
    requests = None  # type: ignore


@dataclass
class StressResult:
    """Result from a single reload attempt."""
    thread_id: int
    iteration: int
    status: str
    load_seq: Optional[int]
    duration_ms: float
    error: Optional[str] = None


@dataclass
class StressStats:
    """Aggregated stress test statistics."""
    total_requests: int = 0
    success_count: int = 0
    failure_count: int = 0
    error_count: int = 0
    load_seq_values: List[int] = field(default_factory=list)
    durations_ms: List[float] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    status_distribution: Dict[str, int] = field(default_factory=dict)


def perform_reload(
    base_url: str,
    api_key: str,
    admin_token: str,
    thread_id: int,
    iteration: int,
) -> StressResult:
    """Perform a single model reload request."""
    if requests is None:
        return StressResult(
            thread_id=thread_id,
            iteration=iteration,
            status="error",
            load_seq=None,
            duration_ms=0,
            error="requests library not installed",
        )

    url = f"{base_url.rstrip('/')}/api/v1/model/reload"
    headers = {
        "X-API-Key": api_key,
        "X-Admin-Token": admin_token,
        "Content-Type": "application/json",
    }
    payload = {"force": True}  # Force reload to stress the lock

    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        duration_ms = (time.perf_counter() - start) * 1000

        if resp.status_code == 200:
            data = resp.json()
            status = data.get("status", "unknown")
            # Try to get load_seq from model info endpoint
            load_seq = None
            try:
                info_resp = requests.get(
                    f"{base_url.rstrip('/')}/api/v1/health/model",
                    headers={"X-API-Key": api_key},
                    timeout=5,
                )
                if info_resp.status_code == 200:
                    load_seq = info_resp.json().get("load_seq")
            except Exception:
                pass

            return StressResult(
                thread_id=thread_id,
                iteration=iteration,
                status=status,
                load_seq=load_seq,
                duration_ms=duration_ms,
            )
        else:
            return StressResult(
                thread_id=thread_id,
                iteration=iteration,
                status=f"http_{resp.status_code}",
                load_seq=None,
                duration_ms=duration_ms,
                error=resp.text[:200] if resp.text else None,
            )
    except requests.RequestException as e:
        duration_ms = (time.perf_counter() - start) * 1000
        return StressResult(
            thread_id=thread_id,
            iteration=iteration,
            status="error",
            load_seq=None,
            duration_ms=duration_ms,
            error=str(e)[:200],
        )


def run_thread_workload(
    base_url: str,
    api_key: str,
    admin_token: str,
    thread_id: int,
    iterations: int,
    results: List[StressResult],
    lock: threading.Lock,
) -> None:
    """Run reload iterations for a single thread."""
    for i in range(iterations):
        result = perform_reload(base_url, api_key, admin_token, thread_id, i)
        with lock:
            results.append(result)
        # Small random jitter to increase contention variety
        time.sleep(0.001 * (thread_id % 5))


def analyze_results(results: List[StressResult]) -> StressStats:
    """Analyze stress test results."""
    stats = StressStats()
    stats.total_requests = len(results)

    for r in results:
        # Count status distribution
        stats.status_distribution[r.status] = stats.status_distribution.get(r.status, 0) + 1

        if r.status == "success":
            stats.success_count += 1
        elif r.status == "error":
            stats.error_count += 1
            if r.error:
                stats.errors.append(r.error)
        else:
            stats.failure_count += 1

        if r.load_seq is not None:
            stats.load_seq_values.append(r.load_seq)

        stats.durations_ms.append(r.duration_ms)

    return stats


def check_monotonicity(seq_values: List[int]) -> "tuple[bool, str]":
    """Check if load_seq values are monotonically non-decreasing."""
    if len(seq_values) < 2:
        return True, "insufficient data"

    # Sort by observation order (already in list order)
    violations = []
    for i in range(1, len(seq_values)):
        if seq_values[i] < seq_values[i - 1]:
            violations.append((i, seq_values[i - 1], seq_values[i]))

    if violations:
        return False, f"{len(violations)} violations: {violations[:5]}"
    return True, f"monotonic ({min(seq_values)} -> {max(seq_values)})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent model reload stress test")
    parser.add_argument("--url", default=os.getenv("STRESS_API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("STRESS_API_KEY", "test-key"))
    parser.add_argument("--admin-token", default=os.getenv("STRESS_ADMIN_TOKEN", "test-admin"))
    parser.add_argument("--threads", type=int, default=int(os.getenv("STRESS_THREADS", "10")))
    parser.add_argument("--iterations", type=int, default=int(os.getenv("STRESS_ITERATIONS", "10")))
    parser.add_argument("--strict", action="store_true", help="Fail on any anomaly")
    args = parser.parse_args()

    if requests is None:
        print("ERROR: requests library not installed. Run: pip install requests")
        return 1

    print(f"Stress test config:")
    print(f"  URL: {args.url}")
    print(f"  Threads: {args.threads}")
    print(f"  Iterations per thread: {args.iterations}")
    print(f"  Total requests: {args.threads * args.iterations}")
    print()

    results: List[StressResult] = []
    lock = threading.Lock()

    start_time = time.perf_counter()

    # Run concurrent reload requests
    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [
            executor.submit(
                run_thread_workload,
                args.url,
                args.api_key,
                args.admin_token,
                tid,
                args.iterations,
                results,
                lock,
            )
            for tid in range(args.threads)
        ]
        # Wait for all threads
        for f in as_completed(futures):
            f.result()  # Raise any exceptions

    total_time = time.perf_counter() - start_time

    # Analyze results
    stats = analyze_results(results)
    monotonic, mono_msg = check_monotonicity(stats.load_seq_values)

    print("=" * 60)
    print("STRESS TEST RESULTS")
    print("=" * 60)
    print(f"Total time: {total_time:.2f}s")
    print(f"Total requests: {stats.total_requests}")
    print(f"Throughput: {stats.total_requests / total_time:.1f} req/s")
    print()
    print("Status distribution:")
    for status, count in sorted(stats.status_distribution.items()):
        pct = count / stats.total_requests * 100
        print(f"  {status}: {count} ({pct:.1f}%)")
    print()

    if stats.durations_ms:
        sorted_dur = sorted(stats.durations_ms)
        p50 = sorted_dur[len(sorted_dur) // 2]
        p95 = sorted_dur[int(len(sorted_dur) * 0.95)]
        p99 = sorted_dur[int(len(sorted_dur) * 0.99)]
        print(f"Latency (ms): p50={p50:.1f}, p95={p95:.1f}, p99={p99:.1f}")
        print()

    print(f"Load sequence monotonicity: {mono_msg}")
    print(f"Monotonic check: {'PASS' if monotonic else 'FAIL'}")
    print()

    if stats.errors:
        print(f"Errors ({len(stats.errors)} total):")
        for err in stats.errors[:5]:
            print(f"  - {err}")
        if len(stats.errors) > 5:
            print(f"  ... and {len(stats.errors) - 5} more")
        print()

    # Final verdict
    passed = True
    issues = []

    if not monotonic:
        issues.append("load_seq not monotonic (possible race condition)")
        passed = False

    if stats.error_count > stats.total_requests * 0.1:
        issues.append(f"high error rate ({stats.error_count / stats.total_requests * 100:.1f}%)")
        if args.strict:
            passed = False

    if passed:
        print("VERDICT: PASS - No concurrency issues detected")
        return 0
    else:
        print(f"VERDICT: FAIL - Issues: {'; '.join(issues)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
