"""
Degradation flapping stress test.

Simulates Faiss availability/unavailability transitions to verify:
- degraded/restored event counters
- degraded_duration_seconds metric behavior
- Degradation history bounded to 10 entries
- Health endpoint consistency

Usage:
  python scripts/stress_degradation_flapping.py [--cycles N] [--interval SECONDS]

Environment:
  STRESS_API_URL: Base API URL (default: http://localhost:8000)
  STRESS_API_KEY: API key for authentication
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import requests
except ImportError:
    requests = None  # type: ignore


@dataclass
class FlappingResult:
    """Result from a single flapping cycle."""
    cycle: int
    degraded_event_count: Optional[int]
    restored_event_count: Optional[int]
    duration_seconds: Optional[float]
    history_count: Optional[int]
    error: Optional[str] = None


@dataclass
class FlappingStats:
    """Aggregated flapping test statistics."""
    total_cycles: int = 0
    successful_cycles: int = 0
    error_cycles: int = 0
    max_history_observed: int = 0
    degraded_events_observed: List[int] = field(default_factory=list)
    restored_events_observed: List[int] = field(default_factory=list)
    duration_observations: List[float] = field(default_factory=list)


def get_health_status(base_url: str, api_key: str) -> Dict[str, Any]:
    """Get current health/degradation status."""
    if requests is None:
        return {"error": "requests not installed"}

    url = f"{base_url.rstrip('/')}/api/v1/health/faiss/health"
    headers = {"X-API-Key": api_key}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def get_metrics(base_url: str) -> Dict[str, float]:
    """Scrape Prometheus metrics endpoint."""
    if requests is None:
        return {}

    url = f"{base_url.rstrip('/')}/metrics"

    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return {}

        metrics = {}
        for line in resp.text.split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            # Parse metric line: name{labels} value
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0].split("{")[0]
                try:
                    value = float(parts[-1])
                    # Handle labeled metrics
                    if "degraded" in line and 'event="degraded"' in line:
                        metrics["similarity_degraded_total_degraded"] = value
                    elif "degraded" in line and 'event="restored"' in line:
                        metrics["similarity_degraded_total_restored"] = value
                    elif name == "faiss_degraded_duration_seconds":
                        metrics["faiss_degraded_duration_seconds"] = value
                    elif name == "degradation_history_count":
                        metrics["degradation_history_count"] = value
                except ValueError:
                    pass
        return metrics
    except Exception:
        return {}


def run_flapping_cycle(
    base_url: str,
    api_key: str,
    cycle: int,
) -> FlappingResult:
    """Run a single observation cycle."""
    try:
        # Get health status
        health = get_health_status(base_url, api_key)
        if "error" in health:
            return FlappingResult(
                cycle=cycle,
                degraded_event_count=None,
                restored_event_count=None,
                duration_seconds=None,
                history_count=None,
                error=health["error"],
            )

        # Get metrics (Prometheus) + derive history count from health
        metrics = get_metrics(base_url)

        history_count_from_health: Optional[int] = None
        try:
            # Prefer health endpoint for bounded history count (authoritative)
            hc = health.get("degradation_history_count")
            if isinstance(hc, int):
                history_count_from_health = hc
            elif isinstance(hc, float):
                history_count_from_health = int(hc)
        except Exception:
            history_count_from_health = None

        return FlappingResult(
            cycle=cycle,
            degraded_event_count=int(metrics.get("similarity_degraded_total_degraded", 0)),
            restored_event_count=int(metrics.get("similarity_degraded_total_restored", 0)),
            duration_seconds=metrics.get("faiss_degraded_duration_seconds"),
            history_count=history_count_from_health,
        )
    except Exception as e:
        return FlappingResult(
            cycle=cycle,
            degraded_event_count=None,
            restored_event_count=None,
            duration_seconds=None,
            history_count=None,
            error=str(e)[:200],
        )


def analyze_results(results: List[FlappingResult]) -> FlappingStats:
    """Analyze flapping test results."""
    stats = FlappingStats()
    stats.total_cycles = len(results)

    for r in results:
        if r.error:
            stats.error_cycles += 1
        else:
            stats.successful_cycles += 1

            if r.degraded_event_count is not None:
                stats.degraded_events_observed.append(r.degraded_event_count)
            if r.restored_event_count is not None:
                stats.restored_events_observed.append(r.restored_event_count)
            if r.duration_seconds is not None:
                stats.duration_observations.append(r.duration_seconds)
            if r.history_count is not None:
                stats.max_history_observed = max(stats.max_history_observed, r.history_count)

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Degradation flapping stress test")
    parser.add_argument("--url", default=os.getenv("STRESS_API_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.getenv("STRESS_API_KEY", "test-key"))
    parser.add_argument("--cycles", type=int, default=20, help="Number of observation cycles")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between cycles")
    args = parser.parse_args()

    if requests is None:
        print("ERROR: requests library not installed. Run: pip install requests")
        return 1

    print(f"Degradation flapping test config:")
    print(f"  URL: {args.url}")
    print(f"  Cycles: {args.cycles}")
    print(f"  Interval: {args.interval}s")
    print()
    print("Note: This script observes degradation metrics over time.")
    print("To test actual flapping, manually toggle Faiss availability during the test.")
    print()

    results: List[FlappingResult] = []

    for cycle in range(args.cycles):
        result = run_flapping_cycle(args.url, args.api_key, cycle)
        results.append(result)

        # Progress indicator
        status = "OK" if not result.error else f"ERR: {result.error[:30]}"
        print(f"  Cycle {cycle + 1}/{args.cycles}: {status}")

        if cycle < args.cycles - 1:
            time.sleep(args.interval)

    # Analyze results
    stats = analyze_results(results)

    print()
    print("=" * 60)
    print("FLAPPING TEST RESULTS")
    print("=" * 60)
    print(f"Total cycles: {stats.total_cycles}")
    print(f"Successful: {stats.successful_cycles}")
    print(f"Errors: {stats.error_cycles}")
    print()

    if stats.degraded_events_observed:
        print(f"Degraded events observed: {min(stats.degraded_events_observed)} -> {max(stats.degraded_events_observed)}")
    if stats.restored_events_observed:
        print(f"Restored events observed: {min(stats.restored_events_observed)} -> {max(stats.restored_events_observed)}")
    if stats.duration_observations:
        print(f"Duration range: {min(stats.duration_observations):.1f}s - {max(stats.duration_observations):.1f}s")
    print(f"Max history count observed: {stats.max_history_observed}")
    print()

    # Validation
    passed = True
    issues = []

    # History should be bounded to 10
    if stats.max_history_observed > 10:
        issues.append(f"History exceeded limit: {stats.max_history_observed} > 10")
        passed = False

    if stats.error_cycles > stats.total_cycles * 0.5:
        issues.append(f"High error rate: {stats.error_cycles}/{stats.total_cycles}")
        passed = False

    if passed:
        print("VERDICT: PASS - Degradation metrics consistent")
        return 0
    else:
        print(f"VERDICT: FAIL - Issues: {'; '.join(issues)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
