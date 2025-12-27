"""Memory stability check with sustained load and periodic sampling."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import re
import statistics
import subprocess
import threading
import time
import urllib.request
from datetime import datetime
from typing import List, Tuple


def _parse_size_to_mib(value: str) -> float:
    match = re.match(r"^([0-9]*\.?[0-9]+)\s*([a-zA-Z]+)$", value.strip())
    if not match:
        return 0.0
    size = float(match.group(1))
    unit = match.group(2).lower()
    if unit in {"b"}:
        return size / 1024.0 / 1024.0
    if unit in {"kb", "kib"}:
        return size / 1024.0
    if unit in {"mb", "mib"}:
        return size
    if unit in {"gb", "gib"}:
        return size * 1024.0
    return size


def _sample_memory(container: str) -> Tuple[float, float]:
    result = subprocess.run(
        [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{.MemUsage}}\t{{.MemPerc}}",
            container,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0 or not result.stdout.strip():
        raise RuntimeError(f"Failed to read docker stats: {result.stderr.strip()}")
    line = result.stdout.strip().splitlines()[0]
    usage, perc = line.split("\t", maxsplit=1)
    used = usage.split("/")[0].strip()
    mem_mib = _parse_size_to_mib(used)
    mem_perc = float(perc.strip().rstrip("%"))
    return mem_mib, mem_perc


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def _make_request(url: str, timeout: float) -> Tuple[int, float]:
    start = time.time()
    code = 500
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            response.read()
            code = response.getcode()
    except Exception:
        code = 500
    return code, (time.time() - start) * 1000.0


class IntervalRecorder:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latencies: List[float] = []
        self._errors = 0
        self._total = 0

    def record(self, code: int, latency_ms: float) -> None:
        with self._lock:
            self._total += 1
            if code != 200:
                self._errors += 1
            self._latencies.append(latency_ms)

    def snapshot_and_reset(self) -> Tuple[int, int, List[float]]:
        with self._lock:
            total = self._total
            errors = self._errors
            latencies = list(self._latencies)
            self._total = 0
            self._errors = 0
            self._latencies = []
        return total, errors, latencies


def _load_loop(
    stop_event: threading.Event,
    url: str,
    concurrency: int,
    timeout: float,
    recorder: IntervalRecorder,
) -> None:
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        while not stop_event.is_set():
            batch_start = time.time()
            futures = [executor.submit(_make_request, url, timeout) for _ in range(concurrency)]
            for future in concurrent.futures.as_completed(futures):
                code, latency_ms = future.result()
                recorder.record(code, latency_ms)
            elapsed = time.time() - batch_start
            if elapsed < 1.0:
                time.sleep(1.0 - elapsed)


def run(args: argparse.Namespace) -> None:
    start = time.time()
    end = start + args.duration_seconds
    next_sample = start
    samples = []
    overall_latencies: List[float] = []
    total_requests = 0
    total_errors = 0

    recorder = IntervalRecorder()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=_load_loop,
        args=(stop_event, args.url, args.concurrency, args.timeout, recorder),
        daemon=True,
    )
    worker.start()

    while time.time() < end:
        now = time.time()
        if now < next_sample:
            time.sleep(min(1.0, next_sample - now))
            continue
        elapsed = int(now - start)
        mem_mib, mem_perc = _sample_memory(args.container)
        interval_total, interval_errors, interval_latencies = recorder.snapshot_and_reset()
        total_requests += interval_total
        total_errors += interval_errors
        overall_latencies.extend(interval_latencies)

        p50 = statistics.median(interval_latencies) if interval_latencies else 0.0
        p95 = _percentile(interval_latencies, 95.0)
        p99 = _percentile(interval_latencies, 99.0)

        samples.append(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "elapsed_seconds": elapsed,
                "requests": interval_total,
                "errors": interval_errors,
                "p50_ms": round(p50, 2),
                "p95_ms": round(p95, 2),
                "p99_ms": round(p99, 2),
                "mem_used_mib": round(mem_mib, 2),
                "mem_percent": round(mem_perc, 2),
            }
        )
        next_sample += args.interval_seconds

    stop_event.set()
    worker.join(timeout=5)

    mem_values = [sample["mem_used_mib"] for sample in samples]
    mem_perc_values = [sample["mem_percent"] for sample in samples]
    overall_p50 = statistics.median(overall_latencies) if overall_latencies else 0.0
    overall_p95 = _percentile(overall_latencies, 95.0)
    overall_p99 = _percentile(overall_latencies, 99.0)

    summary = {
        "start_time": datetime.utcfromtimestamp(start).isoformat() + "Z",
        "duration_seconds": args.duration_seconds,
        "interval_seconds": args.interval_seconds,
        "url": args.url,
        "container": args.container,
        "requests_total": total_requests,
        "errors_total": total_errors,
        "error_rate": round(total_errors / total_requests, 4) if total_requests else 0.0,
        "latency_ms": {
            "p50": round(overall_p50, 2),
            "p95": round(overall_p95, 2),
            "p99": round(overall_p99, 2),
        },
        "memory_mib": {
            "min": round(min(mem_values), 2) if mem_values else 0.0,
            "max": round(max(mem_values), 2) if mem_values else 0.0,
            "avg": round(sum(mem_values) / len(mem_values), 2) if mem_values else 0.0,
        },
        "memory_percent": {
            "min": round(min(mem_perc_values), 2) if mem_perc_values else 0.0,
            "max": round(max(mem_perc_values), 2) if mem_perc_values else 0.0,
            "avg": round(sum(mem_perc_values) / len(mem_perc_values), 2) if mem_perc_values else 0.0,
        },
    }

    with open(args.output_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=samples[0].keys())
        writer.writeheader()
        writer.writerows(samples)

    with open(args.summary_json, "w") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory stability check")
    parser.add_argument("--url", default="http://localhost:8000/health")
    parser.add_argument("--container", default="cad-ml-api")
    parser.add_argument("--duration-seconds", type=int, default=3600)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    args = parser.parse_args()

    run(args)


if __name__ == "__main__":
    main()
