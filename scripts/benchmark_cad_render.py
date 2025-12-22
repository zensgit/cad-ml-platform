"""Simple render latency benchmark for CAD render service."""
from __future__ import annotations

import argparse
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import requests


def _post_render(url: str, file_path: Path, token: str | None) -> Tuple[float, int]:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with file_path.open("rb") as fh:
        files = {"file": (file_path.name, fh, "application/octet-stream")}
        start = time.perf_counter()
        resp = requests.post(url, files=files, headers=headers, timeout=120)
        elapsed = time.perf_counter() - start
    return elapsed, resp.status_code


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:18002/api/v1/render/cad")
    parser.add_argument("--file", required=True)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--token", default=None)
    args = parser.parse_args()

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"File not found: {file_path}")

    durations: List[float] = []
    status_counts = {}

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(_post_render, args.url, file_path, args.token)
                   for _ in range(args.requests)]
        for future in as_completed(futures):
            elapsed, status = future.result()
            durations.append(elapsed)
            status_counts[status] = status_counts.get(status, 0) + 1

    durations.sort()
    p50 = durations[int(0.5 * (len(durations) - 1))]
    p95 = durations[int(0.95 * (len(durations) - 1))]

    print("requests", args.requests)
    print("concurrency", args.concurrency)
    print("status_counts", status_counts)
    print("min", min(durations))
    print("max", max(durations))
    print("mean", statistics.mean(durations))
    print("p50", p50)
    print("p95", p95)


if __name__ == "__main__":
    main()
