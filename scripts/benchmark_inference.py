#!/usr/bin/env python3
"""End-to-end inference benchmark for the full HybridClassifier pipeline (B5.5d).

Measures latency across the complete inference path:
  ezdxf parse → graph conversion → GNN forward → text extraction
  → text keyword matching → fusion → monitor record

Usage:
    # Benchmark 50 files, 5 runs each
    python scripts/benchmark_inference.py \
        --manifest data/graph_cache/cache_manifest.csv \
        --n-files 50 --n-runs 5

    # Benchmark with a specific model
    python scripts/benchmark_inference.py \
        --manifest data/graph_cache/cache_manifest.csv \
        --model models/graph2d_finetuned_24class_v3_int8.pth
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import statistics
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger(__name__)


def benchmark_classify(
    manifest_csv: str,
    n_files: int = 50,
    n_runs: int = 5,
    model_path: str = "",
    seed: int = 42,
) -> dict:
    """Benchmark HybridClassifier.classify() on real DXF files.

    Returns a dict with p50_ms, p95_ms, p99_ms, mean_ms, n_measurements,
    and per-stage timing breakdown.
    """
    # Sample files from manifest
    rows = []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fp = row.get("file_path", "").strip()
            if fp and Path(fp).exists():
                rows.append(row)

    if not rows:
        logger.error("No valid file_path entries found in manifest.")
        return {}

    samples = random.Random(seed).sample(rows, min(n_files, len(rows)))
    logger.info("Benchmarking %d files × %d runs = %d measurements",
                len(samples), n_runs, len(samples) * n_runs)

    # Set model path if specified
    if model_path:
        os.environ["GRAPH2D_MODEL_PATH"] = model_path

    from src.ml.hybrid_classifier import HybridClassifier
    clf = HybridClassifier()

    # Warmup (1 file)
    warmup_fp = samples[0]["file_path"]
    warmup_bytes = Path(warmup_fp).read_bytes()
    clf.classify(filename=Path(warmup_fp).name, file_bytes=warmup_bytes)

    # Benchmark
    latencies: list[float] = []
    text_hits = 0
    errors = 0

    for row in samples:
        fp = row["file_path"]
        file_bytes = Path(fp).read_bytes()
        filename = Path(fp).name

        for _ in range(n_runs):
            t0 = time.perf_counter()
            try:
                result = clf.classify(filename=filename, file_bytes=file_bytes)
                latency = (time.perf_counter() - t0) * 1000
                latencies.append(latency)
                if result.text_content_prediction is not None:
                    text_hits += 1
            except Exception as e:
                errors += 1
                logger.debug("classify failed: %s", e)

    if not latencies:
        logger.error("No successful measurements.")
        return {}

    latencies.sort()
    n = len(latencies)

    stats = {
        "n_files": len(samples),
        "n_runs": n_runs,
        "n_measurements": n,
        "errors": errors,
        "p50_ms": round(latencies[n // 2], 2),
        "p95_ms": round(latencies[int(0.95 * n)], 2),
        "p99_ms": round(latencies[int(0.99 * n)], 2),
        "mean_ms": round(statistics.mean(latencies), 2),
        "min_ms": round(latencies[0], 2),
        "max_ms": round(latencies[-1], 2),
        "text_hit_rate": round(text_hits / n, 4) if n else 0,
    }

    # Monitor summary
    stats["monitor"] = clf.monitor.summary()

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(
        description="End-to-end inference benchmark for HybridClassifier."
    )
    parser.add_argument(
        "--manifest",
        default="data/graph_cache/cache_manifest.csv",
        help="Cache manifest CSV with file_path column.",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Model checkpoint to use (sets GRAPH2D_MODEL_PATH env var).",
    )
    parser.add_argument(
        "--n-files", type=int, default=50,
        help="Number of files to benchmark (default: 50).",
    )
    parser.add_argument(
        "--n-runs", type=int, default=5,
        help="Number of runs per file (default: 5).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for file sampling.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    stats = benchmark_classify(
        manifest_csv=args.manifest,
        n_files=args.n_files,
        n_runs=args.n_runs,
        model_path=args.model,
        seed=args.seed,
    )

    if not stats:
        return 1

    print("\n" + "=" * 60)
    print("HybridClassifier Inference Benchmark")
    print("=" * 60)
    print(f"Files: {stats['n_files']}  Runs: {stats['n_runs']}  "
          f"Measurements: {stats['n_measurements']}  Errors: {stats['errors']}")
    print()
    print(f"  P50  = {stats['p50_ms']:>8.1f} ms")
    print(f"  P95  = {stats['p95_ms']:>8.1f} ms")
    print(f"  P99  = {stats['p99_ms']:>8.1f} ms")
    print(f"  Mean = {stats['mean_ms']:>8.1f} ms")
    print(f"  Min  = {stats['min_ms']:>8.1f} ms")
    print(f"  Max  = {stats['max_ms']:>8.1f} ms")
    print()
    print(f"  Text hit rate: {stats['text_hit_rate'] * 100:.1f}%")
    print()

    monitor = stats.get("monitor", {})
    if monitor:
        print("Monitor summary:")
        print(f"  avg_confidence: {monitor.get('avg_confidence', 0):.4f}")
        print(f"  low_conf_rate:  {monitor.get('low_conf_rate', 0):.4f}")
        print(f"  text_hit_rate:  {monitor.get('text_hit_rate', 0):.4f}")
        print(f"  drift_detected: {monitor.get('drift_detected', False)}")

    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
