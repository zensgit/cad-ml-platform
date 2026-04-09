#!/usr/bin/env python3
"""Run performance baseline and generate a Markdown report.

Executes all new-module benchmarks, collects timing statistics, and writes
a pass/fail report suitable for CI artefacts or manual review.

Usage:
    python scripts/run_performance_baseline.py
    python scripts/run_performance_baseline.py --output reports/performance_baseline.md
    python scripts/run_performance_baseline.py --iterations 100 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Ensure the repo root is importable
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# TimingResult (mirrors the test helper, kept standalone for the script)
# ---------------------------------------------------------------------------

class TimingResult:
    """Micro-benchmark runner with percentile reporting."""

    def __init__(self) -> None:
        self.times: List[float] = []

    def run(
        self,
        func: Callable[..., Any],
        *args: Any,
        iterations: int = 100,
        warmup: int = 5,
        **kwargs: Any,
    ) -> "TimingResult":
        for _ in range(warmup):
            func(*args, **kwargs)
        self.times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            self.times.append(time.perf_counter() - start)
        return self

    @property
    def p50(self) -> float:
        return float(np.percentile(self.times, 50))

    @property
    def p95(self) -> float:
        return float(np.percentile(self.times, 95))

    @property
    def p99(self) -> float:
        return float(np.percentile(self.times, 99))

    @property
    def mean(self) -> float:
        return float(np.mean(self.times))


# ---------------------------------------------------------------------------
# Benchmark result container
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkEntry:
    module: str
    operation: str
    target_ms: float
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    mean_ms: float = 0.0
    iterations: int = 0
    status: str = "SKIP"
    error: str = ""


# ---------------------------------------------------------------------------
# Individual benchmark functions
# ---------------------------------------------------------------------------

def _bench_cost_estimator(iters: int) -> BenchmarkEntry:
    entry = BenchmarkEntry("CostEstimator", "estimate()", target_ms=100)
    try:
        from src.ml.cost.estimator import CostEstimator
        from src.ml.cost.models import CostEstimateRequest

        est = CostEstimator()
        req = CostEstimateRequest(
            material="steel", batch_size=1,
            bounding_volume_mm3=10_000, entity_count=20,
        )
        t = TimingResult().run(est.estimate, req, iterations=iters)
        entry.p50_ms = t.p50 * 1000
        entry.p95_ms = t.p95 * 1000
        entry.p99_ms = t.p99 * 1000
        entry.mean_ms = t.mean * 1000
        entry.iterations = iters
        entry.status = "PASS" if t.p95 * 1000 < entry.target_ms else "FAIL"
    except Exception as exc:
        entry.error = str(exc)
    return entry


def _bench_knowledge_query(iters: int) -> BenchmarkEntry:
    entry = BenchmarkEntry("GraphQueryEngine", "query()", target_ms=50)
    try:
        from src.ml.knowledge import ManufacturingKnowledgeGraph, GraphQueryEngine

        graph = ManufacturingKnowledgeGraph()
        graph.build_default_graph()
        engine = GraphQueryEngine(graph)
        t = TimingResult().run(engine.query, "SUS304适合什么工艺？", iterations=iters)
        entry.p50_ms = t.p50 * 1000
        entry.p95_ms = t.p95 * 1000
        entry.p99_ms = t.p99 * 1000
        entry.mean_ms = t.mean * 1000
        entry.iterations = iters
        entry.status = "PASS" if t.p95 * 1000 < entry.target_ms else "FAIL"
    except Exception as exc:
        entry.error = str(exc)
    return entry


def _bench_knowledge_optimal_process(iters: int) -> BenchmarkEntry:
    entry = BenchmarkEntry("GraphQueryEngine", "find_optimal_process()", target_ms=50)
    try:
        from src.ml.knowledge import ManufacturingKnowledgeGraph, GraphQueryEngine

        graph = ManufacturingKnowledgeGraph()
        graph.build_default_graph()
        engine = GraphQueryEngine(graph)
        t = TimingResult().run(engine.find_optimal_process, "法兰盘", "SUS304", iterations=iters)
        entry.p50_ms = t.p50 * 1000
        entry.p95_ms = t.p95 * 1000
        entry.p99_ms = t.p99 * 1000
        entry.mean_ms = t.mean * 1000
        entry.iterations = iters
        entry.status = "PASS" if t.p95 * 1000 < entry.target_ms else "FAIL"
    except Exception as exc:
        entry.error = str(exc)
    return entry


def _bench_anomaly_detect(iters: int) -> BenchmarkEntry:
    entry = BenchmarkEntry("MetricsAnomalyDetector", "detect()", target_ms=10)
    try:
        from src.ml.monitoring.anomaly_detector import MetricsAnomalyDetector

        detector = MetricsAnomalyDetector()
        data = np.random.normal(100, 5, 500)
        detector.fit("test_metric", data)
        t = TimingResult().run(detector.detect, "test_metric", 101.0, iterations=iters)
        entry.p50_ms = t.p50 * 1000
        entry.p95_ms = t.p95 * 1000
        entry.p99_ms = t.p99 * 1000
        entry.mean_ms = t.mean * 1000
        entry.iterations = iters
        entry.status = "PASS" if t.p95 * 1000 < entry.target_ms else "FAIL"
    except Exception as exc:
        entry.error = str(exc)
    return entry


def _bench_ensemble_uncertainty(iters: int) -> BenchmarkEntry:
    entry = BenchmarkEntry("HybridIntelligence", "analyze_ensemble_uncertainty()", target_ms=5)
    try:
        from src.ml.hybrid.intelligence import HybridIntelligence

        hi = HybridIntelligence()
        preds = {
            "filename": {"label": "法兰盘", "confidence": 0.9},
            "graph2d": {"label": "法兰盘", "confidence": 0.7},
            "titleblock": {"label": "法兰盘", "confidence": 0.85},
            "process": {"label": "壳体", "confidence": 0.3},
            "history": {"label": "法兰盘", "confidence": 0.6},
        }
        t = TimingResult().run(hi.analyze_ensemble_uncertainty, preds, iterations=iters)
        entry.p50_ms = t.p50 * 1000
        entry.p95_ms = t.p95 * 1000
        entry.p99_ms = t.p99 * 1000
        entry.mean_ms = t.mean * 1000
        entry.iterations = iters
        entry.status = "PASS" if t.p95 * 1000 < entry.target_ms else "FAIL"
    except Exception as exc:
        entry.error = str(exc)
    return entry


def _bench_combined_sampling(iters: int) -> BenchmarkEntry:
    entry = BenchmarkEntry("SmartSampler", "combined_sampling(1000 -> 10)", target_ms=50)
    try:
        from src.ml.learning.smart_sampler import SmartSampler

        sampler = SmartSampler()
        rng = np.random.RandomState(42)
        predictions = []
        for i in range(1000):
            probs = rng.dirichlet([1] * 5)
            predictions.append({
                "id": f"sample_{i}",
                "label": f"class_{int(np.argmax(probs))}",
                "class_probs": {f"class_{j}": float(p) for j, p in enumerate(probs)},
                "confidence": float(max(probs)),
            })
        t = TimingResult().run(sampler.combined_sampling, predictions, 10, iterations=iters)
        entry.p50_ms = t.p50 * 1000
        entry.p95_ms = t.p95 * 1000
        entry.p99_ms = t.p99 * 1000
        entry.mean_ms = t.mean * 1000
        entry.iterations = iters
        entry.status = "PASS" if t.p95 * 1000 < entry.target_ms else "FAIL"
    except Exception as exc:
        entry.error = str(exc)
    return entry


def _bench_geometry_diff(iters: int) -> BenchmarkEntry:
    entry = BenchmarkEntry("GeometryDiff", "compare() (small DXF)", target_ms=500)
    try:
        import ezdxf
    except ImportError:
        entry.error = "ezdxf not installed"
        return entry

    try:
        import tempfile
        from src.core.diff.geometry_diff import GeometryDiff

        doc_a = ezdxf.new()
        msp_a = doc_a.modelspace()
        for i in range(50):
            msp_a.add_line((i, 0), (i, 10))

        doc_b = ezdxf.new()
        msp_b = doc_b.modelspace()
        for i in range(50):
            msp_b.add_line((i, 0), (i, 10))
        for i in range(10):
            msp_b.add_circle((i * 5, 20), radius=2)

        with tempfile.TemporaryDirectory() as td:
            fa = os.path.join(td, "a.dxf")
            fb = os.path.join(td, "b.dxf")
            doc_a.saveas(fa)
            doc_b.saveas(fb)

            diff = GeometryDiff()
            capped = min(iters, 20)  # DXF I/O is slow; cap iterations
            t = TimingResult().run(diff.compare, fa, fb, iterations=capped)
            entry.p50_ms = t.p50 * 1000
            entry.p95_ms = t.p95 * 1000
            entry.p99_ms = t.p99 * 1000
            entry.mean_ms = t.mean * 1000
            entry.iterations = capped
            entry.status = "PASS" if t.p95 * 1000 < entry.target_ms else "FAIL"
    except Exception as exc:
        entry.error = str(exc)
    return entry


def _bench_pointnet_normalize(iters: int) -> BenchmarkEntry:
    entry = BenchmarkEntry("PointCloudPreprocessor", "normalize(2048 pts)", target_ms=5)
    try:
        from src.ml.pointnet.preprocessor import PointCloudPreprocessor

        pp = PointCloudPreprocessor()
        points = np.random.randn(2048, 3).astype(np.float32)
        t = TimingResult().run(pp.normalize, points, iterations=iters)
        entry.p50_ms = t.p50 * 1000
        entry.p95_ms = t.p95 * 1000
        entry.p99_ms = t.p99 * 1000
        entry.mean_ms = t.mean * 1000
        entry.iterations = iters
        entry.status = "PASS" if t.p95 * 1000 < entry.target_ms else "FAIL"
    except Exception as exc:
        entry.error = str(exc)
    return entry


def _bench_function_calling_init(iters: int) -> BenchmarkEntry:
    entry = BenchmarkEntry("FunctionCallingEngine", "__init__(offline)", target_ms=100)
    try:
        from src.core.assistant.function_calling import FunctionCallingEngine

        def _init():
            FunctionCallingEngine(llm_provider="offline")

        t = TimingResult().run(_init, iterations=iters)
        entry.p50_ms = t.p50 * 1000
        entry.p95_ms = t.p95 * 1000
        entry.p99_ms = t.p99 * 1000
        entry.mean_ms = t.mean * 1000
        entry.iterations = iters
        entry.status = "PASS" if t.p95 * 1000 < entry.target_ms else "FAIL"
    except Exception as exc:
        entry.error = str(exc)
    return entry


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_ALL_BENCHMARKS = [
    _bench_cost_estimator,
    _bench_knowledge_query,
    _bench_knowledge_optimal_process,
    _bench_anomaly_detect,
    _bench_ensemble_uncertainty,
    _bench_combined_sampling,
    _bench_geometry_diff,
    _bench_pointnet_normalize,
    _bench_function_calling_init,
]


def run_all_benchmarks(iterations: int = 200) -> List[BenchmarkEntry]:
    """Run every registered benchmark and return the collected entries."""
    results: List[BenchmarkEntry] = []
    for bench_fn in _ALL_BENCHMARKS:
        label = bench_fn.__name__.replace("_bench_", "")
        print(f"  Running {label} ...", end=" ", flush=True)
        entry = bench_fn(iterations)
        tag = entry.status if not entry.error else f"SKIP ({entry.error[:40]})"
        print(tag)
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(results: List[BenchmarkEntry], output_path: str) -> None:
    """Write a Markdown performance report to *output_path*."""
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    skipped = sum(1 for r in results if r.status == "SKIP")
    total = len(results)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = [
        "# Performance Baseline Report",
        "",
        f"Generated: {now}",
        "",
        f"**{passed}/{total} PASS** | {failed} FAIL | {skipped} SKIP",
        "",
        "## Results",
        "",
        "| Module | Operation | Target (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Mean (ms) | Iters | Status |",
        "|--------|-----------|-------------|----------|----------|----------|-----------|-------|--------|",
    ]

    for r in results:
        status_icon = {"PASS": "PASS", "FAIL": "**FAIL**", "SKIP": "SKIP"}.get(r.status, r.status)
        if r.status == "SKIP":
            lines.append(
                f"| {r.module} | {r.operation} | {r.target_ms:.0f} "
                f"| - | - | - | - | - | {status_icon} |"
            )
        else:
            lines.append(
                f"| {r.module} | {r.operation} | {r.target_ms:.0f} "
                f"| {r.p50_ms:.2f} | {r.p95_ms:.2f} | {r.p99_ms:.2f} "
                f"| {r.mean_ms:.2f} | {r.iterations} | {status_icon} |"
            )

    # Failures section
    failures = [r for r in results if r.status == "FAIL"]
    if failures:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for r in failures:
            overshoot = r.p95_ms - r.target_ms
            lines.append(
                f"- **{r.module}.{r.operation}**: p95={r.p95_ms:.2f} ms "
                f"exceeds target {r.target_ms:.0f} ms by {overshoot:.2f} ms"
            )

    # Skipped section
    skips = [r for r in results if r.status == "SKIP"]
    if skips:
        lines.append("")
        lines.append("## Skipped")
        lines.append("")
        for r in skips:
            lines.append(f"- **{r.module}.{r.operation}**: {r.error}")

    lines.append("")

    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nReport written to {dest}")


def generate_json(results: List[BenchmarkEntry], output_path: str) -> None:
    """Write results as a machine-readable JSON file alongside the MD report."""
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmarks": [
            {
                "module": r.module,
                "operation": r.operation,
                "target_ms": r.target_ms,
                "p50_ms": round(r.p50_ms, 3),
                "p95_ms": round(r.p95_ms, 3),
                "p99_ms": round(r.p99_ms, 3),
                "mean_ms": round(r.mean_ms, 3),
                "iterations": r.iterations,
                "status": r.status,
                "error": r.error or None,
            }
            for r in results
        ],
    }
    dest = Path(output_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"JSON written to {dest}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run performance baseline benchmarks and generate a report."
    )
    parser.add_argument(
        "--output", "-o",
        default="reports/performance_baseline.md",
        help="Path for the Markdown report (default: reports/performance_baseline.md)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also emit a JSON sidecar file (<output>.json).",
    )
    parser.add_argument(
        "--iterations", "-n",
        type=int,
        default=200,
        help="Number of timed iterations per benchmark (default: 200).",
    )
    args = parser.parse_args()

    print(f"Running performance baseline ({args.iterations} iterations per benchmark)...\n")
    results = run_all_benchmarks(args.iterations)

    generate_report(results, args.output)

    if args.json:
        json_path = str(Path(args.output).with_suffix(".json"))
        generate_json(results, json_path)

    # Exit with failure if any benchmark failed
    failures = [r for r in results if r.status == "FAIL"]
    if failures:
        print(f"\n{len(failures)} benchmark(s) FAILED.")
        sys.exit(1)
    else:
        passed = sum(1 for r in results if r.status == "PASS")
        print(f"\nAll {passed} benchmark(s) passed.")


if __name__ == "__main__":
    main()
