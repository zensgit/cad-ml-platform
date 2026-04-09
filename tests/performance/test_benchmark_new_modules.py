"""
Performance Benchmark Tests for New CAD ML Platform Modules.

Measures latency (p50/p95/p99) of:
- CostEstimator.estimate()
- GraphQueryEngine.query() / find_optimal_process()
- MetricsAnomalyDetector.detect()
- HybridIntelligence.analyze_ensemble_uncertainty()
- SmartSampler.combined_sampling()
- GeometryDiff.compare()
- PointCloudPreprocessor.normalize()
- FunctionCallingEngine (offline mode initialisation)

Each test asserts that p95 latency stays within a documented budget.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
import time
from typing import Any, Callable, Dict, List

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Reusable timing helper
# ---------------------------------------------------------------------------

class TimingResult:
    """Lightweight micro-benchmark runner with percentile reporting.

    Usage::

        t = TimingResult().run(my_func, arg1, arg2, iterations=200)
        print(f"p50={t.p50:.4f}s  p95={t.p95:.4f}s  p99={t.p99:.4f}s")
        assert t.p95 < 0.1
    """

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
        """Time *func* over *iterations* calls (after *warmup* calls).

        Parameters
        ----------
        func:
            Callable to benchmark.
        iterations:
            Number of timed iterations.
        warmup:
            Untimed warm-up calls (JIT, caches, lazy init, etc.).
        """
        # Warm-up phase (results discarded)
        for _ in range(warmup):
            func(*args, **kwargs)

        # Timed phase
        self.times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            self.times.append(time.perf_counter() - start)
        return self

    # -- statistics ---------------------------------------------------------

    @property
    def p50(self) -> float:
        """Median latency in seconds."""
        return float(np.percentile(self.times, 50))

    @property
    def p95(self) -> float:
        """95th percentile latency in seconds."""
        return float(np.percentile(self.times, 95))

    @property
    def p99(self) -> float:
        """99th percentile latency in seconds."""
        return float(np.percentile(self.times, 99))

    @property
    def mean(self) -> float:
        """Arithmetic mean latency in seconds."""
        return float(np.mean(self.times))

    @property
    def std(self) -> float:
        """Standard deviation in seconds."""
        return float(np.std(self.times))

    @property
    def total(self) -> float:
        """Total wall-clock time across all iterations."""
        return float(np.sum(self.times))

    def summary(self, label: str = "") -> str:
        """One-line human-readable summary."""
        tag = f"[{label}] " if label else ""
        return (
            f"{tag}mean={self.mean * 1000:.2f}ms  "
            f"p50={self.p50 * 1000:.2f}ms  "
            f"p95={self.p95 * 1000:.2f}ms  "
            f"p99={self.p99 * 1000:.2f}ms  "
            f"n={len(self.times)}"
        )


# ====================================================================
# Cost Estimator
# ====================================================================

class TestCostEstimatorPerformance:
    """Benchmark CostEstimator.estimate() -- target <100 ms p95."""

    def test_single_estimate_latency(self):
        """Single-part cost estimate must complete in <100 ms (p95)."""
        from src.ml.cost.estimator import CostEstimator
        from src.ml.cost.models import CostEstimateRequest

        est = CostEstimator()
        req = CostEstimateRequest(
            material="steel",
            batch_size=1,
            bounding_volume_mm3=10_000,
            entity_count=20,
        )
        t = TimingResult().run(est.estimate, req, iterations=200)
        print(t.summary("CostEstimator.estimate"))
        assert t.p95 < 0.1, f"Cost estimate p95={t.p95:.4f}s exceeds 100 ms"

    def test_complex_estimate_latency(self):
        """High-complexity estimate (titanium, tight tolerance) stays <100 ms."""
        from src.ml.cost.estimator import CostEstimator
        from src.ml.cost.models import CostEstimateRequest

        est = CostEstimator()
        req = CostEstimateRequest(
            material="titanium",
            batch_size=50,
            bounding_volume_mm3=1_000_000,
            entity_count=500,
            tolerance_grade="IT6",
            surface_finish="Ra0.8",
            complexity_score=0.95,
        )
        t = TimingResult().run(est.estimate, req, iterations=200)
        print(t.summary("CostEstimator.estimate (complex)"))
        assert t.p95 < 0.1, f"Complex estimate p95={t.p95:.4f}s exceeds 100 ms"


# ====================================================================
# Knowledge Graph Query Engine
# ====================================================================

class TestKnowledgeGraphPerformance:
    """Benchmark GraphQueryEngine -- target <50 ms p95."""

    @pytest.fixture(autouse=True)
    def _build_graph(self):
        from src.ml.knowledge import ManufacturingKnowledgeGraph, GraphQueryEngine

        self.graph = ManufacturingKnowledgeGraph()
        self.graph.build_default_graph()
        self.engine = GraphQueryEngine(self.graph)

    def test_query_latency(self):
        """Simple material -> process query under 50 ms."""
        t = TimingResult().run(
            self.engine.query, "SUS304适合什么工艺？", iterations=200,
        )
        print(t.summary("GraphQueryEngine.query"))
        assert t.p95 < 0.05, f"KG query p95={t.p95:.4f}s exceeds 50 ms"

    def test_optimal_process_latency(self):
        """Multi-hop find_optimal_process under 50 ms."""
        t = TimingResult().run(
            self.engine.find_optimal_process, "法兰盘", "SUS304", iterations=200,
        )
        print(t.summary("GraphQueryEngine.find_optimal_process"))
        assert t.p95 < 0.05, f"find_optimal_process p95={t.p95:.4f}s exceeds 50 ms"

    def test_alternative_materials_latency(self):
        """find_alternative_materials under 50 ms."""
        t = TimingResult().run(
            self.engine.find_alternative_materials, "SUS304", iterations=200,
        )
        print(t.summary("GraphQueryEngine.find_alternative_materials"))
        assert t.p95 < 0.05, (
            f"find_alternative_materials p95={t.p95:.4f}s exceeds 50 ms"
        )


# ====================================================================
# Anomaly Detector
# ====================================================================

class TestAnomalyDetectorPerformance:
    """Benchmark MetricsAnomalyDetector.detect() -- target <10 ms p95."""

    def test_detect_latency(self):
        """Single metric anomaly detection under 10 ms."""
        from src.ml.monitoring.anomaly_detector import MetricsAnomalyDetector

        detector = MetricsAnomalyDetector()
        data = np.random.normal(100, 5, 500)
        detector.fit("test_metric", data)

        t = TimingResult().run(detector.detect, "test_metric", 101.0, iterations=500)
        print(t.summary("AnomalyDetector.detect"))
        assert t.p95 < 0.01, f"Anomaly detect p95={t.p95:.4f}s exceeds 10 ms"

    def test_detect_batch_latency(self):
        """Batch detection of 5 metrics under 50 ms."""
        from src.ml.monitoring.anomaly_detector import MetricsAnomalyDetector

        detector = MetricsAnomalyDetector()
        metric_names = [
            "classification_accuracy",
            "cache_hit_rate",
            "p95_latency_seconds",
            "rejection_rate",
            "drift_score",
        ]
        for name in metric_names:
            detector.fit(name, np.random.normal(50, 3, 500))

        batch = {name: 52.0 for name in metric_names}
        t = TimingResult().run(detector.detect_batch, batch, iterations=200)
        print(t.summary("AnomalyDetector.detect_batch (5 metrics)"))
        assert t.p95 < 0.05, f"Batch detect p95={t.p95:.4f}s exceeds 50 ms"


# ====================================================================
# Hybrid Intelligence
# ====================================================================

class TestHybridIntelligencePerformance:
    """Benchmark HybridIntelligence.analyze_ensemble_uncertainty() -- target <5 ms."""

    def test_ensemble_uncertainty_latency(self):
        """Ensemble uncertainty analysis under 5 ms."""
        from src.ml.hybrid.intelligence import HybridIntelligence

        hi = HybridIntelligence()
        preds: Dict[str, Dict[str, Any]] = {
            "filename": {"label": "法兰盘", "confidence": 0.9},
            "graph2d": {"label": "法兰盘", "confidence": 0.7},
            "titleblock": {"label": "法兰盘", "confidence": 0.85},
            "process": {"label": "壳体", "confidence": 0.3},
            "history": {"label": "法兰盘", "confidence": 0.6},
        }
        t = TimingResult().run(
            hi.analyze_ensemble_uncertainty, preds, iterations=500,
        )
        print(t.summary("HybridIntelligence.analyze_ensemble_uncertainty"))
        assert t.p95 < 0.005, (
            f"Ensemble uncertainty p95={t.p95:.4f}s exceeds 5 ms"
        )

    def test_disagreement_report_latency(self):
        """Disagreement analysis under 5 ms."""
        from src.ml.hybrid.intelligence import HybridIntelligence

        hi = HybridIntelligence()
        preds: Dict[str, Dict[str, Any]] = {
            "filename": {"label": "法兰盘", "confidence": 0.9},
            "graph2d": {"label": "壳体", "confidence": 0.7},
            "titleblock": {"label": "法兰盘", "confidence": 0.85},
            "process": {"label": "齿轮", "confidence": 0.4},
            "history": {"label": "法兰盘", "confidence": 0.6},
        }
        # Check if method exists (it may be named differently)
        if hasattr(hi, "detect_disagreement"):
            t = TimingResult().run(hi.detect_disagreement, preds, iterations=500)
            print(t.summary("HybridIntelligence.detect_disagreement"))
            assert t.p95 < 0.005, (
                f"Disagreement p95={t.p95:.4f}s exceeds 5 ms"
            )
        else:
            pytest.skip("detect_disagreement not available")


# ====================================================================
# Smart Sampler
# ====================================================================

class TestSmartSamplerPerformance:
    """Benchmark SmartSampler.combined_sampling() -- target <50 ms for 1000 samples."""

    @staticmethod
    def _make_predictions(n: int = 1000, n_classes: int = 5) -> List[Dict[str, Any]]:
        """Generate *n* synthetic prediction dicts with *n_classes* classes."""
        rng = np.random.RandomState(42)
        preds = []
        for i in range(n):
            probs = rng.dirichlet([1] * n_classes)
            preds.append({
                "id": f"sample_{i}",
                "label": f"class_{int(np.argmax(probs))}",
                "class_probs": {f"class_{j}": float(p) for j, p in enumerate(probs)},
                "confidence": float(np.max(probs)),
            })
        return preds

    def test_combined_sampling_latency(self):
        """Combined sampling of 1000 predictions, selecting 10, under 50 ms."""
        from src.ml.learning.smart_sampler import SmartSampler

        sampler = SmartSampler()
        predictions = self._make_predictions(1000, 5)

        t = TimingResult().run(sampler.combined_sampling, predictions, 10, iterations=50)
        print(t.summary("SmartSampler.combined_sampling (1000 -> 10)"))
        assert t.p95 < 0.05, (
            f"Combined sampling p95={t.p95:.4f}s exceeds 50 ms"
        )

    def test_uncertainty_sampling_latency(self):
        """Uncertainty sampling of 1000 predictions under 20 ms."""
        from src.ml.learning.smart_sampler import SmartSampler

        sampler = SmartSampler()
        predictions = self._make_predictions(1000, 5)

        t = TimingResult().run(
            sampler.uncertainty_sampling, predictions, 10, iterations=100,
        )
        print(t.summary("SmartSampler.uncertainty_sampling (1000 -> 10)"))
        assert t.p95 < 0.02, (
            f"Uncertainty sampling p95={t.p95:.4f}s exceeds 20 ms"
        )


# ====================================================================
# Geometry Diff
# ====================================================================

class TestDiffPerformance:
    """Benchmark GeometryDiff.compare() -- target <500 ms for small DXF files."""

    def test_compare_small_files_latency(self):
        """Diff of two small DXF files (50 lines + 10 circles) under 500 ms."""
        ezdxf = pytest.importorskip("ezdxf")
        from src.core.diff.geometry_diff import GeometryDiff

        # Build two small test DXF files
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
            t = TimingResult().run(diff.compare, fa, fb, iterations=20)
            print(t.summary("GeometryDiff.compare (small)"))
            assert t.p95 < 0.5, f"Diff p95={t.p95:.4f}s exceeds 500 ms"

    def test_compare_medium_files_latency(self):
        """Diff of medium DXF files (200 entities) under 2 s."""
        ezdxf = pytest.importorskip("ezdxf")
        from src.core.diff.geometry_diff import GeometryDiff

        doc_a = ezdxf.new()
        msp_a = doc_a.modelspace()
        for i in range(200):
            msp_a.add_line((i, 0), (i, 10))

        doc_b = ezdxf.new()
        msp_b = doc_b.modelspace()
        for i in range(200):
            msp_b.add_line((i, 0.5), (i, 10.5))

        with tempfile.TemporaryDirectory() as td:
            fa = os.path.join(td, "a_med.dxf")
            fb = os.path.join(td, "b_med.dxf")
            doc_a.saveas(fa)
            doc_b.saveas(fb)

            diff = GeometryDiff()
            t = TimingResult().run(diff.compare, fa, fb, iterations=10)
            print(t.summary("GeometryDiff.compare (medium)"))
            assert t.p95 < 2.0, f"Medium diff p95={t.p95:.4f}s exceeds 2 s"


# ====================================================================
# PointNet Preprocessor
# ====================================================================

class TestPointNetPreprocessorPerformance:
    """Benchmark PointCloudPreprocessor.normalize() -- target <5 ms for 2048 pts."""

    def test_normalize_latency(self):
        """Normalize 2048 points under 5 ms."""
        from src.ml.pointnet.preprocessor import PointCloudPreprocessor

        pp = PointCloudPreprocessor()
        points = np.random.randn(2048, 3).astype(np.float32)

        t = TimingResult().run(pp.normalize, points, iterations=500)
        print(t.summary("PointCloudPreprocessor.normalize (2048 pts)"))
        assert t.p95 < 0.005, (
            f"Normalize p95={t.p95:.4f}s exceeds 5 ms"
        )

    def test_normalize_large_cloud_latency(self):
        """Normalize 10 000 points under 10 ms."""
        from src.ml.pointnet.preprocessor import PointCloudPreprocessor

        pp = PointCloudPreprocessor()
        points = np.random.randn(10_000, 3).astype(np.float32)

        t = TimingResult().run(pp.normalize, points, iterations=200)
        print(t.summary("PointCloudPreprocessor.normalize (10k pts)"))
        assert t.p95 < 0.01, (
            f"Normalize (10k) p95={t.p95:.4f}s exceeds 10 ms"
        )

    def test_augment_latency(self):
        """Augment 2048 points under 5 ms."""
        from src.ml.pointnet.preprocessor import PointCloudPreprocessor

        pp = PointCloudPreprocessor()
        points = np.random.randn(2048, 3).astype(np.float32)

        t = TimingResult().run(pp.augment, points, iterations=500)
        print(t.summary("PointCloudPreprocessor.augment (2048 pts)"))
        assert t.p95 < 0.005, (
            f"Augment p95={t.p95:.4f}s exceeds 5 ms"
        )


# ====================================================================
# Function Calling Engine (offline mode)
# ====================================================================

class TestFunctionCallingEnginePerformance:
    """Benchmark FunctionCallingEngine initialisation in offline mode."""

    def test_offline_init_latency(self):
        """Engine init (offline) under 100 ms."""
        from src.core.assistant.function_calling import FunctionCallingEngine

        def _init():
            FunctionCallingEngine(llm_provider="offline")

        t = TimingResult().run(_init, iterations=100)
        print(t.summary("FunctionCallingEngine.__init__ (offline)"))
        assert t.p95 < 0.1, (
            f"Offline init p95={t.p95:.4f}s exceeds 100 ms"
        )

    def test_system_prompt_generation_latency(self):
        """System prompt generation under 5 ms."""
        from src.core.assistant.function_calling import FunctionCallingEngine

        engine = FunctionCallingEngine(llm_provider="offline")

        t = TimingResult().run(engine.get_system_prompt, iterations=500)
        print(t.summary("FunctionCallingEngine.get_system_prompt"))
        assert t.p95 < 0.005, (
            f"System prompt gen p95={t.p95:.4f}s exceeds 5 ms"
        )
